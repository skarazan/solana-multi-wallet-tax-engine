"""Inter-wallet transfer detection and FIFO lot migration."""

from typing import Dict, List, Set, Tuple

from config import CASH_MINTS, LST_MINTS, SOL_MINT, STABLE_MINTS
from fifo import fifo_transfer
from helius import parse_token_amount
from models import Lot, Transfer, sf, ts_to_date
from prices import PriceService


def detect_inter_wallet(
    all_txs: Dict[str, List[dict]],
    price_service: PriceService,
    my_wallets: Set[str],
) -> Tuple[Set[str], List[Transfer]]:
    """
    Scan all wallets' transactions for direct wallet-to-wallet movements.
    Skips swap transactions (DEX routing touches multiple accounts).

    Returns:
      - sigs: set of signatures to skip during PnL processing
      - transfers: list of Transfer objects for lot migration
    """
    # Build sig -> tx lookup
    sig_tx: Dict[str, dict] = {}
    for w, txs in all_txs.items():
        for tx in txs:
            s = tx.get("signature")
            if s and s not in sig_tx:
                sig_tx[s] = tx

    sigs: Set[str] = set()
    transfers: List[Transfer] = []

    checked_sigs: Set[str] = set()
    for w, txs in all_txs.items():
        for tx in txs:
            sig = tx.get("signature")
            if not sig or sig in checked_sigs:
                continue
            checked_sigs.add(sig)

            # Skip swaps — not inter-wallet transfers even if they
            # touch multiple wallet accounts during routing
            tx_type = tx.get("type", "")
            if tx_type == "SWAP":
                continue
            events = tx.get("events") or {}
            if events.get("swap"):
                continue
            src = tx.get("source", "")
            if src and any(p in src.upper() for p in ["JUPITER", "RAYDIUM", "ORCA"]):
                continue

            ts = tx.get("timestamp", 0)
            sol_usd = price_service.get_sol_usd(ts)
            found_transfer = False

            # Check token transfers for direct wallet -> wallet movement
            for tr in tx.get("tokenTransfers") or []:
                fa = tr.get("fromUserAccount", "")
                ta = tr.get("toUserAccount", "")
                mint = tr.get("mint", "")
                if fa in my_wallets and ta in my_wallets and fa != ta:
                    amt = parse_token_amount(tr)
                    if not amt or amt <= 0:
                        amt = sf(tr.get("tokenAmount", 0))
                    if amt <= 0:
                        continue
                    val = amt * price_service.get_cash_mint_price(mint, sol_usd)
                    transfers.append(Transfer(
                        ts_to_date(ts), fa, ta, mint, amt, val, sig, ts=ts,
                    ))
                    found_transfer = True

            # Check native SOL transfers for direct wallet -> wallet.
            # Don't restrict to TRANSFER/SOL_TRANSFER types — SOL sends
            # can have various type labels. Check nativeBalanceChange
            # across all owned wallets for any non-swap transaction.
            if not found_transfer:
                sender = None
                receiver = None
                sol_amt = 0.0
                for a in tx.get("accountData") or []:
                    acct = a.get("account", "")
                    ch = a.get("nativeBalanceChange", 0)
                    if acct in my_wallets:
                        if ch < -1000:  # negative = SOL left (in lamports)
                            sender = acct
                            sol_amt = abs(ch) / 1e9
                        elif ch > 1000:  # positive = SOL arrived
                            receiver = acct
                if sender and receiver and sender != receiver and sol_amt > 0.001:
                    transfers.append(Transfer(
                        ts_to_date(ts), sender, receiver, SOL_MINT,
                        sol_amt, sol_amt * sol_usd, sig, ts=ts,
                    ))
                    found_transfer = True

            if found_transfer:
                sigs.add(sig)

    # Dedup
    seen: Set[Tuple[str, str, str]] = set()
    uniq: List[Transfer] = []
    for t in transfers:
        k = (t.sig, t.mint, t.from_w)
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    return sigs, uniq


def apply_inter_wallet_transfers(
    transfers: List[Transfer],
    all_lots: Dict[str, Dict[str, List[Lot]]],
):
    """
    Process inter-wallet transfers in chronological order,
    migrating FIFO lots from source wallet to destination wallet.

    Lots carry over with their original cost basis and acquisition
    timestamps — no taxable event is created.

    Bug #10 fix: The original code detected transfers but never
    actually moved the lots between wallets.
    """
    sorted_transfers = sorted(transfers, key=lambda t: t.ts)
    for t in sorted_transfers:
        from_lots = all_lots.get(t.from_w, {}).setdefault(t.mint, [])
        to_lots = all_lots.get(t.to_w, {}).setdefault(t.mint, [])
        fifo_transfer(from_lots, to_lots, t.amount, t.to_w)
