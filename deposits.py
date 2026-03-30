"""Deposit and withdrawal detection for external flows."""

from typing import Dict, List, Set, Tuple

from config import SOL_MINT, STABLE_MINTS
from models import sf, ts_to_date
from prices import PriceService


def calc_deposits_withdrawals(
    txs: List[dict],
    wallet: str,
    price_service: PriceService,
    inter_sigs: Set[str],
    start_ts: int,
    end_ts: int,
) -> Tuple[float, float, List[dict], List[dict]]:
    """
    Detect external deposits and withdrawals for the reporting period.

    Returns:
      - total_deposits_usd
      - total_withdrawals_usd
      - deposit_list (list of dicts)
      - withdrawal_list (list of dicts)
    """
    dt = wt = 0.0
    dl: List[dict] = []
    wl: List[dict] = []

    for tx in sorted(txs, key=lambda x: x.get("timestamp", 0)):
        ts = tx.get("timestamp", 0)
        if not ts or ts < start_ts or ts > end_ts:
            continue
        sig = tx.get("signature", "")
        if sig in inter_sigs:
            continue
        if tx.get("type") == "SWAP":
            continue
        src = tx.get("source", "")
        if src and any(p in src.upper() for p in ["JUPITER", "RAYDIUM", "ORCA"]):
            continue
        ev = tx.get("events", {})
        if ev.get("swap") or ev.get("nft"):
            continue

        sol_usd = price_service.get_sol_usd(ts)
        tt = tx.get("type", "")

        # SOL transfers
        if tt in ["TRANSFER", "SOL_TRANSFER"]:
            if any(t.get("mint") == SOL_MINT for t in tx.get("tokenTransfers", [])):
                continue
            for ad in tx.get("accountData", []):
                if ad.get("account") == wallet:
                    ch = ad.get("nativeBalanceChange", 0) / 1e9
                    if abs(ch) < 0.001:
                        continue
                    v = abs(ch * sol_usd)
                    e = {
                        "date": ts_to_date(ts),
                        "amount": round(abs(ch), 9),
                        "value_usd": round(v, 2),
                        "sig": sig,
                    }
                    if ch > 0:
                        dt += v
                        e["type"] = "SOL Deposit"
                        dl.append(e)
                    else:
                        wt += v
                        e["type"] = "SOL Withdrawal"
                        wl.append(e)

        # Stablecoin transfers
        if tt == "TRANSFER":
            for tr in tx.get("tokenTransfers", []):
                m = tr.get("mint", "")
                if m not in STABLE_MINTS:
                    continue
                if len(tx.get("tokenTransfers", [])) > 1:
                    continue
                amt = sf(tr.get("tokenAmount", 0))
                if amt <= 0:
                    continue
                fa = tr.get("fromUserAccount", "")
                ta = tr.get("toUserAccount", "")
                e = {
                    "date": ts_to_date(ts),
                    "amount": round(amt, 2),
                    "value_usd": round(amt, 2),
                    "sig": sig,
                }
                if ta == wallet and fa != wallet:
                    dt += amt
                    e["type"] = "Stable Deposit"
                    dl.append(e)
                elif fa == wallet and ta != wallet:
                    wt += amt
                    e["type"] = "Stable Withdrawal"
                    wl.append(e)

    return dt, wt, dl, wl
