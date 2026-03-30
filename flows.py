"""Transaction flow extraction — parse Helius tx into outflows/inflows."""

from typing import Optional

from config import SOL_MINT
from helius import parse_token_amount
from models import TransactionFlows


def extract_flows(tx: dict, wallet: str) -> Optional[TransactionFlows]:
    """
    Parse a Helius enhanced transaction into structured flows.

    outflows = tokens LEAVING the wallet (being spent/sold)
    inflows  = tokens ENTERING the wallet (being received/bought)

    Returns None if no meaningful flows detected.
    """
    ts = tx.get("timestamp")
    sig = tx.get("signature")
    src = tx.get("source", "")
    typ = tx.get("type", "")
    is_swap = bool((tx.get("events") or {}).get("swap"))

    # SOL via nativeBalanceChange
    sol_lam = 0
    for a in tx.get("accountData") or []:
        if a.get("account") == wallet:
            sol_lam = int(a.get("nativeBalanceChange") or 0)
            break
    fee_lam = int(tx.get("fee") or 0)
    sol_lam -= fee_lam  # net SOL change excluding fee
    sol_d = sol_lam / 1e9
    fee_sol = fee_lam / 1e9

    outflows = []  # tokens leaving the wallet
    inflows = []   # tokens entering the wallet

    for tr in tx.get("tokenTransfers") or []:
        mint = tr.get("mint")
        fu = tr.get("fromUserAccount")
        tu = tr.get("toUserAccount")
        if not mint or mint == SOL_MINT:
            continue
        amt = parse_token_amount(tr)
        if not amt or amt <= 0:
            continue
        if fu == wallet and tu != wallet:
            outflows.append((mint, amt))
        elif tu == wallet and fu != wallet:
            inflows.append((mint, amt))

    # When swap event exists, its quantities are AUTHORITATIVE.
    # tokenTransfers can have duplicate/wrong amounts due to multi-hop routing.
    swap = (tx.get("events") or {}).get("swap")
    if swap:
        swap_outflows = {}  # mint -> qty
        swap_inflows = {}
        for ti in swap.get("tokenInputs", []):
            mint = ti.get("mint")
            amt = parse_token_amount(ti.get("rawTokenAmount"))
            if mint and amt and amt > 0:
                swap_outflows[mint] = swap_outflows.get(mint, 0) + amt
        for to in swap.get("tokenOutputs", []):
            mint = to.get("mint")
            amt = parse_token_amount(to.get("rawTokenAmount"))
            if mint and amt and amt > 0:
                swap_inflows[mint] = swap_inflows.get(mint, 0) + amt

        if swap_outflows or swap_inflows:
            all_swap_mints = set(swap_outflows.keys()) | set(swap_inflows.keys())
            outflows = [(m, q) for m, q in outflows if m not in all_swap_mints]
            inflows = [(m, q) for m, q in inflows if m not in all_swap_mints]
            for mint, qty in swap_outflows.items():
                if mint != SOL_MINT:
                    outflows.append((mint, qty))
            for mint, qty in swap_inflows.items():
                if mint != SOL_MINT:
                    inflows.append((mint, qty))

    # SOL flows (positive = entering wallet, negative = leaving)
    if sol_d > 1e-9:
        inflows.append((SOL_MINT, sol_d))
    elif sol_d < -1e-9:
        outflows.append((SOL_MINT, abs(sol_d)))

    if not outflows and not inflows:
        return None

    return TransactionFlows(
        ts=ts, sig=sig, src=src, tx_type=typ, is_swap=is_swap,
        outflows=outflows, inflows=inflows, fee_sol=fee_sol,
    )
