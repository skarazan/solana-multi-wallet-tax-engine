#!/usr/bin/env python3
"""
Solana DeFi PnL Calculator v5

Scans Solana wallet transactions via Helius, computes realized and
unrealized PnL using FIFO accounting, and generates a CSV tax report.

Usage:
    python main.py
"""

import base64
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Set

from config import (
    CACHE_DIR,
    CASH_MINTS,
    HELIUS_API_KEY,
    OUTPUT_DIR,
    SOL_MINT,
    START_DATE_UTC,
    END_DATE_UTC,
    USE_CACHE,
    WALLET_NAMES,
    WALLETS,
)
from deposits import calc_deposits_withdrawals
from helius import HeliusClient
from models import (
    IncomeEvent,
    Lot,
    Position,
    Sale,
    Warning,
    ensure_dir,
    sf,
    utc_ts,
)
from pnl import PnLEngine
from prices import PriceService, extract_exact_sol_prices, fetch_kraken_daily
from report import write_csv_report
from transfers import apply_inter_wallet_transfers, detect_inter_wallet


def _is_valid_solana_address(addr: str) -> bool:
    """Check if a string looks like a valid Solana base58 public key (32-44 chars, base58)."""
    if not isinstance(addr, str):
        return False
    addr = addr.strip()
    if not 32 <= len(addr) <= 44:
        return False
    base58_alphabet = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
    return all(c in base58_alphabet for c in addr)


def main():
    start_ts = utc_ts(START_DATE_UTC)
    end_ts = utc_ts(END_DATE_UTC, eod=True)
    ensure_dir(OUTPUT_DIR)

    print("=" * 60)
    print("SOLANA PnL CALCULATOR v5")
    print("=" * 60)
    for i, w in enumerate(WALLETS, 1):
        print(f"  {i}. {WALLET_NAMES.get(w, f'W{i}')}: {w}")
    print(f"Period: {START_DATE_UTC} -> {END_DATE_UTC}")
    print()

    if HELIUS_API_KEY == "insert your free helius rpc api":
        print("ERROR: Set HELIUS_API_KEY in config.py!")
        return

    # ------------------------------------------------------------------
    # 1. Fetch daily SOL prices from Kraken
    # ------------------------------------------------------------------
    print("1. SOL prices...")
    daily = fetch_kraken_daily(utc_ts("2020-01-01"), end_ts)
    print(f"   Done: {len(daily)} daily prices\n")

    # ------------------------------------------------------------------
    # 2. Validate wallets and fetch transaction history
    # ------------------------------------------------------------------
    print("2. Validating wallets & fetching tx history...")
    client = HeliusClient(HELIUS_API_KEY, CACHE_DIR, USE_CACHE)
    all_txs: Dict[str, List[dict]] = {}
    skipped_wallets: List[str] = []

    for i, w in enumerate(WALLETS, 1):
        nm = WALLET_NAMES.get(w, f"W{i}")

        # Validate address format before hitting the API
        if not _is_valid_solana_address(w):
            print(f"   [{i}/{len(WALLETS)}] {nm}: SKIPPED — "
                  f"invalid Solana address ({len(w)} chars: {w[:20]}...)")
            skipped_wallets.append(w)
            all_txs[w] = []
            continue

        print(f"   [{i}/{len(WALLETS)}] {nm}...")
        try:
            all_txs[w] = client.fetch_transactions(w, end_ts)
            print(f"      Done: {len(all_txs[w])} txs")
        except Exception as e:
            print(f"      ERROR fetching {nm}: {e}")
            print(f"      Skipping this wallet — continuing with others")
            skipped_wallets.append(w)
            all_txs[w] = []

    if skipped_wallets:
        print(f"\n   WARNING: {len(skipped_wallets)} wallet(s) skipped:")
        for sw in skipped_wallets:
            print(f"     - {WALLET_NAMES.get(sw, sw[:12] + '...')}: {sw}")
    print()

    # ------------------------------------------------------------------
    # 3. Extract exact SOL prices from swap events
    # ------------------------------------------------------------------
    print("3. Exact SOL prices from swaps...")
    exact: Dict[int, float] = {}
    for txs in all_txs.values():
        exact.update(extract_exact_sol_prices(txs))
    print(f"   Done: {len(exact)} exact prices\n")

    price_service = PriceService(daily, exact)

    # ------------------------------------------------------------------
    # 4. Detect inter-wallet transfers
    # ------------------------------------------------------------------
    print("4. Inter-wallet transfers...")
    my_wallets: Set[str] = set(WALLETS)
    inter_sigs, inter_transfers = detect_inter_wallet(
        all_txs, price_service, my_wallets
    )
    print(f"   Done: {len(inter_transfers)} transfers, {len(inter_sigs)} sigs\n")

    # ------------------------------------------------------------------
    # 5. Pre-seed inter-wallet transfer lots, then compute PnL
    # ------------------------------------------------------------------
    # Build pre-seed map: for each receiving wallet, create lots at FMV
    # from detected inter-wallet transfers. This ensures the receiving
    # wallet has cost basis for tokens sent from other owned wallets,
    # instead of treating them as $0-cost airdrops.
    print("5. Computing PnL...")
    preseed: Dict[str, List[tuple]] = {w: [] for w in WALLETS}
    for t in sorted(inter_transfers, key=lambda x: x.ts):
        sol_usd = price_service.get_sol_usd(t.ts)
        if t.mint in CASH_MINTS:
            cost = t.amount * price_service.get_cash_mint_price(t.mint, sol_usd)
        else:
            cost = t.value_usd  # use the detected transfer value
        preseed.setdefault(t.to_w, []).append((
            t.mint,
            Lot(t.amount, cost, t.ts, f"transfer_from:{t.from_w[:8]}", t.to_w),
        ))
    for w, seeds in preseed.items():
        if seeds:
            nm = WALLET_NAMES.get(w, w[:8] + "...")
            print(f"   Pre-seeding {len(seeds)} transfer lots into {nm}")

    a_sales: Dict[str, List[Sale]] = {}
    a_warnings: Dict[str, List[Warning]] = {}
    a_lots: Dict[str, Dict[str, List[Lot]]] = {}
    a_beg: Dict[str, float] = {}
    a_income: Dict[str, List[IncomeEvent]] = {}

    for w in WALLETS:
        nm = WALLET_NAMES.get(w, w[:8] + "...")
        engine = PnLEngine(w, price_service)
        # Pre-seed lots from transfers received by this wallet
        if preseed.get(w):
            engine.pre_seed_transfer_lots(preseed[w])
        sales, warnings, lots, beg_bal, dbg, income = engine.compute(
            all_txs.get(w, []), inter_sigs
        )
        a_sales[w] = sales
        a_warnings[w] = warnings
        a_lots[w] = lots
        a_beg[w] = beg_bal
        a_income[w] = income

        sol_s = sum(1 for s in sales if s.mint == SOL_MINT)
        tok_s = sum(1 for s in sales if s.mint not in CASH_MINTS)
        swap_s = sum(1 for s in sales if s.tx_type == "SWAP_DISPOSAL")
        total_pnl = sum(s.pnl for s in sales)
        print(f"   {nm}: {len(sales)} sales "
              f"({sol_s} SOL, {tok_s} token, {swap_s} swap disposals), "
              f"PnL=${total_pnl:,.2f}")
        print(f"      Debug: {dbg}")
        if income:
            print(f"      Income events: {len(income)} "
                  f"(${sum(ie.fmv_usd for ie in income):,.2f})")
    print()

    # ------------------------------------------------------------------
    # 6. Inter-wallet lot migration (already pre-seeded above)
    # ------------------------------------------------------------------
    print(f"6. Inter-wallet lots: {len(inter_transfers)} transfers pre-seeded\n")

    # ------------------------------------------------------------------
    # 7. Compute unrealized PnL
    # ------------------------------------------------------------------
    print("7. Unrealized positions...")
    a_unrealized: Dict[str, List[Position]] = {}
    all_mints: Set[str] = set()
    all_balances: Dict[str, Dict[str, float]] = {}

    for w in WALLETS:
        all_balances[w] = client.fetch_balances(w)
        time.sleep(0.05)

    for w, lot_map in a_lots.items():
        bl = all_balances.get(w, {})
        positions = []
        for m in set(lot_map.keys()) | set(bl.keys()):
            qty = bl.get(m, 0.0)
            cost = sum(l.cost_usd for l in lot_map.get(m, []))
            if qty <= 0 and cost <= 0:
                continue
            positions.append(Position(w, m, qty, cost, 0, 0, 0))
            all_mints.add(m)
        a_unrealized[w] = positions
    print(f"   Done: {sum(len(p) for p in a_unrealized.values())} positions\n")

    # ------------------------------------------------------------------
    # 8. Fetch current prices for unrealized PnL
    # ------------------------------------------------------------------
    print("8. Current prices...")
    prices = price_service.fetch_current_prices(list(all_mints))
    for w, pl in a_unrealized.items():
        for p in pl:
            pr = prices.get(p.mint, 0.0)
            p.price = pr
            p.value = p.qty * pr
            p.pnl = p.value - p.cost_basis
    print()

    # ------------------------------------------------------------------
    # 9. Deposits / Withdrawals
    # ------------------------------------------------------------------
    print("9. Deposits/Withdrawals...")
    a_deps: Dict[str, List[dict]] = {}
    a_wds: Dict[str, List[dict]] = {}
    a_end: Dict[str, float] = {}

    for w in WALLETS:
        d, wd, dl, wl = calc_deposits_withdrawals(
            all_txs[w], w, price_service, inter_sigs, start_ts, end_ts,
        )
        a_deps[w] = dl
        a_wds[w] = wl
        a_end[w] = sum(p.value for p in a_unrealized.get(w, []))
        nm = WALLET_NAMES.get(w, w[:8] + "...")
        print(f"   {nm}: +${d:,.2f} / -${wd:,.2f}")
    print()

    # ------------------------------------------------------------------
    # 10. Generate report
    # ------------------------------------------------------------------
    print("10. Report...")
    out = os.path.join(OUTPUT_DIR, "multi_wallet_tax_report.csv")
    write_csv_report(
        all_sales=a_sales,
        all_unrealized=a_unrealized,
        all_warnings=a_warnings,
        all_income=a_income,
        transfers=inter_transfers,
        wallet_names=WALLET_NAMES,
        all_deposits=a_deps,
        all_withdrawals=a_wds,
        all_beginning_bal=a_beg,
        all_ending_bal=a_end,
        filename=out,
        start_ts=start_ts,
        end_ts=end_ts,
    )
    print(f"   Done: {out}\n")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_realized = sum(sum(s.pnl for s in sl) for sl in a_sales.values())
    total_unrealized = sum(sum(p.pnl for p in pl) for pl in a_unrealized.values())
    total_income = sum(
        sum(ie.fmv_usd for ie in il) for il in a_income.values()
    )
    total_trades = sum(len(s) for s in a_sales.values())
    missing_lot_trades = sum(
        sum(1 for s in sl if s.has_missing_lots)
        for sl in a_sales.values()
    )

    print("=" * 60)
    print(f"*** TOTAL P&L: ${total_realized + total_unrealized:,.2f} ***")
    print(f"Realized:     ${total_realized:,.2f}")
    print(f"Unrealized:   ${total_unrealized:,.2f}")
    if total_income > 0:
        print(f"Income:       ${total_income:,.2f}")
    print(f"Trades:       {total_trades}")
    if missing_lot_trades > 0:
        print(f"  ⚠ {missing_lot_trades} trades had missing lot data "
              "(cost basis may be understated)")
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
