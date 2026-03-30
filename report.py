"""CSV report generation."""

import csv
from datetime import datetime, timezone
from typing import Dict, List

from config import CASH_MINTS, SOL_MINT, START_DATE_UTC, END_DATE_UTC, WALLETS
from models import (
    IncomeEvent,
    Lot,
    Position,
    Sale,
    Transfer,
    Warning,
    ts_to_date,
    utc_ts,
)


def write_csv_report(
    all_sales: Dict[str, List[Sale]],
    all_unrealized: Dict[str, List[Position]],
    all_warnings: Dict[str, List[Warning]],
    all_income: Dict[str, List[IncomeEvent]],
    transfers: List[Transfer],
    wallet_names: Dict[str, str],
    all_deposits: Dict[str, List[dict]],
    all_withdrawals: Dict[str, List[dict]],
    all_beginning_bal: Dict[str, float],
    all_ending_bal: Dict[str, float],
    filename: str,
    start_ts: int,
    end_ts: int,
):
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)

        # Compute totals
        tb = sum(all_beginning_bal.values())
        te = sum(all_ending_bal.values())
        td = sum(
            sum(d.get("value_usd", 0) for d in dl)
            for dl in all_deposits.values()
        )
        tw = sum(
            sum(d.get("value_usd", 0) for d in dl)
            for dl in all_withdrawals.values()
        )
        tr = sum(sum(s.pnl for s in sl) for sl in all_sales.values())
        tu = sum(sum(p.pnl for p in pl) for pl in all_unrealized.values())
        ti = sum(
            sum(ie.fmv_usd for ie in il)
            for il in all_income.values()
        )
        nt = sum(len(sl) for sl in all_sales.values())

        # Header
        w.writerow(["SOLANA TAX REPORT v5"])
        w.writerow([
            "Generated:",
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        ])
        w.writerow(["Period:", f"{START_DATE_UTC} to {END_DATE_UTC}"])
        w.writerow([])

        # ===== TOTAL P&L =====
        w.writerow(["*** TOTAL P&L ***", f"${tr + tu:,.2f}"])
        w.writerow(["  Realized P&L:", f"${tr:,.2f}"])
        w.writerow(["  Unrealized P&L:", f"${tu:,.2f}"])
        w.writerow(["  Total Gains (realized):", f"${sum(sum(s.pnl for s in sl if s.pnl > 0) for sl in all_sales.values()):,.2f}"])
        w.writerow(["  Total Losses (realized):", f"${sum(sum(s.pnl for s in sl if s.pnl < 0) for sl in all_sales.values()):,.2f}"])
        if ti > 0:
            w.writerow(["  Ordinary Income (airdrops):", f"${ti:,.2f}"])
        w.writerow([])

        # ===== SUMMARY =====
        w.writerow(["SUMMARY"])
        w.writerow(["Beginning Balance (cost basis):", f"${tb:,.2f}"])
        w.writerow(["+ Deposits:", f"${td:,.2f}"])
        w.writerow(["- Withdrawals:", f"${tw:,.2f}"])
        w.writerow(["Realized P&L:", f"${tr:,.2f}"])
        w.writerow(["Ending Balance (market):", f"${te:,.2f}"])
        w.writerow(["Unrealized P&L:", f"${tu:,.2f}"])
        w.writerow(["Total Trades:", nt])
        w.writerow([])

        # ===== PER-WALLET =====
        w.writerow(["PER-WALLET"])
        w.writerow(["#", "Name", "Addr", "Beg", "Deps", "Wds",
                     "Realized", "End", "#Trades"])
        for i, wlt in enumerate(WALLETS, 1):
            nm = wallet_names.get(wlt, wlt[:8] + "...")
            dep = sum(d.get("value_usd", 0) for d in all_deposits.get(wlt, []))
            wd_ = sum(d.get("value_usd", 0) for d in all_withdrawals.get(wlt, []))
            rl = sum(s.pnl for s in all_sales.get(wlt, []))
            w.writerow([
                i, nm, wlt[:12] + "...",
                f"${all_beginning_bal.get(wlt, 0):,.2f}",
                f"${dep:,.2f}", f"${wd_:,.2f}", f"${rl:,.2f}",
                f"${all_ending_bal.get(wlt, 0):,.2f}",
                len(all_sales.get(wlt, [])),
            ])
        w.writerow([])

        # ===== AGGREGATED P&L BY TOKEN =====
        w.writerow(["REALIZED P&L BY TOKEN (sorted by |P&L|)"])
        w.writerow(["Wallet", "Token", "#Sells", "Total Qty Sold",
                     "Total Proceeds", "Total Cost Basis", "Total P&L",
                     "First Buy", "Last Sell", "Missing Lots?"])
        agg: Dict[tuple, dict] = {}
        for wlt, sl in all_sales.items():
            nm = wallet_names.get(wlt, wlt[:8] + "...")
            for s in sl:
                key = (nm, s.mint, s.label)
                if key not in agg:
                    agg[key] = {
                        "sells": 0, "qty": 0, "proceeds": 0, "cost": 0,
                        "pnl": 0, "first_buy": s.date_acquired,
                        "last_sell": s.date_sold, "missing": False,
                    }
                a = agg[key]
                a["sells"] += 1
                a["qty"] += s.qty
                a["proceeds"] += s.proceeds
                a["cost"] += s.cost_basis
                a["pnl"] += s.pnl
                if s.has_missing_lots:
                    a["missing"] = True
                if s.date_acquired < a["first_buy"]:
                    a["first_buy"] = s.date_acquired
                if s.date_sold > a["last_sell"]:
                    a["last_sell"] = s.date_sold

        agg_list = sorted(agg.items(), key=lambda x: abs(x[1]["pnl"]), reverse=True)
        for (nm, mint, label), a in agg_list:
            w.writerow([
                nm, label, a["sells"], f"{a['qty']:.6f}",
                f"${a['proceeds']:,.2f}", f"${a['cost']:,.2f}",
                f"${a['pnl']:,.2f}", a["first_buy"], a["last_sell"],
                "YES" if a["missing"] else "",
            ])
        w.writerow([])

        # ===== DETAILED TRADES =====
        w.writerow(["DETAILED TRADES (sorted by |gain/loss|)"])
        w.writerow(["Wallet", "Type", "Token", "Acquired", "Sold",
                     "Proceeds", "Cost Basis", "Gain/Loss", "Quantity",
                     "Sig", "Missing Lots?"])
        at = []
        for wlt, sl in all_sales.items():
            for s in sl:
                at.append((wallet_names.get(wlt, wlt[:8] + "..."), s))
        at.sort(key=lambda x: abs(x[1].pnl), reverse=True)
        for nm, s in at:
            w.writerow([
                nm, s.tx_type, s.label, s.date_acquired, s.date_sold,
                f"${s.proceeds:,.2f}", f"${s.cost_basis:,.2f}",
                f"${s.pnl:,.2f}", f"{s.qty:.6f}", s.sig[:8],
                "YES" if s.has_missing_lots else "",
            ])
        w.writerow([])

        # ===== INCOME EVENTS (Airdrops) =====
        all_ie = []
        for wlt, il in all_income.items():
            for ie in il:
                all_ie.append((wallet_names.get(wlt, wlt[:8] + "..."), ie))
        if all_ie:
            w.writerow(["INCOME EVENTS (Airdrops, Staking Rewards)"])
            w.writerow(["Wallet", "Type", "Token", "Date", "Quantity",
                         "FMV (USD)"])
            all_ie.sort(key=lambda x: x[1].ts)
            for nm, ie in all_ie:
                label = "SOL" if ie.mint == SOL_MINT else ie.mint[:16] + "..."
                w.writerow([
                    nm, ie.income_type, label,
                    ts_to_date(ie.ts), f"{ie.qty:.6f}",
                    f"${ie.fmv_usd:,.2f}",
                ])
            w.writerow([])

        # ===== HOLDINGS =====
        w.writerow(["HOLDINGS (sorted by |unrealized P&L|)"])
        w.writerow(["Wallet", "Token", "Qty", "Cost Basis", "Price",
                     "Value", "Unrealized P&L"])
        ap = []
        for wlt, pl in all_unrealized.items():
            for p in pl:
                ap.append((wallet_names.get(wlt, wlt[:8] + "..."), p))
        ap.sort(key=lambda x: abs(x[1].pnl), reverse=True)
        for nm, p in ap:
            w.writerow([
                nm, p.mint[:16] + "...", f"{p.qty:.6f}",
                f"${p.cost_basis:,.2f}", f"${p.price:.8f}",
                f"${p.value:,.2f}", f"${p.pnl:,.2f}",
            ])
        w.writerow([])

        # ===== INTER-WALLET TRANSFERS =====
        pt = [
            t for t in transfers
            if utc_ts(t.date) >= start_ts and utc_ts(t.date) <= end_ts
        ]
        if pt:
            w.writerow(["INTER-WALLET TRANSFERS (period only)"])
            w.writerow(["Date", "From", "To", "Token", "Amount", "Value", "Sig"])
            for t in sorted(pt, key=lambda x: x.date):
                fn = wallet_names.get(t.from_w, t.from_w[:8] + "...")
                tn = wallet_names.get(t.to_w, t.to_w[:8] + "...")
                ml = "SOL" if t.mint == SOL_MINT else t.mint[:12] + "..."
                w.writerow([
                    t.date, fn, tn, ml, f"{t.amount:.6f}",
                    f"${t.value_usd:,.2f}", t.sig[:8],
                ])
            w.writerow([])

        # ===== WARNINGS =====
        all_w = []
        for wlt, wl in all_warnings.items():
            for wr in wl:
                all_w.append((wallet_names.get(wlt, wlt[:8] + "..."), wr))
        if all_w:
            w.writerow(["WARNINGS"])
            w.writerow(["Wallet", "Category", "Token", "Qty", "Date",
                         "Sig", "Note"])
            for nm, wr in all_w[:200]:
                w.writerow([
                    nm, wr.category, wr.mint[:16],
                    f"{wr.quantity:.6f}", wr.date,
                    wr.sig[:8], wr.note,
                ])

        w.writerow([])
        w.writerow(["END OF REPORT"])
