"""
PnL computation engine — core tax logic with FIFO accounting.

Handles five transaction cases:
  A: Token purchase (cash -> tokens) — NOT taxable
  B: Token sale (tokens -> cash) — TAXABLE
  C: Token-to-token swap — TAXABLE (US rules: disposal at FMV)
  D: Cash-to-cash (SOL <-> stables, SOL <-> LST) — conditionally taxable
  E: Fallback (airdrops, unknown patterns)
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple

from config import (
    CASH_MINTS,
    DUST_USD,
    LST_MINTS,
    SOL_MINT,
    STABLE_MINTS,
    START_DATE_UTC,
)
from fifo import FIFOResult, fifo_sell
from flows import extract_flows
from models import (
    IncomeEvent,
    Lot,
    Sale,
    TransactionFlows,
    Warning,
    ts_to_date,
    utc_ts,
)
from prices import PriceService


class PnLEngine:
    def __init__(self, wallet: str, price_service: PriceService):
        self.wallet = wallet
        self.prices = price_service
        self.lots: Dict[str, List[Lot]] = {}
        self.sales: List[Sale] = []
        self.income_events: List[IncomeEvent] = []
        self.warnings: List[Warning] = []
        self.debug_counts: Dict[str, int] = defaultdict(int)

    def _get_lots(self, mint: str) -> List[Lot]:
        return self.lots.setdefault(mint, [])

    def _add_lot(self, mint: str, lot: Lot):
        self.lots.setdefault(mint, []).append(lot)

    def pre_seed_transfer_lots(self, transfer_lots: List[tuple]):
        """
        Pre-seed lots from inter-wallet transfers BEFORE running compute().
        Each entry is (mint, Lot). These are available during processing so
        the receiving wallet has cost basis for tokens sent from other wallets.
        """
        for mint, lot in sorted(transfer_lots, key=lambda x: x[1].acquired_ts):
            self._add_lot(mint, lot)

    def _warn(self, mint: str, qty: float, ts: int, sig: str,
              category: str, note: str):
        self.warnings.append(Warning(
            wallet=self.wallet, mint=mint, quantity=qty,
            date=ts_to_date(ts), sig=sig, category=category, note=note,
        ))

    # ------------------------------------------------------------------
    # Netting: compute net flows per mint to avoid phantom "change" lots
    # ------------------------------------------------------------------

    def _net_outflows(
        self,
        outflows: List[Tuple[str, float]],
        inflows: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """
        For each mint, if it appears on both sides, return only the net
        outflow. This prevents consuming FIFO lots for SOL that immediately
        comes back as "change" (Bug #6 fix).
        """
        out_by_mint: Dict[str, float] = defaultdict(float)
        in_by_mint: Dict[str, float] = defaultdict(float)
        for m, q in outflows:
            out_by_mint[m] += q
        for m, q in inflows:
            in_by_mint[m] += q
        net = []
        for m, total_out in out_by_mint.items():
            net_qty = total_out - in_by_mint.get(m, 0)
            if net_qty > 1e-12:
                net.append((m, net_qty))
        return net

    def _net_inflows(
        self,
        outflows: List[Tuple[str, float]],
        inflows: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """Net inflows — tokens entering that aren't canceled by outflows."""
        out_by_mint: Dict[str, float] = defaultdict(float)
        in_by_mint: Dict[str, float] = defaultdict(float)
        for m, q in outflows:
            out_by_mint[m] += q
        for m, q in inflows:
            in_by_mint[m] += q
        net = []
        for m, total_in in in_by_mint.items():
            net_qty = total_in - out_by_mint.get(m, 0)
            if net_qty > 1e-12:
                net.append((m, net_qty))
        return net

    # ------------------------------------------------------------------
    # Case A: Token Purchase (cash -> tokens, NOT taxable)
    # ------------------------------------------------------------------

    def _case_a_purchase(self, fl: TransactionFlows, sol_usd: float):
        """
        Spending cash (SOL/stables) to buy non-cash tokens.
        Fees ADD to cost basis of received tokens (correct for purchases).
        """
        self.debug_counts["A_purchase"] += 1
        fee_usd = fl.fee_sol * sol_usd

        # Net outflows to avoid consuming SOL that comes back as change
        net_out = self._net_outflows(fl.outflows, fl.inflows)

        total_cost = fee_usd
        for m, q in net_out:
            if m in CASH_MINTS:
                result = fifo_sell(self._get_lots(m), q)
                total_cost += result.cost
                if result.shortfall > 0:
                    # Missing cash lots — use FMV as cost
                    total_cost += result.shortfall * self.prices.get_cash_mint_price(m, sol_usd)

        # Create lots for received non-cash tokens
        net_in = self._net_inflows(fl.outflows, fl.inflows)
        received = [(m, q) for m, q in net_in if m not in CASH_MINTS]
        if received:
            total_qty = sum(q for _, q in received)
            if total_qty > 0:
                for m, q in received:
                    self._add_lot(m, Lot(
                        q, total_cost * (q / total_qty),
                        fl.ts, fl.sig, self.wallet,
                    ))

    # ------------------------------------------------------------------
    # Case B: Token Sale (tokens -> cash, TAXABLE)
    # ------------------------------------------------------------------

    def _case_b_sale(self, fl: TransactionFlows, sol_usd: float, start_ts: int):
        """
        Selling non-cash tokens for cash.
        Bug #2 fix: proceeds allocated by FMV, not cost basis.
        Bug #5 fix: fees reduce proceeds, not inflate cost basis.
        Bug #6 fix: net flows before consuming.
        """
        self.debug_counts["B_sale"] += 1
        fee_usd = fl.fee_sol * sol_usd

        # Net to avoid phantom change lots
        net_out = self._net_outflows(fl.outflows, fl.inflows)
        net_in = self._net_inflows(fl.outflows, fl.inflows)

        # Proceeds = cash received MINUS fees (Bug #5)
        cash_received = sum(
            q * self.prices.get_cash_mint_price(m, sol_usd)
            for m, q in net_in if m in CASH_MINTS
        )
        net_proceeds = cash_received - fee_usd
        if net_proceeds < 0:
            net_proceeds = 0

        # Consume cash outflows (complex multi-leg swaps may have cash on both sides)
        for m, q in net_out:
            if m in CASH_MINTS:
                fifo_sell(self._get_lots(m), q)

        # Dispose non-cash tokens via FIFO
        disposed = []
        for m, q in net_out:
            if m in CASH_MINTS:
                continue
            result = fifo_sell(self._get_lots(m), q)
            disposed.append((m, q, result.cost, result.earliest_acq_date,
                             result.shortfall > 0))
            if result.shortfall > 0:
                self._warn(m, result.shortfall, fl.ts, fl.sig,
                           "MISSING_LOTS", f"No lots for {result.shortfall:.6f} units")

        # Create lots for any non-cash tokens received (partial swap output)
        non_cash_in = [(m, q) for m, q in net_in if m not in CASH_MINTS]
        if non_cash_in:
            # Cost basis = share of net_proceeds (what we paid effectively)
            total_non_cash = len(non_cash_in)
            for m, q in non_cash_in:
                self._add_lot(m, Lot(
                    q, net_proceeds / total_non_cash,
                    fl.ts, fl.sig, self.wallet,
                ))

        # Record sales (only in reporting period)
        # Proceeds allocation: use FMV from a single derive call on the
        # REAL netted flows (not per-token with fake inflows).
        if fl.ts >= start_ts and disposed:
            # For single-token sales (the common case), all proceeds go to it.
            # For multi-token sales, derive FMV shares from the real flows.
            if len(disposed) == 1:
                m, q, cb, acq_date, has_missing = disposed[0]
                if not acq_date:
                    acq_date = ts_to_date(fl.ts)
                pnl = net_proceeds - cb
                label = m[:16] + "..." if len(m) > 16 else m
                self.sales.append(Sale(
                    self.wallet, m, label, q, acq_date, ts_to_date(fl.ts),
                    round(net_proceeds, 2), round(cb, 2), round(pnl, 2),
                    fl.sig, "SELL", has_missing,
                ))
            else:
                # Multiple tokens sold — derive FMV shares from real flows
                fmv_map = self.prices.derive_fmv_from_swap(
                    net_out, net_in, sol_usd, fl.ts)
                total_fmv = sum(
                    fmv_map.get(m, 0) for m, _, _, _, _ in disposed
                )
                for m, q, cb, acq_date, has_missing in disposed:
                    if not acq_date:
                        acq_date = ts_to_date(fl.ts)
                    if total_fmv > 0:
                        alloc = net_proceeds * (fmv_map.get(m, 0) / total_fmv)
                    else:
                        alloc = net_proceeds / len(disposed)
                    pnl = alloc - cb
                    label = m[:16] + "..." if len(m) > 16 else m
                    self.sales.append(Sale(
                        self.wallet, m, label, q, acq_date, ts_to_date(fl.ts),
                        round(alloc, 2), round(cb, 2), round(pnl, 2),
                        fl.sig, "SELL", has_missing,
                    ))

    # ------------------------------------------------------------------
    # Case C: Token-to-Token Swap (TAXABLE under US rules)
    # ------------------------------------------------------------------

    def _case_c_swap(self, fl: TransactionFlows, sol_usd: float, start_ts: int):
        """
        Swapping non-cash token A for non-cash token B.
        Bug #1 fix: This IS a taxable event — disposal of A at FMV.
        Bug #5 fix: Fees reduce disposal proceeds.
        """
        self.debug_counts["C_swap"] += 1
        fee_usd = fl.fee_sol * sol_usd

        net_out = self._net_outflows(fl.outflows, fl.inflows)
        net_in = self._net_inflows(fl.outflows, fl.inflows)

        # Derive FMV using NETTED flows so SOL on both sides cancels out
        fmv_map = self.prices.derive_fmv_from_swap(
            net_out, net_in, sol_usd, fl.ts)

        # Consume any cash outflows
        for m, q in net_out:
            if m in CASH_MINTS:
                fifo_sell(self._get_lots(m), q)

        # Dispose non-cash outflow tokens via FIFO (taxable)
        disposed = []
        total_disposal_fmv = 0.0
        for m, q in net_out:
            if m in CASH_MINTS:
                continue
            result = fifo_sell(self._get_lots(m), q)
            token_fmv = fmv_map.get(m, 0)
            total_disposal_fmv += token_fmv
            disposed.append((m, q, result.cost, result.earliest_acq_date,
                             token_fmv, result.shortfall > 0))
            if result.shortfall > 0:
                self._warn(m, result.shortfall, fl.ts, fl.sig,
                           "MISSING_LOTS", f"No lots for {result.shortfall:.6f} units")

        # Proceeds = FMV of disposed tokens minus fees (Bug #5)
        disposal_proceeds = total_disposal_fmv - fee_usd
        if disposal_proceeds < 0:
            disposal_proceeds = 0

        # Record sales for disposed tokens
        if fl.ts >= start_ts and disposed:
            for m, q, cb, acq_date, fmv, has_missing in disposed:
                if not acq_date:
                    acq_date = ts_to_date(fl.ts)
                if total_disposal_fmv > 0:
                    alloc = disposal_proceeds * (fmv / total_disposal_fmv)
                else:
                    alloc = 0
                pnl = alloc - cb
                label = m[:16] + "..." if len(m) > 16 else m
                self.sales.append(Sale(
                    self.wallet, m, label, q, acq_date, ts_to_date(fl.ts),
                    round(alloc, 2), round(cb, 2), round(pnl, 2),
                    fl.sig, "SWAP_DISPOSAL", has_missing,
                ))

        # Create lots for received tokens at FMV
        for m, q in net_in:
            if m in CASH_MINTS:
                continue
            token_fmv = fmv_map.get(m, 0)
            self._add_lot(m, Lot(q, token_fmv, fl.ts, fl.sig, self.wallet))

    # ------------------------------------------------------------------
    # Case D: Cash-to-Cash (SOL <-> stables, SOL <-> LST)
    # ------------------------------------------------------------------

    def _case_d_cash_swap(self, fl: TransactionFlows, sol_usd: float, start_ts: int):
        """
        SOL -> Stable: taxable (SOL disposal)
        Stable -> SOL: purchase (not taxable)
        SOL <-> LST: taxable (disposal of one for the other)
        Bug #5 fix: fees reduce proceeds on sales, add to cost on purchases.
        Bug #8 fix: LSTs priced independently where possible.
        """
        self.debug_counts["D_cash"] += 1
        fee_usd = fl.fee_sol * sol_usd

        outflow_mints = {m for m, _ in fl.outflows}
        inflow_mints = {m for m, _ in fl.inflows}
        net_out = self._net_outflows(fl.outflows, fl.inflows)
        net_in = self._net_inflows(fl.outflows, fl.inflows)

        has_sol_out = SOL_MINT in outflow_mints or bool(outflow_mints & LST_MINTS)
        has_stable_in = bool(inflow_mints & STABLE_MINTS)
        has_stable_out = bool(outflow_mints & STABLE_MINTS)
        has_sol_in = SOL_MINT in inflow_mints or bool(inflow_mints & LST_MINTS)

        if has_sol_out and has_stable_in:
            # SELLING SOL/LST for stables — TAXABLE
            proceeds = sum(q for m, q in net_in if m in STABLE_MINTS)
            # Bug #5: fees reduce proceeds
            net_proceeds = proceeds - fee_usd
            if net_proceeds < 0:
                net_proceeds = 0

            disposed = []
            for m, q in net_out:
                if m in STABLE_MINTS:
                    fifo_sell(self._get_lots(m), q)
                    continue
                result = fifo_sell(self._get_lots(m), q)
                disposed.append((m, q, result.cost, result.earliest_acq_date,
                                 result.shortfall > 0))
                if result.shortfall > 0:
                    self._warn(m, result.shortfall, fl.ts, fl.sig,
                               "MISSING_LOTS", f"Missing SOL/LST lots")

            if fl.ts >= start_ts and disposed:
                total_cost = sum(c for _, _, c, _, _ in disposed)
                for m, q, cb, acq, has_missing in disposed:
                    if not acq:
                        acq = ts_to_date(fl.ts)
                    # Allocate proportionally if multiple disposals
                    if total_cost > 0 and len(disposed) > 1:
                        alloc = net_proceeds * (cb / total_cost)
                    else:
                        alloc = net_proceeds
                    pnl = alloc - cb
                    label = "SOL" if m == SOL_MINT else m[:12] + "..."
                    self.sales.append(Sale(
                        self.wallet, m, label, q, acq, ts_to_date(fl.ts),
                        round(alloc, 2), round(cb, 2), round(pnl, 2),
                        fl.sig, "SOL_SALE", has_missing,
                    ))

        elif has_stable_out and has_sol_in:
            # BUYING SOL/LST with stables — NOT taxable
            total_cost = fee_usd  # fees add to cost basis on purchases
            for m, q in net_out:
                result = fifo_sell(self._get_lots(m), q)
                total_cost += result.cost

            for m, q in net_in:
                if m not in STABLE_MINTS:
                    self._add_lot(m, Lot(q, total_cost, fl.ts, fl.sig, self.wallet))

        else:
            # SOL <-> LST or other cash-cash — taxable disposal
            # The outflow asset is disposed at FMV of what's received
            disposed = []
            total_received_value = 0.0

            for m, q in net_in:
                val = q * self.prices.get_cash_mint_price(m, sol_usd)
                total_received_value += val

            net_proceeds = total_received_value - fee_usd
            if net_proceeds < 0:
                net_proceeds = 0

            for m, q in net_out:
                result = fifo_sell(self._get_lots(m), q)
                disposed.append((m, q, result.cost, result.earliest_acq_date,
                                 result.shortfall > 0))

            if fl.ts >= start_ts and disposed:
                total_cost = sum(c for _, _, c, _, _ in disposed)
                for m, q, cb, acq, has_missing in disposed:
                    if not acq:
                        acq = ts_to_date(fl.ts)
                    if total_cost > 0 and len(disposed) > 1:
                        alloc = net_proceeds * (cb / total_cost)
                    else:
                        alloc = net_proceeds
                    pnl = alloc - cb
                    label = "SOL" if m == SOL_MINT else m[:12] + "..."
                    self.sales.append(Sale(
                        self.wallet, m, label, q, acq, ts_to_date(fl.ts),
                        round(alloc, 2), round(cb, 2), round(pnl, 2),
                        fl.sig, "CASH_SWAP", has_missing,
                    ))

            # Create lots for received cash
            for m, q in net_in:
                val = q * self.prices.get_cash_mint_price(m, sol_usd)
                self._add_lot(m, Lot(q, val, fl.ts, fl.sig, self.wallet))

    # ------------------------------------------------------------------
    # Case E: Fallback (airdrops, transfers, unknown)
    # ------------------------------------------------------------------

    def _case_e_fallback(self, fl: TransactionFlows, sol_usd: float, start_ts: int):
        """
        Fallback for transfers, airdrops, and unknown patterns.

        Key distinction:
        - Cash inflows (SOL, stables, LSTs) are DEPOSITS, not income.
          They get lots at FMV but NO income event recorded.
        - Non-cash token inflows with no outflows may be airdrops.
          Only record income if we can verify FMV from cache.
        """
        self.debug_counts["E_fallback"] += 1

        net_out = self._net_outflows(fl.outflows, fl.inflows)
        net_in = self._net_inflows(fl.outflows, fl.inflows)

        has_non_cash_outflow = any(m not in CASH_MINTS for m, _ in net_out)

        # Separate cash inflows (deposits) from non-cash (potential airdrops)
        cash_received = [(m, q) for m, q in net_in if m in CASH_MINTS]
        token_received = [(m, q) for m, q in net_in if m not in CASH_MINTS]

        # --- Handle cash inflows as deposits (NOT income) ---
        for m, q in cash_received:
            fmv = q * self.prices.get_cash_mint_price(m, sol_usd)
            self._add_lot(m, Lot(q, fmv, fl.ts, fl.sig, self.wallet))

        # --- Consume any outflows ---
        total_cost = fl.fee_sol * sol_usd
        for m, q in net_out:
            result = fifo_sell(self._get_lots(m), q)
            total_cost += result.cost

        # --- Handle non-cash token inflows ---
        if token_received and not has_non_cash_outflow and not fl.is_swap:
            # Potential airdrop: tokens arrived, nothing meaningful sent.
            # Only record as income if we have a reliable FMV.
            for m, q in token_received:
                fmv = self.prices.lookup_token_price_at(m, fl.ts) * q
                if fmv <= 0:
                    # No price data — create lot at $0 (conservative).
                    # NOT recorded as income to avoid phantom income.
                    fmv = 0
                self._add_lot(m, Lot(q, fmv, fl.ts, "airdrop", self.wallet))
                if fmv > 0 and fl.ts >= start_ts:
                    self.income_events.append(IncomeEvent(
                        self.wallet, m, q, fmv, fl.ts, "AIRDROP"))
        elif token_received:
            # Unknown pattern with outflows — allocate outflow cost to inflows
            if total_cost > 0:
                total_qty = sum(q for _, q in token_received)
                if total_qty > 0:
                    for m, q in token_received:
                        cost_alloc = total_cost * (q / total_qty)
                        self._add_lot(m, Lot(q, cost_alloc, fl.ts, fl.sig, self.wallet))
            else:
                for m, q in token_received:
                    self._add_lot(m, Lot(q, 0, fl.ts, fl.sig, self.wallet))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute(
        self, txs: List[dict], inter_sigs: Set[str]
    ) -> Tuple[
        List[Sale], List[Warning], Dict[str, List[Lot]], float,
        Dict[str, int], List[IncomeEvent],
    ]:
        flows = []
        for tx in txs:
            fl = extract_flows(tx, self.wallet)
            if fl:
                flows.append(fl)
        flows.sort(key=lambda x: x.ts)

        start_ts = utc_ts(START_DATE_UTC)
        beg_bal = 0.0
        got_beg = False
        self.debug_counts["total"] = len(flows)

        for fl in flows:
            # Capture beginning balance at start of reporting period
            if not got_beg and fl.ts >= start_ts:
                beg_bal = sum(l.cost_usd for ll in self.lots.values() for l in ll)
                got_beg = True

            # Skip inter-wallet transfers (lots migrated separately)
            if fl.sig in inter_sigs:
                self.debug_counts["inter_skip"] += 1
                continue

            sol_usd = self.prices.get_sol_usd(fl.ts)

            # --- Dust filter ---
            total_val = 0.0
            has_token = False
            for m, q in fl.outflows + fl.inflows:
                if m in CASH_MINTS:
                    total_val += q * self.prices.get_cash_mint_price(m, sol_usd)
                else:
                    has_token = True
                    total_val += 999  # any real token = not dust
            if not has_token and total_val < DUST_USD:
                self.debug_counts["dust"] += 1
                continue

            # --- Classify ---
            outflow_mints = {m for m, _ in fl.outflows}
            inflow_mints = {m for m, _ in fl.inflows}
            cash_out = outflow_mints & CASH_MINTS    # cash leaving wallet
            cash_in = inflow_mints & CASH_MINTS      # cash entering wallet
            tok_out = outflow_mints - CASH_MINTS      # non-cash tokens leaving
            tok_in = inflow_mints - CASH_MINTS        # non-cash tokens entering

            if cash_out and tok_in and not tok_out:
                self._case_a_purchase(fl, sol_usd)
            elif tok_out and cash_in:
                self._case_b_sale(fl, sol_usd, start_ts)
            elif tok_out and tok_in and not cash_in:
                self._case_c_swap(fl, sol_usd, start_ts)
            elif cash_out and cash_in and not tok_out and not tok_in:
                self._case_d_cash_swap(fl, sol_usd, start_ts)
            else:
                self._case_e_fallback(fl, sol_usd, start_ts)

        if not got_beg:
            beg_bal = sum(l.cost_usd for ll in self.lots.values() for l in ll)

        return (
            self.sales, self.warnings, self.lots, beg_bal,
            dict(self.debug_counts), self.income_events,
        )
