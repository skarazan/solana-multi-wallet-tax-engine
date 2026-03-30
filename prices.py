"""Price service — SOL daily prices, token FMV derivation, current price lookups."""

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import requests

from config import (
    CASH_MINTS,
    HELIUS_API_KEY,
    LST_MINTS,
    SOL_MINT,
    STABLE_MINTS,
)
from helius import _h_post, parse_token_amount
from models import sf, ts_to_date


# =========================
# Standalone price fetchers
# =========================

def fetch_kraken_daily(start_ts: int, end_ts: int) -> Dict[str, float]:
    r = requests.get(
        "https://api.kraken.com/0/public/OHLC",
        params={"pair": "SOLUSD", "interval": 1440, "since": start_ts - 86400},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(f"Kraken: {data['error']}")
    result = data.get("result", {})
    pk = next((k for k in result if k != "last"), None)
    if not pk:
        raise RuntimeError("No Kraken data")
    out = {}
    for row in result[pk]:
        c = sf(row[4])
        if c > 0:
            out[ts_to_date(int(row[0]))] = c
    return out


def extract_exact_sol_prices(txs: List[dict]) -> Dict[int, float]:
    out = {}
    for tx in txs:
        ts = tx.get("timestamp")
        if not ts:
            continue
        swap = (tx.get("events") or {}).get("swap")
        if not swap:
            continue
        ni = _native_to_sol(swap.get("nativeInput"))
        no = _native_to_sol(swap.get("nativeOutput"))
        si = so = 0.0
        for ti in swap.get("tokenInputs") or []:
            if ti.get("mint") in STABLE_MINTS:
                a = parse_token_amount(ti.get("rawTokenAmount"))
                if a:
                    si += a
        for to in swap.get("tokenOutputs") or []:
            if to.get("mint") in STABLE_MINTS:
                a = parse_token_amount(to.get("rawTokenAmount"))
                if a:
                    so += a
        p = None
        if ni >= 0.01 and so > 0:
            p = so / ni
        elif si > 0 and no >= 0.01:
            p = si / no
        if p and 5 <= p <= 500:
            out[ts] = p
    return out


def _native_to_sol(obj) -> float:
    if not isinstance(obj, dict):
        return 0.0
    a = obj.get("amount")
    if a is None:
        return 0.0
    try:
        f = float(a)
    except Exception:
        return 0.0
    return f / 1e9 if f >= 1000 else f


def _fetch_dex_price(mint: str) -> float:
    try:
        r = requests.get(
            f"https://api.dexscreener.com/latest/dex/tokens/{mint}", timeout=10
        )
        if r.status_code != 200:
            return 0.0
        pairs = r.json().get("pairs", [])
        if not pairs:
            return 0.0
        best = max(pairs, key=lambda p: sf(p.get("liquidity", {}).get("usd"), 0))
        return sf(best.get("priceUsd"), 0)
    except Exception:
        return 0.0


# =========================
# PriceService
# =========================

class PriceService:
    def __init__(self, daily_sol: Dict[str, float], exact_sol: Dict[int, float]):
        self.daily_sol = daily_sol
        self.exact_sol = exact_sol
        self.token_price_cache: Dict[Tuple[str, str], float] = {}  # (mint, date) -> usd
        self.session = requests.Session()

    def get_sol_usd(self, ts: int) -> float:
        if ts in self.exact_sol:
            return self.exact_sol[ts]
        d = ts_to_date(ts)
        if d in self.daily_sol:
            return self.daily_sol[d]
        for i in range(1, 8):
            dd = ts_to_date(ts - i * 86400)
            if dd in self.daily_sol:
                return self.daily_sol[dd]
        return list(self.daily_sol.values())[-1] if self.daily_sol else 0.0

    def get_cash_mint_price(self, mint: str, sol_usd: float) -> float:
        """Price of a cash-like mint in USD."""
        if mint in STABLE_MINTS:
            return 1.0
        if mint == SOL_MINT:
            return sol_usd
        if mint in LST_MINTS:
            # Try cached LST price; fall back to SOL price
            d = ts_to_date(0)  # current
            cached = self.token_price_cache.get((mint, d))
            if cached and cached > 0:
                return cached
            return sol_usd  # fallback
        return 0.0

    def cache_derived_price(self, mint: str, ts: int, price_per_unit: float):
        """Cache a derived per-unit price with sanity bounds."""
        if not (price_per_unit > 0):
            return
        # Sanity: reject obviously wrong prices.
        # No single Solana token has ever been worth > $100k/unit;
        # anything above that is almost certainly a calculation error.
        if price_per_unit > 100_000:
            return
        d = ts_to_date(ts)
        # If we already have a cached price for this (mint, date),
        # keep it only if the new one is in the same order of magnitude.
        # This prevents a single bad swap from overwriting good data.
        existing = self.token_price_cache.get((mint, d))
        if existing and existing > 0:
            ratio = price_per_unit / existing
            if ratio > 100 or ratio < 0.01:
                return  # too far off from existing — likely bad data
        self.token_price_cache[(mint, d)] = price_per_unit

    def lookup_token_price_at(self, mint: str, ts: int) -> float:
        """Try to find a cached price for a token near a timestamp."""
        d = ts_to_date(ts)
        cached = self.token_price_cache.get((mint, d))
        if cached and cached > 0:
            return cached
        # Check nearby days (only 1 day out to limit stale price propagation)
        for offset in range(1, 2):
            for dd in [ts_to_date(ts - offset * 86400), ts_to_date(ts + offset * 86400)]:
                cached = self.token_price_cache.get((mint, dd))
                if cached and cached > 0:
                    return cached
        return 0.0

    def derive_fmv_from_swap(
        self,
        outflows: List[Tuple[str, float]],
        inflows: List[Tuple[str, float]],
        sol_usd: float,
        ts: int,
    ) -> Dict[str, float]:
        """
        Derive FMV for all tokens in a swap using conservation of value:
        total outflow USD == total inflow USD.

        Returns dict mapping mint -> total FMV in USD for that mint's
        quantity in this transaction.

        Strategy:
        1. Compute known USD on each side (cash mints).
        2. Try to price unknowns from the cached price history.
        3. Use the conservation equation to fill in the remaining unknowns.
        4. If both sides are fully unknown, return 0 (no inflation).
        """
        result: Dict[str, float] = {}

        cash_outflow_usd = 0.0
        cash_inflow_usd = 0.0
        non_cash_outflows: List[Tuple[str, float]] = []
        non_cash_inflows: List[Tuple[str, float]] = []

        for m, q in outflows:
            if m in CASH_MINTS:
                cash_outflow_usd += q * self.get_cash_mint_price(m, sol_usd)
            else:
                non_cash_outflows.append((m, q))

        for m, q in inflows:
            if m in CASH_MINTS:
                cash_inflow_usd += q * self.get_cash_mint_price(m, sol_usd)
            else:
                non_cash_inflows.append((m, q))

        # Try to price each unknown token from cache
        out_known: Dict[str, float] = {}  # mint -> total FMV
        out_unknown: List[Tuple[str, float]] = []
        for m, q in non_cash_outflows:
            p = self.lookup_token_price_at(m, ts)
            if p > 0:
                out_known[m] = p * q
            else:
                out_unknown.append((m, q))

        in_known: Dict[str, float] = {}
        in_unknown: List[Tuple[str, float]] = []
        for m, q in non_cash_inflows:
            p = self.lookup_token_price_at(m, ts)
            if p > 0:
                in_known[m] = p * q
            else:
                in_unknown.append((m, q))

        # Total known value on each side
        known_outflow = cash_outflow_usd + sum(out_known.values())
        known_inflow = cash_inflow_usd + sum(in_known.values())

        # Conservation: total_outflow == total_inflow
        # Use whichever side has MORE known value as the anchor.

        # Case 1: Outflow side has unknowns, inflow side is fully known
        #   => unknown outflow FMV = known_inflow - (cash_outflow + known_outflow_tokens)
        # Case 2: Inflow side has unknowns, outflow side is fully known
        #   => unknown inflow FMV = known_outflow - (cash_inflow + known_inflow_tokens)
        # Case 3: Both sides have unknowns
        #   => Can't solve without external price — return 0 for unknowns

        # Populate result with what we already know
        result.update(out_known)
        result.update(in_known)

        if out_unknown and not in_unknown:
            # All inflows are known — derive unknown outflows
            residual = known_inflow - cash_outflow_usd - sum(out_known.values())
            if residual > 0:
                self._distribute_fmv(out_unknown, residual, ts, result)
        elif in_unknown and not out_unknown:
            # All outflows are known — derive unknown inflows
            residual = known_outflow - cash_inflow_usd - sum(in_known.values())
            if residual > 0:
                self._distribute_fmv(in_unknown, residual, ts, result)
        elif out_unknown and in_unknown:
            # Both sides have unknowns — try using the known side as anchor
            # If one side has at least some cash, use that
            if known_outflow > known_inflow and known_outflow > 0:
                residual = known_outflow - cash_inflow_usd - sum(in_known.values())
                if residual > 0:
                    self._distribute_fmv(in_unknown, residual, ts, result)
                # Outflow unknowns: derive from inflow total
                total_inflow_now = cash_inflow_usd + sum(
                    result.get(m, 0) for m, _ in non_cash_inflows
                )
                out_residual = total_inflow_now - cash_outflow_usd - sum(out_known.values())
                if out_residual > 0:
                    self._distribute_fmv(out_unknown, out_residual, ts, result)
            elif known_inflow > 0:
                residual = known_inflow - cash_outflow_usd - sum(out_known.values())
                if residual > 0:
                    self._distribute_fmv(out_unknown, residual, ts, result)
                total_outflow_now = cash_outflow_usd + sum(
                    result.get(m, 0) for m, _ in non_cash_outflows
                )
                in_residual = total_outflow_now - cash_inflow_usd - sum(in_known.values())
                if in_residual > 0:
                    self._distribute_fmv(in_unknown, in_residual, ts, result)
            # else: both sides totally unknown — leave FMV as 0 (safe)

        return result

    def _distribute_fmv(
        self,
        tokens: List[Tuple[str, float]],
        total_fmv: float,
        ts: int,
        result: Dict[str, float],
    ):
        """Distribute a total FMV across one or more unknown tokens."""
        if len(tokens) == 1:
            m, q = tokens[0]
            result[m] = total_fmv
            if q > 0:
                self.cache_derived_price(m, ts, total_fmv / q)
        else:
            # Equal split (we have no basis to weight differently)
            each = total_fmv / len(tokens)
            for m, q in tokens:
                result[m] = each
                if q > 0:
                    self.cache_derived_price(m, ts, each / q)

    def fetch_current_prices(self, mints: List[str]) -> Dict[str, float]:
        """Fetch current prices for unrealized PnL calculation."""
        prices = {}
        print(f"   Fetching prices for {len(mints)} tokens...")
        for i, m in enumerate(mints, 1):
            if m in STABLE_MINTS:
                prices[m] = 1.0
                continue
            if m == SOL_MINT or m in LST_MINTS:
                try:
                    r = self.session.get(
                        "https://api.kraken.com/0/public/Ticker",
                        params={"pair": "SOLUSD"},
                        timeout=10,
                    )
                    d = r.json()
                    if not d.get("error"):
                        rk = list(d["result"].keys())[0]
                        sol_price = sf(d["result"][rk]["c"][0])
                        if m == SOL_MINT:
                            prices[m] = sol_price
                        else:
                            # For LSTs, try DexScreener first for actual price
                            lst_price = _fetch_dex_price(m)
                            prices[m] = lst_price if lst_price > 0 else sol_price
                        continue
                except Exception:
                    pass

            p = _fetch_dex_price(m)
            if p > 0:
                prices[m] = p
            else:
                try:
                    r = self.session.get(
                        f"https://price.jup.ag/v4/price?ids={m}", timeout=10
                    )
                    if r.status_code == 200:
                        jd = r.json()
                        if "data" in jd and m in jd["data"]:
                            jp = sf(jd["data"][m].get("price"))
                            if jp > 0:
                                prices[m] = jp
                                continue
                except Exception:
                    pass
                try:
                    payload = {
                        "jsonrpc": "2.0", "id": "p", "method": "getAsset",
                        "params": {"id": m},
                    }
                    r = _h_post(
                        f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}",
                        payload,
                        self.session,
                    )
                    if "result" in r:
                        hp = sf(
                            r["result"]
                            .get("token_info", {})
                            .get("price_info", {})
                            .get("price_per_token"),
                            0,
                        )
                        if hp > 0:
                            prices[m] = hp
                            continue
                except Exception:
                    pass
                prices[m] = 0.0

            if i % 20 == 0:
                print(f"      {i}/{len(mints)}")
            time.sleep(0.03)
        return prices
