#!/usr/bin/env python3


import csv
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

import requests

# =========================
# CONFIG
# =========================

HELIUS_API_KEY = "insert your free helius rpc api"

WALLETS = [
"insert wallet address",

]

WALLET_NAMES = {
"insert the wallet address": "Main Wallet",

}

START_DATE_UTC = "2025-01-01" # insert start date for repro
END_DATE_UTC = "2026-02-03" # insert end date 

HELIUS_MIN_DELAY_SEC = 0.12
HELIUS_MAX_RETRIES = 8
HELIUS_PAGE_LIMIT = 100

OUTPUT_DIR = "./solana_tax_output_multi_wallet"
CACHE_DIR = "./cache_helius_2025"
USE_CACHE = True

USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
STABLE_MINTS = {USDC_MINT, USDT_MINT}
SOL_MINT = "So11111111111111111111111111111111111111112"

JITO_SOL = "J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn"
MSOL = "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So"
BSOL = "bSo13r4TkiE4KumL71LsHTPpL2euBYLFx6h9HP3piy1"
LST_MINTS = {JITO_SOL, MSOL, BSOL}

# "Cash" = SOL + stablecoins + LSTs. Selling TO these = taxable event.
CASH_MINTS = {SOL_MINT} | STABLE_MINTS | LST_MINTS

DUST_USD = 0.50

# =========================
# Helpers
# =========================

def utc_ts(s: str, eod: bool = False) -> int:
    dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if eod:
        dt = dt.replace(hour=23, minute=59, second=59)
    return int(dt.timestamp())

def ts_to_date(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def sf(x, d=0.0) -> float:
    try: return float(x)
    except: return d

# =========================
# Data
# =========================

@dataclass
class Lot:
    amount: float
    cost_usd: float
    acquired_ts: int
    via: str = ""
    wallet: str = ""

@dataclass
class Sale:
    wallet: str
    mint: str
    label: str
    qty: float
    date_acquired: str
    date_sold: str
    proceeds: float
    cost_basis: float
    pnl: float
    sig: str
    tx_type: str  # SELL_TO_SOL, SELL_TO_STABLE, SOL_TO_STABLE

@dataclass
class Position:
    wallet: str
    mint: str
    qty: float
    cost_basis: float
    price: float
    value: float
    pnl: float

@dataclass
class Transfer:
    date: str
    from_w: str
    to_w: str
    mint: str
    amount: float
    value_usd: float
    sig: str

# =========================
# Prices
# =========================

def fetch_kraken_daily(start_ts: int, end_ts: int) -> Dict[str, float]:
    r = requests.get("https://api.kraken.com/0/public/OHLC",
                     params={"pair": "SOLUSD", "interval": 1440, "since": start_ts - 86400},
                     timeout=30)
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

def get_sol_price(ts: int, daily: Dict[str, float], exact: Dict[int, float] = None) -> float:
    if exact and ts in exact:
        return exact[ts]
    d = ts_to_date(ts)
    if d in daily:
        return daily[d]
    for i in range(1, 8):
        dd = ts_to_date(ts - i * 86400)
        if dd in daily:
            return daily[dd]
    return list(daily.values())[-1] if daily else 0.0

def mint_price_at_swap(mint: str, sol_usd: float) -> float:
    """Price of a cash-like mint at swap time."""
    if mint == SOL_MINT or mint in LST_MINTS:
        return sol_usd
    if mint in STABLE_MINTS:
        return 1.0
    return 0.0

def native_to_sol(obj) -> float:
    if not isinstance(obj, dict):
        return 0.0
    a = obj.get("amount")
    if a is None:
        return 0.0
    try:
        f = float(a)
    except:
        return 0.0
    return f / 1e9 if f >= 1000 else f

def extract_exact_sol_prices(txs: List[dict]) -> Dict[int, float]:
    out = {}
    for tx in txs:
        ts = tx.get("timestamp")
        if not ts:
            continue
        swap = (tx.get("events") or {}).get("swap")
        if not swap:
            continue
        ni = native_to_sol(swap.get("nativeInput"))
        no = native_to_sol(swap.get("nativeOutput"))
        si = so = 0.0
        for ti in swap.get("tokenInputs") or []:
            if ti.get("mint") in STABLE_MINTS:
                a = parse_token_amount(ti.get("rawTokenAmount"))
                if a: si += a
        for to in swap.get("tokenOutputs") or []:
            if to.get("mint") in STABLE_MINTS:
                a = parse_token_amount(to.get("rawTokenAmount"))
                if a: so += a
        p = None
        if ni >= 0.01 and so > 0:
            p = so / ni
        elif si > 0 and no >= 0.01:
            p = si / no
        if p and 5 <= p <= 500:
            out[ts] = p
    return out

# =========================
# Helius
# =========================

def h_get(url, session):
    for i in range(HELIUS_MAX_RETRIES):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 429:
                time.sleep(HELIUS_MIN_DELAY_SEC * (2 ** i))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  Helius err ({i+1}): {e}")
            time.sleep(HELIUS_MIN_DELAY_SEC * (2 ** i))
    raise RuntimeError("Helius failed")

def h_post(url, payload, session):
    for i in range(HELIUS_MAX_RETRIES):
        try:
            r = session.post(url, json=payload, timeout=60)
            if r.status_code == 429:
                time.sleep(HELIUS_MIN_DELAY_SEC * (2 ** i))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  Helius err ({i+1}): {e}")
            time.sleep(HELIUS_MIN_DELAY_SEC * (2 ** i))
    raise RuntimeError("Helius failed")

def fetch_txs(wallet: str, end_ts: int) -> List[dict]:
    cd = os.path.join(CACHE_DIR, wallet[:8])
    ensure_dir(cd)
    session = requests.Session()
    before = None
    all_txs = []
    page = 0
    while True:
        page += 1
        base = (f"https://api-mainnet.helius-rpc.com/v0/addresses/{wallet}/transactions"
                f"?api-key={HELIUS_API_KEY}&limit={HELIUS_PAGE_LIMIT}")
        url = base + (f"&before={before}" if before else "")
        cp = os.path.join(cd, f"page_{page:04d}.json")
        if USE_CACHE and os.path.exists(cp):
            with open(cp) as f:
                txs = json.load(f)
        else:
            txs = h_get(url, session)
            if USE_CACHE:
                with open(cp, "w") as f:
                    json.dump(txs, f)
            time.sleep(HELIUS_MIN_DELAY_SEC)
        if not txs:
            break
        for tx in txs:
            ts = tx.get("timestamp")
            if isinstance(ts, int) and ts <= end_ts:
                all_txs.append(tx)
        before = txs[-1].get("signature")
        if not before:
            break
        if page % 10 == 0:
            print(f"    p{page}: {len(all_txs)} txs")
    return all_txs

def parse_token_amount(raw) -> Optional[float]:
    if not isinstance(raw, dict):
        return None
    inner = raw.get("rawTokenAmount") or raw
    if isinstance(inner, dict):
        ui = inner.get("uiAmount")
        if ui is not None:
            try: return float(ui)
            except: pass
    ta = raw.get("tokenAmount")
    dec = raw.get("decimals")
    if ta is None:
        return None
    try:
        return int(str(ta)) / (10 ** int(dec))
    except:
        return None

def fetch_dex_price(mint: str) -> float:
    try:
        r = requests.get(f"https://api.dexscreener.com/latest/dex/tokens/{mint}", timeout=10)
        if r.status_code != 200:
            return 0.0
        pairs = r.json().get("pairs", [])
        if not pairs:
            return 0.0
        best = max(pairs, key=lambda p: sf(p.get("liquidity", {}).get("usd"), 0))
        return sf(best.get("priceUsd"), 0)
    except:
        return 0.0

def fetch_prices(mints: List[str]) -> Dict[str, float]:
    session = requests.Session()
    prices = {}
    print(f"   Fetching prices for {len(mints)} tokens...")
    for i, m in enumerate(mints, 1):
        if m in STABLE_MINTS:
            prices[m] = 1.0
            continue
        if m == SOL_MINT or m in LST_MINTS:
            try:
                r = session.get("https://api.kraken.com/0/public/Ticker",
                                params={"pair": "SOLUSD"}, timeout=10)
                d = r.json()
                if not d.get("error"):
                    rk = list(d["result"].keys())[0]
                    prices[m] = sf(d["result"][rk]["c"][0])
                    continue
            except:
                pass
        p = fetch_dex_price(m)
        if p > 0:
            prices[m] = p
        else:
            try:
                r = session.get(f"https://price.jup.ag/v4/price?ids={m}", timeout=10)
                if r.status_code == 200:
                    jd = r.json()
                    if "data" in jd and m in jd["data"]:
                        jp = sf(jd["data"][m].get("price"))
                        if jp > 0:
                            prices[m] = jp
                            continue
            except:
                pass
            try:
                payload = {"jsonrpc": "2.0", "id": "p", "method": "getAsset", "params": {"id": m}}
                r = h_post(f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}",
                           payload, session)
                if "result" in r:
                    hp = sf(r["result"].get("token_info", {}).get("price_info", {}).get("price_per_token"))
                    if hp > 0:
                        prices[m] = hp
                        continue
            except:
                pass
            prices[m] = 0.0
        if i % 20 == 0:
            print(f"      {i}/{len(mints)}")
        time.sleep(0.03)
    return prices

# =========================
# Inter-Wallet Detection
# =========================

def detect_inter_wallet(all_txs: Dict[str, List[dict]], daily, my_wallets, exact=None):
 
    # Build sig -> tx lookup from all wallets
    sig_tx: Dict[str, dict] = {}
    for w, txs in all_txs.items():
        for tx in txs:
            s = tx.get("signature")
            if s and s not in sig_tx:
                sig_tx[s] = tx

    sigs = set()
    transfers = []

    # Check every transaction for direct wallet-to-wallet movements
    checked_sigs = set()
    for w, txs in all_txs.items():
        for tx in txs:
            sig = tx.get("signature")
            if not sig or sig in checked_sigs:
                continue
            checked_sigs.add(sig)

            # SKIP swaps - these are NOT inter-wallet transfers even if
            # they touch multiple wallet accounts during routing
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
            sol_usd = get_sol_price(ts, daily, exact)
            found_transfer = False

            # Check token transfers for direct wallet->wallet movement
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
                    val = amt * mint_price_at_swap(mint, sol_usd)
                    transfers.append(Transfer(ts_to_date(ts), fa, ta, mint, amt, val, sig))
                    found_transfer = True

            # Check native SOL transfers for direct wallet->wallet
            if not found_transfer and tx_type in ["TRANSFER", "SOL_TRANSFER"]:
                sender = None
                receiver = None
                sol_amt = 0.0
                for a in tx.get("accountData") or []:
                    acct = a.get("account", "")
                    ch = a.get("nativeBalanceChange", 0)
                    if acct in my_wallets:
                        if ch < 0:
                            sender = acct
                            sol_amt = abs(ch) / 1e9
                        elif ch > 0:
                            receiver = acct
                if sender and receiver and sender != receiver and sol_amt > 0.001:
                    transfers.append(Transfer(ts_to_date(ts), sender, receiver, SOL_MINT,
                                              sol_amt, sol_amt * sol_usd, sig))
                    found_transfer = True

            # ONLY add to inter-wallet sigs if we found an actual transfer
            if found_transfer:
                sigs.add(sig)

    # Dedup
    seen = set()
    uniq = []
    for t in transfers:
        k = (t.sig, t.mint, t.from_w)
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    return sigs, uniq

# =========================
# Flow Extraction
# =========================

def extract_flows(tx: dict, wallet: str) -> Optional[dict]:
    ts = tx.get("timestamp")
    sig = tx.get("signature")
    src = tx.get("source", "")
    typ = tx.get("type", "")
    is_swap = bool((tx.get("events") or {}).get("swap"))

    # SOL via nativeBalanceChange ONLY
    sol_lam = 0
    for a in tx.get("accountData") or []:
        if a.get("account") == wallet:
            sol_lam = int(a.get("nativeBalanceChange") or 0)
            break
    fee_lam = int(tx.get("fee") or 0)
    sol_lam -= fee_lam
    sol_d = sol_lam / 1e9
    fee_sol = fee_lam / 1e9

    ins = []
    outs = []
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
            ins.append((mint, amt))
        elif tu == wallet and fu != wallet:
            outs.append((mint, amt))

    # CRITICAL: When swap event exists, its quantities are AUTHORITATIVE.
    # tokenTransfers can have duplicate/wrong amounts due to multi-hop routing.
    # Replace any tokenTransfer entries with swap event quantities.
    swap = (tx.get("events") or {}).get("swap")
    if swap:
        swap_ins = {}   # mint -> qty from swap event
        swap_outs = {}
        for ti in swap.get("tokenInputs", []):
            mint = ti.get("mint")
            amt = parse_token_amount(ti.get("rawTokenAmount"))
            if mint and amt and amt > 0:
                swap_ins[mint] = swap_ins.get(mint, 0) + amt
        for to in swap.get("tokenOutputs", []):
            mint = to.get("mint")
            amt = parse_token_amount(to.get("rawTokenAmount"))
            if mint and amt and amt > 0:
                swap_outs[mint] = swap_outs.get(mint, 0) + amt

        # Remove tokenTransfer entries for mints that appear in swap event
        # and replace with swap event quantities
        if swap_ins or swap_outs:
            all_swap_mints = set(swap_ins.keys()) | set(swap_outs.keys())
            ins = [(m, q) for m, q in ins if m not in all_swap_mints]
            outs = [(m, q) for m, q in outs if m not in all_swap_mints]
            for mint, qty in swap_ins.items():
                if mint != SOL_MINT:
                    ins.append((mint, qty))
            for mint, qty in swap_outs.items():
                if mint != SOL_MINT:
                    outs.append((mint, qty))

    if sol_d > 1e-9:
        outs.append((SOL_MINT, sol_d))
    elif sol_d < -1e-9:
        ins.append((SOL_MINT, abs(sol_d)))

    if not ins and not outs:
        return None
    return {"ts": ts, "sig": sig, "src": src, "type": typ, "is_swap": is_swap,
            "ins": ins, "outs": outs, "fee_sol": fee_sol}

# =========================
# FIFO
# =========================

def fifo_sell(lots: List[Lot], qty: float) -> Tuple[float, str]:
    """Returns (cost_consumed, earliest_acquire_date)"""
    remaining = qty
    cost = 0.0
    earliest = ""
    while remaining > 1e-12 and lots:
        lot = lots[0]
        if lot.amount <= 1e-12:
            lots.pop(0)
            continue
        take = min(lot.amount, remaining)
        frac = take / lot.amount
        portion = lot.cost_usd * frac
        cost += portion
        if not earliest:
            earliest = ts_to_date(lot.acquired_ts)
        lot.amount -= take
        lot.cost_usd -= portion
        remaining -= take
        if lot.amount <= 1e-12:
            lots.pop(0)
    return cost, earliest

def fifo_transfer(from_lots: List[Lot], to_lots: List[Lot], qty: float, to_wallet: str):
    remaining = qty
    while remaining > 1e-12 and from_lots:
        lot = from_lots[0]
        if lot.amount <= 1e-12:
            from_lots.pop(0)
            continue
        take = min(lot.amount, remaining)
        frac = take / lot.amount
        cm = lot.cost_usd * frac
        to_lots.append(Lot(take, cm, lot.acquired_ts, lot.via, to_wallet))
        lot.amount -= take
        lot.cost_usd -= cm
        remaining -= take
        if lot.amount <= 1e-12:
            from_lots.pop(0)

# =========================
# P&L ENGINE
# =========================

def compute_pnl(wallet: str, txs: List[dict], daily, inter_sigs, exact=None):
    lots: Dict[str, List[Lot]] = {}
    sales: List[Sale] = []
    warns: List[dict] = []
    if not exact:
        exact = {}

    flows = []
    for tx in txs:
        f = extract_flows(tx, wallet)
        if f:
            flows.append(f)
    flows.sort(key=lambda x: x["ts"])

    start_ts = utc_ts(START_DATE_UTC)
    beg_bal = 0.0
    got_beg = False

    # Debug counters
    dbg = {"A_purchase": 0, "B_sale": 0, "C_swap": 0, "D_cash": 0,
           "E_fallback": 0, "dust": 0, "inter_skip": 0, "no_flows": 0,
           "total": len(flows)}

    for fl in flows:
        ts = fl["ts"]
        sig = fl["sig"]

        if not got_beg and ts >= start_ts:
            beg_bal = sum(l.cost_usd for ll in lots.values() for l in ll)
            got_beg = True

        if sig in inter_sigs:
            dbg["inter_skip"] += 1
            continue

        sol_usd = get_sol_price(ts, daily, exact)
        fee_usd = fl["fee_sol"] * sol_usd
        ins = fl["ins"]
        outs = fl["outs"]

        # --- Dust filter ---
        total_val = 0.0
        has_token = False
        for m, q in ins + outs:
            if m in CASH_MINTS:
                total_val += q * mint_price_at_swap(m, sol_usd)
            else:
                has_token = True
                total_val += 999  # any real token = not dust
        if not has_token and total_val < DUST_USD:
            dbg["dust"] += 1
            continue

        # --- Classify ---
        in_mints = {m for m, _ in ins}
        out_mints = {m for m, _ in outs}
        cash_in = in_mints & CASH_MINTS      # cash leaving wallet
        cash_out = out_mints & CASH_MINTS     # cash entering wallet
        tok_in = in_mints - CASH_MINTS        # tokens leaving wallet
        tok_out = out_mints - CASH_MINTS      # tokens entering wallet

        # Calculate cash values
        cash_out_usd = sum(q * mint_price_at_swap(m, sol_usd) for m, q in outs if m in CASH_MINTS)
        cash_in_usd = sum(q * mint_price_at_swap(m, sol_usd) for m, q in ins if m in CASH_MINTS)

        # ============================================================
        # CASE A: TOKEN PURCHASE (cash out -> tokens in)
        # Spending SOL/stable to buy tokens. NOT taxable.
        # Cost basis of new tokens = cost of cash spent.
        # ============================================================
        if cash_in and tok_out and not tok_in:
            dbg["A_purchase"] += 1
            # Consume cash lots
            total_cost = fee_usd
            for m, q in ins:
                if m in STABLE_MINTS:
                    total_cost += q  # stables at face value
                else:
                    c, _ = fifo_sell(lots.setdefault(m, []), q)
                    total_cost += c

            # Create token lots
            received = [(m, q) for m, q in outs if m not in CASH_MINTS]
            if received:
                wt = sum(q for _, q in received)
                if wt > 0:
                    for m, q in received:
                        lots.setdefault(m, []).append(
                            Lot(q, total_cost * (q / wt), ts, sig, wallet))

            # Also create lots for any SOL/LST received (change back)
            for m, q in outs:
                if m in CASH_MINTS and m not in STABLE_MINTS:
                    # SOL received back as "change" - lot at current price
                    lots.setdefault(m, []).append(
                        Lot(q, q * mint_price_at_swap(m, sol_usd), ts, sig, wallet))

            continue  # NO sale recorded

        # ============================================================
        # CASE B: TOKEN SALE (tokens in -> cash out)
        # Selling tokens for SOL/stable. TAXABLE.
        # ============================================================
        if tok_in and cash_out:
            dbg["B_sale"] += 1
            proceeds = cash_out_usd
            total_cost = fee_usd

            disposed = []
            for m, q in ins:
                if m in CASH_MINTS:
                    # Cash also going out (complex swap) - consume those lots too
                    if m in STABLE_MINTS:
                        total_cost += q
                    else:
                        c, d = fifo_sell(lots.setdefault(m, []), q)
                        total_cost += c
                else:
                    c, d = fifo_sell(lots.setdefault(m, []), q)
                    total_cost += c
                    disposed.append((m, q, c, d))
                    if not d:
                        warns.append({"wallet": wallet, "mint": m, "quantity": q,
                                      "date_sold": ts_to_date(ts), "signature": sig,
                                      "note": "Missing lots"})

            # Create lots for any non-cash outputs (partial swap)
            received = [(m, q) for m, q in outs if m not in CASH_MINTS]
            if received:
                wt = sum(q for _, q in received)
                if wt > 0:
                    for m, q in received:
                        lots.setdefault(m, []).append(
                            Lot(q, total_cost * (q / wt), ts, sig, wallet))

            # Also create lots for received cash (SOL received)
            for m, q in outs:
                if m == SOL_MINT or m in LST_MINTS:
                    lots.setdefault(m, []).append(
                        Lot(q, q * mint_price_at_swap(m, sol_usd), ts, sig, wallet))

            # Record sales (only in reporting period)
            if ts >= start_ts and disposed:
                total_tok_cost = sum(c for _, _, c, _ in disposed)
                for m, q, cb, acq_date in disposed:
                    if not acq_date:
                        acq_date = ts_to_date(ts)
                    # Proportional proceeds
                    alloc = proceeds * (cb / total_tok_cost) if total_tok_cost > 0 else proceeds / len(disposed)
                    pnl = alloc - cb
                    label = m[:16] + "..." if len(m) > 16 else m
                    sales.append(Sale(wallet, m, label, q, acq_date, ts_to_date(ts),
                                      round(alloc, 2), round(cb, 2), round(pnl, 2),
                                      sig, "SELL"))
            continue

        # ============================================================
        # CASE C: TOKEN-TO-TOKEN SWAP (no cash involved)
        # Cost basis transfers. NOT taxable.
        # ============================================================
        if tok_in and tok_out and not cash_out:
            dbg["C_swap"] += 1
            total_cost = fee_usd
            for m, q in ins:
                if m in CASH_MINTS:
                    if m in STABLE_MINTS:
                        total_cost += q
                    else:
                        c, _ = fifo_sell(lots.setdefault(m, []), q)
                        total_cost += c
                else:
                    c, _ = fifo_sell(lots.setdefault(m, []), q)
                    total_cost += c

            received = [(m, q) for m, q in outs if m not in STABLE_MINTS]
            if received:
                wt = sum(q for _, q in received)
                if wt > 0:
                    for m, q in received:
                        lots.setdefault(m, []).append(
                            Lot(q, total_cost * (q / wt), ts, sig, wallet))
            continue  # NO sale

        # ============================================================
        # CASE D: SOL <-> STABLE
        # ============================================================
        if cash_in and cash_out and not tok_in and not tok_out:
            dbg["D_cash"] += 1
            has_sol_in = SOL_MINT in in_mints or bool(in_mints & LST_MINTS)
            has_stable_out = bool(out_mints & STABLE_MINTS)
            has_stable_in = bool(in_mints & STABLE_MINTS)
            has_sol_out = SOL_MINT in out_mints or bool(out_mints & LST_MINTS)

            if has_sol_in and has_stable_out:
                # SELLING SOL for stables - taxable on SOL
                proceeds = sum(q for m, q in outs if m in STABLE_MINTS)
                total_cost = fee_usd
                disposed = []
                for m, q in ins:
                    if m in STABLE_MINTS:
                        total_cost += q
                    else:
                        c, d = fifo_sell(lots.setdefault(m, []), q)
                        total_cost += c
                        disposed.append((m, q, c, d))
                        if not d:
                            warns.append({"wallet": wallet, "mint": m, "quantity": q,
                                          "date_sold": ts_to_date(ts), "signature": sig,
                                          "note": "Missing SOL lots"})

                if ts >= start_ts and disposed:
                    total_d_cost = sum(c for _, _, c, _ in disposed)
                    for m, q, cb, acq in disposed:
                        if not acq:
                            acq = ts_to_date(ts)
                        alloc = proceeds * (cb / total_d_cost) if total_d_cost > 0 else proceeds
                        pnl = alloc - cb
                        sales.append(Sale(wallet, m, "SOL", q, acq, ts_to_date(ts),
                                          round(alloc, 2), round(cb, 2), round(pnl, 2),
                                          sig, "SOL_SALE"))

            elif has_stable_in and has_sol_out:
                # BUYING SOL with stables - not taxable, just create SOL lot
                total_cost = fee_usd
                for m, q in ins:
                    if m in STABLE_MINTS:
                        total_cost += q
                    else:
                        c, _ = fifo_sell(lots.setdefault(m, []), q)
                        total_cost += c
                for m, q in outs:
                    if m not in STABLE_MINTS:
                        lots.setdefault(m, []).append(Lot(q, total_cost, ts, sig, wallet))

            else:
                # Other cash<->cash (SOL<->LST, etc) - just transfer lots
                total_cost = fee_usd
                for m, q in ins:
                    c, _ = fifo_sell(lots.setdefault(m, []), q)
                    total_cost += c
                for m, q in outs:
                    lots.setdefault(m, []).append(Lot(q, total_cost, ts, sig, wallet))

            continue

        # ============================================================
        # CASE E: FALLBACK (transfers, airdrops, etc)
        # ============================================================
        dbg["E_fallback"] += 1
        # Consume any lots for outgoing assets
        total_cost = fee_usd
        for m, q in ins:
            if m in STABLE_MINTS:
                total_cost += q
            else:
                c, _ = fifo_sell(lots.setdefault(m, []), q)
                total_cost += c

        # Create lots for incoming assets
        received = [(m, q) for m, q in outs if m not in STABLE_MINTS]
        if received:
            wt = sum(q for _, q in received)
            if wt > 0:
                for m, q in received:
                    cost_alloc = total_cost * (q / wt) if total_cost > 0 else 0
                    lots.setdefault(m, []).append(Lot(q, cost_alloc, ts, sig, wallet))

    if not got_beg:
        beg_bal = sum(l.cost_usd for ll in lots.values() for l in ll)

    return sales, warns, lots, beg_bal, dbg

# =========================
# Balances
# =========================

def fetch_balances(wallet):
    session = requests.Session()
    url = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"
    bals = {}
    try:
        r = h_post(url, {"jsonrpc": "2.0", "id": "s", "method": "getBalance",
                          "params": [wallet]}, session)
        sol = r.get("result", {}).get("value", 0) / 1e9
        if sol > 0:
            bals[SOL_MINT] = sol
        r = h_post(url, {"jsonrpc": "2.0", "id": "t", "method": "getTokenAccountsByOwner",
                          "params": [wallet, {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
                                     {"encoding": "jsonParsed"}]}, session)
        for a in r.get("result", {}).get("value", []):
            info = a.get("account", {}).get("data", {}).get("parsed", {}).get("info", {})
            m = info.get("mint")
            ui = sf(info.get("tokenAmount", {}).get("uiAmount"), 0)
            if m and ui > 1e-6:
                bals[m] = ui
        return bals
    except Exception as e:
        print(f"  Bal error: {e}")
        return {}

# =========================
# Deposits/Withdrawals
# =========================

def calc_deps_wds(txs, wallet, daily, inter_sigs, start_ts, end_ts, exact=None):
    dt = wt = 0.0
    dl, wl = [], []
    if not exact: exact = {}
    for tx in sorted(txs, key=lambda x: x.get("timestamp", 0)):
        ts = tx.get("timestamp", 0)
        if not ts or ts < start_ts or ts > end_ts: continue
        sig = tx.get("signature", "")
        if sig in inter_sigs: continue
        if tx.get("type") == "SWAP": continue
        src = tx.get("source", "")
        if src and any(p in src.upper() for p in ["JUPITER", "RAYDIUM", "ORCA"]): continue
        ev = tx.get("events", {})
        if ev.get("swap") or ev.get("nft"): continue
        sol_usd = get_sol_price(ts, daily, exact)
        tt = tx.get("type", "")
        if tt in ["TRANSFER", "SOL_TRANSFER"]:
            if any(t.get("mint") == SOL_MINT for t in tx.get("tokenTransfers", [])): continue
            for ad in tx.get("accountData", []):
                if ad.get("account") == wallet:
                    ch = ad.get("nativeBalanceChange", 0) / 1e9
                    if abs(ch) < 0.001: continue
                    v = abs(ch * sol_usd)
                    e = {"date": ts_to_date(ts), "amount": round(abs(ch), 9),
                         "value_usd": round(v, 2), "sig": sig}
                    if ch > 0:
                        dt += v; e["type"] = "SOL Deposit"; dl.append(e)
                    else:
                        wt += v; e["type"] = "SOL Withdrawal"; wl.append(e)
        if tt == "TRANSFER":
            for tr in tx.get("tokenTransfers", []):
                m = tr.get("mint", "")
                if m not in STABLE_MINTS: continue
                if len(tx.get("tokenTransfers", [])) > 1: continue
                amt = sf(tr.get("tokenAmount", 0))
                if amt <= 0: continue
                fa = tr.get("fromUserAccount", "")
                ta = tr.get("toUserAccount", "")
                e = {"date": ts_to_date(ts), "amount": round(amt, 2),
                     "value_usd": round(amt, 2), "sig": sig}
                if ta == wallet and fa != wallet:
                    dt += amt; e["type"] = "Stable Deposit"; dl.append(e)
                elif fa == wallet and ta != wallet:
                    wt += amt; e["type"] = "Stable Withdrawal"; wl.append(e)
    return dt, wt, dl, wl

# =========================
# CSV
# =========================

def write_csv(all_sales, all_unreal, all_warns, transfers, wnames,
              all_deps, all_wds, all_beg, all_end, fname, start_ts, end_ts):
    with open(fname, "w", newline="") as f:
        w = csv.writer(f)
        tb = sum(all_beg.values())
        te = sum(all_end.values())
        td = sum(sum(d.get("value_usd", 0) for d in dl) for dl in all_deps.values())
        tw = sum(sum(d.get("value_usd", 0) for d in dl) for dl in all_wds.values())
        tr = sum(sum(s.pnl for s in sl) for sl in all_sales.values())
        tu = sum(sum(p.pnl for p in pl) for pl in all_unreal.values())
        nt = sum(len(sl) for sl in all_sales.values())

        w.writerow(["SOLANA TAX REPORT v4"])
        w.writerow(["Generated:", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")])
        w.writerow(["Period:", f"{START_DATE_UTC} to {END_DATE_UTC}"])
        w.writerow([])

        # ===== TOTAL P&L =====
        w.writerow(["*** TOTAL P&L ***", f"${tr + tu:,.2f}"])
        w.writerow(["  Realized P&L:", f"${tr:,.2f}"])
        w.writerow(["  Unrealized P&L:", f"${tu:,.2f}"])
        w.writerow(["  Total Gains (realized):",
                     f"${sum(sum(s.pnl for s in sl if s.pnl > 0) for sl in all_sales.values()):,.2f}"])
        w.writerow(["  Total Losses (realized):",
                     f"${sum(sum(s.pnl for s in sl if s.pnl < 0) for sl in all_sales.values()):,.2f}"])
        w.writerow([])

        w.writerow(["SUMMARY"])
        w.writerow(["Beginning Balance (cost basis):", f"${tb:,.2f}"])
        w.writerow(["+ Deposits:", f"${td:,.2f}"])
        w.writerow(["- Withdrawals:", f"${tw:,.2f}"])
        w.writerow(["Realized P&L:", f"${tr:,.2f}"])
        w.writerow(["Ending Balance (market):", f"${te:,.2f}"])
        w.writerow(["Unrealized P&L:", f"${tu:,.2f}"])
        w.writerow(["Total Trades:", nt])
        w.writerow([])

        w.writerow(["PER-WALLET"])
        w.writerow(["#", "Name", "Addr", "Beg", "Deps", "Wds", "Realized", "End", "#Trades"])
        for i, wlt in enumerate(WALLETS, 1):
            nm = wnames.get(wlt, wlt[:8]+"...")
            dep = sum(d.get("value_usd", 0) for d in all_deps.get(wlt, []))
            wd_ = sum(d.get("value_usd", 0) for d in all_wds.get(wlt, []))
            rl = sum(s.pnl for s in all_sales.get(wlt, []))
            w.writerow([i, nm, wlt[:12]+"...", f"${all_beg.get(wlt,0):,.2f}",
                         f"${dep:,.2f}", f"${wd_:,.2f}", f"${rl:,.2f}",
                         f"${all_end.get(wlt,0):,.2f}", len(all_sales.get(wlt, []))])
        w.writerow([])

        # ===== AGGREGATED P&L BY TOKEN (one row per token per wallet) =====
        w.writerow(["REALIZED P&L BY TOKEN (sorted by |P&L|)"])
        w.writerow(["Wallet", "Token", "#Sells", "Total Qty Sold", "Total Proceeds",
                     "Total Cost Basis", "Total P&L", "First Buy", "Last Sell"])
        # Aggregate: (wallet, mint) -> totals
        agg = {}
        for wlt, sl in all_sales.items():
            nm = wnames.get(wlt, wlt[:8]+"...")
            for s in sl:
                key = (nm, s.mint, s.label)
                if key not in agg:
                    agg[key] = {"sells": 0, "qty": 0, "proceeds": 0, "cost": 0, "pnl": 0,
                                "first_buy": s.date_acquired, "last_sell": s.date_sold}
                a = agg[key]
                a["sells"] += 1
                a["qty"] += s.qty
                a["proceeds"] += s.proceeds
                a["cost"] += s.cost_basis
                a["pnl"] += s.pnl
                if s.date_acquired < a["first_buy"]:
                    a["first_buy"] = s.date_acquired
                if s.date_sold > a["last_sell"]:
                    a["last_sell"] = s.date_sold
        agg_list = [(k, v) for k, v in agg.items()]
        agg_list.sort(key=lambda x: abs(x[1]["pnl"]), reverse=True)
        for (nm, mint, label), a in agg_list:
            w.writerow([nm, label, a["sells"], f"{a['qty']:.6f}",
                         f"${a['proceeds']:,.2f}", f"${a['cost']:,.2f}",
                         f"${a['pnl']:,.2f}", a["first_buy"], a["last_sell"]])
        w.writerow([])

        # ===== DETAILED TRADES (individual transactions) =====
        w.writerow(["DETAILED TRADES (sorted by |gain/loss|)"])
        w.writerow(["Wallet", "Type", "Token", "Acquired", "Sold", "Proceeds",
                     "Cost Basis", "Gain/Loss", "Quantity", "Sig"])
        at = []
        for wlt, sl in all_sales.items():
            for s in sl:
                at.append((wnames.get(wlt, wlt[:8]+"..."), s))
        at.sort(key=lambda x: abs(x[1].pnl), reverse=True)
        for nm, s in at:
            w.writerow([nm, s.tx_type, s.label, s.date_acquired, s.date_sold,
                         f"${s.proceeds:,.2f}", f"${s.cost_basis:,.2f}",
                         f"${s.pnl:,.2f}", f"{s.qty:.6f}", s.sig[:8]])
        w.writerow([])

        # Holdings
        w.writerow(["HOLDINGS (sorted by |unrealized P&L|)"])
        w.writerow(["Wallet", "Token", "Qty", "Cost Basis", "Price", "Value", "Unrealized P&L"])
        ap = []
        for wlt, pl in all_unreal.items():
            for p in pl:
                ap.append((wnames.get(wlt, wlt[:8]+"..."), p))
        ap.sort(key=lambda x: abs(x[1].pnl), reverse=True)
        for nm, p in ap:
            w.writerow([nm, p.mint[:16]+"...", f"{p.qty:.6f}", f"${p.cost_basis:,.2f}",
                         f"${p.price:.8f}", f"${p.value:,.2f}", f"${p.pnl:,.2f}"])
        w.writerow([])

        # Inter-wallet (period only)
        pt = [t for t in transfers if utc_ts(t.date) >= start_ts and utc_ts(t.date) <= end_ts]
        if pt:
            w.writerow(["INTER-WALLET TRANSFERS (period only)"])
            w.writerow(["Date", "From", "To", "Token", "Amount", "Value", "Sig"])
            for t in sorted(pt, key=lambda x: x.date):
                fn = wnames.get(t.from_w, t.from_w[:8]+"...")
                tn = wnames.get(t.to_w, t.to_w[:8]+"...")
                ml = "SOL" if t.mint == SOL_MINT else t.mint[:12]+"..."
                w.writerow([t.date, fn, tn, ml, f"{t.amount:.6f}",
                             f"${t.value_usd:,.2f}", t.sig[:8]])
            w.writerow([])

        aw = []
        for wlt, wl in all_warns.items():
            for wr in wl:
                wr["wallet"] = wnames.get(wlt, wlt[:8]+"...")
                aw.append(wr)
        if aw:
            w.writerow(["WARNINGS"])
            w.writerow(["Wallet", "Token", "Qty", "Date", "Sig", "Note"])
            for wr in aw[:200]:
                w.writerow([wr.get("wallet",""), wr.get("mint","")[:16],
                             f"{wr.get('quantity',0):.6f}", wr.get("date_sold",""),
                             wr.get("signature","")[:8], wr.get("note","")])

        w.writerow([])
        w.writerow(["END OF REPORT"])

# =========================
# Main
# =========================

def main():
    start_ts = utc_ts(START_DATE_UTC)
    end_ts = utc_ts(END_DATE_UTC, eod=True)
    ensure_dir(OUTPUT_DIR)

    print("=" * 60)
    print("SOLANA TAX CALCULATOR v4")
    print("=" * 60)
    for i, w in enumerate(WALLETS, 1):
        print(f"  {i}. {WALLET_NAMES.get(w, f'W{i}')}: {w}")
    print(f"Period: {START_DATE_UTC} -> {END_DATE_UTC}")
    print()

    if HELIUS_API_KEY == "your-api-key-here":
        print("ERROR: Set HELIUS_API_KEY!"); return

    print("1. SOL prices...")
    daily = fetch_kraken_daily(utc_ts("2020-01-01"), end_ts)
    print(f"   Done: {len(daily)} prices\n")

    print("2. Fetching tx history...")
    atx: Dict[str, List[dict]] = {}
    for i, w in enumerate(WALLETS, 1):
        print(f"   [{i}/{len(WALLETS)}] {WALLET_NAMES.get(w, f'W{i}')}...")
        atx[w] = fetch_txs(w, end_ts)
        print(f"      Done: {len(atx[w])} txs")
    print()

    print("2.5. Exact SOL prices...")
    exact: Dict[int, float] = {}
    for txs in atx.values():
        exact.update(extract_exact_sol_prices(txs))
    print(f"   Done: {len(exact)} prices\n")

    print("3. Inter-wallet...")
    mw = set(WALLETS)
    isigs, itrans = detect_inter_wallet(atx, daily, mw, exact)
    print(f"   Done: {len(itrans)} transfers, {len(isigs)} sigs\n")

    print("4. P&L...")
    a_sales: Dict[str, List[Sale]] = {}
    a_warns: Dict[str, List[dict]] = {}
    a_lots: Dict[str, Dict[str, List[Lot]]] = {}
    a_beg: Dict[str, float] = {}
    a_deps: Dict[str, List[dict]] = {}
    a_wds: Dict[str, List[dict]] = {}
    a_end: Dict[str, float] = {}

    for w in WALLETS:
        nm = WALLET_NAMES.get(w, w[:8]+"...")
        sl, wr, lt, bg, dbg = compute_pnl(w, atx.get(w, []), daily, isigs, exact)
        a_sales[w] = sl; a_warns[w] = wr; a_lots[w] = lt; a_beg[w] = bg
        sol_s = sum(1 for s in sl if s.mint == SOL_MINT)
        tok_s = sum(1 for s in sl if s.mint not in CASH_MINTS)
        print(f"   {nm}: {len(sl)} sales ({sol_s} SOL, {tok_s} token), "
              f"P&L=${sum(s.pnl for s in sl):,.2f}")
        print(f"      Debug: {dbg}")
    print()

    print("5. Unrealized...")
    a_unreal: Dict[str, List[Position]] = {}
    am = set()
    ab: Dict[str, Dict[str, float]] = {}
    for w in WALLETS:
        ab[w] = fetch_balances(w)
        time.sleep(0.05)
    for w, lm in a_lots.items():
        bl = ab.get(w, {})
        pos = []
        for m in set(lm.keys()) | set(bl.keys()):
            q = bl.get(m, 0.0)
            c = sum(l.cost_usd for l in lm.get(m, []))
            if q <= 0 and c <= 0: continue
            pos.append(Position(w, m, q, c, 0, 0, 0))
            am.add(m)
        a_unreal[w] = pos
    print(f"   Done: {sum(len(p) for p in a_unreal.values())} positions\n")

    print("6. Prices...")
    prices = fetch_prices(list(am))
    for w, pl in a_unreal.items():
        for p in pl:
            pr = prices.get(p.mint, 0.0)
            p.price = pr; p.value = p.qty * pr; p.pnl = p.value - p.cost_basis
    print()

    print("7. Deposits/Withdrawals...")
    for w in WALLETS:
        d, wd, dl, wl = calc_deps_wds(atx[w], w, daily, isigs, start_ts, end_ts, exact)
        a_deps[w] = dl; a_wds[w] = wl
        a_end[w] = sum(p.value for p in a_unreal.get(w, []))
        print(f"   {WALLET_NAMES.get(w, w[:8]+'...')}: +${d:,.2f} / -${wd:,.2f}")
    print()

    print("8. Report...")
    out = os.path.join(OUTPUT_DIR, "multi_wallet_tax_report.csv")
    write_csv(a_sales, a_unreal, a_warns, itrans, WALLET_NAMES,
              a_deps, a_wds, a_beg, a_end, out, start_ts, end_ts)
    print(f"   Done: {out}\n")

    tr = sum(sum(s.pnl for s in sl) for sl in a_sales.values())
    tu = sum(sum(p.pnl for p in pl) for pl in a_unreal.values())
    print("=" * 60)
    print(f"*** TOTAL P&L: ${tr + tu:,.2f} ***")
    print(f"Realized:   ${tr:,.2f}")
    print(f"Unrealized: ${tu:,.2f}")
    print(f"Trades:     {sum(len(s) for s in a_sales.values())}")
    print("=" * 60)
    print("Done!")

if __name__ == "__main__":
    main()