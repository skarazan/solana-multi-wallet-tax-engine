"""Data models and utility helpers."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Tuple


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


def sf(x, d=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return d


def ensure_dir(p: str):
    import os
    os.makedirs(p, exist_ok=True)


# =========================
# Dataclasses
# =========================

@dataclass
class Lot:
    amount: float
    cost_usd: float
    acquired_ts: int
    via: str = ""
    wallet: str = ""
    is_synthetic: bool = False


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
    tx_type: str  # SELL, SWAP_DISPOSAL, SOL_SALE
    has_missing_lots: bool = False


@dataclass
class IncomeEvent:
    wallet: str
    mint: str
    qty: float
    fmv_usd: float
    ts: int
    income_type: str  # AIRDROP, STAKING_REWARD


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
    ts: int = 0


@dataclass
class Warning:
    wallet: str
    mint: str
    quantity: float
    date: str
    sig: str
    category: str  # MISSING_LOTS, NO_FMV, UNKNOWN_PATTERN
    note: str


@dataclass
class TransactionFlows:
    ts: int
    sig: str
    src: str
    tx_type: str
    is_swap: bool
    outflows: List[Tuple[str, float]]  # (mint, qty) tokens LEAVING the wallet
    inflows: List[Tuple[str, float]]   # (mint, qty) tokens ENTERING the wallet
    fee_sol: float
