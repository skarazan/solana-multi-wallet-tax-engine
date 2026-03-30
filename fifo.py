"""FIFO accounting engine — lot consumption and transfer."""

from dataclasses import dataclass, field
from typing import List, Tuple

from models import Lot, ts_to_date


@dataclass
class FIFOResult:
    cost: float = 0.0
    earliest_acq_date: str = ""
    shortfall: float = 0.0


def fifo_sell(lots: List[Lot], qty: float) -> FIFOResult:
    """
    Consume lots in FIFO order (oldest first).

    Returns FIFOResult with:
      - cost: total cost basis consumed
      - earliest_acq_date: date string of the oldest lot touched
      - shortfall: quantity not covered by available lots (0 if fully covered)
    """
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

    shortfall = remaining if remaining > 1e-12 else 0.0
    return FIFOResult(cost=cost, earliest_acq_date=earliest, shortfall=shortfall)


def fifo_transfer(
    from_lots: List[Lot], to_lots: List[Lot], qty: float, to_wallet: str
):
    """
    Move lots from one wallet to another, preserving cost basis
    and acquisition timestamps. Used for inter-wallet transfers.
    """
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
