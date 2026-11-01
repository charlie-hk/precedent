"""Market-structure recognition.

Swings, break of structure (BOS), change of character (CHOCH), supply/demand
zones, liquidity sweeps and a trend/range read. Everything here is causal: a
feature at bar i uses only bars <= i, so the output can be used in a backtest
without look-ahead.

What ships here is a solid baseline. The calibrated thresholds and the richer
zone-quality model used in production are reserved (see README and
`learning.py`).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class Trend(str, Enum):
    UP = "up"
    DOWN = "down"
    RANGE = "range"


@dataclass
class Swing:
    index: int
    time: pd.Timestamp
    price: float
    kind: str  # "high" or "low"


@dataclass
class Zone:
    kind: str  # "supply" or "demand"
    start_index: int
    top: float
    bottom: float
    created_time: pd.Timestamp
    strength: float


def find_swings(df: pd.DataFrame, left: int = 2, right: int = 2) -> list[Swing]:
    """Fractal swing points: a high with `left`/`right` lower highs around it (and vice versa).

    The `right` lookahead is only used to *confirm* a swing; a swing at bar i is
    not known until bar i+right, and downstream code respects that delay.
    """
    highs, lows = df["high"].to_numpy(), df["low"].to_numpy()
    out: list[Swing] = []
    for i in range(left, len(df) - right):
        wh = highs[i - left : i + right + 1]
        wl = lows[i - left : i + right + 1]
        if highs[i] == wh.max() and (wh.argmax() == left):
            out.append(Swing(i, df.index[i], float(highs[i]), "high"))
        elif lows[i] == wl.min() and (wl.argmin() == left):
            out.append(Swing(i, df.index[i], float(lows[i]), "low"))
    return out


def label_structure(df: pd.DataFrame, swings: list[Swing]) -> pd.DataFrame:
    """Walk confirmed swings and mark BOS / CHOCH events and the running trend.

    Returns a frame indexed like `df` with columns: trend, event ("BOS"/"CHOCH"/None).
    BOS = continuation (new HH in an uptrend, new LL in a downtrend).
    CHOCH = first break against the active trend (the character change).
    """
    trend = pd.Series(Trend.RANGE.value, index=df.index, dtype=object)
    event = pd.Series(None, index=df.index, dtype=object)

    last_high: float | None = None
    last_low: float | None = None
    cur = Trend.RANGE

    for sw in swings:
        # A swing is only actionable at confirmation time (right offset already applied upstream).
        t = df.index[sw.index]
        if sw.kind == "high":
            if last_high is not None and sw.price > last_high:
                if cur == Trend.DOWN:
                    event.loc[t] = "CHOCH"
                    cur = Trend.UP
                else:
                    event.loc[t] = "BOS"
                    cur = Trend.UP
            last_high = sw.price
        else:
            if last_low is not None and sw.price < last_low:
                if cur == Trend.UP:
                    event.loc[t] = "CHOCH"
                    cur = Trend.DOWN
                else:
                    event.loc[t] = "BOS"
                    cur = Trend.DOWN
            last_low = sw.price
        trend.loc[t:] = cur.value

    return pd.DataFrame({"trend": trend, "event": event})


@dataclass
class StructureEvent:
    index: int
    time: pd.Timestamp
    kind: str  # "BOS" or "CHOCH"
    price: float  # the swing price that did the breaking
    level: float  # the prior swing level that was broken
    trend_after: str


def events(df: pd.DataFrame, swings: list[Swing]) -> list[StructureEvent]:
    """Precise BOS/CHOCH list: each with the bar, the broken level and the new trend.

    Same walk as `label_structure`, but it records *where* each break happened and
    which prior swing level it broke, so the output can be shown or drawn exactly.
    """
    out: list[StructureEvent] = []
    last_high: float | None = None
    last_low: float | None = None
    cur = Trend.RANGE

    for sw in swings:
        if sw.kind == "high":
            if last_high is not None and sw.price > last_high:
                kind = "CHOCH" if cur == Trend.DOWN else "BOS"
                cur = Trend.UP
                out.append(StructureEvent(sw.index, df.index[sw.index], kind,
                                          float(sw.price), float(last_high), cur.value))
            last_high = sw.price
        else:
            if last_low is not None and sw.price < last_low:
                kind = "CHOCH" if cur == Trend.UP else "BOS"
                cur = Trend.DOWN
                out.append(StructureEvent(sw.index, df.index[sw.index], kind,
                                          float(sw.price), float(last_low), cur.value))
            last_low = sw.price
    return out


def supply_demand_zones(df: pd.DataFrame, swings: list[Swing], max_zones: int = 40) -> list[Zone]:
    """Zones from the base of a strong move off a confirmed swing.

    Demand = the candle range at a swing low that launched an up move; supply =
    at a swing high that launched a down move. Strength is the size of the move
    relative to the zone height (baseline proxy; production uses a richer score).
    """
    zones: list[Zone] = []
    n = len(df)
    opens, closes = df["open"].to_numpy(), df["close"].to_numpy()
    highs, lows = df["high"].to_numpy(), df["low"].to_numpy()

    for sw in swings:
        i = sw.index
        fwd = min(i + 6, n - 1)
        if fwd <= i:
            continue
        move = closes[fwd] - closes[i]
        if sw.kind == "low" and move > 0:
            top, bottom = max(opens[i], closes[i]), lows[i]
            height = max(top - bottom, 1e-9)
            zones.append(Zone("demand", i, top, bottom, df.index[i], float(move / height)))
        elif sw.kind == "high" and move < 0:
            top, bottom = highs[i], min(opens[i], closes[i])
            height = max(top - bottom, 1e-9)
            zones.append(Zone("supply", i, top, bottom, df.index[i], float(-move / height)))

    zones.sort(key=lambda z: z.strength, reverse=True)
    return zones[:max_zones]


def liquidity_sweeps(df: pd.DataFrame, swings: list[Swing], tol: float = 0.0005) -> list[int]:
    """Bars that poke beyond a prior swing then close back inside (stop run).

    Returns bar indices of detected sweeps.
    """
    out: list[int] = []
    highs, lows, closes = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    prior_highs = [s for s in swings if s.kind == "high"]
    prior_lows = [s for s in swings if s.kind == "low"]

    for s in prior_highs:
        for j in range(s.index + 1, min(s.index + 30, len(df))):
            if highs[j] > s.price * (1 + tol) and closes[j] < s.price:
                out.append(j)
                break
    for s in prior_lows:
        for j in range(s.index + 1, min(s.index + 30, len(df))):
            if lows[j] < s.price * (1 - tol) and closes[j] > s.price:
                out.append(j)
                break
    return sorted(set(out))


def analyze(df: pd.DataFrame, left: int = 2, right: int = 2) -> dict:
    """Run the full structure pass and return swings, labels, zones and sweeps."""
    swings = find_swings(df, left, right)
    labels = label_structure(df, swings)
    evs = events(df, swings)
    zones = supply_demand_zones(df, swings)
    sweeps = liquidity_sweeps(df, swings)
    return {"swings": swings, "labels": labels, "events": evs, "zones": zones, "sweeps": sweeps}
