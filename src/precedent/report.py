"""Turn a structure analysis into a precise, readable report.

Shows the current state, the most recent BOS/CHOCH events with their broken
levels and times, the active zones relative to the current price, and the latest
liquidity sweeps. This is the information the raw counts were missing.
"""
from __future__ import annotations

import pandas as pd


def text_report(df: pd.DataFrame, result: dict, recent: int = 8) -> str:
    price = float(df["close"].iloc[-1])
    trend = result["labels"]["trend"].iloc[-1]
    evs = result["events"]
    zones = result["zones"]
    sweeps = result["sweeps"]

    lines = []
    lines.append(f"Current price: {price:.2f}   trend: {trend}")

    if evs:
        last = evs[-1]
        lines.append(f"Last structure event: {last.kind} at {last.time}  "
                     f"broke level {last.level:.2f}  -> trend {last.trend_after}")
    else:
        lines.append("No confirmed BOS/CHOCH yet.")

    lines.append("")
    lines.append(f"Recent structure events (last {recent}):")
    lines.append(f"  {'time':<25} {'type':<6} {'broke level':>12} {'-> trend':>10}")
    for e in evs[-recent:]:
        lines.append(f"  {str(e.time):<25} {e.kind:<6} {e.level:>12.2f} {e.trend_after:>10}")

    lines.append("")
    lines.append("Active zones nearest current price:")
    ranked = sorted(zones, key=lambda z: abs((z.top + z.bottom) / 2 - price))
    for z in ranked[:4]:
        where = "above" if z.bottom > price else "below" if z.top < price else "price inside"
        dist = 0.0 if where == "price inside" else min(abs(z.bottom - price), abs(z.top - price))
        lines.append(f"  {z.kind:<7} {z.bottom:.2f}-{z.top:.2f}  ({where}, {dist:.2f} away)  "
                     f"strength {z.strength:.2f}")

    if sweeps:
        lines.append("")
        last_sweeps = sweeps[-3:]
        times = ", ".join(str(df.index[i]) for i in last_sweeps)
        lines.append(f"Latest liquidity sweeps at bars: {times}")

    return "\n".join(lines)


def events_frame(result: dict) -> pd.DataFrame:
    """The full event list as a DataFrame, for inspection or export."""
    evs = result["events"]
    return pd.DataFrame(
        [{"time": e.time, "type": e.kind, "price": e.price, "broke_level": e.level,
          "trend_after": e.trend_after} for e in evs]
    )
