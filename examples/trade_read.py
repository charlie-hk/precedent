"""A concrete use case: read one moment the way a trader would.

It finds a realistic trigger (price returning into the strongest supply/demand
zone), then at that exact bar prints:
  - the structure context (trend, which zone, recent liquidity sweep)
  - the historical-analog verdict (what similar past windows did next)
  - a plain-language read of how to use it
  - and, because this is historical sample data, what actually happened after.

The read itself uses only information available at that bar (no look-ahead).
The "what actually happened" line is hindsight, shown only to make the example
tangible.

    python examples/trade_read.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from precedent import data_pipeline as dp  # noqa: E402
from precedent import structure as st  # noqa: E402
from precedent.similarity import AnalogSearch  # noqa: E402

WINDOW, HORIZON = 32, 12


def find_retest(df, zone, start_from):
    """First bar after a zone is created where price trades back into it."""
    lows, highs = df["low"].to_numpy(), df["high"].to_numpy()
    for i in range(max(zone.start_index + 6, start_from), len(df) - HORIZON - 1):
        if lows[i] <= zone.top and highs[i] >= zone.bottom:
            return i
    return None


def main():
    df = dp.load_csv(str(Path(__file__).resolve().parents[1] / "data/sample_xauusd_h1.csv"))
    res = st.analyze(df)
    labels, zones, sweeps = res["labels"], res["zones"], res["sweeps"]

    # Pick the strongest zone that gets retested with enough history behind it.
    trigger = None
    for z in zones:
        i = find_retest(df, z, start_from=WINDOW + 50)
        if i is not None:
            trigger, zone = i, z
            break
    if trigger is None:
        print("No clean retest found in the sample.")
        return

    i = trigger
    trend = labels["trend"].iloc[i]
    bias = "short" if zone.kind == "supply" else "long"
    recent_sweep = any(abs(i - s) <= 5 for s in sweeps)

    print("== The moment ==")
    print(f"time: {df.index[i]}   price: {df['close'].iloc[i]:.2f}")
    print(f"trend read: {trend}")
    print(f"at a {zone.kind} zone {zone.bottom:.2f}-{zone.top:.2f} (strength {zone.strength:.2f})")
    print(f"recent liquidity sweep nearby: {'yes' if recent_sweep else 'no'}")
    print(f"=> structure bias here: {bias.upper()}")

    print("\n== What history says (analogs) ==")
    search = AnalogSearch(df, window=WINDOW, horizon=HORIZON)
    q = search.query(i, k=25)
    if not q.analogs:
        print("not enough history for an analog read at this bar")
        return
    mfr, wr, disp = q.mean_forward_return, q.win_rate, q.dispersion
    print(f"considered {q.n_considered} past windows, kept {len(q.analogs)}")
    print(f"mean forward return of analogs: {mfr:+.4f}")
    print(f"share that rose afterwards:     {wr:.0%}")
    print(f"dispersion (lower = more reliable): {disp:.4f}")

    # Turn the numbers into a confidence read.
    analog_dir = "up" if mfr > 0 else "down"
    agree = (bias == "short" and analog_dir == "down") or (bias == "long" and analog_dir == "up")
    disp_thr = 0.008  # baseline; the calibrated threshold is reserved
    reliable = disp < disp_thr

    print("\n== How to read it ==")
    if agree and reliable:
        verdict = (f"History leans the SAME way as the structure ({bias}), and analogs "
                   f"agree ({disp:.4f} dispersion). Precedent supports the setup.")
    elif agree and not reliable:
        verdict = (f"History leans {bias} too, but analogs are scattered "
                   f"({disp:.4f}). Weak precedent: smaller size or wait for more confirmation.")
    elif not agree and reliable:
        verdict = (f"Warning: structure says {bias}, but similar pasts moved {analog_dir}. "
                   f"Conflicting read: stand aside.")
    else:
        verdict = "No clear precedent (mixed direction and high dispersion). Stand aside."
    print(verdict)

    realised = float(np.log(df["close"].iloc[i + HORIZON] / df["close"].iloc[i]))
    print("\n== What actually happened after (hindsight only) ==")
    print(f"forward return over next {HORIZON} bars: {realised:+.4f} "
          f"({'up' if realised > 0 else 'down'})")


if __name__ == "__main__":
    main()
