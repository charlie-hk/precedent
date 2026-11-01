"""Run Precedent on live data from your own MetaTrader 5 terminal.

Windows only, MT5 terminal open and logged in, and `pip install MetaTrader5`.

    python scripts/run_mt5.py --symbol XAUUSD --timeframe H1 --bars 5000
    python scripts/run_mt5.py --list gold          # find your broker's gold symbol

It prints the structure read of the latest bars and the historical-analog
verdict for the most recent completed bar, using only past data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from precedent import data_pipeline as dp  # noqa: E402
from precedent import structure as st  # noqa: E402
from precedent.mt5_loader import available_symbols, load_mt5  # noqa: E402
from precedent.similarity import AnalogSearch  # noqa: E402

WINDOW, HORIZON = 32, 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--timeframe", default="H1")
    ap.add_argument("--bars", type=int, default=5000)
    ap.add_argument("--list", metavar="TEXT", help="list broker symbols containing TEXT and exit")
    args = ap.parse_args()

    if args.list is not None:
        for s in available_symbols(args.list):
            print(s)
        return

    df = load_mt5(args.symbol, args.timeframe, args.bars)
    rep = dp.check_integrity(df, args.timeframe)
    print(f"== {args.symbol} {args.timeframe} ==")
    print(f"rows={rep.rows}  {rep.start} -> {rep.end}  gaps={rep.gaps}  bad_ohlc={rep.bad_ohlc}")

    res = st.analyze(df)
    labels = res["labels"]
    print("\n== Structure (now) ==")
    print(f"trend: {labels['trend'].iloc[-1]}  "
          f"BOS={(labels['event']=='BOS').sum()}  CHOCH={(labels['event']=='CHOCH').sum()}  "
          f"zones={len(res['zones'])}  sweeps={len(res['sweeps'])}")
    if res["zones"]:
        z = res["zones"][0]
        print(f"strongest zone: {z.kind} {z.bottom:.2f}-{z.top:.2f} (strength {z.strength:.2f})")

    print("\n== Analog read (latest completed bar) ==")
    search = AnalogSearch(df, window=WINDOW, horizon=HORIZON)
    i = len(df) - HORIZON - 1
    q = search.query(i, k=25)
    if not q.analogs:
        print("not enough history for an analog read")
        return
    print(f"as of {df.index[i]}  price {df['close'].iloc[i]:.2f}")
    print(f"analog mean forward return: {q.mean_forward_return:+.4f}  "
          f"win_rate={q.win_rate:.0%}  dispersion={q.dispersion:.4f}")
    lean = "up" if q.mean_forward_return > 0 else "down"
    reliable = q.dispersion < 0.008
    print(f"=> precedent leans {lean}; "
          f"{'reliable (tight analogs)' if reliable else 'weak (scattered analogs) - caution'}")
    print("\nResearch read only. Not a signal, no guarantee. You decide and manage the risk.")


if __name__ == "__main__":
    main()
