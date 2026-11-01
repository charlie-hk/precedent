"""End-to-end demo on the bundled synthetic sample.

Runs the full pipeline: load -> integrity -> structure -> one analog query ->
walk-forward evaluation, and prints a short report. Everything shown is real
output computed from the data, not canned.

    python demo.py            # uses data/sample_xauusd_h1.csv
    python demo.py --csv my_xauusd.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from precedent import data_pipeline as dp
from precedent import report
from precedent import structure as st
from precedent.backtest import walk_forward
from precedent.similarity import AnalogSearch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/sample_xauusd_h1.csv")
    ap.add_argument("--timeframe", default="H1")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=12)
    args = ap.parse_args()

    df = dp.load_csv(args.csv)
    rep = dp.check_integrity(df, args.timeframe)
    print("== Data ==")
    print(f"rows={rep.rows}  {rep.start.date()} -> {rep.end.date()}  "
          f"gaps={rep.gaps}  bad_ohlc={rep.bad_ohlc}  integrity_ok={rep.ok()}")
    for w in rep.warnings:
        print(f"  warning: {w}")

    res = st.analyze(df)
    print("\n== Structure ==")
    print(report.text_report(df, res))
    try:
        from precedent.viz import plot_structure
        out = plot_structure(df, res, "out/structure.png")
        print(f"\nchart with marked BOS/CHOCH saved to: {out}")
    except Exception as e:
        print(f"\n(chart skipped: {e.__class__.__name__}. pip install matplotlib to enable)")

    print("\n== Historical analog (latest bar) ==")
    search = AnalogSearch(df, window=args.window, horizon=args.horizon)
    q = search.query(len(df) - args.horizon - 1, k=25)
    if q.analogs:
        print(f"considered {q.n_considered} past windows, kept {len(q.analogs)}")
        print(f"analog mean forward return: {q.mean_forward_return:+.4f}  "
              f"win_rate={q.win_rate:.2f}  dispersion={q.dispersion:.4f}")
        print("closest 3 precedents:")
        for a in q.analogs[:3]:
            print(f"  {a.time.date()}  dist={a.distance:.3f}  fwd={a.forward_return:+.4f}")
    else:
        print("not enough history for an analog query")

    print("\n== Walk-forward evaluation (out-of-sample half) ==")
    wf = walk_forward(df, window=args.window, horizon=args.horizon)
    for k, v in wf.as_dict().items():
        print(f"  {k}: {v}")
    print("\nNote: synthetic data. On a near-random series the analog edge should be")
    print("near zero; that is the honest baseline. Real edge comes from the reserved")
    print("learned metric and feature set (see README).")


if __name__ == "__main__":
    main()
