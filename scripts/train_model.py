"""Train and evaluate the direction model, walk-forward, and report honestly.

    python scripts/train_model.py                      # sample data
    python scripts/train_model.py --csv your.csv
    python scripts/train_model.py --mt5 --symbol XAUUSD --timeframe H1 --bars 20000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from precedent import data_pipeline as dp  # noqa: E402
from precedent.ml import walk_forward_fit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/sample_xauusd_h1.csv")
    ap.add_argument("--mt5", action="store_true")
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--timeframe", default="H1")
    ap.add_argument("--bars", type=int, default=20000)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=12)
    args = ap.parse_args()

    if args.mt5:
        from precedent.mt5_loader import load_mt5
        df = load_mt5(args.symbol, args.timeframe, args.bars)
        label = f"{args.symbol} {args.timeframe} (MT5)"
    else:
        df = dp.load_csv(args.csv)
        label = args.csv

    res = walk_forward_fit(df, window=args.window, horizon=args.horizon)
    print(f"== Direction model, walk-forward on {label} ==")
    d = res.as_dict()
    for k, v in d.items():
        print(f"  {k}: {v}")

    print("\nHow to read this:")
    print(f"- 'accuracy' is out-of-sample. 'base_rate' is what you get by always")
    print(f"  guessing the more common direction.")
    edge = d["edge_over_base"]
    if edge <= 0.01:
        print(f"- edge over base is {edge:+.4f}: essentially no predictive edge here.")
        print("  That is the honest result for a model like this on this data.")
    else:
        print(f"- edge over base is {edge:+.4f}: a small edge. Treat with caution and")
        print("  re-test on more data before trusting it.")
    print("- A real trading edge, if any, needs the reserved feature set and far more")
    print("  data; this is the honest baseline a trained model gives.")


if __name__ == "__main__":
    main()
