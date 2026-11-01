"""Generate a synthetic XAU/USD H1 OHLCV sample for the demo.

This is SYNTHETIC data for demonstration only, not real market data. It is
built to contain trends, ranges and pullbacks so the structure detector and
similarity search have realistic shapes to work on. Point the pipeline at your
own broker/MT5 export (same columns) to run on real data.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n: int = 3000, seed: int = 7, start_price: float = 2000.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # A slowly changing drift produces alternating trend/range regimes.
    regime = np.cumsum(rng.normal(0, 0.0004, n))
    regime = np.tanh(regime - regime.mean()) * 0.0012
    vol = 0.0018 + 0.0010 * np.abs(np.sin(np.linspace(0, 9 * np.pi, n)))

    returns = regime + rng.normal(0, 1, n) * vol
    close = start_price * np.exp(np.cumsum(returns))

    # Build OHLC around the close path with plausible w", intrabar ranges.
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]
    bar_range = np.abs(returns) * close * rng.uniform(1.2, 3.0, n) + close * 0.0004
    high = np.maximum(open_, close) + bar_range * rng.uniform(0.2, 1.0, n)
    low = np.minimum(open_, close) - bar_range * rng.uniform(0.2, 1.0, n)
    volume = (rng.gamma(2.0, 1.0, n) * 500 + 100).round()

    idx = pd.date_range("2023-01-02 00:00", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "time": idx,
            "open": open_.round(2),
            "high": high.round(2),
            "low": low.round(2),
            "close": close.round(2),
            "volume": volume.astype(int),
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="data/sample_xauusd_h1.csv")
    args = p.parse_args()

    df = generate(args.n, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows to {out}")
