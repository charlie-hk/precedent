"""Encode a window of price action into a fixed-length vector.

The point of the brief is to compare the *shape and development* of price, not
single candles, so a window is normalised to be invariant to absolute price and
to its own volatility. Two moves with the same structure but different price
levels should land close together.

This is the BASELINE encoder. It is deliberately simple and readable. The tuned
feature set that carries the real retrieval edge (microstructure, multi-timeframe
alignment, volume-shape terms) is reserved; see README and `learning.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def window_embedding(df: pd.DataFrame, end_index: int, window: int = 32) -> np.ndarray | None:
    """Vector for the `window` bars ending at `end_index` (inclusive).

    Uses only bars <= end_index, so it is safe to call bar-by-bar in a backtest.
    Returns None if there is not enough history.
    """
    start = end_index - window + 1
    if start < 0:
        return None

    w = df.iloc[start : end_index + 1]
    close = w["close"].to_numpy()
    high = w["high"].to_numpy()
    low = w["low"].to_numpy()

    # Log returns, standardised: removes price level and scales by local vol.
    rets = np.diff(np.log(close), prepend=np.log(close[0]))
    sd = rets.std() or 1e-9
    rets_z = rets / sd

    # Normalised path relative to the window's first close.
    path = (close / close[0]) - 1.0
    rng = (high - low) / close  # per-bar range, a volatility-shape term

    feat = np.concatenate([rets_z, path, rng])
    return feat.astype(np.float32)


def build_library(df: pd.DataFrame, window: int = 32, step: int = 1, horizon: int = 12):
    """Build the historical library of (embedding, metadata) for every eligible bar.

    `horizon` is how far forward we later measure the outcome. We stop `horizon`
    bars before the end so every library entry has a realised, past outcome.
    """
    X, meta = [], []
    last = len(df) - horizon - 1
    for i in range(window - 1, last, step):
        emb = window_embedding(df, i, window)
        if emb is None:
            continue
        X.append(emb)
        meta.append({"index": i, "time": df.index[i], "close": float(df["close"].iloc[i])})
    return (np.vstack(X) if X else np.empty((0,))), meta
