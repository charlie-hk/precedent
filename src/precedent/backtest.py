"""Walk-forward evaluation.

Two things are measured, both out-of-sample and causal:

1. Recognition: how stable the structure labels are, and how often a detected
   liquidity sweep is followed by a reversion (a check that the detector marks
   something real, not noise).
2. Analog usefulness: do the historical-analog outcomes carry any signal about
   the forward return, measured only on data after a warm-up split.

This is an honest measurement harness, not a trading strategy. It reports what
is there, including when the edge is weak. The production report adds precision/
recall against a labelled set and transaction-cost-aware expectancy; that
labelled set is reserved.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .similarity import AnalogSearch


@dataclass
class WalkForwardResult:
    n_queries: int
    directional_accuracy: float  # analog-predicted sign vs realised sign
    mean_abs_forward: float
    correlation: float  # analog mean forward vs realised forward
    low_dispersion_accuracy: float  # accuracy on the most-agreeing analogs only

    def as_dict(self) -> dict:
        return {
            "n_queries": self.n_queries,
            "directional_accuracy": round(self.directional_accuracy, 4),
            "mean_abs_forward": round(self.mean_abs_forward, 6),
            "analog_vs_realised_corr": round(self.correlation, 4),
            "low_dispersion_accuracy": round(self.low_dispersion_accuracy, 4),
        }


def walk_forward(
    df: pd.DataFrame,
    window: int = 32,
    horizon: int = 12,
    k: int = 25,
    warmup_frac: float = 0.5,
    step: int = 3,
) -> WalkForwardResult:
    """Query analogs across the out-of-sample half and compare predicted vs realised."""
    search = AnalogSearch(df, window=window, horizon=horizon, step=1)

    start = int(len(df) * warmup_frac)
    end = len(df) - horizon - 1

    preds, reals, disps = [], [], []
    for i in range(start, end, step):
        res = search.query(i, k=k)
        if not res.analogs:
            continue
        realised = float(np.log(df["close"].iloc[i + horizon] / df["close"].iloc[i]))
        preds.append(res.mean_forward_return)
        reals.append(realised)
        disps.append(res.dispersion)

    if len(preds) < 5:
        return WalkForwardResult(len(preds), float("nan"), float("nan"), float("nan"), float("nan"))

    preds, reals, disps = np.array(preds), np.array(reals), np.array(disps)
    directional = float(np.mean(np.sign(preds) == np.sign(reals)))
    mean_abs = float(np.mean(np.abs(reals)))
    corr = float(np.corrcoef(preds, reals)[0, 1]) if np.std(preds) > 0 else float("nan")

    # Restrict to the third of queries where analogs agreed most (lowest dispersion).
    thr = np.nanquantile(disps, 0.33)
    mask = disps <= thr
    low_disp_acc = (
        float(np.mean(np.sign(preds[mask]) == np.sign(reals[mask]))) if mask.sum() else float("nan")
    )

    return WalkForwardResult(len(preds), directional, mean_abs, corr, low_disp_acc)
