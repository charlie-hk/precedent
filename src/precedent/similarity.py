"""Historical-analog search.

Given the window ending "now", find the most structurally similar windows in the
past and summarise what happened *after* each of them. The hard rule from the
brief is enforced: a query at bar i may only retrieve library entries whose
outcome completed at or before bar i. Nothing uses future information to pick or
score an analog.

The retrieval here is an honest k-nearest-neighbour baseline. The learned metric
and the setup-quality scoring that rank analogs in production are reserved.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import build_library, window_embedding


@dataclass
class Analog:
    index: int
    time: pd.Timestamp
    distance: float
    forward_return: float  # realised return over the horizon after this analog


@dataclass
class AnalogResult:
    query_index: int
    query_time: pd.Timestamp
    analogs: list[Analog]
    n_considered: int

    @property
    def mean_forward_return(self) -> float:
        if not self.analogs:
            return float("nan")
        return float(np.mean([a.forward_return for a in self.analogs]))

    @property
    def win_rate(self) -> float:
        """Share of analogs whose forward return was positive. Direction-agnostic
        callers can interpret relative to their intended side."""
        if not self.analogs:
            return float("nan")
        pos = sum(1 for a in self.analogs if a.forward_return > 0)
        return pos / len(self.analogs)

    @property
    def dispersion(self) -> float:
        """Std of analog outcomes: high dispersion = unreliable precedent."""
        if len(self.analogs) < 2:
            return float("nan")
        return float(np.std([a.forward_return for a in self.analogs]))


def _forward_return(df: pd.DataFrame, i: int, horizon: int) -> float:
    c0 = df["close"].iloc[i]
    c1 = df["close"].iloc[i + horizon]
    return float(np.log(c1 / c0))


class AnalogSearch:
    def __init__(self, df: pd.DataFrame, window: int = 32, horizon: int = 12, step: int = 1):
        self.df = df
        self.window = window
        self.horizon = horizon
        self.X, self.meta = build_library(df, window, step, horizon)
        if len(self.X):
            # Precompute realised outcomes for every library entry.
            self.fwd = np.array([_forward_return(df, m["index"], horizon) for m in self.meta])
            # Cache L2-normalised vectors for cosine distance.
            norms = np.linalg.norm(self.X, axis=1, keepdims=True)
            self._Xn = self.X / np.clip(norms, 1e-9, None)
        else:
            self.fwd = np.empty(0)
            self._Xn = self.X

    def query(self, query_index: int, k: int = 25) -> AnalogResult:
        """Return the k closest past analogs to the window ending at `query_index`.

        Only analogs whose outcome window finished at/before `query_index` are
        eligible, which is what keeps the search free of look-ahead.
        """
        qt = self.df.index[query_index]
        q = window_embedding(self.df, query_index, self.window)
        if q is None or len(self.X) == 0:
            return AnalogResult(query_index, qt, [], 0)

        qn = q / max(np.linalg.norm(q), 1e-9)

        # Eligibility: library entry's outcome ends before the query bar.
        idx = np.array([m["index"] for m in self.meta])
        eligible = (idx + self.horizon) <= query_index
        if not eligible.any():
            return AnalogResult(query_index, qt, [], 0)

        dists = 1.0 - (self._Xn[eligible] @ qn)
        sub_meta = [m for m, e in zip(self.meta, eligible) if e]
        sub_fwd = self.fwd[eligible]

        order = np.argsort(dists)[:k]
        analogs = [
            Analog(sub_meta[j]["index"], sub_meta[j]["time"], float(dists[j]), float(sub_fwd[j]))
            for j in order
        ]
        return AnalogResult(query_index, qt, analogs, int(eligible.sum()))
