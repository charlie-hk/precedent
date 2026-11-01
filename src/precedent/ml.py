"""A real, trained machine-learning model for next-move direction.

This is not a stub and not a nearest-neighbour lookup: it fits a gradient-boosted
classifier on causal window features to predict whether the forward return over
`horizon` bars is positive, and evaluates it walk-forward (train on the past,
test on the future it never saw).

It is deliberately honest. On real markets this kind of model usually lands near
chance, and the report says so plainly, including a comparison to the naive base
rate. A model that is only shown its out-of-sample accuracy cannot flatter itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import window_embedding


@dataclass
class MLResult:
    n_train: int
    n_test: int
    accuracy: float
    base_rate: float  # accuracy of always predicting the majority class
    high_conf_accuracy: float  # accuracy where the model was >= conf_threshold sure
    n_high_conf: int

    def as_dict(self) -> dict:
        return {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "accuracy": round(self.accuracy, 4),
            "base_rate": round(self.base_rate, 4),
            "edge_over_base": round(self.accuracy - self.base_rate, 4),
            "high_conf_accuracy": round(self.high_conf_accuracy, 4),
            "n_high_conf": self.n_high_conf,
        }


def build_dataset(df: pd.DataFrame, window: int = 32, horizon: int = 12, step: int = 1):
    """Causal features X and forward-direction labels y for every eligible bar."""
    X, y, idx = [], [], []
    last = len(df) - horizon - 1
    close = df["close"].to_numpy()
    for i in range(window - 1, last, step):
        emb = window_embedding(df, i, window)
        if emb is None:
            continue
        fwd = np.log(close[i + horizon] / close[i])
        X.append(emb)
        y.append(1 if fwd > 0 else 0)
        idx.append(i)
    return np.vstack(X), np.array(y), np.array(idx)


def walk_forward_fit(df: pd.DataFrame, window: int = 32, horizon: int = 12,
                     train_frac: float = 0.6, conf_threshold: float = 0.60) -> MLResult:
    """Train on the first `train_frac` of the data, test on the rest.

    The split is chronological, and the test labels' outcomes all occur after the
    training window, so the model is judged only on data it could not have seen.
    """
    X, y, idx = build_dataset(df, window, horizon)
    if len(y) < 200:
        raise ValueError("Not enough data to train and test meaningfully.")

    cut = int(len(y) * train_frac)
    # Leave a horizon-sized gap so no training label overlaps the test window.
    gap = horizon
    Xtr, ytr = X[:cut], y[:cut]
    Xte, yte = X[cut + gap:], y[cut + gap:]

    model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                           max_iter=300, l2_regularization=1.0,
                                           random_state=0)
    model.fit(Xtr, ytr)

    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = float((pred == yte).mean())
    base = float(max(yte.mean(), 1 - yte.mean()))

    conf_mask = (proba >= conf_threshold) | (proba <= 1 - conf_threshold)
    if conf_mask.any():
        hc_acc = float((pred[conf_mask] == yte[conf_mask]).mean())
    else:
        hc_acc = float("nan")

    return MLResult(len(ytr), len(yte), acc, base, hc_acc, int(conf_mask.sum()))
