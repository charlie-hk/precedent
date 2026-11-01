import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from precedent import data_pipeline as dp  # noqa: E402
from precedent import structure as st  # noqa: E402
from precedent.features import window_embedding  # noqa: E402
from precedent.similarity import AnalogSearch  # noqa: E402
from precedent import learning  # noqa: E402


def _frame(n=400, seed=1):
    rng = np.random.default_rng(seed)
    close = 2000 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    high = close * (1 + rng.uniform(0, 0.002, n))
    low = close * (1 - rng.uniform(0, 0.002, n))
    opn = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open": opn, "high": np.maximum(high, np.maximum(opn, close)),
                         "low": np.minimum(low, np.minimum(opn, close)), "close": close,
                         "volume": rng.integers(100, 1000, n)}, index=idx)


def test_integrity_flags_bad_ohlc():
    df = _frame()
    df.iloc[10, df.columns.get_loc("high")] = df["low"].iloc[10] - 1  # high < low
    rep = dp.check_integrity(df, "H1")
    assert rep.bad_ohlc >= 1
    assert not rep.ok()


def test_resample_aggregates_correctly():
    df = _frame(n=240)
    h4 = dp.resample(df, "H4")
    # Each H4 bar's high is the max of its four H1 highs.
    assert h4["high"].iloc[0] == df["high"].iloc[0:4].max()
    assert h4["open"].iloc[0] == df["open"].iloc[0]


def test_embedding_is_causal_and_fixed_length():
    df = _frame()
    e1 = window_embedding(df, 100, window=32)
    # Changing a FUTURE bar must not change an embedding ending at bar 100.
    df2 = df.copy()
    df2.iloc[200, df2.columns.get_loc("close")] *= 1.05
    e2 = window_embedding(df2, 100, window=32)
    assert e1 is not None and np.allclose(e1, e2)
    assert len(e1) == 32 * 3


def test_analog_search_has_no_lookahead():
    df = _frame()
    search = AnalogSearch(df, window=32, horizon=12)
    i = 200
    res = search.query(i, k=10)
    # Every returned analog's outcome window must finish at or before the query bar.
    for a in res.analogs:
        assert a.index + 12 <= i


def test_structure_runs_and_labels_trend():
    df = _frame()
    res = st.analyze(df)
    assert set(res["labels"].columns) == {"trend", "event"}
    assert res["labels"]["trend"].iloc[-1] in {"up", "down", "range"}


def test_reserved_components_raise():
    with pytest.raises(learning.ReservedComponent):
        learning.LearnedMetric().fit(np.zeros((2, 2)), [])
    with pytest.raises(learning.ReservedComponent):
        learning.SetupQualityModel().score(np.zeros(2))
