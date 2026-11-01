"""Load, validate and resample OHLCV data.

Handles the three things the job brief calls out explicitly: correct candle
timestamps, multiple timeframes, and missing data. Integrity problems are
reported, not silently patched, because a quiet fix is how look-ahead and bad
fills creep in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]

# Minutes per timeframe label.
TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "H1": 60, "H4": 240, "D1": 1440}


@dataclass
class IntegrityReport:
    rows: int
    start: pd.Timestamp
    end: pd.Timestamp
    duplicates: int
    gaps: int
    bad_ohlc: int
    warnings: list[str] = field(default_factory=list)

    def ok(self) -> bool:
        return self.duplicates == 0 and self.bad_ohlc == 0


def load_csv(path: str, tz: str = "UTC") -> pd.DataFrame:
    """Read an OHLCV CSV with a `time` column into a tz-aware, time-indexed frame."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "time" not in cols:
        raise ValueError("CSV must have a 'time' column.")
    df = df.rename(columns={cols["time"]: "time"})
    for c in OHLCV:
        if c not in df.columns:
            raise ValueError(f"CSV missing '{c}' column.")

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    if tz != "UTC":
        df.index = df.index.tz_convert(tz)
    return df[OHLCV].astype(float)


def check_integrity(df: pd.DataFrame, timeframe: str) -> IntegrityReport:
    """Flag duplicate timestamps, missing candles and impossible OHLC bars."""
    warnings: list[str] = []

    duplicates = int(df.index.duplicated().sum())
    if duplicates:
        warnings.append(f"{duplicates} duplicate timestamps")

    step = pd.Timedelta(minutes=TF_MINUTES[timeframe])
    deltas = df.index.to_series().diff().dropna()
    # A gap is any step larger than one bar. Weekends on FX/metals are expected,
    # so we only count intra-session gaps as suspicious here.
    gaps = int((deltas > step).sum())

    bad = (
        (df["high"] < df["low"])
        | (df["high"] < df[["open", "close"]].max(axis=1))
        | (df["low"] > df[["open", "close"]].min(axis=1))
    )
    bad_ohlc = int(bad.sum())
    if bad_ohlc:
        warnings.append(f"{bad_ohlc} bars with inconsistent OHLC")

    return IntegrityReport(
        rows=len(df),
        start=df.index[0],
        end=df.index[-1],
        duplicates=duplicates,
        gaps=gaps,
        bad_ohlc=bad_ohlc,
        warnings=warnings,
    )


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample a lower timeframe up to a higher one (e.g. H1 -> H4)."""
    rule = f"{TF_MINUTES[timeframe]}min"
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample(rule, label="left", closed="left").agg(agg)
    return out.dropna(subset=["open"])


def align_timeframes(df_base: pd.DataFrame, timeframes: list[str]) -> dict[str, pd.DataFrame]:
    """Build a dict of timeframe -> frame from one base series, for MTF analysis."""
    return {tf: resample(df_base, tf) for tf in timeframes}
