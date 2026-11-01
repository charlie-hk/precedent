"""Load OHLCV straight from a running MetaTrader 5 terminal.

Returns the same frame shape as `data_pipeline.load_csv`, so the rest of
Precedent (structure, analog search, backtest) works on live data unchanged.

Requirements:
  - Windows, with the MetaTrader 5 terminal installed and logged in to your broker.
  - `pip install MetaTrader5`.
  - The terminal running while this is called.

The MetaTrader5 package is Windows-only, so this module imports it lazily and
is never needed on other platforms.
"""
from __future__ import annotations

import pandas as pd

TIMEFRAMES = ("M1", "M5", "M15", "H1", "H4", "D1")


def load_mt5(symbol: str = "XAUUSD", timeframe: str = "H1", bars: int = 5000) -> pd.DataFrame:
    """Pull the most recent `bars` candles of `symbol` at `timeframe` from MT5.

    `symbol` must match your broker's exact name (common variants: XAUUSD, GOLD,
    XAUUSD.m). If you are unsure, print `available_symbols()` below.
    """
    try:
        import MetaTrader5 as mt5
    except ImportError as e:
        raise RuntimeError(
            "MetaTrader5 is not installed. On Windows: pip install MetaTrader5, "
            "and make sure the MT5 terminal is installed and logged in."
        ) from e

    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    if timeframe not in tf_map:
        raise ValueError(f"timeframe must be one of {TIMEFRAMES}")

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}. Is the terminal open?")

    try:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(
                f"Symbol '{symbol}' not found. Check the exact name in your broker's "
                f"Market Watch (it may be GOLD or XAUUSD.m). See available_symbols()."
            )
        rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, bars)
    finally:
        mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError("No candles returned. Check the symbol, timeframe and that history is loaded.")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"tick_volume": "volume"}).set_index("time").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def available_symbols(contains: str = "") -> list[str]:
    """List broker symbols (optionally filtered), to find the exact gold name."""
    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}.")
    try:
        syms = [s.name for s in mt5.symbols_get()]
    finally:
        mt5.shutdown()
    c = contains.upper()
    return [s for s in syms if c in s.upper()] if c else syms
