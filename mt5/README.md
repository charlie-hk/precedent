# PrecedentStructure (MT5 indicator)

Marks swings, **BOS** and **CHOCH** directly on your MetaTrader 5 chart, using the
same structure logic as the Python project. It draws on the chart itself, not an
image.

## Install

1. In MetaTrader 5: **File -> Open Data Folder**.
2. Go to `MQL5\Indicators`.
3. Copy `PrecedentStructure.mq5` there.
4. In MT5 open **MetaEditor** (the IDE button), open the file, press **Compile**
   (F7). It should compile with no errors.
5. Back in MT5, open a gold (XAUUSD) chart. In the **Navigator**, under
   **Indicators**, drag **PrecedentStructure** onto the chart.

## What you see

- Small dots at swing highs (red) and lows (green).
- A dashed line at the level that was broken, with a label:
  - **BOS** (blue): break in the direction of the trend (continuation).
  - **CHOCH** (orange): first break against the trend (possible turn).

## Inputs

- `InpSwing` — swing strength (bars on each side). Higher = fewer, bigger swings.
- `InpBars` — how many recent bars to analyse.
- `InpShowSwings` — show the swing dots.
- `InpLevelBack` — how far back to draw the broken-level line.
- Colors for BOS, CHOCH and swings.

It only reads the chart and draws. It places no trades.
