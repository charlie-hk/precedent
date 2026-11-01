"""Draw the structure on a real candlestick chart.

Candles, swing points, supply/demand zones, and BOS/CHOCH each drawn as a line
at the level that was broken, labelled at the bar where it was confirmed. This is
"mark the CHOCHs" done properly, not a text list.

Needs matplotlib (`pip install matplotlib`).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BG = "#0f1115"
GRID = "#1b1f29"
UP = "#36c28b"
DOWN = "#e5534b"
BOS_C = "#3a8bff"
CHOCH_C = "#f5a623"
TXT = "#c7cad1"


def plot_structure(df: pd.DataFrame, result: dict, out_path: str, last: int = 180) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    view = df.iloc[-last:]
    i0 = len(df) - len(view)
    o = view["open"].to_numpy()
    h = view["high"].to_numpy()
    low = view["low"].to_numpy()
    c = view["close"].to_numpy()

    fig, ax = plt.subplots(figsize=(15, 7.5), dpi=120)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, lw=0.6)

    # Candles.
    for k in range(len(view)):
        col = UP if c[k] >= o[k] else DOWN
        ax.plot([k, k], [low[k], h[k]], color=col, lw=0.8, zorder=2)
        lo, hi = min(o[k], c[k]), max(o[k], c[k])
        ax.add_patch(Rectangle((k - 0.3, lo), 0.6, max(hi - lo, 1e-6),
                               facecolor=col, edgecolor=col, zorder=3))

    # Supply/demand zones that overlap the visible price range.
    pmin, pmax = low.min(), h.max()
    for z in result["zones"][:14]:
        if z.top < pmin or z.bottom > pmax:
            continue
        col = DOWN if z.kind == "supply" else UP
        ax.axhspan(z.bottom, z.top, color=col, alpha=0.07, zorder=1)

    # BOS / CHOCH: draw the broken level as a line up to the break bar + a label.
    for e in result["events"]:
        if e.index < i0:
            continue
        xi = e.index - i0
        col = CHOCH_C if e.kind == "CHOCH" else BOS_C
        x_from = max(0, xi - 18)
        ax.plot([x_from, xi], [e.level, e.level], color=col, ls="--", lw=1.1, alpha=0.9, zorder=4)
        ax.scatter([xi], [e.level], color=col, s=18, zorder=5)
        dy = 10 if e.trend_after == "up" else -16
        ax.annotate(e.kind, (xi, e.level), color=col, fontsize=8, fontweight="bold",
                    xytext=(2, dy), textcoords="offset points", zorder=6)

    trend = result["labels"]["trend"].iloc[-1]
    ax.set_title(f"{last} bars   ·   current trend: {trend}   "
                 f"(blue = BOS, orange = CHOCH, dashed line = broken level)",
                 color=TXT, fontsize=11)
    ax.tick_params(colors="#8b93a1", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#262b36")
    ax.set_xlim(-1, len(view))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    return str(out)
