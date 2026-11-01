"""Draw the structure on the chart: swings, BOS/CHOCH with broken levels, zones.

This is what "mark the CHOCHs" means in practice. Saves a PNG so you can see
exactly where each event was detected and which level it broke.

Needs matplotlib (`pip install matplotlib`).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BG = "#0f1115"
LINE = "#c7cad1"
UP = "#36c28b"
DOWN = "#e5534b"
BOS_C = "#3a8bff"
CHOCH_C = "#f5a623"


def plot_structure(df: pd.DataFrame, result: dict, out_path: str, last: int = 300) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    view = df.iloc[-last:]
    i0 = len(df) - len(view)
    x = range(len(view))
    close = view["close"].to_numpy()

    fig, ax = plt.subplots(figsize=(14, 7), dpi=110)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.plot(x, close, color=LINE, lw=1.1)

    # Swings in view.
    for s in result["swings"]:
        if s.index >= i0:
            ax.scatter(s.index - i0, s.price, s=12,
                       color=UP if s.kind == "low" else DOWN, zorder=3)

    # Zones overlapping the view.
    for z in result["zones"][:12]:
        c = DOWN if z.kind == "supply" else UP
        ax.axhspan(z.bottom, z.top, color=c, alpha=0.08)

    # BOS / CHOCH markers with the broken level drawn as a dashed line.
    for e in result["events"]:
        if e.index < i0:
            continue
        xi = e.index - i0
        c = CHOCH_C if e.kind == "CHOCH" else BOS_C
        ax.axhline(e.level, xmin=max(0, (xi - 20) / len(view)), xmax=min(1, (xi + 2) / len(view)),
                   color=c, ls="--", lw=1.0, alpha=0.7)
        ax.annotate(e.kind, (xi, e.price), color=c, fontsize=8, fontweight="bold",
                    xytext=(0, 8 if e.kind == "BOS" else -14), textcoords="offset points",
                    ha="center")

    ax.set_title(f"Structure  ·  last {len(view)} bars  ·  trend: {result['labels']['trend'].iloc[-1]}",
                 color=LINE, fontsize=12)
    ax.tick_params(colors="#8b93a1", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#262b36")
    ax.set_xlim(0, len(view))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    return str(out)
