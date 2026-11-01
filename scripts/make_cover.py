"""Render a social-preview cover image for the repo (1280x640).

Uses the project's own structure + analog search on the sample data, so the
picture shows the real idea: a current window and its nearest historical analog.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from precedent import data_pipeline as dp  # noqa: E402
from precedent import structure as st  # noqa: E402
from precedent.similarity import AnalogSearch  # noqa: E402

BG = "#0f1115"
PANEL = "#0b0d12"
LINE = "#c7cad1"
ACCENT = "#3a8bff"
ACCENT2 = "#f5a623"
MUTED = "#8b93a1"

W = 32
H = 12

df = dp.load_csv(str(Path(__file__).resolve().parents[1] / "data/sample_xauusd_h1.csv"))
# Use a clean recent slice for the backdrop.
view = df.iloc[-520:].reset_index(drop=True)
close = view["close"].to_numpy()

# Real analog: query the last completed window, find its nearest past analog.
search = AnalogSearch(view, window=W, horizon=H)
qi = len(view) - H - 1
res = search.query(qi, k=10)
analog_i = res.analogs[0].index if res.analogs else len(view) // 3

res_struct = st.analyze(view)
swings = res_struct["swings"]

fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_facecolor(BG)
ax.axis("off")

x = np.arange(len(close))
# Price line sits in the lower 60% so text has room up top.
ymin, ymax = close.min(), close.max()
pad = (ymax - ymin) * 0.08
ax.set_xlim(-5, len(close) + 5)
ax.set_ylim(ymin - pad, ymin + (ymax - ymin) * 2.6)

ax.plot(x, close, color=LINE, lw=1.4, alpha=0.9)

# Swing dots.
for s in swings:
    if s.index < len(close):
        ax.scatter(s.index, close[s.index], s=9,
                   color=ACCENT if s.kind == "low" else ACCENT2, alpha=0.65, zorder=3)


def box(center, color, label, align="center"):
    a, b = center - W + 1, center
    ylo = close[a:b + 1].min() - pad * 0.6
    yhi = close[a:b + 1].max() + pad * 0.6
    ax.add_patch(Rectangle((a, ylo), (b - a), (yhi - ylo), fill=True,
                           facecolor=color, alpha=0.12, edgecolor=color, lw=1.6, zorder=2))
    if align == "left":
        tx, ha = max(a, 8), "left"
    elif align == "right":
        tx, ha = min(b, len(close) - 8), "right"
    else:
        tx, ha = (a + b) / 2, "center"
    ax.text(tx, yhi + pad * 0.5, label, color=color, fontsize=11,
            ha=ha, va="bottom", fontweight="bold")


box(qi, ACCENT, "now", align="right")
box(analog_i, ACCENT2, "nearest precedent", align="left")

# Connecting arrow between the two windows.
ax.add_patch(FancyArrowPatch(
    (analog_i, close[analog_i] + pad * 2.2), (qi, close[qi] + pad * 2.2),
    connectionstyle="arc3,rad=-0.35", arrowstyle="-|>", mutation_scale=16,
    color=MUTED, lw=1.2, alpha=0.8, zorder=4))

# Title block.
ax.text(12, ymin + (ymax - ymin) * 2.42, "Precedent", color="white",
        fontsize=56, fontweight="bold", ha="left", va="top")
ax.text(14, ymin + (ymax - ymin) * 2.02,
        "Market-structure recognition  ·  historical-analog search",
        color=LINE, fontsize=19, ha="left", va="top")
ax.text(14, ymin + (ymax - ymin) * 1.80,
        "When has price looked structurally like this before, and what happened next?",
        color=MUTED, fontsize=14, ha="left", va="top", style="italic")

# Accent underline, placed clear of the text.
ax.plot([14, 210], [ymin + (ymax - ymin) * 1.68] * 2, color=ACCENT, lw=3, alpha=0.9)

# Footer credit.
ax.text(len(close) - 6, ymin - pad * 0.2, "XAU/USD   ·   Python   ·   A. Amini",
        color=MUTED, fontsize=12, ha="right", va="bottom")

out = Path(__file__).resolve().parents[1] / "docs/cover.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, facecolor=BG)
print(f"wrote {out}")
