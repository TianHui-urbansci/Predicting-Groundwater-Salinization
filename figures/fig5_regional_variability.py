"""Figure 5 - Regional variability of chloride and predictors by state.

One value per well (its median), so states are not weighted by sampling frequency. For
each state the dot is the median, the bar the 25th-75th percentile and the whisker the
75th-95th percentile, on linear axes. The chloride axis is clipped at 400 mg/L; where a
state's 95th percentile lies beyond it, an arrow marks the edge and the value is printed.
The tide panel shows Daily_Max, the daily maximum water level (NAVD88) at each well's
tide gauge. Pennsylvania is excluded.

Usage:  python figures/fig5_regional_variability.py [--data <dataset.csv>]
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FIGURES, data_path, in_population   # noqa: E402
FIGURES.mkdir(exist_ok=True)

SRC = data_path()
OUT = FIGURES / "Figure5_regional_variability.png"

BAR, DOT, WHIS = "#9ec5f4", "#1c5cab", "#6f9fd8"
INK, INK2 = "#0b0b0b", "#52514e"
PANELS = [("Chloride", "Chloride", "mg/L"),
          ("Well depth", "WellDepthM", "m"),
          ("Surface elevation", "Surf_Ele", "m"),
          ("Distance to inundation line", "SLR0_DIST", "km"),
          ("Distance to surface water", "Hydro_DIST", "km"),
          ("Distance to salt dome", "Salt_DIST", "km"),
          ("Water-table elevation", "WaterEle_m", "m"),
          ("Daily maximum tide level", "Daily_Max", "m")]
KM = {"SLR0_DIST", "Hydro_DIST", "Salt_DIST"}      # metres are unreadable at this range
# Chloride only: the axis cannot hold every state's tail (see docstring). The clip is set
# just above the largest interquartile range in the panel so the bars stay legible.
CLIP = {"Chloride": 400.0}
COLS = [c for _, c, _ in PANELS]

df = pd.read_csv(SRC, dtype={"SiteNo": str}, low_memory=False)
for c in COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df[in_population(df) & (df["State"] != "PA")]
W = df.groupby(["State", "SiteNo"])[COLS].median().reset_index()
for c in KM:
    W[c] = W[c] / 1000.0
order = W.groupby("State").Chloride.median().sort_values().index.tolist()   # low to high
y = {s: i for i, s in enumerate(order)}

fig, axes = plt.subplots(2, 4, figsize=(15.0, 8.4))
for ax, (title, col, unit) in zip(axes.ravel(), PANELS):
    stats, lo_all, hi_all = [], [], []
    for st in order:
        v = W.loc[W.State == st, col].dropna()
        if len(v) < 5:
            continue
        q1, med, q3, p95 = v.quantile([.25, .5, .75, .95])
        stats.append((st, q1, med, q3, p95))
        lo_all.append(q1); hi_all.append(p95)
    lo, hi = min(lo_all), max(hi_all)
    clip = CLIP.get(col)
    if clip is not None:
        hi = clip
    pad = .06 * (hi - lo) if hi > lo else abs(hi) * .2 + 1
    xlo = min(lo - pad, 0) if lo >= 0 else lo - pad
    xhi = hi + pad
    ax.set_xlim(xlo, xhi)

    for st, q1, med, q3, p95 in stats:
        yy = y[st]
        # whisker first, so the IQR bar sits on top of its left end
        wend = min(p95, hi) if clip is not None else p95
        if wend > q3:
            ax.plot([q3, wend], [yy, yy], color=WHIS, lw=1.5,
                    solid_capstyle="butt", zorder=2)
            if clip is None or p95 <= hi:
                ax.plot([p95], [yy], marker="|", ms=5.5, mec=WHIS, mew=1.5, zorder=2)
        if clip is not None and p95 > hi:
            # tail runs past the view: arrowhead at the edge, value printed
            ax.plot([xhi], [yy], marker=">", ms=4.4, mfc=WHIS, mec="none",
                    clip_on=False, zorder=4)
            ax.text(xhi - .012 * (xhi - xlo), yy + .36, f"{p95:,.0f}", ha="right",
                    va="bottom", fontsize=9.9, color=INK2, zorder=4)
        ax.plot([q1, q3], [yy, yy], color=BAR, lw=7, solid_capstyle="butt", zorder=3)
        ax.plot([med], [yy], marker="o", ms=5.2, mfc=DOT, mec="#fcfcfb", mew=1.0,
                zorder=4)

    if lo < 0:
        ax.axvline(0, color="#b9b8b3", lw=.8, ls=":", zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=13.5, color=INK)
    ax.set_ylim(-.7, len(order) - .25)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.tick_params(axis="x", labelsize=12.6, colors=INK2)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#ecebe7", lw=.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#cfcec9")
    sub = unit + ("  ·  clipped; arrows = 95th pct" if clip else "")
    ax.set_title(title, fontsize=15.9, color=INK, pad=20, loc="left")
    ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=12.15, color=INK2, va="bottom")

handles = [Line2D([], [], ls="", marker="o", mfc=DOT, mec="#fcfcfb", mew=1.0, ms=6,
                  label="Median"),
           Line2D([], [], color=BAR, lw=7, label="25th–75th percentile"),
           Line2D([], [], color=WHIS, lw=1.5, marker="|", ms=6, mec=WHIS, mew=1.5,
                  label="75th–95th percentile")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=14.1,
           bbox_to_anchor=(.5, -.035))
fig.subplots_adjust(left=.035, right=.985, top=.93, bottom=.075, wspace=.17, hspace=.36)
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="#fcfcfb")
print("wrote")
