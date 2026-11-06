"""Figure 6 - Shapley attribution and response direction of the projection models.

Input: results/shap_importance.csv (step 7). Predictors are rows and states columns.
Marker area is the predictor's share of mean |SHAP| in that state; colour is the
direction of response, the Spearman correlation between the predictor and its SHAP
value (blue: predicted chloride falls as the predictor rises). Aquifer is drawn in grey
because its ordinal codes have no direction.

Usage:  python figures/fig6_shap_attribution.py
"""
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FIGURES, RESULTS, data_path   # noqa: E402
FIGURES.mkdir(exist_ok=True)

S = pd.read_csv(RESULTS / "shap_importance.csv")

REGIONS = [("Mid-Atlantic", ["NJ", "DE", "MD", "VA"], "#2a78d6"),
           ("South Atlantic", ["SC", "NC", "GA", "FL"], "#eb6834"),
           ("Gulf", ["AL", "MS", "LA", "TX"], "#1baf7a")]
ORDER = [s for _, sts, _ in REGIONS for s in sts]
ROWS = [("Elevation", ["Surf_Ele", "Ele_abv_MWL_1y"]),
        ("Well depth", ["WellDepthM"]),
        ("Distance to\ninundation line", ["SLR0_DIST"]),
        ("Distance to\nsurface water", ["Hydro_DIST"]),
        ("Salt-dome\ndistance", ["Salt_DIST"]),
        ("Aquifer type", ["Aquifer"])]
NO_DIR = {"Aquifer"}
NEUTRAL, INK, INK2 = "#f0efec", "#0b0b0b", "#52514e"
DIV = ["#0d366b", "#1c5cab", "#2a78d6", "#86b6ef", NEUTRAL,
       "#f0a9a8", "#e34948", "#c22e2d", "#8f1f1f"]
cmap = LinearSegmentedColormap.from_list("div", DIV)
norm = Normalize(-1, 1)

fig, ax = plt.subplots(figsize=(12.0, 5.2))
ax.set_facecolor("#fcfcfb")
SMAX = 55.0           # share (%) mapped to the largest marker
AMAX = 1250.0         # marker area in points^2 at SMAX

for i, (lab, preds) in enumerate(ROWS):
    y = len(ROWS) - 1 - i
    for j, st in enumerate(ORDER):
        k = S[(S.State == st) & (S.predictor.isin(preds))]
        if k.empty:
            continue
        r = k.iloc[0]
        share = 100 * r.shap_share
        area = max(AMAX * share / SMAX, 25)
        if r.predictor in NO_DIR or pd.isna(r.direction_rho):
            ax.scatter(j, y, s=area, marker="s", facecolor="#d7d6d2",
                       edgecolor="#b4b3ae", lw=.7, zorder=3)
        else:
            ax.scatter(j, y, s=area, marker="s",
                       facecolor=cmap(norm(r.direction_rho)),
                       edgecolor="#fcfcfb", lw=1.2, zorder=3)
        if share >= 16:          # label only where the cell can hold it
            lum = (np.mean(cmap(norm(r.direction_rho))[:3])
                   if r.predictor not in NO_DIR else .84)
            ax.text(j, y, f"{share:.0f}", ha="center", va="center", fontsize=10.5,
                    color="#ffffff" if lum < .55 else INK, zorder=4)

ax.set_xlim(-.6, len(ORDER) - .4); ax.set_ylim(-.75, len(ROWS) - .4)
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels(ORDER, fontsize=13, fontweight="bold", color=INK)
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([lab for lab, _ in ROWS][::-1], fontsize=12.5, color=INK, linespacing=1.15)
ax.tick_params(length=0)
for s_ in ax.spines.values():
    s_.set_visible(False)
ax.grid(axis="both", color="#eceae6", lw=.7, zorder=0)
ax.set_axisbelow(True)

# Regional bands sit ABOVE the grid as column-group headers. Placing them below runs
# into the tick-label zone, whose extent is set at draw time and cannot be cleared
# reliably by nudging a data-coordinate offset.
tr = ax.get_xaxis_transform()        # x in data units, y as an axes fraction
x0 = 0
for name, sts, col in REGIONS:
    x1 = x0 + len(sts) - 1
    ax.plot([x0 - .36, x1 + .36], [1.035, 1.035], color=col, lw=3.4,
            solid_capstyle="butt", transform=tr, clip_on=False, zorder=5)
    ax.text((x0 + x1) / 2, 1.06, name, ha="center", va="bottom", fontsize=12.5,
            style="italic", color=col, transform=tr, clip_on=False)
    x0 = x1 + 1

cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                  pad=.012, fraction=.030, aspect=22, ticks=[-1, 0, 1])
cb.set_label("Direction of response (Spearman ρ)\n"
             "← expected: chloride falls as predictor rises        inverted →",
             fontsize=10.5, color=INK2)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=10, colors=INK2)

kax = fig.add_axes([0.16, -0.20, 0.20, 0.10])
kax.set_facecolor("#fcfcfb")
for sp in kax.spines.values():
    sp.set_visible(False)
kax.set_xticks([]); kax.set_yticks([])
size_keys = [10, 25, 45]
for n, v in enumerate(size_keys):
    kax.scatter(n, 0, s=AMAX * v / SMAX * .85, marker="s", facecolor="#c9c8c4",
                edgecolor="#fcfcfb", lw=1.0, clip_on=False)
    kax.text(n, -.75, f"{v}%", ha="center", va="top", fontsize=10.5, color=INK2)
kax.text(-.62, 0, "Attribution\nshare", ha="right", va="center", fontsize=12,
         color=INK2)
kax.set_xlim(-.75, len(size_keys) - .3); kax.set_ylim(-1, 1)

fig.savefig(FIGURES / "Figure6_shap_attribution.png", dpi=300, bbox_inches="tight",
            facecolor="#fcfcfb")
print("wrote Figure6_shap_attribution.png")
