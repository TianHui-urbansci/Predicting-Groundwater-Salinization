"""Figures 7 and 8 - Monitoring wells at projected exposure under SSP2-4.5 and SSP5-8.5.

Input: results/figure_wells_<SSP>.csv (step 8). Inundated wells are hollow triangles
around a dot coloured by observed chloride; fresh-to-saline wells are rings around a dot
coloured by projected chloride; wells in states without a projection model are grey.

Usage:  python figures/fig7_8_projection_maps.py SSP245   (Figure 7)
        python figures/fig7_8_projection_maps.py SSP585   (Figure 8)
"""
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FIGURES, RESULTS, data_path   # noqa: E402
FIGURES.mkdir(exist_ok=True)

SSP = sys.argv[1] if len(sys.argv) > 1 else "SSP245"
CFG = {"SSP245": dict(fig=7, title="SSP2-4.5",
                      sub="Atlantic +4 ft  ·  Alabama and Mississippi +4 ft  ·  Louisiana and Texas +5 ft", rng="4–5 ft"),
       "SSP585": dict(fig=8, title="SSP5-8.5",
                      sub="Atlantic +7 ft  ·  Alabama and Mississippi +7 ft  ·  Louisiana and Texas +9 ft", rng="7–9 ft")}[SSP]
W = pd.read_csv(RESULTS / f"figure_wells_{SSP}.csv", dtype={"SiteNo": str})
W["has_model"] = W["has_model"].astype(bool)
W["fresh_to_saline"] = W["fresh_to_saline"].astype(bool)
OUT = FIGURES / f"Figure{CFG['fig']}_projection_{SSP}.png"

fig = plt.figure(figsize=(15.5, 9.2))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-98.5, -73.5, 24.4, 41.5], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor="#f4f3ee", zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor="#dbe9f4", zorder=0)
ax.add_feature(cfeature.LAKES, facecolor="#dbe9f4", edgecolor="none", zorder=1)
ax.add_feature(cfeature.STATES, edgecolor="#a8a8a8", linewidth=.5, zorder=2)
ax.add_feature(cfeature.COASTLINE, edgecolor="#5a5a5a", linewidth=.6, zorder=2)
gl = ax.gridlines(draw_labels=True, linewidth=.4, color="#cfcfcf", ls=":")
gl.top_labels = gl.right_labels = False
gl.xlabel_style = gl.ylabel_style = {"size": 10}

norm, cmap = LogNorm(vmin=3, vmax=500), "coolwarm"
M, NM = W[W.has_model], W[~W.has_model]

# Semi-transparent outlines (set through the edge colour's alpha, not an `alpha` kwarg).
EDGE = (0, 0, 0, 0.5)
EDGE_TRI = (0, 0, 0, 0.42)    # the Case 1 triangle only

# Inundated wells, including those in states without a projection model.
k1 = pd.concat([M[M.case == 1], NM[NM.case == 1]], ignore_index=True)
k2, k3 = M[M.case == 2], M[M.case == 3]
ft = M[M.fresh_to_saline]

# Other wells in states without a projection model.
nmp = NM[NM.case != 1]
ax.scatter(nmp.lon, nmp.lat, c="#cfcfcf", s=4, marker="o", lw=0,
           zorder=2.5, transform=ccrs.PlateCarree())

# Fresh-to-saline: ring around a dot coloured by projected chloride.
ax.scatter(ft.lon, ft.lat, facecolors="none", edgecolors=[EDGE], s=46, lw=.45,
           zorder=4, transform=ccrs.PlateCarree())
sc = ax.scatter(ft.lon, ft.lat, c=np.clip(ft.pred_Cl, 3, 500), cmap=cmap, norm=norm,
                s=10, marker="o", lw=0, zorder=4.5, transform=ccrs.PlateCarree())
# Inundated: hollow triangle around a dot coloured by observed chloride.
ax.scatter(k1.lon, k1.lat, c=np.clip(k1.obs_Cl, 3, 500), cmap=cmap, norm=norm,
           s=10, marker="o", lw=0, zorder=5, transform=ccrs.PlateCarree())
ax.scatter(k1.lon, k1.lat, facecolors="none", s=40, marker="^", edgecolors=[EDGE_TRI],
           lw=.45, zorder=6, transform=ccrs.PlateCarree())

# Colour bar aligned with the map frame.
fig.canvas.draw()
pos = ax.get_position()
cax = fig.add_axes([pos.x1 + 0.014, pos.y0, 0.017, pos.height])
cb = fig.colorbar(sc, cax=cax, ticks=[5, 10, 50, 100, 250, 500])
cb.set_label("Chloride (mg/L): observed for inundated wells, projected for fresh→saline wells",
             fontsize=10.5)
cb.ax.set_yticklabels(["5", "10", "50", "100", "250", "500"])
cb.outline.set_linewidth(.6)

h = [Line2D([], [], ls="", marker="^", mfc="none", mec=EDGE_TRI, mew=1.0, ms=10,
            label=f"Inundated (SLR = {CFG['rng']}); dot = observed chloride"),
     Line2D([], [], ls="", marker="o", mfc="none", mec=EDGE, mew=1.0, ms=10,
            label="Fresh→saline transition (obs ≤250 → pred >250 mg/L); dot = projected chloride"),
     Line2D([], [], ls="", marker="o", mfc="#cfcfcf", mec="none", ms=7,
            label="No model – chloride not projected")]
leg = ax.legend(handles=h, loc="upper left", bbox_to_anchor=(0.012, 0.985), fontsize=9.5,
                framealpha=1.0, title="Well Classification", title_fontsize=10.5,
                borderpad=.7, labelspacing=.55)
leg.get_frame().set_edgecolor("#999999")

ax.set_title(f"Monitoring wells at salinization exposure under {CFG['title']}\n"
             f"{CFG['sub']}", fontsize=14, fontweight="bold", pad=12)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT.name}: inundated {len(k1):,}, fresh->saline {len(ft):,}, grey {len(nmp):,}")
