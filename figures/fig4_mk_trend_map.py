"""
Figure 4 - Mann-Kendall trend in groundwater chloride (input: results/mk_results.csv from
step 1, well coordinates from the dataset)
Significant increasing wells: warm colors (4 levels by tau strength)
Significant decreasing wells: cool/blue colors (4 levels by |tau| strength)
Non-significant wells: light gray background dots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.geodesic import Geodesic
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FIGURES, RESULTS, data_path   # noqa: E402
FIGURES.mkdir(exist_ok=True)

OUT = FIGURES / "Figure4_mk_trend_map.png"

# ── Load MK results with well coordinates ─────────────────────────────────────
print("Loading data ...", flush=True)
mk = pd.read_csv(RESULTS / "mk_results.csv", dtype={"SiteNo": str})

# Get well coordinates from main dataset
df = pd.read_csv(data_path(), dtype={"SiteNo": str}, usecols=["SiteNo", "lat", "lon"])
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
coords = df.groupby("SiteNo")[["lat","lon"]].first().reset_index()

mk = mk.merge(coords, on="SiteNo", how="left")
mk = mk.dropna(subset=["lat","lon"])
print(f"  MK wells with coords: {len(mk):,}", flush=True)

# ── Classify wells ────────────────────────────────────────────────────────────
# Significant increasing: p<0.05, tau>0
# Significant decreasing: p<0.05, tau<0
# Non-significant: p>=0.05

inc = mk[(mk["p_value"] < 0.05) & (mk["tau"] > 0)].copy()
dec = mk[(mk["p_value"] < 0.05) & (mk["tau"] < 0)].copy()
ns  = mk[mk["p_value"] >= 0.05].copy()

print(f"  Increasing (sig): {len(inc):,}", flush=True)
print(f"  Decreasing (sig): {len(dec):,}", flush=True)
print(f"  No trend:         {len(ns):,}",  flush=True)

# ── Tau strength bins: 4 levels (0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0) ──────
TAU_BINS  = [0.0, 0.25, 0.50, 0.75, 1.01]
TAU_LABELS = ["Weak (0–0.25)", "Moderate (0.25–0.50)",
              "Strong (0.50–0.75)", "Very Strong (0.75–1.0)"]

# Increasing: warm palette (light peach → dark red), ColorBrewer Reds
INC_COLORS = ["#FDBB84", "#FC8D59", "#E34A33", "#B30000"]

# Decreasing: cool palette (light blue → dark blue), ColorBrewer Blues
DEC_COLORS = ["#BDD7E7", "#6BAED6", "#2171B5", "#08306B"]

# Bin tau values
inc["tau_bin"] = pd.cut(inc["tau"].abs(),    bins=TAU_BINS, labels=[0,1,2,3], right=False).astype(int)
dec["tau_bin"] = pd.cut(dec["tau"].abs(),    bins=TAU_BINS, labels=[0,1,2,3], right=False).astype(int)

# ── Projection & figure ───────────────────────────────────────────────────────
EXTENT   = [-100.8, -73.5, 24.2, 42.5]
PROJ     = ccrs.AlbersEqualArea(central_longitude=-88, central_latitude=33,
                                standard_parallels=(29.5, 45.5))
DATA_CRS = ccrs.PlateCarree()

MAP_L, MAP_B, MAP_W, MAP_H = 0.01, 0.05, 0.80, 0.90

fig = plt.figure(figsize=(15, 9.5), dpi=150)
ax  = fig.add_axes([MAP_L, MAP_B, MAP_W, MAP_H], projection=PROJ)
ax.set_extent(EXTENT, crs=DATA_CRS)

# ── Base map ──────────────────────────────────────────────────────────────────
ax.add_feature(cfeature.OCEAN.with_scale("50m"),  facecolor="#c8e8f4", zorder=0)
ax.add_feature(cfeature.LAND.with_scale("50m"),   facecolor="#f5f5f0", zorder=1)
ax.add_feature(cfeature.LAKES.with_scale("50m"),  facecolor="#c8e8f4", zorder=2)
ax.add_feature(cfeature.RIVERS.with_scale("50m"),
               edgecolor="#a0c8e0", linewidth=0.4, zorder=3)
ax.add_feature(cfeature.NaturalEarthFeature(
    "cultural", "admin_1_states_provinces_lines", "50m",
    edgecolor="#888888", facecolor="none", linewidth=0.6), zorder=4)
ax.add_feature(cfeature.NaturalEarthFeature(
    "cultural", "admin_1_states_provinces", "50m",
    edgecolor="#666666", facecolor="none", linewidth=0.5), zorder=4)
ax.add_feature(cfeature.BORDERS.with_scale("50m"),
               edgecolor="#555555", linewidth=0.9, zorder=4)
ax.coastlines(resolution="50m", color="#444444", linewidth=0.8, zorder=5)

# ── Non-significant wells (gray background) ───────────────────────────────────
ax.scatter(ns["lon"].values, ns["lat"].values,
           c="#cccccc", s=5, linewidths=0, alpha=0.45,
           transform=DATA_CRS, zorder=6, rasterized=True)

# ── Decreasing wells (blue, draw first so increasing sits on top) ─────────────
for lvl in range(4):
    sub = dec[dec["tau_bin"] == lvl]
    if len(sub) == 0:
        continue
    ax.scatter(sub["lon"].values, sub["lat"].values,
               c=DEC_COLORS[lvl], s=18, linewidths=0.2,
               edgecolors="#333333", alpha=0.88,
               transform=DATA_CRS, zorder=7, rasterized=True)

# ── Increasing wells (warm, drawn on top) ─────────────────────────────────────
for lvl in range(4):
    sub = inc[inc["tau_bin"] == lvl]
    if len(sub) == 0:
        continue
    ax.scatter(sub["lon"].values, sub["lat"].values,
               c=INC_COLORS[lvl], s=18, linewidths=0.2,
               edgecolors="#333333", alpha=0.88,
               transform=DATA_CRS, zorder=8, rasterized=True)

# ── State labels ──────────────────────────────────────────────────────────────
STATE_CENTROIDS = {
    "TX": (-98.0, 29.5), "LA": (-92.3, 30.9), "MS": (-89.6, 32.5),
    "AL": (-86.7, 32.8), "FL": (-83.5, 28.0), "GA": (-83.5, 32.8),
    "SC": (-81.2, 33.9), "NC": (-79.4, 35.4), "VA": (-78.5, 37.5),
    "MD": (-76.8, 39.0), "DE": (-75.5, 39.1), "NJ": (-74.4, 40.1),
    "PA": (-77.5, 40.9),
}
for state, (cx, cy) in STATE_CENTROIDS.items():
    ax.text(cx, cy, state, fontsize=18, fontweight="bold", color="#222222",
            ha="center", va="center", transform=DATA_CRS, zorder=10,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.55))

# ── Title ─────────────────────────────────────────────────────────────────────
n_inc = len(inc)
n_dec = len(dec)
n_ns  = len(ns)
ax.set_title(
    "Mann-Kendall Trend in Groundwater Chloride Concentration — 13-State Coastal Study Area\n"
    f"Significant increasing: n={n_inc:,}  |  Significant decreasing: n={n_dec:,}  |  "
    f"No significant trend: n={n_ns:,}  (p < 0.05)",
    fontsize=11, fontweight="bold", pad=8)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_elements = []

# Section header — increasing
legend_elements.append(
    Line2D([0],[0], marker='', linestyle='', label='Significant Increasing Trend',
           color='none'))
for lvl, (col, lbl) in enumerate(zip(INC_COLORS, TAU_LABELS)):
    legend_elements.append(
        Line2D([0],[0], marker='o', linestyle='', markersize=8,
               markerfacecolor=col, markeredgecolor='#333333', markeredgewidth=0.4,
               label=f'  {lbl}'))

# Spacer
legend_elements.append(
    Line2D([0],[0], marker='', linestyle='', label='', color='none'))

# Section header — decreasing
legend_elements.append(
    Line2D([0],[0], marker='', linestyle='', label='Significant Decreasing Trend',
           color='none'))
for lvl, (col, lbl) in enumerate(zip(DEC_COLORS, TAU_LABELS)):
    legend_elements.append(
        Line2D([0],[0], marker='o', linestyle='', markersize=8,
               markerfacecolor=col, markeredgecolor='#333333', markeredgewidth=0.4,
               label=f'  {lbl}'))

# Spacer
legend_elements.append(
    Line2D([0],[0], marker='', linestyle='', label='', color='none'))

# No trend
legend_elements.append(
    Line2D([0],[0], marker='o', linestyle='', markersize=7,
           markerfacecolor='#cccccc', markeredgecolor='none',
           label='No Significant Trend (p >= 0.05)'))

leg = ax.legend(handles=legend_elements,
                loc='lower right',
                fontsize=9.5,
                framealpha=0.92,
                edgecolor='#aaaaaa',
                borderpad=0.8,
                labelspacing=0.45,
                handletextpad=0.5,
                title="Kendall's tau strength",
                title_fontsize=10,
                bbox_to_anchor=(0.995, 0.01))

# Bold the section headers
for text in leg.get_texts():
    t = text.get_text()
    if t in ('Significant Increasing Trend', 'Significant Decreasing Trend'):
        text.set_fontweight('bold')

# ── North arrow ───────────────────────────────────────────────────────────────
na_cx  = 0.055
na_tip = 0.94
na_aw  = 0.014
na_ah  = 0.021
ax.add_patch(MplPolygon(
    [(na_cx,         na_tip),
     (na_cx - na_aw, na_tip - na_ah),
     (na_cx + na_aw, na_tip - na_ah)],
    closed=True, facecolor="black", edgecolor="black", linewidth=0.4,
    transform=ax.transAxes, zorder=12))
ax.text(na_cx, na_tip - na_ah - 0.007, "N",
        transform=ax.transAxes,
        fontsize=11, fontweight="bold", ha="center", va="top", color="black", zorder=12)

# ── Scale bar ─────────────────────────────────────────────────────────────────
geod = Geodesic()
bar_lon0, bar_lat0 = -96.5, 25.6
breaks_km  = [0, 100, 300, 500]
tick_h_ax  = 0.013
bar_y_ax   = 0.085

def lon_at_km(lon0, lat0, km):
    if km == 0:
        return lon0
    pt = np.array(geod.direct([[lon0, lat0]], 90.0, km * 1000.0)).flatten()
    return float(pt[0])

inv = ax.transAxes.inverted()

def to_ax_x(lon, lat=bar_lat0):
    disp = ax.transData.transform(
        ax.projection.transform_point(lon, lat, DATA_CRS))
    return inv.transform(disp)[0]

bx = [to_ax_x(lon_at_km(bar_lon0, bar_lat0, k)) for k in breaks_km]

ax.plot([bx[0], bx[-1]], [bar_y_ax, bar_y_ax],
        color="black", lw=2.5, transform=ax.transAxes, zorder=10,
        solid_capstyle="butt")
for x in bx:
    ax.plot([x, x], [bar_y_ax - tick_h_ax, bar_y_ax + tick_h_ax],
            color="black", lw=1.5, transform=ax.transAxes, zorder=10)
for km, x in zip(breaks_km, bx):
    label = f"{km} km" if km == 500 else str(km)
    ax.text(x, bar_y_ax - tick_h_ax - 0.006, label,
            fontsize=8.5, ha="center", va="top",
            transform=ax.transAxes, color="black", zorder=11)

# ── Source note ───────────────────────────────────────────────────────────────
fig.text(0.01, 0.005,
         "Data: USGS NWIS / WQP groundwater chloride records  |  "
         "Mann-Kendall test on annual mean chloride, wells with >= 5 years of record, "
         "coastal-plain aquifers within 20 km of the NOAA 0-ft extent  |  "
         "Projection: Albers Equal Area (NAD83)",
         fontsize=7.5, color="#666666")

plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved -> {OUT}", flush=True)
plt.close()
print("Done.", flush=True)
