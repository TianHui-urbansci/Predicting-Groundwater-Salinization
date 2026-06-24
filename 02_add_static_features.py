"""
02_add_static_features.py
=========================
Adds static/geological features to the final salinity dataset.

PYTHON (this script):
  A. SLR distances      — planar distance (m) from each well to the nearest
                          NOAA SLR inundation boundary, per scenario.
                          Hybrid approach: wells inside the inundation zone → 0;
                          outside wells → KD-tree on boundary pixels (Albers m).
                          Source: NOAA Sea Level Rise Inundation Rasters (2019)
                          Columns: SLR0_DIST, SLR4_DIST, SLR5_DIST, SLR7_DIST,
                                   SLR9_DIST  (NaN for non-applicable state/level)

  B. Aquifer fill       — spatial join to USGS Principal Aquifers shapefile for
                          wells where Aquifer is still NA after WQP / NGWMN.
                          Primary: sjoin(predicate='within'); fallback:
                          sjoin_nearest for coastal / offshore points.
                          Source: USGS Principal Aquifers of the US (2003)
                          Columns: Aquifer (updated), AQ_CODE

  C. Salt_DIST          — planar distance (m) to nearest Gulf Coast salt dome.
                          Applied only to TX, LA, MS, AL.
                          Source: Gulf Coast salt dome locations (data.shp)
                          Column: Salt_DIST

  D. NOAA tide stations — 3 nearest NOAA tide gauge stations per well, used to
                          join Daily_Mean / Daily_Max tide levels in the dynamic
                          feature script. If the nearest station has no data for
                          a given date, fall back to the 2nd or 3rd nearest.
                          Source: NOAA Tides & Currents station selection (2024)
                          Columns: TideSta_1, TideSta_2, TideSta_3  (Station_ID)
                                   TideDist_1, TideDist_2, TideDist_3 (km)

ARCGIS (manual workflow — see note at bottom of file):
  E. Surf_Ele           — well surface elevation from NED 30 m DEM
                          Source: National Elevation Dataset (USDA, 1999)
  F. Hydro_DIST         — planar distance to nearest surface water body (m)
                          Source: NHD Best Resolution (USGS, 2023)

"""

import re
import shutil
import time
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from pyproj import Transformer
from scipy import ndimage
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")


# ── USER CONFIG — update these to your local paths ────────────────────────────
# NOAA SLR rasters: one sub-folder per state (e.g., SLR_DIR / "TX" / "*.tif")
# Download: https://coast.noaa.gov/slrdata/
SLR_DIR = Path(r"path/to/NOAA_SLR_Inundation")

# USGS Principal Aquifers of the United States shapefile (2003)
# Download: https://water.usgs.gov/GIS/metadata/usgswrd/XML/aquifers_us.xml
AQ_SHP  = Path(r"path/to/us_aquifers.shp")
# ──────────────────────────────────────────────────────────────────────────────

# ── Paths (relative to this script's directory) ────────────────────────────────
DATA          = Path(__file__).resolve().parent
FINAL_CSV     = DATA / "ComprehensiveData" / "daily_chloride_final_20260620.csv"
SALT_SHP      = DATA / "SaltDomes" / "Data" / "data.shp"
# NOAA tide gauge station list: CSV with Station_ID, lat, lon columns
# Download from https://tidesandcurrents.noaa.gov/stations.html (select all stations → export)
# or via API: https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json
TIDE_STA_CSV  = DATA / "NOAA_Tide_Stations" / "stations.csv"


# ── Constants ──────────────────────────────────────────────────────────────────
# NAD83 geographic → NAD83 Albers Equal Area (metres); xy = lon, lat order
TRANSFORMER = Transformer.from_crs("EPSG:4269", "EPSG:5070", always_xy=True)
BLOCK_ROWS  = 2000    # raster rows per read chunk
SUBSAMPLE   = 10      # 1-in-10 boundary pixels (~30 m at 3 m raster resolution)
N_TIDE_STA      = 3        # number of nearest tide stations to record (primary + backups)
MAX_TIDE_DIST_M = 160_000  # coastal study area limit: wells > 160 km (~100 mi) from any
                            # tide gauge are too far inland; TideSta/TideDist set to NaN

# SLR inundation scenarios (feet) per state.
# All states include 0 ft (current sea-level baseline) → SLR0_DIST is the
# reference shoreline distance used as a predictor in all feature sets.
# Atlantic states additionally use 4 ft and 7 ft scenarios.
# Gulf states (TX, LA) use 5 ft and 9 ft instead of 4 ft and 7 ft.
STATE_SLR_SCENARIOS = {
    "AL": [0, 4, 7],
    "DE": [0, 4, 7],
    "FL": [0, 4, 7],
    "GA": [0, 4, 7],
    "LA": [0, 5, 9],
    "MD": [0, 4, 7],
    "MS": [0, 4, 7],
    "NC": [0, 4, 7],
    "NJ": [0, 4, 7],
    "PA": [0, 4, 7],
    "SC": [0, 4, 7],
    "TX": [0, 5, 9],
    "VA": [0, 4, 7],
}

SALT_STATES = {"TX", "LA", "MS", "AL"}   # Gulf states with salt dome data


# ══════════════════════════════════════════════════════════════════════════════
# A. SLR DISTANCES
# ══════════════════════════════════════════════════════════════════════════════

def _find_slr_tiles(state: str, slr_level: int) -> list:
    """Return sorted list of SLR inundation .tif tiles for a state and level."""
    state_dir = SLR_DIR / state
    pattern   = re.compile(rf"_connectRaster_{slr_level}[_.]", re.IGNORECASE)
    return sorted(p for p in state_dir.glob("*.tif") if pattern.search(p.name))


def _sample_inundation(tiles: list,
                       lons: np.ndarray,
                       lats: np.ndarray) -> np.ndarray:
    """
    Sample SLR raster at well locations.
    Returns bool array — True where the well pixel == 1 (inside inundation zone).
    """
    n         = len(lons)
    inundated = np.zeros(n, dtype=bool)
    for tile in tiles:
        with rasterio.open(tile) as src:
            b = src.bounds
            in_tile = ((lons >= b.left   - 0.001) & (lons <= b.right  + 0.001) &
                       (lats >= b.bottom - 0.001) & (lats <= b.top    + 0.001))
            if not in_tile.any():
                continue
            idx    = np.where(in_tile)[0]
            coords = list(zip(lons[idx], lats[idx]))
            try:
                vals = np.array([v[0] for v in src.sample(coords, masked=False)])
                inundated[idx] |= (vals == 1)
            except Exception:
                pass
    return inundated


def _build_boundary_kdtree(state: str, slr_level: int):
    """
    Build a KD-tree on the SLR inundation boundary pixels (subsampled).
    Boundary pixels: inundated cells adjacent to non-inundated (binary erosion).
    Tree coordinates are in Albers metres (EPSG:5070).
    Returns (tree, tiles) or (None, tiles) if no tiles / boundary found.
    """
    tiles = _find_slr_tiles(state, slr_level)
    if not tiles:
        print(f"    [WARN] No tiles: {state} SLR={slr_level} ft", flush=True)
        return None, tiles

    all_lons, all_lats = [], []
    for tile in tiles:
        with rasterio.open(tile) as src:
            n_rows, n_cols = src.height, src.width
            transform      = src.transform
            for r0 in range(0, n_rows, BLOCK_ROWS):
                r_start = max(0, r0 - 1)
                r_end   = min(n_rows, r0 + BLOCK_ROWS + 1)
                win     = rasterio.windows.Window(
                              0, r_start, n_cols, r_end - r_start)
                data    = src.read(1, window=win)
                mask    = (data == 1).astype(np.uint8)
                eroded  = ndimage.binary_erosion(mask, border_value=0)
                bnd     = mask & ~eroded
                rows_b, cols_b = np.where(bnd)
                if len(rows_b) == 0:
                    continue
                idx    = np.arange(0, len(rows_b), SUBSAMPLE)
                rows_g = (rows_b[idx] + r_start).astype(np.float64)
                xs, ys = rasterio.transform.xy(
                             transform,
                             rows_g,
                             cols_b[idx].astype(np.float64))
                all_lons.extend(xs)
                all_lats.extend(ys)
        print(f"      {tile.name}: boundary pts={len(all_lons):,}", flush=True)

    if not all_lons:
        return None, tiles

    bx, by = TRANSFORMER.transform(np.array(all_lons), np.array(all_lats))
    tree   = cKDTree(np.column_stack([bx, by]))
    print(f"    KD-tree: {len(bx):,} pts  SLR={slr_level} ft", flush=True)
    return tree, tiles


def calc_slr_distances(wells: pd.DataFrame) -> pd.DataFrame:
    """
    Add SLR{level}_DIST columns (metres) to the wells DataFrame.
    Wells inside the inundation zone → 0.
    Wells outside → KD-tree nearest boundary pixel distance.
    Non-applicable state/level combinations remain NaN.
    """
    print("\n" + "=" * 65, flush=True)
    print("A. SLR DISTANCES", flush=True)
    print("=" * 65, flush=True)

    lons = wells["lon"].values
    lats = wells["lat"].values

    for state, levels in STATE_SLR_SCENARIOS.items():
        w_idx = wells.index[wells["State"] == state].tolist()
        if not w_idx:
            continue
        print(f"\n  {state}  wells={len(w_idx):,}  SLR scenarios={levels} ft",
              flush=True)
        s_lons = lons[w_idx]
        s_lats = lats[w_idx]

        for lvl in levels:
            col = f"SLR{lvl}_DIST"
            t0  = time.time()

            tree, tiles = _build_boundary_kdtree(state, lvl)
            if tree is None:
                wells.loc[w_idx, col] = np.nan
                continue

            inundated = _sample_inundation(tiles, s_lons, s_lats)
            n_in      = int(inundated.sum())
            n_out     = len(w_idx) - n_in
            print(f"    inside (dist=0)={n_in:,}  outside={n_out:,}", flush=True)

            dists = np.zeros(len(w_idx), dtype=np.float64)
            if n_out > 0:
                out_mask        = ~inundated
                wx, wy          = TRANSFORMER.transform(s_lons[out_mask],
                                                        s_lats[out_mask])
                d, _            = tree.query(np.column_stack([wx, wy]))
                dists[out_mask] = d

            wells.loc[w_idx, col] = dists
            mean_out = dists[~inundated].mean() if n_out > 0 else 0.0
            print(f"    {col}: range 0–{dists.max():.0f} m  "
                  f"mean(outside)={mean_out:.0f} m  [{time.time()-t0:.0f}s]",
                  flush=True)

    slr_cols = sorted(c for c in wells.columns
                      if c.startswith("SLR") and c.endswith("_DIST"))
    print(f"\n  SLR columns added: {slr_cols}", flush=True)
    for col in slr_cols:
        n_notna = int(wells[col].notna().sum())
        print(f"    {col:14s}  non-null={n_notna:,} / {len(wells):,}", flush=True)

    return wells


# ══════════════════════════════════════════════════════════════════════════════
# B. AQUIFER SPATIAL JOIN
# ══════════════════════════════════════════════════════════════════════════════

def fill_aquifer_spatial(wells: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing Aquifer values from WQP and NGWMN datasets using the USGS Principal Aquifers shapefile.
    Primary join: sjoin(predicate='within').
    Fallback:     sjoin_nearest for coastal / offshore points that fall outside
                  all polygons.
    Updates the Aquifer column in-place; adds AQ_CODE for newly filled rows.
    """
    print("\n" + "=" * 65, flush=True)
    print("B. AQUIFER SPATIAL JOIN", flush=True)
    print("=" * 65, flush=True)

    mask_na = wells["Aquifer"].isna()
    n_na    = int(mask_na.sum())
    print(f"  Aquifer already filled: {(~mask_na).sum():,}  |  NA: {n_na:,}",
          flush=True)

    if n_na == 0:
        print("  All wells have Aquifer — skipping spatial join.", flush=True)
        if "AQ_CODE" not in wells.columns:
            wells["AQ_CODE"] = pd.NA
        return wells

    print(f"  Loading aquifer shapefile ...", flush=True)
    aq_shp = gpd.read_file(AQ_SHP)[["AQ_NAME", "AQ_CODE", "geometry"]]  # EPSG:4269

    # Build GeoDataFrame for NA wells: WGS84 → NAD83 to match shapefile CRS
    wells_na = wells[mask_na].copy()
    gdf_na   = gpd.GeoDataFrame(
        wells_na,
        geometry=gpd.points_from_xy(wells_na["lon"], wells_na["lat"]),
        crs="EPSG:4326"
    ).to_crs("EPSG:4269")

    # Primary: point-within-polygon join
    joined = gpd.sjoin(gdf_na, aq_shp, how="left", predicate="within")

    # Fallback: nearest polygon for coastal / offshore points
    outside = joined["AQ_NAME"].isna()
    if outside.any():
        n_outside = int(outside.sum())
        print(f"  sjoin_nearest fallback: {n_outside:,} coastal/offshore wells ...",
              flush=True)
        joined_near = gpd.sjoin_nearest(
            gdf_na[outside].drop(columns=["index_right"], errors="ignore"),
            aq_shp, how="left"
        )
        joined.loc[outside, ["AQ_NAME", "AQ_CODE"]] = (
            joined_near[["AQ_NAME", "AQ_CODE"]].values
        )

    spatial_filled = int(joined["AQ_NAME"].notna().sum())
    print(f"  Matched: {spatial_filled:,} / {n_na:,}", flush=True)

    # Write back into wells
    wells.loc[mask_na, "Aquifer"] = joined["AQ_NAME"].values
    wells.loc[mask_na, "AQ_CODE"] = joined["AQ_CODE"].values

    final_na  = int(wells["Aquifer"].isna().sum())
    final_pct = final_na / len(wells) * 100
    print(f"  Final Aquifer NA: {final_na:,} ({final_pct:.1f}%)", flush=True)

    return wells

# ══════════════════════════════════════════════════════════════════════════════
# C. SALT DOME DISTANCE  (TX, LA, MS, AL)
# ══════════════════════════════════════════════════════════════════════════════

def calc_salt_dist(wells: pd.DataFrame) -> pd.DataFrame:
    """
    Add Salt_DIST column: planar distance (m) to the nearest Gulf Coast
    salt dome for wells in TX, LA, MS, AL.
    Wells in other states receive Salt_DIST = NaN.
    Source: Gulf Coast salt dome shapefile (data.shp).
    """
    print("\n" + "=" * 65, flush=True)
    print("C. SALT DOME DISTANCE  (TX, LA, MS, AL)", flush=True)
    print("=" * 65, flush=True)

    wells["Salt_DIST"] = np.nan

    gulf_mask = wells["State"].isin(SALT_STATES)
    n_gulf    = int(gulf_mask.sum())
    print(f"  Gulf Coast wells: {n_gulf:,}", flush=True)

    if n_gulf == 0:
        print("  No Gulf Coast wells found — skipping.", flush=True)
        return wells

    print(f"  Loading salt dome shapefile ...", flush=True)
    salt_gdf  = gpd.read_file(SALT_SHP).to_crs("EPSG:5070")

    # Use centroids (handles both Point and Polygon geometries)
    salt_pts  = salt_gdf.geometry.centroid
    sx        = salt_pts.x.values
    sy        = salt_pts.y.values
    salt_tree = cKDTree(np.column_stack([sx, sy]))
    print(f"  Salt dome locations: {len(sx):,}", flush=True)

    # Transform Gulf well coordinates → Albers metres
    gulf_wells = wells[gulf_mask]
    wx, wy     = TRANSFORMER.transform(gulf_wells["lon"].values,
                                       gulf_wells["lat"].values)
    d, _       = salt_tree.query(np.column_stack([wx, wy]))
    wells.loc[gulf_mask, "Salt_DIST"] = d

    for state in sorted(SALT_STATES & set(wells["State"].unique())):
        s_mask = gulf_mask & (wells["State"] == state)
        ds     = wells.loc[s_mask, "Salt_DIST"].dropna()
        print(f"  {state}: n={len(ds):,}  "
              f"min={ds.min():.0f}  max={ds.max():.0f}  mean={ds.mean():.0f} m",
              flush=True)

    return wells

# ══════════════════════════════════════════════════════════════════════════════
# D. NOAA TIDE GAUGE STATIONS — nearest 3 per well
# ══════════════════════════════════════════════════════════════════════════════

def find_nearest_tide_stations(wells: pd.DataFrame,
                               n: int = N_TIDE_STA) -> pd.DataFrame:
    """
    Find the n nearest NOAA tide gauge stations for each well.

    Adds TideSta_1 / _2 / _3 (Station_ID string) and TideDist_1 / _2 / _3
    (planar distance in km) to the wells DataFrame.

    In the dynamic feature script, Daily_Mean and Daily_Max tide levels are
    joined by matching the well's date against the primary station (TideSta_1).
    TideSta_2 and TideSta_3 serve as fallbacks for dates when the primary
    station has no observation.

    Station CSV expected columns (case-insensitive):
      Station_ID  — NOAA station identifier (e.g. "8534720")
      lat         — station latitude  (decimal degrees, WGS84)
      lon         — station longitude (decimal degrees, WGS84)

    Source: NOAA National Oceanic and Atmospheric Administration (2024).
            Water levels — Station selection [Data set].
            NOAA Tides & Currents. https://tidesandcurrents.noaa.gov
    """
    print("\n" + "=" * 65, flush=True)
    print(f"D. NOAA TIDE STATIONS — {n} nearest per well", flush=True)
    print("=" * 65, flush=True)

    if not TIDE_STA_CSV.exists():
        print(f"  [WARN] Station file not found: {TIDE_STA_CSV}", flush=True)
        print(f"  Skipping — TideSta / TideDist columns will be missing.", flush=True)
        return wells

    # ── Load station list ──────────────────────────────────────────────────────
    sta = pd.read_csv(TIDE_STA_CSV, dtype=str)
    sta.columns = sta.columns.str.strip()

    # Normalise column names to lowercase for flexible matching
    col_map = {c: c.lower().replace(" ", "_") for c in sta.columns}
    sta.rename(columns=col_map, inplace=True)

    # Accept common variants: station_id / id / stationid
    for id_cand in ("station_id", "id", "stationid", "station"):
        if id_cand in sta.columns:
            sta.rename(columns={id_cand: "Station_ID"}, inplace=True)
            break
    for lat_cand in ("lat", "latitude"):
        if lat_cand in sta.columns:
            sta.rename(columns={lat_cand: "sta_lat"}, inplace=True)
            break
    for lon_cand in ("lon", "lng", "longitude"):
        if lon_cand in sta.columns:
            sta.rename(columns={lon_cand: "sta_lon"}, inplace=True)
            break

    required = {"Station_ID", "sta_lat", "sta_lon"}
    missing  = required - set(sta.columns)
    if missing:
        raise ValueError(
            f"Station CSV missing columns: {missing}. "
            f"Found: {list(sta.columns)}"
        )

    sta["sta_lat"] = pd.to_numeric(sta["sta_lat"], errors="coerce")
    sta["sta_lon"] = pd.to_numeric(sta["sta_lon"], errors="coerce")
    sta = sta.dropna(subset=["sta_lat", "sta_lon"]).reset_index(drop=True)
    print(f"  Stations loaded: {len(sta):,}", flush=True)

    # ── Build KD-tree on stations (Albers metres) ──────────────────────────────
    # Stations are in WGS84; transformer accepts NAD83 ≈ WGS84 for this purpose
    sx, sy   = TRANSFORMER.transform(sta["sta_lon"].values, sta["sta_lat"].values)
    sta_tree = cKDTree(np.column_stack([sx, sy]))
    sta_ids  = sta["Station_ID"].values

    # ── Query n nearest stations for each well ─────────────────────────────────
    wx, wy = TRANSFORMER.transform(wells["lon"].values, wells["lat"].values)
    dists, idxs = sta_tree.query(np.column_stack([wx, wy]), k=n)
    # dists are in Albers metres (EPSG:5070); stored as metres throughout

    for rank in range(1, n + 1):
        d_m  = np.round(dists[:, rank - 1], 1)
        s_id = sta_ids[idxs[:, rank - 1]].copy().astype(object)
        # Wells beyond the coastal study area limit → NaN (no usable tide station)
        beyond = d_m > MAX_TIDE_DIST_M
        d_m[beyond]  = np.nan
        s_id[beyond] = np.nan
        wells[f"TideSta_{rank}"]  = s_id
        wells[f"TideDist_{rank}"] = d_m   # metres

    # ── Summary ───────────────────────────────────────────────────────────────
    n_valid  = wells["TideSta_1"].notna().sum()
    n_beyond = wells["TideSta_1"].isna().sum()
    print(f"  Assigned {n} nearest stations to {n_valid:,} wells "
          f"(within {MAX_TIDE_DIST_M/1000:.0f} km)", flush=True)
    if n_beyond:
        print(f"  [{n_beyond:,} wells beyond {MAX_TIDE_DIST_M/1000:.0f} km → NaN]",
              flush=True)
    d1 = wells["TideDist_1"].dropna()
    print(f"  TideDist_1 (nearest):  "
          f"min={d1.min():.0f} m  median={d1.median():.0f} m  "
          f"max={d1.max():.0f} m", flush=True)

    # Show unique primary stations used
    n_uniq = wells["TideSta_1"].nunique()
    print(f"  Unique primary stations (TideSta_1): {n_uniq}", flush=True)

    return wells


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_global = time.time()

    # ── Backup ────────────────────────────────────────────────────────────────
    if not BAK_CSV.exists():
        print(f"Backing up -> {BAK_CSV.name}")
        shutil.copy2(FINAL_CSV, BAK_CSV)
    else:
        print(f"Backup already exists: {BAK_CSV.name} (skipping copy)")

    # ── Load full dataset ──────────────────────────────────────────────────────
    print(f"\nLoading {FINAL_CSV.name} ...", flush=True)
    df = pd.read_csv(FINAL_CSV, dtype={"SiteNo": str}, low_memory=False)
    for col in ("lon", "lat"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"  {len(df):,} rows  x  {len(df.columns)} cols", flush=True)

    # ── Extract one row per unique well (best non-null Aquifer per SiteNo) ─────
    # Keep first non-null Aquifer value across all rows for each site
    aq_best = (df[df["Aquifer"].notna()]
               .groupby("SiteNo")["Aquifer"]
               .first()
               .reset_index())
    wells = (df[["SiteNo", "State", "lon", "lat"]]
             .drop_duplicates("SiteNo")
             .dropna(subset=["lon", "lat"])
             .reset_index(drop=True))
    wells = wells.merge(aq_best, on="SiteNo", how="left")
    print(f"  Unique wells: {len(wells):,}  "
          f"Aquifer filled: {wells['Aquifer'].notna().sum():,}", flush=True)

    # ── Compute static features ────────────────────────────────────────────────
    wells = calc_slr_distances(wells)
    wells = fill_aquifer_spatial(wells)
    wells = calc_salt_dist(wells)
    wells = find_nearest_tide_stations(wells)

    # ── Join features back to full dataset by SiteNo ───────────────────────────
    print("\n" + "=" * 65, flush=True)
    print("Joining features to full dataset ...", flush=True)

    # Columns to join (everything except shared identifier fields)
    join_cols = [c for c in wells.columns
                 if c not in {"SiteNo", "State", "lon", "lat"}]
    # Separate Aquifer update from new columns
    new_cols  = [c for c in join_cols if c != "Aquifer"]

    # Update Aquifer: only fill rows that were NA (preserve original values)
    aq_map  = wells.set_index("SiteNo")["Aquifer"]
    null_aq = df["Aquifer"].isna()
    df.loc[null_aq, "Aquifer"] = df.loc[null_aq, "SiteNo"].map(aq_map)

    # Merge new feature columns
    df = df.merge(wells[["SiteNo"] + new_cols], on="SiteNo", how="left")

    print(f"  Final: {len(df):,} rows  x  {len(df.columns)} cols", flush=True)

    added = [c for c in new_cols if c in df.columns]
    print(f"\n  New / updated columns:", flush=True)
    for col in sorted(added):
        pct = df[col].notna().mean() * 100
        print(f"    {col:18s}  non-null={pct:.1f}%", flush=True)
    aq_pct = df["Aquifer"].notna().mean() * 100
    print(f"    {'Aquifer':18s}  non-null={aq_pct:.1f}%  (updated)", flush=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving -> {FINAL_CSV.name} ...", flush=True)
    df.to_csv(FINAL_CSV, index=False)
    print(f"Done.  {len(df):,} rows  x  {len(df.columns)} cols", flush=True)
    print(f"Total time: {(time.time()-t_global)/60:.1f} min", flush=True)


main()


# ══════════════════════════════════════════════════════════════════════════════
# E & F — ARCGIS WORKFLOW  (performed manually in ArcGIS Pro)
# ══════════════════════════════════════════════════════════════════════════════
#
# Input: unique well locations exported from daily_chloride_final_20260620.csv
#        as a point feature class (WGS84 / EPSG:4326).
#
# E. Surf_Ele — Surface Elevation (m)
#    Tool   : Extract Values to Points (Spatial Analyst)
#    DEM    : National Elevation Dataset (NED) 30 m mosaic (USDA, 1999)
#             https://www.usgs.gov/core-science-systems/ngp/3dep
#    CRS    : reproject well points to match DEM projection before extraction
#    Output : attribute field RASTERVALU → rename to Surf_Ele (m)
#
# F. Hydro_DIST — Distance to Nearest Surface Water Body (m)
#    Tool   : Near (Analysis) with method = PLANAR
#    Input  : well point feature class
#    Near   : National Hydrography Dataset (NHD) Best Resolution flowlines
#             and water-body polygons, merged; clipped to 13-state study area
#             Source: https://www.usgs.gov/national-hydrography/national-hydrography-dataset
#             (USGS, 2023)
#    Output : NEAR_DIST field → rename to Hydro_DIST (m)
#             NEAR_DIST = -1 (no feature within search radius) → set to NaN
#
# After ArcGIS processing:
#    Export attribute table to CSV → join Surf_Ele and Hydro_DIST to
#    daily_chloride_final_20260620.csv by SiteNo using pd.merge.
