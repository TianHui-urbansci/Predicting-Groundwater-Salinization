"""
03_add_dynamic_features.py
==========================
Adds time-varying (dynamic) features to the final salinity dataset.

A. Tide download — Hourly water level data fetched from NOAA CO-OPS API via
                   the noaa_coops library, year-by-year per station.
                   Datum: MHHW (water_level = 0 at Mean Higher High Water).
                   Units: metres.  Time zone: GMT.
                   One CSV per station saved to TideMHW_toMHWW/{station_id}.csv.
                   Stations already downloaded are skipped automatically.
                   Source: NOAA CO-OPS (https://tidesandcurrents.noaa.gov)

   Tide levels  — Daily aggregates (mean / max / min) joined using the
                   TideSta_1/2/3 fallback chain from 02_add_static_features.py.
                   Columns: Daily_Mean, Daily_Max, Daily_Min  (m, rel. MHHW)
                            Station_ID  — station that provided each row's data

B. Precipitation — Daily precipitation and rolling accumulations matched to
                   the nearest nClimGrid-Daily land grid cell (KD-tree).
                   Source: NOAA nClimGrid-Daily (Durre et al., 2022)
                   Columns: daily_precip  (mm)
                            {1,2,3,4,5,6,7,14}d_Before_prec  (mm, rolling sum)
                            {1m,3m,6m,1Y}_Before_prec         (mm, rolling sum)

C. Derived tide features:
   Ele_MHHW    — land surface elevation relative to the observed daily mean
                 tidal water level (not relative to the fixed MHHW datum zero).
                 Ele_MHHW = Surf_Ele (NAVD88) − Daily_Mean
                 Daily_Mean is the observed mean water level for that day,
                 reported in the MHHW datum frame (negative when water is
                 below the MHHW benchmark; positive when above).
                 Large positive → land well above that day's mean tide level.
                 Near zero / negative → water approaching or above land surface.
                 Note: mixes NAVD88 and MHHW datum zeros; the implicit offset
                 (MHHW_NAVD88, ~0.2–1.5 m per station) is constant per station
                 and does not affect temporal dynamics used by the model.
   Ele_MAX     — same concept using daily maximum tide instead of mean:
                 Ele_MAX = Surf_Ele − Daily_Max  (MHHW datum)
                 Negative → well surface inundated at daily high tide.

"""

import gc
import shutil
import time
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from noaa_coops import Station
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")


# ── USER CONFIG — update these to your local paths ────────────────────────────
# nClimGrid-Daily NetCDF files: one file per month, pattern ncdd-YYYYMM-grd-*.nc
# Download from NOAA: https://www.ncei.noaa.gov/products/land-based-station/nclimgrid-daily
PREC_NC_DIR = Path(r"path/to/nClimGrid_Daily")
# ──────────────────────────────────────────────────────────────────────────────

# ── Paths (relative to this script's directory) ────────────────────────────────
DATA         = Path(__file__).resolve().parent
FINAL_CSV    = DATA / "ComprehensiveData" / "daily_chloride_final_20260620.csv"
BAK_CSV      = DATA / "ComprehensiveData" / "daily_chloride_final_20260620_bak.csv"
TIDE_RAW_DIR = DATA / "TideMHW_toMHWW"       # raw NOAA CO-OPS water level CSVs


# ── Constants ──────────────────────────────────────────────────────────────────
TIDE_COLS = ["Daily_Mean", "Daily_Max", "Daily_Min"]

PREC_WINDOWS = {
    "1d_Before_prec" :   1,
    "2d_Before_prec" :   2,
    "3d_Before_prec" :   3,
    "4d_Before_prec" :   4,
    "5d_Before_prec" :   5,
    "6d_Before_prec" :   6,
    "7d_Before_prec" :   7,
    "14d_Before_prec":  14,
    "1m_Before_prec" :  30,
    "3m_Before_prec" :  90,
    "6m_Before_prec" : 180,
    "1Y_Before_prec" : 365,
}
PREC_COLS  = ["daily_precip"] + list(PREC_WINDOWS.keys())
LEAD_DAYS  = max(PREC_WINDOWS.values())
BB_BUFFER  = 1.0   # degrees buffer around state bounding box for grid clip


# ══════════════════════════════════════════════════════════════════════════════
# A. TIDE DATA DOWNLOAD  (run once; skips stations already downloaded)
# ══════════════════════════════════════════════════════════════════════════════

def _download_station(station_id: str, year_min: int, year_max: int,
                      out_path: Path) -> bool:
    """
    Download hourly water level data for one NOAA CO-OPS station year-by-year.

    Parameters
    ----------
    station_id : str   e.g. "8720030"
    year_min   : int   first year to download (inclusive)
    year_max   : int   last  year to download (inclusive)
    out_path   : Path  destination CSV file

    API settings
    ------------
    product   = "water_level"
    datum     = "MHHW"   → water_level = 0 at Mean Higher High Water
    interval  = "h"      → hourly readings
    units     = "Metric" → metres
    time_zone = "GMT"

    Returns True on success, False if all years failed.
    """
    sta     = Station(id=station_id)
    results = []

    for year in range(year_min, year_max + 1):
        begin = f"{year}0101"
        end   = f"{year}1231"
        for attempt in range(3):           # up to 3 retries per year
            try:
                df_yr = sta.get_data(
                    begin_date = begin,
                    end_date   = end,
                    product    = "water_level",
                    datum      = "MHHW",
                    interval   = "h",
                    units      = "Metric",
                    time_zone  = "GMT",
                )
                if df_yr is not None and len(df_yr) > 0:
                    df_yr["Station ID"] = station_id
                    results.append(df_yr)
                break                      # success — move to next year
            except Exception as e:
                if attempt == 2:
                    print(f"      [{station_id}] {year} failed after 3 attempts: {e}",
                          flush=True)
                else:
                    time.sleep(2 ** attempt)   # exponential back-off

    if not results:
        return False

    combined = pd.concat(results)
    # Flatten date_time index → column for consistent CSV storage
    combined.index.name = "date_time"
    combined = combined.reset_index()
    combined.to_csv(out_path, index=False)
    return True


def download_tide_data(df: pd.DataFrame) -> None:
    """
    Identify all unique NOAA station IDs assigned in TideSta_1/2/3, then
    download hourly water level data via the NOAA CO-OPS API for any station
    that does not yet have a CSV file in TIDE_RAW_DIR.

    The date range is derived from the dataset's Date column.

    Downloaded CSVs are saved as TIDE_RAW_DIR / {station_id}.csv and are
    automatically read by _build_tide_lookup() in the next step.
    Stations whose CSV already exists are skipped.

    Data source: NOAA CO-OPS (https://tidesandcurrents.noaa.gov)
    """
    print("\n" + "=" * 65, flush=True)
    print("A. TIDE DATA DOWNLOAD", flush=True)
    print("=" * 65, flush=True)

    TIDE_RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all unique station IDs from the three ranked columns
    sta_cols = [c for c in ["TideSta_1", "TideSta_2", "TideSta_3"]
                if c in df.columns]
    all_ids  = (pd.concat([df[c].dropna() for c in sta_cols])
                  .unique()
                  .tolist())
    all_ids  = [str(int(float(s))) for s in all_ids]   # normalise to "8720030" format
    print(f"  Unique station IDs: {len(all_ids)}", flush=True)

    # Year range from dataset
    date_col = pd.to_datetime(df["Date"], errors="coerce").dropna()
    year_min = date_col.dt.year.min() - 1
    year_max = date_col.dt.year.max()
    print(f"  Date range: {year_min} – {year_max}", flush=True)

    # Skip stations already downloaded ({station_id}.csv present)
    existing = {fp.stem for fp in TIDE_RAW_DIR.glob("*.csv")}
    to_fetch = [s for s in all_ids if s not in existing]
    print(f"  Already downloaded: {len(existing)}  |  To fetch: {len(to_fetch)}",
          flush=True)

    for i, sid in enumerate(sorted(to_fetch), 1):
        out = TIDE_RAW_DIR / f"{sid}.csv"
        print(f"\n  [{i}/{len(to_fetch)}] Station {sid} ...", flush=True)
        t0 = time.time()
        ok = _download_station(sid, year_min, year_max, out)
        status = f"saved -> {out.name}" if ok else "no data returned — skipped"
        print(f"    {status}  [{time.time()-t0:.0f}s]", flush=True)
        time.sleep(0.5)    # polite pause between stations

    print(f"\n  Download complete. "
          f"CSV files in {TIDE_RAW_DIR.name}/: "
          f"{len(list(TIDE_RAW_DIR.glob('*.csv'))):,}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# A. TIDE LEVELS
# ══════════════════════════════════════════════════════════════════════════════

def _build_tide_lookup() -> pd.DataFrame:
    """
    Read all per-station NOAA CO-OPS water level CSVs from TIDE_RAW_DIR
    (downloaded by download_tide_data) and build a daily lookup table per
    (Station_ID, Date) with:
      Daily_Mean, Daily_Max, Daily_Min

    Each CSV contains hourly readings with columns:
      date_time  — "YYYY-MM-DD HH:MM:SS" (floored to calendar date)
      water_level — metres relative to MHHW datum
      Station ID  — NOAA station identifier
    """
    print("  Building tide lookup from raw CSVs ...", flush=True)
    raw_frames = []
    for fp in sorted(TIDE_RAW_DIR.glob("*.csv")):
        df = pd.read_csv(fp, low_memory=False)
        df.columns = df.columns.str.strip()

        # Detect columns by substring to be robust to minor naming variations
        sid_col = next((c for c in df.columns if "station" in c.lower()), None)
        wl_col  = next((c for c in df.columns if "water"   in c.lower()), None)
        dt_col  = next((c for c in df.columns
                        if c.lower() in ("date_time", "date")), None)

        if sid_col is None or wl_col is None or dt_col is None:
            print(f"    [SKIP] {fp.name}: missing station / water_level / date column",
                  flush=True)
            continue

        tmp = df[[sid_col, dt_col, wl_col]].copy()
        tmp.columns = ["Station_ID", "Date", "wl"]
        tmp["Station_ID"] = pd.to_numeric(tmp["Station_ID"], errors="coerce").astype("Int64")
        # Floor hourly timestamps (e.g. "1998-01-01 00:00:00") to calendar date
        tmp["Date"]       = (pd.to_datetime(tmp["Date"], errors="coerce", format="mixed")
                               .dt.normalize())
        tmp["wl"]         = pd.to_numeric(tmp["wl"], errors="coerce")
        raw_frames.append(tmp.dropna(subset=["Date", "wl"]))

    if not raw_frames:
        raise FileNotFoundError(f"No usable tide CSVs found in {TIDE_RAW_DIR}")

    raw = pd.concat(raw_frames, ignore_index=True)
    daily = (raw.groupby(["Station_ID", "Date"])["wl"]
                .agg(Daily_Mean="mean", Daily_Max="max", Daily_Min="min")
                .reset_index())

    lut = daily.copy()
    lut["Station_ID"] = lut["Station_ID"].astype("Int64")
    print(f"  Tide lookup: {len(lut):,} station-date rows  "
          f"| {lut['Station_ID'].nunique()} stations", flush=True)
    return lut


def _merge_tide(df: pd.DataFrame,
                lut: pd.DataFrame,
                sta_col: str) -> pd.DataFrame:
    """
    Left-merge df against tide lookup on (sta_col, Date).
    Returns only the TIDE_COLS values for rows that matched.
    """
    tmp = df[["_idx", sta_col, "Date"]].rename(columns={sta_col: "Station_ID"})
    tmp["Station_ID"] = pd.to_numeric(tmp["Station_ID"], errors="coerce").astype("Int64")
    merged = tmp.merge(lut, on=["Station_ID", "Date"], how="left")
    return merged.set_index("_idx")


def join_tide_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Join Daily_Mean / Daily_Max / Daily_Min to df using
    the TideSta_1 / TideSta_2 / TideSta_3 fallback chain.

    Station_ID column records which station provided data for each row.
    Rows whose nearest three stations all lack data for the observation date
    remain NA.
    """
    print("\n" + "=" * 65, flush=True)
    print("A. TIDE LEVELS", flush=True)
    print("=" * 65, flush=True)

    lut = _build_tide_lookup()

    # Initialize output columns
    for col in TIDE_COLS:
        if col not in df.columns:
            df[col] = np.nan
    if "Station_ID" not in df.columns:
        df["Station_ID"] = pd.NA

    df["_idx"] = np.arange(len(df))

    filled_total = 0
    for rank in range(1, 4):               # TideSta_1, TideSta_2, TideSta_3
        sta_col  = f"TideSta_{rank}"
        if sta_col not in df.columns:
            break

        na_mask  = df["Daily_Mean"].isna() & df[sta_col].notna()
        n_need   = na_mask.sum()
        if n_need == 0:
            continue

        print(f"  TideSta_{rank}: attempting {n_need:,} still-NA rows ...", flush=True)
        sub    = df.loc[na_mask].copy()
        merged = _merge_tide(sub, lut, sta_col)

        filled_mask = merged["Daily_Mean"].notna()
        n_filled    = int(filled_mask.sum())

        if n_filled:
            fill_idx = merged.index[filled_mask]
            for col in TIDE_COLS:
                df.loc[fill_idx, col] = merged.loc[filled_mask, col].values
            df.loc[fill_idx, "Station_ID"] = sub.loc[fill_idx, sta_col].values
            filled_total += n_filled

        print(f"    filled={n_filled:,}  still-NA={n_need - n_filled:,}", flush=True)

    df.drop(columns=["_idx"], inplace=True)

    # Summary
    for col in TIDE_STAT_COLS:
        pct = df[col].notna().mean() * 100
        print(f"  {col:14s}  non-null={pct:.1f}%", flush=True)
    print(f"  Station_ID assigned: {df['Station_ID'].notna().sum():,} rows  "
          f"| {df['Station_ID'].nunique()} unique stations", flush=True)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# B. PRECIPITATION  (NOAA nClimGrid-Daily)
# ══════════════════════════════════════════════════════════════════════════════

def _nc_index() -> dict:
    """Build {YYYYMM: Path} index from nClimGrid-Daily NetCDF files."""
    idx = {}
    for nc in PREC_NC_DIR.rglob("ncdd-*-grd-*.nc"):
        ym     = nc.stem[5:11]          # characters 5-10: YYYYMM
        status = "scaled" if "scaled" in nc.stem else "prelim"
        if ym not in idx or status == "scaled":
            idx[ym] = nc
    return idx


def _ref_grid(nc_idx: dict):
    """Return reference lat/lon arrays and a land-cell mask from any NC file."""
    ds        = xr.open_dataset(next(iter(nc_idx.values())), engine="h5netcdf")
    lats      = ds["lat"].values.copy()
    lons      = ds["lon"].values.copy()
    land_mask = ~np.all(np.isnan(ds["prcp"].values), axis=0)
    ds.close()
    return lats, lons, land_mask


def _state_cells(df_s: pd.DataFrame, ref_lats, ref_lons, land_mask):
    """
    Clip the nClimGrid reference grid to the state bounding box and map each
    unique well (SiteNo) to the nearest valid land grid cell.
    """
    lat_min = df_s["lat"].min() - BB_BUFFER
    lat_max = df_s["lat"].max() + BB_BUFFER
    lon_min = df_s["lon"].min() - BB_BUFFER
    lon_max = df_s["lon"].max() + BB_BUFFER

    lat_idx = np.where((ref_lats >= lat_min) & (ref_lats <= lat_max))[0]
    lon_idx = np.where((ref_lons >= lon_min) & (ref_lons <= lon_max))[0]

    clip_lats = ref_lats[lat_idx]
    clip_lons = ref_lons[lon_idx]

    lon_grid, lat_grid = np.meshgrid(clip_lons, clip_lats)
    flat_lats = lat_grid.ravel()
    flat_lons = lon_grid.ravel()

    clip_mask     = land_mask[np.ix_(lat_idx, lon_idx)].ravel()
    land_flat_idx = np.where(clip_mask)[0]
    land_lats     = flat_lats[clip_mask]
    land_lons     = flat_lons[clip_mask]

    if len(land_lats) == 0:
        raise ValueError("No valid land cells in state bounding box.")

    tree  = cKDTree(np.column_stack([land_lats, land_lons]))
    sites = (df_s[["SiteNo", "lat", "lon"]]
             .drop_duplicates("SiteNo")
             .reset_index(drop=True))
    _, nn_idx = tree.query(sites[["lat", "lon"]].values.astype(float))

    matched_flat  = land_flat_idx[nn_idx]
    n_clip_lon    = len(clip_lons)
    sites["ilat"] = (matched_flat // n_clip_lon).astype(int)
    sites["ilon"] = (matched_flat  % n_clip_lon).astype(int)

    snap_deg = np.sqrt((sites["lat"].values - land_lats[nn_idx])**2 +
                       (sites["lon"].values - land_lons[nn_idx])**2)
    if (snap_deg > 0.15).any():
        print(f"    Coastal snap: {(snap_deg > 0.15).sum()} well(s) "
              f"> 0.15° to nearest land cell", flush=True)

    return sites[["SiteNo", "ilat", "ilon"]], clip_lats, clip_lons


def _extract_matrix(date_min, date_max, sites_df, clip_lats, clip_lons,
                    nc_idx: dict, nc_start: pd.Timestamp):
    """
    Load nClimGrid NetCDF files and return:
      dates_arr : np.array[datetime64], shape (n_days,)
      data_mat  : float64 array,        shape (n_days, n_cells)
    """
    months  = pd.period_range(pd.Timestamp(date_min).to_period("M"),
                               pd.Timestamp(date_max).to_period("M"), freq="M")
    unique  = sites_df[["ilat", "ilon"]].drop_duplicates().reset_index(drop=True)
    lat_da  = xr.DataArray(unique["ilat"].tolist(), dims="cell")
    lon_da  = xr.DataArray(unique["ilon"].tolist(), dims="cell")

    date_chunks, data_chunks = [], []
    for m in months:
        ym = m.strftime("%Y%m")
        if ym not in nc_idx:
            continue
        ds  = xr.open_dataset(nc_idx[ym], engine="h5netcdf")
        da  = ds["prcp"].sel(lat=clip_lats, lon=clip_lons, method="nearest")
        arr = da.isel(lat=lat_da, lon=lon_da).values.astype(np.float32)
        date_chunks.append(pd.to_datetime(da["time"].values))
        data_chunks.append(arr)
        ds.close()

    if not date_chunks:
        return None, None

    return (np.concatenate(date_chunks),
            np.vstack(data_chunks).astype(np.float64))


def _build_precip_result(df_s, sites_df, dates_arr, data_mat):
    """
    Map (n_days × n_cells) matrix back to well-date rows; compute rolling
    accumulations one window at a time to control peak memory.
    """
    unique    = sites_df[["ilat", "ilon"]].drop_duplicates().reset_index(drop=True)
    cell_idx  = {(int(il), int(io)): i
                 for i, (il, io) in enumerate(unique[["ilat","ilon"]].values)}
    date_idx  = pd.Series(np.arange(len(dates_arr)),
                          index=pd.DatetimeIndex(dates_arr))

    merged  = df_s.merge(sites_df[["SiteNo", "ilat", "ilon"]], on="SiteNo", how="left")
    d_pos   = date_idx.reindex(merged["Date"]).values
    c_pos   = merged.apply(
        lambda r: cell_idx.get((int(r["ilat"]), int(r["ilon"])), -1), axis=1).values

    valid     = (~np.isnan(d_pos.astype(float))) & (c_pos >= 0)
    d_int     = np.where(valid, d_pos.astype(float).astype(int), 0)
    c_int     = np.where(valid, c_pos.astype(int), 0)

    merged["daily_precip"] = np.where(valid, data_mat[d_int, c_int], np.nan)

    # Rolling sums (shifted by 1: window ends the day BEFORE the sample date)
    df_mat  = pd.DataFrame(data_mat)
    shifted = df_mat.shift(1)
    del df_mat; gc.collect()

    for col, w in PREC_WINDOWS.items():
        rolled         = shifted.rolling(window=w, min_periods=1).sum()
        merged[col]    = np.where(valid, rolled.values[d_int, c_int], np.nan)
        del rolled; gc.collect()

    del shifted; gc.collect()
    return merged.drop(columns=["ilat", "ilon"], errors="ignore")


def join_precipitation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Match each well-date row to its nearest nClimGrid-Daily land grid cell
    and add daily_precip + rolling accumulation columns.
    """
    print("\n" + "=" * 65, flush=True)
    print("B. PRECIPITATION  (nClimGrid-Daily)", flush=True)
    print("=" * 65, flush=True)

    if not PREC_NC_DIR.exists():
        print(f"  [WARN] PREC_NC_DIR not found: {PREC_NC_DIR}", flush=True)
        print("  Skipping precipitation — columns will be NA.", flush=True)
        for col in PREC_COLS:
            if col not in df.columns:
                df[col] = np.nan
        return df

    nc_idx   = _nc_index()
    nc_start = pd.Timestamp(f"{min(nc_idx)[:4]}-{min(nc_idx)[4:]}-01")
    print(f"  NC index: {len(nc_idx)} months ({min(nc_idx)} – {max(nc_idx)})",
          flush=True)

    ref_lats, ref_lons, land_mask = _ref_grid(nc_idx)

    for col in PREC_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df["Date"] = pd.to_datetime(df["Date"])

    states      = sorted(df["State"].dropna().unique())
    result_rows = []

    for state in states:
        t0    = time.time()
        df_s  = df[df["State"] == state].copy()
        print(f"\n  [{state}]  rows={len(df_s):,}", flush=True)

        date_min = max(df_s["Date"].min() - timedelta(days=LEAD_DAYS), nc_start)
        date_max = df_s["Date"].max()

        try:
            sites_df, clip_lats, clip_lons = _state_cells(
                df_s, ref_lats, ref_lons, land_mask)
        except ValueError as e:
            print(f"    [SKIP] {e}", flush=True)
            result_rows.append(df_s)
            continue

        dates_arr, data_mat = _extract_matrix(
            date_min, date_max, sites_df, clip_lats, clip_lons, nc_idx, nc_start)

        if dates_arr is None:
            print(f"    No NC data for date range — precip stays NA", flush=True)
            result_rows.append(df_s)
            continue

        result = _build_precip_result(df_s, sites_df, dates_arr, data_mat)
        del data_mat; gc.collect()

        pct    = result["daily_precip"].notna().mean() * 100
        print(f"    daily_precip filled={pct:.1f}%  [{time.time()-t0:.0f}s]",
              flush=True)
        result_rows.append(result)

    df_out = pd.concat(result_rows, ignore_index=True)
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# C. DERIVED TIDE FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def add_tide_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Ele_MHHW and Ele_MAX columns.

    Ele_MHHW  = Surf_Ele − Daily_Mean
                Surf_Ele is in NAVD88 (m); Daily_Mean is the observed daily
                mean water level reported in the MHHW datum frame (not the
                fixed MHHW benchmark itself — it varies daily with the tide).
                Captures how far the land surface sits above/below that day's
                mean tide level.  Large positive → land well above water;
                near zero / negative → water approaching land surface.
                The implicit MHHW-to-NAVD88 offset (~0.2–1.5 m per station) is
                constant per station and does not affect within-site temporal 
                dynamics used by the model.

    Ele_MAX   = Surf_Ele − Daily_Max
                Same concept using the daily maximum tide height.
                Negative values indicate the surface would be inundated at peak tide.
    """
    print("\n" + "=" * 65, flush=True)
    print("C. DERIVED TIDE FEATURES", flush=True)
    print("=" * 65, flush=True)

    for col in ("Surf_Ele", "Daily_Mean", "Daily_Max"):
        if col not in df.columns:
            print(f"  [WARN] Required column '{col}' not found — "
                  f"some derived features will be NA.", flush=True)

    for col in ("lon", "lat", "Daily_Mean", "Surf_Ele", "Daily_Max"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Ele_MHHW ──────────────────────────────────────────────────────────────
    # Ele_MHHW = Surf_Ele (NAVD88) − Daily_Mean (MHHW datum)
    #   > 0 → land surface above current mean tide level
    #   ≈ 0 → land at mean tide level
    #   < 0 → land surface below mean tide level (frequent tidal inundation)
    if "Surf_Ele" in df.columns and "Daily_Mean" in df.columns:
        df["Ele_MHHW"] = df["Surf_Ele"] - df["Daily_Mean"]
        pct = df["Ele_MHHW"].notna().mean() * 100
        print(f"  Ele_MHHW  non-null={pct:.1f}%  "
              f"median={df['Ele_MHHW'].median():.2f} m", flush=True)
    else:
        df["Ele_MHHW"] = np.nan
        print("  Ele_MHHW  skipped (Surf_Ele or Daily_Mean missing)", flush=True)

    # ── Ele_MAX ───────────────────────────────────────────────────────────────
    # Ele_MAX = Surf_Ele (NAVD88) − Daily_Max (MHHW datum)
    #   > 0 → land surface above today's maximum water level
    #   < 0 → land surface would be inundated at daily high tide
    if "Surf_Ele" in df.columns and "Daily_Max" in df.columns:
        df["Ele_MAX"] = df["Surf_Ele"] - df["Daily_Max"]
        pct = df["Ele_MAX"].notna().mean() * 100
        print(f"  Ele_MAX   non-null={pct:.1f}%  "
              f"median={df['Ele_MAX'].median():.2f} m", flush=True)
    else:
        df["Ele_MAX"] = np.nan

    return df


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

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\nLoading {FINAL_CSV.name} ...", flush=True)
    df = pd.read_csv(FINAL_CSV, dtype={"SiteNo": str}, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["lon"]  = pd.to_numeric(df["lon"],  errors="coerce")
    df["lat"]  = pd.to_numeric(df["lat"],  errors="coerce")
    print(f"  {len(df):,} rows  x  {len(df.columns)} cols", flush=True)

    # ── A: Download raw hourly tide data (skips already-downloaded stations) ──
    download_tide_data(df)

    # ── A: Aggregate to daily stats, join with fallback ───────────────────────
    df = join_tide_levels(df)

    # ── B: Precipitation ──────────────────────────────────────────────────────
    df = join_precipitation(df)

    # ── C: Derived tide features ──────────────────────────────────────────────
    df = add_tide_derived(df)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65, flush=True)
    new_cols = TIDE_COLS + ["Station_ID"] + PREC_COLS + ["Ele_MHHW", "Ele_MAX"]
    # TIDE_COLS = Daily_Mean, Daily_Max, Daily_Min only (rolling lags excluded)
    print(f"Final: {len(df):,} rows  x  {len(df.columns)} cols", flush=True)
    print(f"\nNew / updated columns:", flush=True)
    for col in new_cols:
        if col in df.columns:
            pct = df[col].notna().mean() * 100
            print(f"  {col:24s}  non-null={pct:.1f}%", flush=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving -> {FINAL_CSV.name} ...", flush=True)
    df.to_csv(FINAL_CSV, index=False)
    print(f"Done.  {len(df):,} rows  x  {len(df.columns)} cols", flush=True)
    print(f"Total time: {(time.time()-t_global)/60:.1f} min", flush=True)


main()
