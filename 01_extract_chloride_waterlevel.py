"""
01_extract_chloride_waterlevel.py
==================================
Extracts Chloride, WaterLevel, WellDepth, and Aquifer from federal and
state-specific groundwater databases, and saves one clean CSV per state.

Data sources
------------
  Federal (all states):
    A. Water Quality Portal (WQP)      waterqualitydata.us
    B. National Groundwater Monitoring Network (NGWMN)  cida.usgs.gov/ngwmn

  State supplements:
    TX  — TWDB (Texas Water Development Board, Groundwater Database)
    DE  — DGS (Delaware Geological Survey, DGS Report of Investigations 85)
    SC  — SCDES (South Carolina Department of Environmental Services,
    Saltwater Intrusion Monitoring Network, Groundwater Level Monitoring Network) 
    FL  — DBHYDRO (South Florida Water Management District, DBHYDRO Wells and Boreholes)
    NC  — NCDEQ—DWR (North Carolina Division of Water Resources,Groundwater Levels & Quality)
    GA  — GEPD (Georgia Environmental Protection Division, Georgia Environmental Monitoring and Assessment System)

Output schema (all sources unified before concat)
--------------------------------------------------
  SiteNo, Date, Chloride, WaterLevel_m, WellDepth, Aquifer, geometry

"""

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA     = Path(__file__).parent
COUNTIES = DATA / "study_area_counties"

STATE_NAMES = {
    "NJ": "New Jersey",    "PA": "Pennsylvania", "DE": "Delaware",
    "MD": "Maryland",      "VA": "Virginia",     "NC": "North Carolina",
    "SC": "South Carolina","GA": "Georgia",       "FL": "Florida",
    "AL": "Alabama",       "MS": "Mississippi",   "LA": "Louisiana",
    "TX": "Texas",
}

# Unified output columns — every track normalises to these before concat
SCHEMA = ["SiteNo", "Date", "Chloride", "WaterLevel_m",
          "WellDepth", "Aquifer", "geometry"]

FT_TO_M = 0.3048   # feet → metres for WaterLevel

# ── Shared helpers ─────────────────────────────────────────────────────────────

def load_counties(state: str) -> gpd.GeoDataFrame:
    ct = gpd.read_file(str(COUNTIES), crs="EPSG:4326")
    return ct[ct["STATE"] == STATE_NAMES[state]].to_crs(4326)


def to_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce a DataFrame to SCHEMA:
      - Add missing columns as NaN
      - Keep only SCHEMA columns
      - SiteNo → str, Date → datetime, numeric coercion for Chloride/WellDepth
    """
    for col in SCHEMA:
        if col not in df.columns:
            df[col] = np.nan
    df = df[SCHEMA].copy()
    df["SiteNo"]    = df["SiteNo"].astype(str).str.strip()
    df["Date"]      = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["Chloride"]  = pd.to_numeric(df["Chloride"],    errors="coerce")
    df["WellDepth"] = pd.to_numeric(df["WellDepth"],   errors="coerce")
    return (df.dropna(subset=["SiteNo", "Date", "Chloride"])
              .drop_duplicates(subset=["SiteNo", "Date"], keep="first")
              .sort_values("Date")
              .reset_index(drop=True))


def first_valid(*cols) -> pd.Series:
    """Return first non-null value across multiple Series (for Aquifer merging)."""
    return pd.concat(cols, axis=1).bfill(axis=1).iloc[:, 0]


# ── Track A: WQP ──────────────────────────────────────────────────────────────

def extract_wqp(state: str, counties: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    WQP station + results → Chloride with WaterLevel, WellDepth, Aquifer.
    WellDepth: prefers WellDepthMeasure, falls back to WellHoleDepthMeasure.
    WaterLevel: CharacteristicName in {'Depth, from ground surface…',
                                       'Depth to water level below…'}.
    """
    wqp = DATA / "WQP"
    station = pd.read_csv(wqp / f"station_{state}.csv",
                          encoding="unicode_escape", low_memory=False)
    results = pd.read_csv(wqp / f"resultphyschem_{state}.csv",
                          encoding="unicode_escape", low_memory=False)

    gdf = gpd.GeoDataFrame(
        station,
        geometry=gpd.points_from_xy(station["LongitudeMeasure"],
                                     station["LatitudeMeasure"]), crs=4326)
    data = gpd.clip(
        gpd.GeoDataFrame(pd.merge(results, gdf, on="MonitoringLocationIdentifier",
                                  how="left"), crs=4326),
        counties)

    # Aquifer: prefer AquiferName, fallback FormationTypeText
    aq = (gdf[["MonitoringLocationIdentifier", "AquiferName", "FormationTypeText"]]
          .drop_duplicates("MonitoringLocationIdentifier"))
    aq["Aquifer"] = first_valid(aq["AquiferName"], aq["FormationTypeText"])

    # Chloride rows
    CHL = data[data["CharacteristicName"] == "Chloride"].copy()
    CHL.rename(columns={"MonitoringLocationIdentifier": "SiteNo",
                        "ActivityStartDate":            "Date",
                        "ResultMeasureValue":           "Chloride"}, inplace=True)
    # WellDepth: measured depth preferred, hole depth as fallback; convert ft → m
    CHL["WellDepth"] = pd.to_numeric(
        CHL["WellDepthMeasure/MeasureValue"].combine_first(
        CHL["WellHoleDepthMeasure/MeasureValue"]), errors="coerce")
    unit = CHL.get("WellDepthMeasure/MeasureUnitCode", pd.Series(dtype=str))
    ft_mask = unit.str.lower().isin(["ft", "feet"]) | unit.isna()  # default ft for US
    CHL.loc[ft_mask, "WellDepth"] = CHL.loc[ft_mask, "WellDepth"] * FT_TO_M
    CHL = pd.merge(CHL[["SiteNo","Date","Chloride","WellDepth","geometry"]],
                   aq.rename(columns={"MonitoringLocationIdentifier":"SiteNo"})
                     [["SiteNo","Aquifer"]],
                   on="SiteNo", how="left")

    # Water level rows
    LEV = data[data["CharacteristicName"].isin([
        "Depth, from ground surface to well water level"])].copy()
    LEV.rename(columns={"MonitoringLocationIdentifier": "SiteNo",
                        "ActivityStartDate":            "Date",
                        "ResultMeasureValue":           "WaterLevel_m"}, inplace=True)
    LEV = (LEV[["SiteNo","Date","WaterLevel_m"]]
           .drop_duplicates(["SiteNo","Date"], keep="first"))

    out = pd.merge(CHL, LEV, on=["SiteNo","Date"], how="left")
    print(f"  [WQP]   {state}: {len(out):,} rows")
    return to_schema(out)


# ── Track B: NGWMN ────────────────────────────────────────────────────────────

def extract_ngwmn(state: str, counties: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    NGWMN SITE_INFO + QUALITY + WATERLEVEL → Chloride with WaterLevel.
    WaterLevel converted ft → m (NAVD88 relative).
    Aquifer: prefers NatAqfrDesc, fallback LocalAquiferName.
    """
    ng = DATA / "NGWMN"
    site = pd.read_csv(ng / "SITE_INFO.csv", encoding="unicode_escape",
                       low_memory=False)
    qual = pd.read_csv(ng / "QUALITY.csv",   encoding="unicode_escape",
                       low_memory=False)
    lev  = pd.read_csv(ng / "WATERLEVEL.csv",encoding="unicode_escape",
                       low_memory=False)

    gdf = gpd.GeoDataFrame(
        site,
        geometry=gpd.points_from_xy(site["DecLongVa"], site["DecLatVa"]),
        crs=4326)
    data = gpd.clip(
        gpd.GeoDataFrame(pd.merge(qual, gdf, on="SiteNo", how="left"), crs=4326),
        counties)

    CHL = data[data["CharacteristicName"].str.lower() == "chloride"].copy()
    CHL.rename(columns={"ActivityStartDate": "Date",
                        "ResultMeasureValue": "Chloride"}, inplace=True)
    CHL["Aquifer"] = first_valid(CHL.get("NatAqfrDesc",   pd.Series(dtype=str)),
                                 CHL.get("LocalAquiferName", pd.Series(dtype=str)))
    CHL = CHL[["SiteNo","Date","Chloride","WellDepth","Aquifer","geometry"]].copy()
    # NGWMN reports well depth in feet → convert to metres
    CHL["WellDepth"] = pd.to_numeric(CHL["WellDepth"], errors="coerce").mul(FT_TO_M)

    lev["Date"] = (pd.to_datetime(lev["Time"], format="mixed", utc=True)
                   .dt.tz_localize(None).dt.normalize())
    # Convert ft → m; use NAVD88 elevation if depth-below-surface is null
    lev["WaterLevel_m"] = (
        pd.to_numeric(lev["Depth to Water Below Land Surface in ft."],
                      errors="coerce").mul(FT_TO_M)
        .combine_first(
        pd.to_numeric(lev["Water level in feet relative to NAVD88"],
                      errors="coerce").mul(FT_TO_M)))
    LEV = (lev[lev["SiteNo"].astype(str).isin(data["SiteNo"].astype(str))]
           [["SiteNo","Date","WaterLevel_m"]]
           .drop_duplicates(["SiteNo","Date"], keep="first"))

    out = pd.merge(CHL, LEV, on=["SiteNo","Date"], how="left")
    print(f"  [NGWMN] {state}: {len(out):,} rows")
    return to_schema(out)


# ── State supplements ──────────────────────────────────────────────────────────
# TX  — TWDB (Texas Water Development Board, Groundwater Database)
def extract_twdb(counties: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    TX — TWDB water quality + water level (4 pipe-delimited files each).

    SiteNo format: 7-digit numeric (e.g. '8737703'), distinct from USGS 'USGS-XXXXXXXXX'.

    Water level column choice:
      WaterElevation  = LandElevation - DepthFromLSD  (absolute NAVD88 ft) -- NOT used
      DepthFromLSD    = depth to water below land surface (ft, positive = deeper)
                        -- consistent with WQP 'Depth, from ground surface to
                           well water level' and NGWMN 'Depth to Water Below
                           Land Surface in ft.' -- USED HERE, converted ft → m
    """
    td = DATA / "TWDB2026"

    def _pipe(f): return pd.read_csv(td / f, sep="|", encoding="latin1",
                                     on_bad_lines="skip", low_memory=False)

    # Well metadata → geometry + Aquifer
    well = gpd.read_file(str(td / "TWDB_Groundwater" / "TWDB_Groundwater.shp"))
    well = gpd.clip(well.to_crs(4326), counties).drop_duplicates("StateWellN")
    well.rename(columns={"StateWellN":"SiteNo","WellDepth":"WellDepth",
                         "AquiferCod":"Aquifer"}, inplace=True)
    # TWDB well depth is in feet → convert to metres
    well["WellDepth"] = pd.to_numeric(well["WellDepth"], errors="coerce").mul(FT_TO_M)
    aoi  = well["SiteNo"].astype(str).tolist()

    # Chloride
    wq = pd.concat([_pipe(f) for f in [
        "WaterQualityMajor.txt","WaterQualityMinor.txt",
        "WaterQualityOtherUnassigned.txt","WaterQualityCombination.txt"]])
    CHL = wq[wq["ParameterDescription"].str.contains("CHLORIDE, DISSOLVED",
              case=False, na=False)].copy()
    CHL = CHL[CHL["StateWellNumber"].astype(str).isin(aoi)]
    CHL.rename(columns={"StateWellNumber":"SiteNo","SampleDate":"Date",
                        "ParameterValue":"Chloride"}, inplace=True)

    # Water level: use DepthFromLSD (depth below land surface, ft) → m
    wl = pd.concat([_pipe(f) for f in [
        "WaterLevelsMajor.txt","WaterLevelsMinor.txt",
        "WaterLevelsOtherUnassigned.txt","WaterLevelsCombination.txt"]])
    wl = wl.drop_duplicates()
    LEV = wl[wl["StateWellNumber"].astype(str).isin(aoi)].copy()
    LEV.rename(columns={"StateWellNumber":"SiteNo",
                        "MeasurementDate":"Date"}, inplace=True)
    LEV["WaterLevel_m"] = pd.to_numeric(LEV["DepthFromLSD"],
                                         errors="coerce").mul(FT_TO_M)
    LEV = LEV.dropna(subset=["WaterLevel_m"])[["SiteNo","Date","WaterLevel_m"]]

    out = (pd.merge(CHL[["SiteNo","Date","Chloride"]], LEV,
                    on=["SiteNo","Date"], how="left")
             .merge(well[["SiteNo","WellDepth","Aquifer","geometry"]],
                    on="SiteNo", how="left"))
    print(f"  [TWDB]  TX: {len(out):,} rows")
    return to_schema(out)

# DE  — DGS (Delaware Geological Survey, DGS Report of Investigations 85)
def extract_dgs() -> pd.DataFrame:
    """DE — DGS Report of Investigations 85 (Lab + Field + Site)."""
    dgs = DATA / "DGS"
    site = pd.read_csv(dgs / "B21C_SiteData.csv",
                       encoding="unicode_escape", low_memory=False)
    lab  = pd.read_csv(dgs / "B21C_LabData.csv",
                       encoding="unicode_escape", low_memory=False)
    fld  = pd.read_csv(dgs / "B21C_FieldData.csv",
                       encoding="unicode_escape", low_memory=False)

    # Flexible column name match for site file (headers may contain \n)
    def _col(df, *keywords):
        for c in df.columns:
            if all(k in c.lower() for k in keywords):
                return c
        return None

    site_id = _col(site, "well", "ident") or site.columns[0]
    gdf = gpd.GeoDataFrame(site, geometry=gpd.points_from_xy(
        site["longitude"], site["latitude"]), crs=4326)

    # Rename site columns to schema names
    renames = {site_id: "SiteNo"}
    for pat, name in [("land surface", "Surf_Ele"),
                      ("well depth",   "WellDepth"),
                      ("aquifer",      "Aquifer")]:
        c = _col(site, *pat.split())
        if c: renames[c] = name
    gdf.rename(columns=renames, inplace=True)
    # DGS site data reports well depth in feet → convert to metres
    if "WellDepth" in gdf.columns:
        gdf["WellDepth"] = pd.to_numeric(gdf["WellDepth"], errors="coerce").mul(FT_TO_M)

    lab = lab[["Site Identifier","Date Sampled","Cl (mg/L)"]].copy()
    lab.rename(columns={"Site Identifier":"SiteNo","Date Sampled":"Date",
                        "Cl (mg/L)":"Chloride"}, inplace=True)

    # Field water level (depth to water at sampling time)
    wl_col = _col(fld, "water") or None
    fld_sel = fld[["DGSWell_Identifier","Date Sampled"]
                  + ([wl_col] if wl_col else [])].copy()
    fld_sel.rename(columns={"DGSWell_Identifier":"SiteNo",
                             "Date Sampled":"Date",
                             **({"wl_col":"WaterLevel_m"} if wl_col else {})},
                   inplace=True)

    out = (lab.merge(gdf[["SiteNo","WellDepth","Aquifer","geometry"]],
                     on="SiteNo", how="left")
              .merge(fld_sel, on=["SiteNo","Date"], how="left"))
    print(f"  [DGS]   DE: {len(out):,} rows")
    return to_schema(out)

# SC  — SCDES (Saltwater Intrusion Monitoring Network) 
def _calibrate_spc(wqp_df: pd.DataFrame, min_pairs: int = 3) -> pd.DataFrame:
    """Fit Chloride = a·SpC + b per site from co-located WQP pairs."""
    paired = wqp_df[["SiteNo","SpecificConduct","Chloride"]].dropna()
    rows   = []
    for site, g in paired.groupby("SiteNo"):
        if len(g) < min_pairs: continue
        try:
            popt, _ = curve_fit(lambda x,a,b: a*x+b,
                                g["SpecificConduct"], g["Chloride"])
            rows.append(dict(SiteNo=site, slope=popt[0], intercept=popt[1],
                             r2=round(r2_score(g["Chloride"],
                                 popt[0]*g["SpecificConduct"]+popt[1]),4),
                             n=len(g)))
        except RuntimeError:
            pass
    # global fallback
    popt_g, _ = curve_fit(lambda x,a,b: a*x+b,
                          paired["SpecificConduct"], paired["Chloride"])
    rows.append(dict(SiteNo="__global__", slope=popt_g[0],
                     intercept=popt_g[1], r2=np.nan, n=len(paired)))
    return pd.DataFrame(rows).set_index("SiteNo")


def extract_scdnr(wqp_df: pd.DataFrame) -> pd.DataFrame:
    """
    SC — SCDNR daily SpC converted to Chloride via per-site WQP calibration.
    Water level gap-filled from per-well CSVs (daily logger + manual).
    """
    sd = DATA / "SCDNR"

    # SpC daily data + geometry
    site = pd.read_csv(sd / "SCDNR_Wells.csv",
                       encoding="unicode_escape", low_memory=False)
    data = pd.read_csv(sd / "SCDNR_DailyData.csv",
                       encoding="unicode_escape", low_memory=False)
    gdf  = gpd.GeoDataFrame(site,
            geometry=gpd.points_from_xy(site["Longitude"], site["Latitude"]),
            crs=4326)
    df = (pd.merge(data, gdf, on="SiteNo", how="left")
            .rename(columns={"Daily Ave SpC UScm": "SpecificConduct"})
            .drop(columns=["Latitude","Longitude"], errors="ignore"))

    # SpC → Chloride
    coef = _calibrate_spc(wqp_df)
    coef.to_csv(sd / "SpC_Chloride_calibration.csv")

    def _predict(row):
        if pd.isna(row.get("SpecificConduct")): return np.nan
        c = coef.loc[row["SiteNo"]] if row["SiteNo"] in coef.index \
            else coef.loc["__global__"]
        return max(0.0, c["slope"] * row["SpecificConduct"] + c["intercept"])

    df["Chloride"] = df.apply(_predict, axis=1)

    # Water level: SCDES Groundwater level monitoring network
    wl_dir = sd / "SC_WaterLevel"
    if wl_dir.exists():
        wl_frames = []
        for fp in sorted(wl_dir.rglob("*.csv")):
            w = pd.read_csv(fp)
            w.columns = [c.strip() for c in w.columns]
            dcol = next((c for c in w.columns if "date" in c.lower()), None)
            if not dcol: continue
            w.rename(columns={dcol: "Date"}, inplace=True)
            w["Date"] = pd.to_datetime(w["Date"], errors="coerce").dt.normalize()
            DAILY, MANUAL = "Daily average water levels", "Manual Water Levels"
            if DAILY not in w.columns or MANUAL not in w.columns: continue
            for c in [DAILY, MANUAL]:
                w[c] = pd.to_numeric(w[c], errors="coerce")
            # Gap-fill daily with same-date manual
            manual_map = (w.groupby("Date")[MANUAL]
                           .apply(lambda s: s.dropna().iloc[0]
                                  if s.notna().any() else np.nan))
            w[DAILY] = w[DAILY].fillna(w["Date"].map(manual_map))
            w["SiteNo"] = fp.stem.split("_")[0]
            wl_frames.append(w[["SiteNo","Date",DAILY]].rename(
                columns={DAILY: "WaterLevel_m"}))

        if wl_frames:
            wl = (pd.concat(wl_frames, ignore_index=True)
                    .drop_duplicates(["SiteNo","Date"], keep="first"))
            wl["WaterLevel_m"] = wl["WaterLevel_m"].mul(FT_TO_M)
            df = pd.merge(df, wl, on=["SiteNo","Date"], how="left")

    # Aquifer from site file
    aq_col = next((c for c in site.columns if "aquifer" in c.lower()), None)
    if aq_col:
        df = df.merge(site[["SiteNo", aq_col]].rename(
            columns={aq_col:"Aquifer"}), on="SiteNo", how="left")

    print(f"  [SCDNR] SC: {len(df):,} rows")
    return to_schema(df)

# FL  — DBHYDRO (South Florida Water Management District, DBHYDRO Wells and Boreholes)

def extract_dbhydro(counties: gpd.GeoDataFrame) -> pd.DataFrame:
    """FL — DBHYDRO chloride; NAD83 → WGS84; WellDepth from well CSV."""
    db = DATA / "DBHYDRO"
    chl = pd.read_csv(db / "FL_All_GW.csv",
                      encoding="unicode_escape", low_memory=False)
    chl["Date"]    = pd.to_datetime(chl["Collection_Date"], errors="coerce").dt.normalize()
    chl["Chloride"] = pd.to_numeric(chl["Value"], errors="coerce")
    chl.rename(columns={"Station ID": "SiteNo"}, inplace=True)

    wells = gpd.GeoDataFrame(
        pd.read_csv(db / "DBHYDRO_Wells_and_Boreholes.csv",
                    encoding="unicode_escape", low_memory=False),
        geometry=gpd.points_from_xy(
            pd.read_csv(db / "DBHYDRO_Wells_and_Boreholes.csv",
                        encoding="unicode_escape")["LON_DD"],
            pd.read_csv(db / "DBHYDRO_Wells_and_Boreholes.csv",
                        encoding="unicode_escape")["LAT_DD"]),
        crs=4269).to_crs(4326).drop_duplicates("STATION")
    wells.rename(columns={"STATION":"SiteNo","DEPTH_DRILLED":"WellDepth"},
                 inplace=True)
    # DBHYDRO DEPTH_DRILLED is in feet → convert to metres
    wells["WellDepth"] = pd.to_numeric(wells["WellDepth"], errors="coerce").mul(FT_TO_M)

    out = gpd.clip(
        gpd.GeoDataFrame(
            pd.merge(chl[["SiteNo","Date","Chloride"]],
                     wells[["SiteNo","WellDepth","geometry"]],
                     on="SiteNo", how="left"), crs=4326),
        counties)
    print(f"  [DBHYDRO] FL: {len(out):,} rows")
    return to_schema(out)

# NC  — NCDEQ—DWR (North Carolina Division of Water Resources,Groundwater Levels & Quality)
def extract_nc_state(counties: gpd.GeoDataFrame) -> pd.DataFrame:
    """NC — DWR chloride + DEQ ambient monitoring chloride, combined."""
    frames = []

    # NC DWR
    try:
        dwr = pd.read_csv(DATA / "NCDWR" / "nc_dwr_groundwater_chloride_data.csv",
                          encoding="unicode_escape", low_memory=False)
        dwr = gpd.GeoDataFrame(dwr,
              geometry=gpd.points_from_xy(dwr["Longitude"], dwr["Latitude"]),
              crs=4326)
        dwr.rename(columns={"Well ID":              "SiteNo",
                             "Date Sampled":         "Date",
                             "Chloride (mg/L)":      "Chloride",
                             "Total Depth  (ft bgs)":"WellDepth",
                             "Aquifer":              "Aquifer"}, inplace=True)
        # Column name explicitly states ft bgs → convert to metres
        dwr["WellDepth"] = pd.to_numeric(dwr["WellDepth"], errors="coerce").mul(FT_TO_M)
        frames.append(dwr[SCHEMA].copy())
        print(f"  [NCDWR] NC: {len(dwr):,} rows")
    except FileNotFoundError:
        print("  [NCDWR] file not found — skipped")

    # NC DEQ
    try:
        deq = pd.read_csv(
            DATA / "NCDEQ" / "DEQAmbientGroundwaterQualityMonitoringNetwork2.csv",
            encoding="unicode_escape", low_memory=False)
        deq = gpd.clip(gpd.GeoDataFrame(
            deq, geometry=gpd.points_from_xy(deq["Longitude"], deq["Latitude"]),
            crs=4326), counties)
        deq = deq[deq["Parameter"] == "Chloride"].copy()
        deq["SiteNo"]   = (deq["Site Name"].astype(str).str.strip() + "-" +
                           deq["Quad"].astype(str).str.strip())
        deq["Date"]     = pd.to_datetime(deq["Collection Date"], errors="coerce")
        deq.rename(columns={"Result":         "Chloride",
                             "Aquifier":       "Aquifer",   # typo in source
                             "Depth (ft bgs)": "WellDepth"}, inplace=True)
        # Column name explicitly states ft bgs → convert to metres
        deq["WellDepth"] = pd.to_numeric(deq["WellDepth"], errors="coerce").mul(FT_TO_M)
        frames.append(deq[SCHEMA].copy())
        print(f"  [NCDEQ] NC: {len(deq):,} rows")
    except FileNotFoundError:
        print("  [NCDEQ] file not found — skipped")

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return to_schema(out)

# GA  — GEPD (Georgia Environmental Monitoring and Assessment System)

def extract_gomas(counties: gpd.GeoDataFrame) -> pd.DataFrame:
    """GA — GOMAS chloride; geometry + WellDepth joined from WQP station + local CSV."""
    gm = DATA / "GOMAS"
    PREFIX = "21GAEPD_WQX-"

    gomas = pd.read_csv(gm / "Report_2023-01-20.csv",
                        encoding="unicode_escape", low_memory=False)
    gomas = gomas[["Monitoring Location ID","Date","Result (Value)","Units"]].dropna()
    gomas["SiteNo"] = PREFIX + gomas["Monitoring Location ID"].astype(str).str.strip()

    # Geometry from WQP station file (reused)
    ga_site = pd.read_csv(DATA / "WQP" / "station_GA.csv",
                          encoding="unicode_escape", low_memory=False)
    geo = (gpd.GeoDataFrame(ga_site,
            geometry=gpd.points_from_xy(ga_site["LongitudeMeasure"],
                                         ga_site["LatitudeMeasure"]), crs=4326)
           [["MonitoringLocationIdentifier","geometry"]]
           .drop_duplicates("MonitoringLocationIdentifier")
           .rename(columns={"MonitoringLocationIdentifier":"SiteNo"}))

    # WellDepth from local station CSV (row 0 is sub-header → skip)
    local = pd.read_csv(gm / "GA_Local_Station.csv",
                        encoding="unicode_escape", low_memory=False).iloc[1:]
    local["SiteNo"]    = PREFIX + local["Well ID"].astype(str).str.strip()
    # Column name explicitly states ft → convert to metres
    local["WellDepth"] = pd.to_numeric(local["Well Depth (ft.)"],
                                        errors="coerce").mul(FT_TO_M)

    out = (gomas.rename(columns={"Result (Value)":"Chloride"})
                .merge(geo,   on="SiteNo", how="left")
                .merge(local[["SiteNo","WellDepth"]], on="SiteNo", how="left"))
    out = gpd.clip(gpd.GeoDataFrame(out.dropna(subset=["geometry"]),
                   crs=4326), counties)
    print(f"  [GOMAS] GA: {len(out):,} rows")
    return to_schema(out)


# ── Main ───────────────────────────────────────────────────────────────────────

# State-specific supplement functions (None = WQP+NGWMN only)
STATE_SUPPLEMENTS = {
    "TX": lambda ct, _:   extract_twdb(ct),
    "DE": lambda ct, _:   extract_dgs(),
    "SC": lambda ct, wqp: extract_scdnr(wqp),
    "FL": lambda ct, _:   extract_dbhydro(ct),
    "NC": lambda ct, _:   extract_nc_state(ct),
    "GA": lambda ct, _:   extract_gomas(ct),
}


def process_state(state: str) -> None:
    print(f"\n{'='*55}\n  {state}\n{'='*55}")
    counties = load_counties(state)
    frames   = []

    for label, fn in [("WQP",   lambda: extract_wqp(state, counties)),
                      ("NGWMN", lambda: extract_ngwmn(state, counties))]:
        try:
            frames.append(fn())
        except FileNotFoundError as e:
            print(f"  [{label}] skipped — {e}")

    wqp_df = frames[0] if frames else pd.DataFrame()

    if state in STATE_SUPPLEMENTS:
        try:
            frames.append(STATE_SUPPLEMENTS[state](counties, wqp_df))
        except FileNotFoundError as e:
            print(f"  [State supplement] skipped — {e}")

    if not frames:
        print(f"  No data found for {state}"); return

    combined = (pd.concat(frames, ignore_index=True)
                  .drop_duplicates(subset=["SiteNo","Date"], keep="first")
                  .sort_values("Date")
                  .reset_index(drop=True))

    out = DATA / f"{state}_NGlevel2026.csv"
    combined.drop(columns="geometry", errors="ignore").to_csv(out, index=False)
    wl_pct = combined["WaterLevel_m"].notna().mean() * 100
    print(f"  -> {out.name}: {len(combined):,} rows | "
          f"WaterLevel {wl_pct:.0f}% filled")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="ALL")
    args = ap.parse_args()
    states = list(STATE_NAMES) if args.state.upper() == "ALL" \
             else [args.state.upper()]
    for st in states:
        if st not in STATE_NAMES:
            print(f"Unknown state: {st}"); continue
        process_state(st)
    print("\nDone.")
