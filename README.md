# Coastal Groundwater Salinity Intrusion — Data & Scripts

Companion code for:

> Tian, H., Lassiter, A., Shen, C. (2026). Predicting salinization of groundwater along the U.S. Atlantic and Gulf coasts with machine learning. *Water Resources Research*, 2025WR042359.

---

## Overview

This repository provides the Python scripts used to compile the dataset, train
XGBoost salinity models, interpret results with SHAP, and project chloride
concentrations under sea-level rise (SLR) scenarios.

The study covers **13 Atlantic and Gulf Coast states** (NJ, PA, DE, MD, VA, NC,
SC, GA, FL, AL, MS, LA, TX) using **243,754 chloride observations** from
**46,289 monitoring wells** spanning 1906–2022.

---

## Introduction

Coastal groundwater salinization is an escalating threat to freshwater resources
along the U.S. Atlantic and Gulf coasts. Salt enters coastal aquifers through
multiple pathways: lateral saltwater intrusion driven by sea-level rise or
over-pumping, storm-surge and tidal inundation, reduced freshwater recharge
during droughts, and dissolution or upward migration from geological salt
bodies such as Pleistocene salt domes. Understanding *which* mechanism dominates
and *where* is critical for regional adaptation planning, yet prior studies have
largely been limited to individual aquifer systems, restricting the ability to
compare processes across the diverse hydrogeological settings of the Atlantic
and Gulf coasts.

This study addresses that gap by combining a large multi-source chloride
dataset with machine learning to (1) characterize historical salinity trends
across 13 states, (2) identify the dominant salinization drivers per state and
region, (3) interpret model behavior through SHAP analysis, and (4) project
chloride concentrations under two sea-level rise scenarios linked to SSP2-4.5
and SSP5-8.5 emission pathways.

**Study area** spans three regions within 160 km of the coastline:

| Region | States |
|--------|--------|
| Mid-Atlantic | NJ, PA, DE, MD, VA |
| South Atlantic | NC, SC, GA, FL |
| Gulf Coast | AL, MS, LA, TX |

These regions face above-average relative sea-level rise rates — up to
8–10 mm/year in Louisiana and 3–5 mm/year along the Northeast Atlantic coast —
and overlie five principal aquifer systems (Surficial, Floridan, Coastal
Lowlands, Northern Atlantic Coastal Plain, and Valley and Ridge) with
contrasting permeability and salinity vulnerability.

**Key findings:**
- **55.6%** of 5,899 monitored wells showed rising chloride trends;
  Georgia, New Jersey, and Louisiana were most affected.
- XGBoost models achieved strong performance in 8 of 13 states
  (R² = 0.75–0.98); dominant drivers varied systematically by region —
  static hydrogeology in the Mid-Atlantic, groundwater withdrawal in the
  South Atlantic, and compound tidal–geological processes in the Gulf Coast.
- Under SSP5-8.5, thousands of freshwater wells face inundation or
  fresh-to-saline transition, with Louisiana most exposed to inundation
  and Texas most susceptible to inland saltwater migration.

---

## Experiment Process

The analysis follows six sequential steps, each implemented in a dedicated
script. The figure below summarizes the workflow:

```
Raw federal &               Compiled         Final analysis-ready
state databases   ──01──►  state CSVs  ──►  dataset (244,768 rows)
                                │
                               02  Static features (SLR dist, aquifer,
                                │  salt dome dist, tide stations)
                                │  + ArcGIS: Surf_Ele, Hydro_DIST
                               03  Dynamic features (tide levels,
                                │  precipitation, Ele_MHHW, Ele_MAX)
                                │
                    ┌──────────┼──────────┐
                   04          05         06
             Model         SHAP       SLR
           comparison    analysis  projections
```

### Step 1 — Data Compilation (`01_extract_chloride_waterlevel.py`)
Chloride observations, water-level records, well depth, and aquifer
classifications were extracted from the Water Quality Portal (WQP) and the
National Groundwater Monitoring Network (NGWMN) for all 13 states, supplemented
by state agency databases for TX, DE, SC, FL, NC, and GA. Records were unified
to a common schema, spatially clipped to study-area county boundaries, and
de-duplicated by site and date. Only records with `Chloride > 0 mg/L` were
retained in the final dataset.

**Output:** one CSV per state — `{STATE}_NGlevel2026.csv`

### Step 2 — Static Features (`02_add_static_features.py`)
Time-invariant predictors were added to the dataset. NOAA Sea Level Rise
inundation rasters (2019) were used to compute planar distances from each well
to current and projected inundation boundaries (0, 4, 5, 7, 9 ft scenarios)
via a KD-tree on raster boundary pixels projected to Albers Equal Area (m).
Aquifer system was assigned from the USGS Principal Aquifers shapefile using
spatial join with a nearest-neighbor fallback for coastal wells. Distance to
Gulf Coast paleo-saltwater extent (`Salt_DIST`) was computed for TX, LA, MS,
and AL. The three nearest NOAA tide gauge stations per well were identified to
support the fallback chain in Step 3. Surface elevation (`Surf_Ele`) and
distance to surface water (`Hydro_DIST`) were extracted manually in ArcGIS Pro
using the NED 30 m DEM and NHD Best Resolution, respectively.

**Output:** updated `ComprehensiveData/All_daily_full_Lev_final_*.csv`

### Step 3 — Dynamic Features (`03_add_dynamic_features.py`)
Hourly water-level data were downloaded from the NOAA CO-OPS API for all
required tide gauge stations and aggregated to daily mean, maximum, and minimum
tide levels plus N-day rolling statistics (N = 1–7, 14 days). A three-station
fallback chain filled dates where the nearest station had missing observations.
Daily precipitation and rolling accumulations (1–7 days, 14 days, 1, 3, 6, and
12 months) were extracted from NOAA nClimGrid-Daily NetCDF files by matching
each well to the nearest valid land grid cell via a KD-tree. Two derived tidal
elevation metrics were computed: `Ele_MHHW` (surface elevation minus daily mean
tide, m) and `Ele_MAX` (surface elevation minus daily tidal maximum, m).

**Output:** updated `ComprehensiveData/All_daily_full_Lev_final_*.csv`

### Step 4 — Model Comparison (`04_model_comparison.py`)
Six feature sets encoding distinct salinization mechanisms were evaluated for
each of the 13 states and 3 regional aggregates. Four algorithms were tested
(Random Forest, XGBoost, Extra Trees, ANN); XGBoost was selected based on
consistently superior back-transformed R² and computational efficiency. Each
model used a per-site chronological 70/15/15% train/validation/test split to
respect temporal structure. Two XGBoost depth variants (depth-6 and depth-8)
were evaluated per scope; the variant with higher test R² was reported. The
target variable was `log1p(Chloride)`; R² and RMSE were back-transformed to the
original chloride scale for reporting.

**Output:** `ComprehensiveData/ml_results/model_comparison.csv`,
`model_comparison_best.csv`

### Step 5 — SHAP Analysis (`05_shap_analysis.py`)
The best-performing model per state was retrained on the combined train +
validation set and evaluated on the held-out test set. SHAP TreeExplainer was
used to decompose each test-set prediction into additive feature contributions.
Feature importance was ranked by mean absolute SHAP value; effect direction was
assessed by Pearson correlation between feature values and SHAP values. Three
plot types were produced per state: feature importance bar chart, beeswarm
summary, and dependence plots for the top-4 features.

**Output:** `ComprehensiveData/ml_results/shap/{STATE}/`

### Step 6 — SLR Projections (`06_slr_projection.py`)
Chloride concentrations were projected under SSP2-4.5 and SSP5-8.5 for the
8 states with model R² ≥ 0.75. Projected regional SLR values by 2100 (relative
to 2000 baseline) were mapped to the nearest available NOAA integer-foot
inundation contour: +4 ft / +7 ft for Atlantic and eastern Gulf states;
+5 ft / +9 ft for Louisiana and Texas. Each scenario updated only the
sea-level-related predictors (`SLR0_DIST` → scenario distance; `Ele_MHHW` and
`Ele_MAX` reduced by rise magnitude for LA/TX Coastal+Paleo models); all other
features were held at site medians. Wells were classified as Inundated (Case 1),
Extrapolation risk (Case 2), or Reliable prediction (Case 3). Fresh-to-saline
transitions were flagged where observed chloride was below the EPA secondary
drinking-water standard (250 mg/L) and the scenario-predicted chloride exceeded
it.

**Output:** `ComprehensiveData/ml_results/slr_predictions/`

---

## Repository Structure

```
├── 01_extract_chloride_waterlevel.py   # Step 1 — compile chloride & water-level records
├── 02_add_static_features.py           # Step 2 — add static/geological features
├── 03_add_dynamic_features.py          # Step 3 — add tide levels & precipitation features
├── 04_model_comparison.py              # Step 4 — XGBoost feature-set comparison
├── 05_shap_analysis.py                 # Step 5 — SHAP feature importance & effects
├── 06_slr_projection.py                # Step 6 — SLR scenario chloride projections
└── README.md
```

---

## Script Descriptions

### `01_extract_chloride_waterlevel.py`
Extracts chloride, water level, well depth, and aquifer information from
federal and state groundwater databases for all 13 states. Outputs one CSV
per state.

**Federal sources (all states):**
- Water Quality Portal (WQP) — waterqualitydata.us
- National Groundwater Monitoring Network (NGWMN) — cida.usgs.gov/ngwmn

**State supplements:**
| State | Source |
|-------|--------|
| TX | Texas Water Development Board (TWDB) Groundwater Database |
| DE | Delaware Geological Survey (DGS) Report of Investigations 85 |
| SC | SCDNR Saltwater Intrusion & Groundwater Level Monitoring Networks |
| FL | SFWMD DBHYDRO Wells and Boreholes |
| NC | NCDEQ–DWR Groundwater Levels & Quality |
| GA | Georgia EPD Environmental Monitoring and Assessment System (GOMAS) |

**Usage:**
```bash
python 01_extract_chloride_waterlevel.py              # all 13 states
python 01_extract_chloride_waterlevel.py --state TX   # single state
```

**Required local data** (place in the script's directory):
- `WQP/station_{STATE}.csv` and `WQP/resultphyschem_{STATE}.csv` — downloaded from waterqualitydata.us
- `NGWMN/SITE_INFO.csv`, `QUALITY.csv`, `WATERLEVEL.csv`
- State-specific folders: `TWDB2026/`, `DGS/`, `SCDNR/`, `DBHYDRO/`, `NCDWR/`, `NCDEQ/`, `GOMAS/`
- `study_area_counties/` — county boundary shapefile for spatial clipping

---

### `02_add_static_features.py`
Adds time-invariant predictors to the compiled dataset. Operates on
`ComprehensiveData/All_daily_full_Lev_final_*.csv`.

**Features added (Python):**
| Column | Description | Source |
|--------|-------------|--------|
| `SLR0_DIST` | Distance to current sea-level inundation boundary (m) | NOAA SLR Rasters (2019) |
| `SLR4_DIST`, `SLR7_DIST` | Distance under 4 ft / 7 ft SLR (Atlantic states) | NOAA SLR Rasters |
| `SLR5_DIST`, `SLR9_DIST` | Distance under 5 ft / 9 ft SLR (Gulf states) | NOAA SLR Rasters |
| `Aquifer` / `AQ_CODE` | Principal aquifer system | USGS Principal Aquifers (2003) |
| `Salt_DIST` | Distance to nearest paleo-saltwater extent (m); Gulf Coast only | Andrews (2023); Beckman & Williamson (1990); Schuba et al. (2025) |
| `TideSta_1/2/3` | Nearest 3 NOAA tide gauge station IDs (fallback chain) | NOAA Tides & Currents (2024) |
| `TideDist_1/2/3` | Distance to those stations (m) | — |

**Features added (ArcGIS — manual workflow described at end of script):**
| Column | Description | Source |
|--------|-------------|--------|
| `Surf_Ele` | Well surface elevation (m, NAVD88) | NED 30 m DEM (USDA, 1999) |
| `Hydro_DIST` | Distance to nearest surface water body (m) | NHD Best Resolution (USGS, 2023) |

**USER CONFIG required** — set these paths at the top of the script before running:
```python
SLR_DIR = Path(r"path/to/NOAA_SLR_Inundation")   # coast.noaa.gov/slrdata
AQ_SHP  = Path(r"path/to/us_aquifers.shp")        # USGS Principal Aquifers shapefile
```
The NOAA tide station list (`NOAA_Tide_Stations/stations.csv`) should be placed
in the script's directory (downloadable from tidesandcurrents.noaa.gov).

---

### `03_add_dynamic_features.py`
Adds time-varying predictors to the dataset.

**Features added:**
| Column(s) | Description | Source |
|-----------|-------------|--------|
| `Daily_Mean`, `Daily_Max`, `Daily_Min` | Daily tide water level at nearest NOAA station (m, MHHW datum) | NOAA CO-OPS API |
| `{N}day_mean/max/min` (N = 1–7, 14) | Rolling N-day tide statistics | — |
| `Station_ID` | Which station provided each row's tide data | — |
| `daily_precip` | Daily precipitation (mm) | NOAA nClimGrid-Daily (Durre et al., 2022) |
| `{1–7,14}d_Before_prec` | Rolling precipitation sums (mm) | — |
| `{1m,3m,6m,1Y}_Before_prec` | Monthly–annual precipitation accumulations (mm) | — |
| `Ele_MHHW` | Surface elevation minus daily mean tide (m) | Derived |
| `Ele_MAX` | Surface elevation minus daily maximum tide (m) | Derived |

Tide data are downloaded automatically from the NOAA CO-OPS API (via
`noaa_coops`); stations already downloaded are skipped. A three-station
fallback chain (`TideSta_1/2/3` from Step 2) fills gaps where the nearest
station has missing data on a given date.

**USER CONFIG required:**
```python
PREC_NC_DIR = Path(r"path/to/nClimGrid_Daily")   # NOAA nClimGrid-Daily NetCDF files
                                                   # ncei.noaa.gov/products/land-based-station/nclimgrid-daily
```

---

### `04_model_comparison.py`
Trains XGBoost models for each of six feature sets across all 13 states and
three regional aggregates, and identifies the best-performing feature set per
scope.

**Feature sets:**
| Name | Features |
|------|----------|
| Static | `Surf_Ele`, `WellDepthM`, `SLR0_DIST`, `Aquifer`, `Hydro_DIST` |
| Withdrawal | Static + `WaterEle_m` (water-table elevation) |
| Storm | Static + `daily_precip`, `7d_Before_prec`, `1m_Before_prec` |
| Coastal | `Ele_MAX`, `Ele_MHHW`, `WellDepthM`, `SLR0_DIST`, `Aquifer`, `Hydro_DIST` |
| Paleo | Static + `Salt_DIST` (Gulf states only) |
| Coastal+Paleo | Coastal + `Salt_DIST` (Gulf states only) |

**Methodology:**
- Target: `log1p(Chloride)`, back-transformed for reported R² and RMSE
- 70 / 15 / 15 % chronological split per site (train / validation / test)
- Two XGBoost variants evaluated (depth-6 and depth-8); best reported
- NA dropped on target + all numeric features before splitting (no imputation)

**Outputs** → `ComprehensiveData/ml_results/`:
- `model_comparison.csv` — full results per scope × feature set
- `model_comparison_best.csv` — best feature set per scope

---

### `05_shap_analysis.py`
Retrains the best model per state on train+validation combined and computes
SHAP values on the held-out test set.

**Outputs** → `ComprehensiveData/ml_results/shap/{STATE}/`:
- `{STATE}_importance.png` — mean |SHAP| bar chart
- `{STATE}_beeswarm.png` — feature effect direction (beeswarm plot)
- `{STATE}_dependence_{feat}.png` — dependence plots for top-4 features
- `shap_values_{STATE}.csv` — raw SHAP values on test set
- `shap_importance_{STATE}.csv` — mean |SHAP| summary

**Usage:**
```bash
python 05_shap_analysis.py                        # all 13 states
python 05_shap_analysis.py --states NJ FL GA      # specific states
```

---

### `06_slr_projection.py`
Projects chloride concentrations under two SLR scenarios (SSP2-4.5 and
SSP5-8.5) for the 8 states with the best-performing models (R² ≥ 0.75).

**SLR scenarios** (NOAA nearest integer-foot conversion):

| Scenario | Atlantic + Eastern Gulf | Western Gulf (LA, TX) |
|----------|------------------------|----------------------|
| SSP2-4.5 | 4 ft (1.22 m) | 5 ft (1.52 m) |
| SSP5-8.5 | 7 ft (2.13 m) | 9 ft (2.74 m) |

**Feature adjustments under SLR:**
- All states: `SLR0_DIST` → scenario distance column
- LA, TX (Coastal+Paleo): `Ele_MHHW` and `Ele_MAX` reduced by rise magnitude

**Well classification:**
- Case 1 — Inundated: surface elevation ≤ SLR rise
- Case 2 — Extrapolation risk: scenario distance < 5th percentile of training range
- Case 3 — Reliable prediction: all others (used for summary statistics)

**Outputs** → `ComprehensiveData/ml_results/slr_predictions/`:
- `{STATE}_slr_predictions.csv` — per-site predictions for all scenarios
- `slr_summary_all_states.csv` — cross-state summary statistics

---

## Finalized Dataset

The compiled, cleaned dataset used for all modeling is:

```
ComprehensiveData/All_daily_full_Lev_final_20260602.csv
```

- **243,754 chloride observations** from **46,289 monitoring wells**
- **13 states**, date range **1906–2022**
- **65 columns** covering raw measurements, static features, tide levels,
  precipitation, and derived tidal elevation metrics
- Only records with `Chloride > 0` are retained (physically impossible negative
  values and below-detection entries removed)

This file is the input for scripts 04–06. It is **not** included in this
repository due to size; contact the authors or reconstruct it by running
scripts 01–03 in sequence.

---

## Dependencies

```bash
pip install pandas numpy scipy scikit-learn xgboost shap \
            geopandas rasterio pyproj noaa_coops xarray h5netcdf
```

ArcGIS Pro is required for the manual `Surf_Ele` (NED DEM) and `Hydro_DIST`
(NHD) extraction steps documented at the end of `02_add_static_features.py`.

---

## Running Order

```
01 → 02 → [ArcGIS: Surf_Ele, Hydro_DIST] → 03 → 04 → 05 → 06
```

Scripts 04–06 can be run independently on the finalized dataset without
re-running the data compilation pipeline.

---

## Citation

If you use these scripts or the dataset, please cite:

Hui Tian*; Allison Lassiter; Chaopeng Shen. (2026). “Predicting salinization of groundwater along the U.S. Atlantic and Gulf coasts with machine learning, Water Resources Research [under review]

---

## Contact

Hui Tian — University of Pennsylvania

huitian@upenn.edu
