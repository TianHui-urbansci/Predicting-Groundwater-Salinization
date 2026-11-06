# Predicting Groundwater Salinization Along the U.S. Atlantic and Gulf Coasts

Code and data for:

> Tian, H., Lassiter, A., & Shen, C. (2026). Predicting salinization of groundwater along the U.S. Atlantic and Gulf coasts with machine learning. *Water Resources Research*.

The scripts reproduce the paper's results from the published dataset:

- long-term chloride trends;
- state-level XGBoost models, with a paired comparison that separates what added predictors contribute from changes in sample;
- SHAP interpretation;
- projections of which monitoring wells face new salinization risk under sea-level rise.

---

## Data

`data/` holds the analysis dataset, also archived on Zenodo (<https://zenodo.org/records/20707353>):

| File | Contents |
|---|---|
| `Coastal_Groundwater_Chloride_AtlanticGulf_1906_2023.csv` | 182,998 chloride observations at 23,192 monitoring wells, 1906–2023, 28 columns |
| `Coastal_Groundwater_Chloride_AtlanticGulf_1906_2023_DataDictionary.csv` | Definition, units, source and range of each column |

The observations were compiled from the Water Quality Portal, the National Ground-Water Monitoring Network and six state agencies (Texas, Delaware, South Carolina, Florida, North Carolina, Georgia). Site attributes come from:

- NOAA sea-level-rise inundation extents;
- the USGS principal aquifers map;
- the National Elevation Dataset and the National Hydrography Dataset;
- mapped salt domes.

Time-varying predictors come from nClimGrid-Daily precipitation and NOAA tide gauges (converted to NAVD88). The dataset contains the analysis population of Section 2.1:

- chloride above 0 mg/L, with two physically impossible records removed;
- wells in Atlantic and Gulf Coastal Plain aquifers;
- wells within 20 km of the present-day NOAA inundation extent.

Pennsylvania (85 observations) is pooled into the Mid-Atlantic scope but not modelled on its own.

---

## Scripts

Run from this folder. Each script reads `data/`, writes tables to `results/` and prints a summary.

| Script | Paper | What it does |
|---|---|---|
| `common.py` | — | Shared settings: the analysis population, predictor sets, the chronological 70/15/15 split within each well, XGBoost settings, random seeds, and each state's projection model |
| `01_trend_analysis.py` | 2.2, 3.1 | Mann-Kendall test and Sen's slope on annual mean chloride, for wells with at least 5 years of data |
| `02_algorithm_comparison.py` | 2.4, 3.3.1 | Random Forest, XGBoost, Extra Trees and a neural network on the base predictors |
| `03_state_models.py` | 3.3.1, Table 3 | Base model and four alternative specifications (groundwater elevation, precipitation, salt-dome distance, tide-referenced elevation) in each state and region |
| `04_paired_complete_case.py` | 2.5.2, 3.3.2, Table 4 | Splits each specification's difference from the base model into a predictor effect and a sample effect, and measures run-to-run noise |
| `05_groundwater_head_sensitivity.py` | 2.5.3, 3.3.3, Tables 5–6 | Six representations of groundwater head by principal aquifer, with bootstrap confidence intervals over wells |
| `06_projection_model_selection.py` | 2.6.1, 3.4.1 | Searches predictor configurations and compares land-surface with tide-referenced elevation to choose each state's projection model |
| `07_shap_analysis.py` | 2.5.4, 3.3.4, Figure 6 | SHAP attribution share and response direction of each projection model, and the matching correlations in the observed data |
| `08_slr_projection.py` | 2.6.2, 3.4.2, Table 7, Figures 7–8 | Applies the projection criteria, then counts inundated wells and fresh-to-saline transitions under SSP2-4.5 (+4/+5 ft) and SSP5-8.5 (+7/+9 ft) |

**Figures** (`figures/`, written to `figures/output/`):

| Script | Figure | Needs |
|---|---|---|
| `fig4_mk_trend_map.py` | 4 | `01_trend_analysis.py` |
| `fig5_regional_variability.py` | 5 | the dataset only |
| `fig6_shap_attribution.py` | 6 | `07_shap_analysis.py` |
| `fig7_8_projection_maps.py SSP245` / `SSP585` | 7 / 8 | `08_slr_projection.py` |

**Supplementary** (`supplementary/`):

| Script | What it tests |
|---|---|
| `S1_cross_state_transfer.py` | Whether a model trained on neighbouring states predicts a held-out state (Key Points; Section 4.2.1) |

---

## Running

```bash
pip install pandas numpy scipy scikit-learn xgboost shap matplotlib cartopy
```

```bash
python 01_trend_analysis.py
python 02_algorithm_comparison.py
python 03_state_models.py
python 04_paired_complete_case.py
python 05_groundwater_head_sensitivity.py
python 06_projection_model_selection.py
python 07_shap_analysis.py
python 08_slr_projection.py

python figures/fig4_mk_trend_map.py
python figures/fig5_regional_variability.py
python figures/fig6_shap_attribution.py
python figures/fig7_8_projection_maps.py SSP245
python figures/fig7_8_projection_maps.py SSP585
python supplementary/S1_cross_state_transfer.py
```

Scripts 01–07 are independent of each other. Script 08 needs the output of 07, and each figure needs the script listed above. Scripts 04–08 fit many models (several random seeds and bootstrap resamples), so expect minutes to tens of minutes each on a laptop.

To read the data from elsewhere, pass `--data <path>` or set `SALINITY_DATA`. Set `SALINITY_RESULTS` or `SALINITY_FIGURES` to write outputs elsewhere.

---

## Notes on reproducibility

- **Model results vary slightly.** XGBoost subsamples rows at random, so model results depend on row order and library versions as well as the seed.
  - Running these scripts on the published dataset reproduces the inundation counts in Table 7 exactly.
  - Fresh-to-saline counts come within a few wells per state.
  - Model R² values come within about 0.01 for most states.
- **Single runs are noisy.** On identical data, R²_orig can change by more than 0.1 with the random seed or predictor order (Section 3.3.2). Scripts 06–08 therefore average over 5 or 10 seeds and choose tree depth on the validation partition.
- **Projections are exploratory.** Only the predictors that sea-level rise changes are updated: distance to the inundation line, and tide-referenced elevation where used. Storm surge, pumping and land use stay at present conditions.
- **Tested with** Python 3.14, xgboost 3.4.1, scikit-learn 1.9.1, shap 0.52.0.

---

## Citation

Please cite the paper above and the dataset (<https://zenodo.org/records/20707353>) if you use this code or data.
