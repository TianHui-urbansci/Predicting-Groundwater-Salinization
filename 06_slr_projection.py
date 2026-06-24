"""
06_slr_projection.py
====================
Sea-level rise (SLR) scenario projection of groundwater chloride concentration
for 13 Atlantic / Gulf-Coast states using the confirmed best XGBoost models.

Best models (from 04_model_comparison.py)
------------------------------------------
  NJ  Withdrawal   R²=0.372   NC  Withdrawal   R²=0.836
  PA  Static       R²=0.592   GA  Static       R²=0.858
  DE  Static       R²=0.525   FL  Withdrawal   R²=0.794
  MD  Static       R²=0.804   AL  Paleo        R²=0.464
  VA  Static       R²=0.851   MS  Storm        R²=0.408
  SC  Withdrawal   R²=0.982   LA  Coastal + Paleo R²=0.845
                              TX  Coastal + Paleo R²=0.752

Feature sets
------------
  Static         : Surf_Ele, WellDepthM, SLR0_DIST, Aquifer, Hydro_DIST
  Withdrawal     : Static + WaterEle_m (= Surf_Ele - WaterLevel_m)
  Storm          : Static + daily_precip, 7d_Before_prec, 1m_Before_prec
  Paleo          : Static + Salt_DIST
  Coastal+Paleo  : Ele_MAX (=Surf_Ele-Daily_Max), Ele_MHHW (=Surf_Ele-Daily_Mean),
                   WellDepthM, SLR0_DIST, Aquifer, Hydro_DIST, Salt_DIST

SLR scenarios
-------------
  Atlantic + AL, MS : SLR4ft (+1.219 m via SLR_4_DIST),
                      SLR7ft (+2.134 m via SLR_7_DIST)
  LA, TX            : SLR5ft (+1.524 m via SLR_5_DIST),
                      SLR9ft (+2.743 m via SLR_9_DIST)

Feature adjustments under SLR
-------------------------------
  All states  : SLR0_DIST replaced by scenario distance column
  LA, TX only : Ele_MHHW -= rise_m  (freeboard above mean tide shrinks)
                Ele_MAX  -= rise_m  (freeboard above daily tidal max shrinks)
  All others  : tide-derived and non-distance features held at site median

Well classification
-------------------
  Case 1 — Inundated    : Surf_Ele <= rise_m
  Case 2 — Extrap risk  : SLR_DIST < 5th-percentile of training SLR0_DIST
  Case 3 — Reliable     : all other wells (used for summary statistics)

Training strategy
-----------------
  Full dataset (all rows passing NA filter) used for training.
  No train/test holdout — R² values reported are from 04_model_comparison.py.

Outputs (saved to ml_results/slr_predictions/)
-----------------------------------------------
  {STATE}_slr_predictions.csv   -- per-site predictions for all scenarios
  slr_summary_all_states.csv    -- cross-state summary statistics

Usage
-----
  python 04_slr_projection.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA = Path(__file__).parent / "ComprehensiveData"
OUT  = DATA / "ml_results" / "slr_predictions"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = 'log_Chloride'
EPA_CL = 250.0          # EPA secondary drinking-water standard (mg/L)

# SLR scenario rise magnitudes (feet → metres)
SLR4_M = 4 * 0.3048    # 1.2192 m
SLR5_M = 5 * 0.3048    # 1.5240 m
SLR7_M = 7 * 0.3048    # 2.1336 m
SLR9_M = 9 * 0.3048    # 2.7432 m

# ── Feature sets ──────────────────────────────────────────────────────────────
STATIC       = ['Surf_Ele', 'WellDepthM', 'SLR0_DIST', 'Aquifer', 'Hydro_DIST']
WITHDRAWAL   = STATIC + ['WaterEle_m']
STORM        = STATIC + ['daily_precip', '7d_Before_prec', '1m_Before_prec']
PALEO        = STATIC + ['Salt_DIST']
COASTAL_PALEO = ['Ele_MAX', 'Ele_MHHW', 'WellDepthM', 'SLR0_DIST',
                 'Aquifer', 'Hydro_DIST', 'Salt_DIST']

# ── XGBoost hyperparameters ───────────────────────────────────────────────────
_COMMON  = dict(subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=1.0, tree_method='hist', n_jobs=-1,
                random_state=42, verbosity=0)
XGB_BASE = dict(n_estimators=1000, learning_rate=0.05, max_depth=6,
                min_child_weight=5, **_COMMON)
XGB_DEEP = dict(n_estimators=1000, learning_rate=0.05, max_depth=8,
                min_child_weight=3, **_COMMON)

# ── Per-state configuration ───────────────────────────────────────────────────
# Updated R² and XGB variants from 04_model_comparison.py (20260620 dataset)
ATL_SCENS = [('SLR4ft', 'SLR_4_DIST', SLR4_M), ('SLR7ft', 'SLR_7_DIST', SLR7_M)]
GUL_SCENS = [('SLR5ft', 'SLR_5_DIST', SLR5_M), ('SLR9ft', 'SLR_9_DIST', SLR9_M)]

STATE_CFG = {
    # Mid-Atlantic
    'MD': dict(feats=STATIC,        params=XGB_DEEP, r2=0.804, region='Mid-Atlantic',   scens=ATL_SCENS),
    'VA': dict(feats=STATIC,        params=XGB_DEEP, r2=0.851, region='Mid-Atlantic',   scens=ATL_SCENS),
    # South Atlantic
    'SC': dict(feats=WITHDRAWAL,    params=XGB_DEEP, r2=0.992, region='South Atlantic', scens=ATL_SCENS),
    'NC': dict(feats=WITHDRAWAL,    params=XGB_BASE, r2=0.836, region='South Atlantic', scens=ATL_SCENS),
    'GA': dict(feats=STATIC,        params=XGB_DEEP, r2=0.857, region='South Atlantic', scens=ATL_SCENS),
    'FL': dict(feats=WITHDRAWAL,    params=XGB_DEEP, r2=0.794, region='South Atlantic', scens=ATL_SCENS),
    # Gulf Coast
    'LA': dict(feats=COASTAL_PALEO, params=XGB_BASE, r2=0.845, region='Gulf Coast',     scens=GUL_SCENS),
    'TX': dict(feats=COASTAL_PALEO, params=XGB_DEEP, r2=0.752, region='Gulf Coast',     scens=GUL_SCENS),
}

# ── Load & prepare ────────────────────────────────────────────────────────────
print("Loading data ...", flush=True)
df = pd.read_csv(DATA / "daily_chloride_final_20260620.csv",
                 dtype={"SiteNo": str}, low_memory=False)
_STR = {'SiteNo', 'Date', 'State', 'Aquifer', 'geometry'}
for c in df.columns:
    if c not in _STR:
        df[c] = pd.to_numeric(df[c], errors='coerce')
df = df[df['Chloride'] > 0].copy()
df['log_Chloride'] = np.log1p(df['Chloride'])
df['WaterEle_m']   = df['Surf_Ele'] - df['WaterLevel_m']
print(f"  {len(df):,} rows loaded", flush=True)


# ── Training helpers ──────────────────────────────────────────────────────────
def train_full(sub, feats, params):
    """
    Train on all rows (no holdout). Returns model, encoder, feature medians,
    available features, training-set size, and SLR0_DIST 5th percentile.
    """
    avail     = [f for f in feats if f in sub.columns]
    avail_num = [f for f in avail if f != 'Aquifer']
    sub2 = sub[avail + [TARGET]].dropna(subset=[TARGET] + avail_num).copy()

    med = {f: float(sub2[f].median()) for f in avail_num}

    enc = None
    if 'Aquifer' in avail:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        sub2['Aquifer'] = enc.fit_transform(sub2[['Aquifer']]).ravel().astype(float)
        med['Aquifer']  = float(sub2['Aquifer'].median())

    mdl = xgb.XGBRegressor(**params)
    mdl.fit(sub2[avail].values, sub2[TARGET].values, verbose=False)

    slr_p5 = float(sub['SLR0_DIST'].quantile(0.05))
    return mdl, enc, med, avail, len(sub2), slr_p5


def build_site_median(sub, avail, enc, scens):
    """
    Aggregate all records to one representative row per SiteNo using
    the median for numeric features and mode for Aquifer.
    Also retains SLR scenario distance columns and key diagnostic columns.
    """
    dist_cols  = [dc for _, dc, _ in scens]
    extra_cols = dist_cols + ['Surf_Ele', 'Chloride', 'Ele_MAX', 'Ele_MHHW',
                              'WaterEle_m', 'Daily_Mean', 'Daily_Max']
    num_feats  = [f for f in avail if f != 'Aquifer']
    keep       = list(dict.fromkeys(num_feats + extra_cols))
    keep       = [c for c in keep if c in sub.columns]

    site = sub.groupby('SiteNo')[keep].median()

    if enc is not None and 'Aquifer' in avail:
        def _mode(x):
            v = x.dropna()
            return v.mode().iloc[0] if len(v) > 0 and len(v.mode()) > 0 else np.nan
        aq_mode       = sub.groupby('SiteNo')['Aquifer'].agg(_mode)
        site['Aquifer'] = enc.transform(
            aq_mode.values.reshape(-1, 1)).ravel().astype(float)

    return site


def make_scenario_X(site_df, avail, med, dist_col, rise_m):
    """
    Build the feature matrix for one SLR scenario:
      - Replace SLR0_DIST with the scenario distance column.
      - For Derived_Salt (LA/TX): adjust Ele_MHHW and Ele_MAX downward by rise_m
        because sea level rise reduces the well's freeboard above tidal datum.
      - All other features held at site median values.
    """
    X = site_df[avail].copy()

    # Replace baseline distance with scenario distance
    if dist_col in site_df.columns:
        X['SLR0_DIST'] = site_df[dist_col].clip(lower=1.0)
    else:
        X['SLR0_DIST'] = (site_df['SLR0_DIST'].clip(lower=1.0) - rise_m / 0.001)

    # Tidal freeboard adjustment (LA, TX — Derived_Salt features only)
    if rise_m > 0:
        if 'Ele_MHHW' in avail:
            X['Ele_MHHW'] = (site_df['Ele_MHHW']
                             .fillna(med.get('Ele_MHHW', 0.0)) - rise_m)
        if 'Ele_MAX' in avail:
            X['Ele_MAX']  = (site_df['Ele_MAX']
                             .fillna(med.get('Ele_MAX',  0.0)) - rise_m)

    # Fill any remaining NaN with training medians
    for f in avail:
        if f in X.columns and X[f].isna().any():
            X[f] = X[f].fillna(med.get(f, 0.0))

    return X[avail]


def classify_wells(site_df, slr_dist_vals, rise_m, slr_p5):
    """
    Case 1 — Inundated  : surface elevation <= SLR rise
    Case 2 — Extrap risk: scenario distance < training 5th percentile
    Case 3 — Reliable   : all others (used for summary statistics)
    """
    surf      = (site_df['Surf_Ele'].values
                 if 'Surf_Ele' in site_df.columns
                 else np.full(len(site_df), 9999.0))
    inundated = surf <= rise_m
    extrap    = (~inundated) & (slr_dist_vals < slr_p5)
    return np.where(inundated, 1, np.where(extrap, 2, 3))


# ── Main projection loop ──────────────────────────────────────────────────────
all_summary = []

for state, cfg in STATE_CFG.items():
    feats, params = cfg['feats'], cfg['params']
    pub_r2, region, scens = cfg['r2'], cfg['region'], cfg['scens']

    sub = df[df['State'] == state].copy()
    print(f"\n{'='*70}", flush=True)
    print(f"  {state}  [{region}]  R2={pub_r2}  n_rows={len(sub):,}", flush=True)
    print(f"  Features : {feats}", flush=True)
    print(f"  Scenarios: {[s for s, _, _ in scens]}", flush=True)

    mdl, enc, med, avail, n_train, slr_p5 = train_full(sub, feats, params)
    print(f"  Trained on {n_train:,} rows  |  SLR0_DIST p5 = {slr_p5:.0f} m",
          flush=True)

    site_df = build_site_median(sub, avail, enc, scens)
    n_sites = len(site_df)
    obs_cl  = site_df['Chloride'].values

    # ── Baseline prediction ───────────────────────────────────────────────────
    X_base   = make_scenario_X(site_df, avail, med, 'SLR0_DIST', 0.0)
    cl_base  = np.expm1(np.clip(mdl.predict(X_base.values), 0, None))

    # Per-site result dataframe
    res = pd.DataFrame(index=site_df.index)
    res['State']         = state
    res['Region']        = region
    res['Obs_Cl']        = np.round(obs_cl, 2)
    res['Obs_class']     = np.where(obs_cl < EPA_CL, 'Fresh', 'Saline')
    res['Cl_baseline']   = np.round(cl_base, 2)
    res['Case_baseline'] = classify_wells(site_df,
                                          site_df['SLR0_DIST'].clip(lower=1.0).values,
                                          0.0, slr_p5)
    if 'Surf_Ele' in site_df.columns:
        res['Surf_Ele']  = site_df['Surf_Ele'].values
    res['SLR0_DIST']     = site_df['SLR0_DIST'].values

    # ── Scenario predictions ──────────────────────────────────────────────────
    for scen, dist_col, rise_m in scens:
        if dist_col not in site_df.columns:
            print(f"  SKIP {scen}: {dist_col} not available for {state}",
                  flush=True)
            continue

        X_s    = make_scenario_X(site_df, avail, med, dist_col, rise_m)
        cl_s   = np.expm1(np.clip(mdl.predict(X_s.values), 0, None))
        dist_s = site_df[dist_col].clip(lower=1.0).values
        case_s = classify_wells(site_df, dist_s, rise_m, slr_p5)

        res[f'Cl_{scen}']      = np.round(np.where(case_s == 1, np.nan, cl_s), 2)
        res[f'Case_{scen}']    = case_s
        res[f'dCl_{scen}']     = np.where(case_s == 3,
                                           np.round(cl_s - cl_base, 2), np.nan)
        res[f'pct_chg_{scen}'] = np.where(case_s == 3,
                                           np.round((cl_s - cl_base)
                                                    / np.clip(cl_base, 1, None) * 100, 1),
                                           np.nan)
        res[dist_col] = site_df[dist_col].values

        # ── Summary statistics ────────────────────────────────────────────────
        n1, n2, n3 = int((case_s==1).sum()), int((case_s==2).sum()), int((case_s==3).sum())
        mask3      = case_s == 3
        obs3       = obs_cl[mask3]
        pct_chg    = (cl_s[mask3] - cl_base[mask3]) / np.clip(cl_base[mask3], 1, None) * 100

        # Existing saline: observed median Cl >= 250 across ALL sites
        n_exist_saline = int((obs_cl >= EPA_CL).sum())

        # Fresh->Saline: Case 3 only (reliable predictions)
        n_fresh = int((obs3 < EPA_CL).sum())
        f2s     = int(((obs3 < EPA_CL) & (cl_s[mask3] >= EPA_CL)).sum())

        # Total new at risk: inundated (Case1) + fresh-to-saline (Case3)
        n_new_risk = n1 + f2s

        print(f"\n  {scen} (rise={rise_m:.3f} m):", flush=True)
        print(f"    Total wells={n_sites:,}  Existing saline={n_exist_saline:,}  "
              f"Inundated(Case1)={n1:,}  Case2_extrap={n2:,}  "
              f"Case3_reliable={n3:,}", flush=True)
        print(f"    Fresh->Saline (Case3): {f2s:,} of {n_fresh:,} fresh wells "
              f"({f2s/max(n_fresh,1)*100:.1f}%)", flush=True)
        print(f"    Total new at risk (Case1+F->S): {n_new_risk:,}  "
              f"mean Cl chg {pct_chg.mean():+.1f}%  median {np.median(pct_chg):+.1f}%",
              flush=True)

        all_summary.append(dict(
            State                    = state,
            Model_R2                 = pub_r2,
            SLR_Scenario             = scen,
            Total_Wells              = n_sites,
            Existing_Saline_Wells    = n_exist_saline,
            Inundated_Wells_at_SLR   = n1,
            Fresh_to_Saline_at_SLR   = f2s,
            Total_New_Salinization_Risk = n_new_risk,
            # additional detail columns
            Case2_Extrap_Wells       = n2,
            Case3_Reliable_Wells     = n3,
            pct_Inundated            = round(n1/n_sites*100, 1),
            pct_Fresh_to_Saline      = round(f2s/max(n_fresh,1)*100, 1),
            mean_pct_Cl_change       = round(float(pct_chg.mean()), 1),
            median_pct_Cl_change     = round(float(np.median(pct_chg)), 1),
            Feature_Set              = ', '.join(avail),
            SLR_rise_m               = round(rise_m, 3),
        ))

    out_f = OUT / f"{state}_slr_predictions.csv"
    res.to_csv(out_f)
    print(f"  Saved -> {out_f}", flush=True)

# ── Cross-state summary ───────────────────────────────────────────────────────
summary = pd.DataFrame(all_summary)
summary_path = OUT / "slr_summary_all_states.csv"
summary.to_csv(summary_path, index=False)
print(f"\n\nSummary saved -> {summary_path}", flush=True)

# ── Final table in requested format ──────────────────────────────────────────
W = [5, 8, 9, 9, 16, 18, 21, 27]
print(f"\n{'='*130}", flush=True)
print("SLR PROJECTION RESULTS", flush=True)
print(f"{'='*130}", flush=True)
hdr = (f"{'State':<5} {'R2':>7} {'Scenario':<9} "
       f"{'Total':>7} {'Exist.':>9} {'Inundated':>10} "
       f"{'Fresh->Saline':>14} {'Total New Risk':>15}")
print(hdr, flush=True)
print(f"{'':5} {'':>7} {'':9} "
      f"{'Wells':>7} {'Saline':>9} {'Wells@SLR':>10} "
      f"{'Wells@SLR':>14} {'Wells@SLR':>15}", flush=True)
print('-' * 80, flush=True)

prev_state = None
for _, r in summary.iterrows():
    if prev_state and prev_state != r['State']:
        print(flush=True)
    prev_state = r['State']
    print(
        f"{r['State']:<5} {r['Model_R2']:>7.3f} {r['SLR_Scenario']:<9} "
        f"{int(r['Total_Wells']):>7,} {int(r['Existing_Saline_Wells']):>9,} "
        f"{int(r['Inundated_Wells_at_SLR']):>10,} "
        f"{int(r['Fresh_to_Saline_at_SLR']):>14,} "
        f"{int(r['Total_New_Salinization_Risk']):>15,}",
        flush=True,
    )
