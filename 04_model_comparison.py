"""
04_model_comparison.py
======================
XGBoost feature-set comparison for coastal groundwater chloride prediction.
Covers all 13 Atlantic/Gulf-Coast US states and 3 regional aggregates.

Feature sets tested
-------------------
  Static          : Surf_Ele, WellDepthM, SLR0_DIST, Aquifer, Hydro_DIST
  Withdrawal      : Static + WaterEle_m  (= Surf_Ele - WaterLevel_m)
  Storm           : Static + daily_precip, 7d_Before_prec, 1m_Before_prec
  Coastal         : Ele_MAX, Ele_MHHW, WellDepthM, SLR0_DIST, Aquifer, Hydro_DIST
  Paleo           : Static + Salt_DIST                    [Gulf states / region only]
  Coastal+Paleo   : Coastal + Salt_DIST                   [Gulf states / region only]

Feature definitions
-------------------
  Surf_Ele       : Well surface elevation above mean sea level (m)
  WellDepthM     : Well screen depth below surface (m)
  SLR0_DIST      : Distance to nearest sea-level-rise reference shoreline (m)
  Hydro_DIST     : Distance to nearest surface water body (m)
  Aquifer        : Aquifer system name (ordinal-encoded; unknown -> -1)
  WaterEle_m     : Water-table elevation = Surf_Ele - WaterLevel_m (m asl)
  daily_precip   : Precipitation on sampling date (mm)
  7d_Before_prec : Cumulative precipitation in the 7 days before sampling (mm)
  1m_Before_prec : Cumulative precipitation in the 30 days before sampling (mm)
  Ele_MAX        : Land surface elevation minus the observed daily tidal maximum (m)
  Ele_MHHW       : Land surface elevation minus the observed daily mean tidal water level (m),
                   where water level is reported in the MHHW datum frame (not relative to
                   the fixed MHHW benchmark itself — varies daily with the tide)
  Salt_DIST      : Distance to nearest Pleistocene salt-water intrusion extent (m)

Methodology
-----------
  • NA dropped on the target AND all numeric model features BEFORE splitting
    (filter-then-split; no imputation)
  • 70 / 15 / 15 % chronological split per SiteNo (train / validation / test)
  • XGBoost base (max_depth=6) and deep (max_depth=8) both evaluated;
    the variant with higher test R² is reported
  • Aquifer encoded with OrdinalEncoder; unknown categories mapped to -1
  • Target  : log1p(Chloride)
  • Reported R² and RMSE are back-transformed to the original Chloride scale

Outputs
-------
  ml_results/model_comparison.csv      -- full per-scope × per-featset results
  ml_results/model_comparison_best.csv -- best feature set per scope

Usage
-----
  python 04_model_comparison.py
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA = Path(__file__).parent / "ComprehensiveData"
OUT  = DATA / "ml_results"
OUT.mkdir(exist_ok=True)

TARGET    = "log_Chloride"
SEED      = 42
MIN_TRAIN = 50
MIN_TEST  = 20

# ── Feature-set definitions ────────────────────────────────────────────────────
STATIC  = ["Surf_Ele", "WellDepthM", "SLR0_DIST", "Aquifer", "Hydro_DIST"]
FS = {
    "Static":        STATIC,
    "Withdrawal":    STATIC + ["WaterEle_m"],
    "Storm":         STATIC + ["daily_precip", "7d_Before_prec", "1m_Before_prec"],
    "Coastal":       ["Ele_MAX", "Ele_MHHW", "WellDepthM", "SLR0_DIST",
                      "Aquifer", "Hydro_DIST"],
    "Paleo":         STATIC + ["Salt_DIST"],           # Gulf only
    "Coastal+Paleo": ["Ele_MAX", "Ele_MHHW", "WellDepthM", "SLR0_DIST",
                      "Aquifer", "Hydro_DIST", "Salt_DIST"],  # Gulf only
}
FS_ORDER = list(FS.keys())

# Paleo and Coastal+Paleo require Salt_DIST which is only meaningful in Gulf
GULF_SCOPES = {"AL", "MS", "LA", "TX", "Gulf"}

# ── Scope definitions ─────────────────────────────────────────────────────────
ALL_STATES = ["NJ", "PA", "DE", "MD", "VA",
              "SC", "NC", "GA", "FL",
              "AL", "MS", "LA", "TX"]
REGIONS = {
    "Mid_Atlantic":   ["NJ", "PA", "DE", "MD", "VA"],
    "South_Atlantic": ["SC", "NC", "GA", "FL"],
    "Gulf":           ["AL", "MS", "LA", "TX"],
}
SCOPE_ORDER = [
    "NJ", "PA", "DE", "MD", "VA", "Mid_Atlantic",
    "SC", "NC", "GA", "FL", "South_Atlantic",
    "AL", "MS", "LA", "TX", "Gulf",
]

# ── XGBoost hyperparameters ───────────────────────────────────────────────────
_COMMON = dict(
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    tree_method="hist", early_stopping_rounds=30,
    n_jobs=-1, random_state=SEED, verbosity=0,
)
XGB_VARIANTS = {
    "base": dict(n_estimators=1000, learning_rate=0.05, max_depth=6,
                 min_child_weight=5, **_COMMON),
    "deep": dict(n_estimators=1000, learning_rate=0.05, max_depth=8,
                 min_child_weight=3, **_COMMON),
}

# ── Load and prepare data ─────────────────────────────────────────────────────
print("Loading data ...", flush=True)
df = pd.read_csv(
    DATA / "daily_chloride_final_20260620.csv",
    dtype={"SiteNo": str}, low_memory=False,
)
_STR_COLS = {"SiteNo", "Date", "State", "Aquifer", "geometry"}
for col in df.columns:
    if col not in _STR_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[df["Chloride"] > 0].copy()
df["log_Chloride"] = np.log1p(df["Chloride"])
df["WaterEle_m"]   = df["Surf_Ele"] - df["WaterLevel_m"]   # water-table elevation (m asl)
df["Date_parsed"]  = pd.to_datetime(df["Date"], errors="coerce")
df["Date_ord"]     = df["Date_parsed"].astype(np.int64)
print(f"  {len(df):,} rows loaded ({df['State'].nunique()} states)", flush=True)


# ── Chronological 70 / 15 / 15 split per SiteNo ───────────────────────────────
def split_703(scope_df: pd.DataFrame):
    """
    Per-site chronological split: 70 % train, 15 % validation, 15 % test.
    Sites with only 1 row go to train; sites with 2 rows go to train + test.
    """
    tr, va, te = [], [], []
    for _, grp in scope_df.groupby("SiteNo", sort=False):
        g = grp.sort_values("Date_ord")
        n = len(g)
        if n == 1:
            tr.append(g)
        elif n == 2:
            tr.append(g.iloc[:1])
            te.append(g.iloc[1:])
        else:
            n_tr = max(1, int(n * 0.70))
            n_va = max(1, int(n * 0.15))
            tr.append(g.iloc[:n_tr])
            va.append(g.iloc[n_tr : n_tr + n_va])
            te.append(g.iloc[n_tr + n_va :])
    cat = lambda lst: pd.concat(lst).drop(columns="Date_ord") if lst else pd.DataFrame()
    return cat(tr), cat(va), cat(te)


# ── Train and evaluate one XGBoost variant ────────────────────────────────────
def run_xgb(tr, va, te, features, params):
    """
    Train on tr, use va for early stopping, evaluate on te.
    Returns a dict with R², RMSE, and sample counts; or None if too few samples.
    """
    avail = [f for f in features if f in tr.columns]

    def _xy(d):
        cols = [c for c in avail + [TARGET] if c in d.columns]
        return d[cols].copy() if len(d) else pd.DataFrame()

    tr2, va2, te2 = _xy(tr), _xy(va), _xy(te)
    if len(tr2) < MIN_TRAIN or len(te2) < MIN_TEST:
        return None

    # Encode Aquifer (ordinal; unknown categories -> -1)
    if "Aquifer" in avail:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        tr2["Aquifer"] = enc.fit_transform(tr2[["Aquifer"]]).ravel().astype(float)
        if len(va2):
            va2["Aquifer"] = enc.transform(va2[["Aquifer"]]).ravel().astype(float)
        if len(te2):
            te2["Aquifer"] = enc.transform(te2[["Aquifer"]]).ravel().astype(float)

    X_tr = tr2[avail].values
    X_va = va2[avail].values if len(va2) else None
    X_te = te2[avail].values
    y_tr, y_te = tr2[TARGET].values, te2[TARGET].values
    y_va = va2[TARGET].values if len(va2) else None

    mdl = xgb.XGBRegressor(**params)
    eval_set = [(X_va, y_va)] if X_va is not None else [(X_tr, y_tr)]
    mdl.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)

    # Back-transform predictions to original Chloride scale
    y_true  = np.expm1(y_te)
    y_pred  = np.expm1(np.clip(mdl.predict(X_te), 0, None))
    return dict(
        R2      = round(r2_score(y_true, y_pred), 4),
        RMSE    = round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        n_train = len(tr2),
        n_val   = len(va2) if len(va2) else 0,
        n_test  = len(te2),
    )


def valid_and_run(sub_raw: pd.DataFrame, features: list):
    """
    1. Drop NA on target + all numeric model features  →  n_valid rows retained
    2. Chronological 70/15/15 split
    3. Try XGB base (depth-6) and deep (depth-8); keep variant with higher test R²
    Returns (n_valid, result_dict | None)
    """
    num_feats = [f for f in features if f in sub_raw.columns and f != "Aquifer"]
    sub = sub_raw.dropna(subset=[TARGET] + num_feats).copy()
    n_valid = len(sub)
    if n_valid < MIN_TRAIN + MIN_TEST:
        return n_valid, None

    tr, va, te = split_703(sub)
    best_res, best_variant = None, None
    for vname, params in XGB_VARIANTS.items():
        res = run_xgb(tr, va, te, features, params)
        if res is not None and (best_res is None or res["R2"] > best_res["R2"]):
            best_res, best_variant = res, vname

    if best_res is not None:
        best_res["model"] = best_variant
    return n_valid, best_res


# ── Main comparison loop ──────────────────────────────────────────────────────
scopes = [(st, "State",  df[df["State"] == st].copy())         for st in ALL_STATES]
for rname, sts in REGIONS.items():
    scopes.append((rname, "Region", df[df["State"].isin(sts)].copy()))
scopes.sort(key=lambda x: SCOPE_ORDER.index(x[0]))

all_rows = []
t0 = time.time()
SEP_SCOPES = {"Mid_Atlantic", "South_Atlantic", "Gulf"}

# ── Console header ────────────────────────────────────────────────────────────
COL_W = 16
HDR_COLS = ["Static", "Withdrawal", "Storm", "Coastal", "Paleo", "Coastal+Paleo"]
print(f"\n{'Scope':<16} {'Type':<7} " +
      "  ".join(f"{h:>16}" for h in HDR_COLS) + "  Best", flush=True)
print(" " * 23 + "  ".join(f"{'n_valid   R²':>16}" for _ in HDR_COLS), flush=True)
print("-" * 160, flush=True)

for scope_name, scope_type, sub_raw in scopes:
    if scope_name in SEP_SCOPES:
        print(flush=True)

    # Gulf-only feature sets are skipped for non-Gulf scopes
    fs_to_run = {
        k: v for k, v in FS.items()
        if k not in ("Paleo", "Coastal+Paleo") or scope_name in GULF_SCOPES
    }

    results, n_valids = {}, {}
    for fs_name, features in fs_to_run.items():
        nv, res           = valid_and_run(sub_raw, features)
        n_valids[fs_name] = nv
        results[fs_name]  = res

    # Identify best feature set by test R²
    r2_vals = {k: results[k]["R2"] for k in fs_to_run if results.get(k)}
    best_fs = max(r2_vals, key=r2_vals.get) if r2_vals else "N/A"
    best_r2 = r2_vals[best_fs] if r2_vals else np.nan

    def _r2(k):   return results[k]["R2"]     if results.get(k) else np.nan
    def _nv(k):   return n_valids.get(k, 0)
    def _rmse(k): return results[k]["RMSE"]   if results.get(k) else np.nan
    def _nte(k):  return results[k]["n_test"] if results.get(k) else 0
    def _mdl(k):  return results[k]["model"]  if results.get(k) else ""

    def _fmt(k):
        if k not in fs_to_run or results.get(k) is None:
            return f"{'N/A':>16}"
        marker = "*" if k == best_fs else " "
        return f"{_nv(k):>7,} {_r2(k):>6.3f}{marker}"

    print(
        f"{scope_name:<16} {scope_type:<7} " +
        "  ".join(_fmt(k) for k in HDR_COLS) +
        f"  {best_fs} ({best_r2:.3f})",
        flush=True,
    )

    # Accumulate row for CSV output
    row = dict(Scope=scope_name, ScopeType=scope_type,
               Best_FeatSet=best_fs, Best_R2=best_r2)
    for fs_name in FS_ORDER:
        row[f"{fs_name}_n"]     = _nv(fs_name)
        row[f"{fs_name}_n_test"]= _nte(fs_name)
        row[f"{fs_name}_R2"]    = _r2(fs_name)
        row[f"{fs_name}_RMSE"]  = _rmse(fs_name)
        row[f"{fs_name}_model"] = _mdl(fs_name)
    all_rows.append(row)

# ── Save full results ─────────────────────────────────────────────────────────
res_df = pd.DataFrame(all_rows)
res_df.to_csv(OUT / "model_comparison.csv", index=False)
print(f"\nDone in {(time.time()-t0)/60:.1f} min", flush=True)
print(f"Saved -> {OUT / 'model_comparison.csv'}", flush=True)

# ── Best-model summary table ──────────────────────────────────────────────────
print("\n" + "=" * 80, flush=True)
print("BEST MODEL PER SCOPE", flush=True)
print("=" * 80, flush=True)
print(f"\n{'Scope':<16} {'n_valid':>8} {'n_test':>7}  "
      f"{'Best_FeatSet':<16} {'R²':>7} {'RMSE':>8}  XGB_variant", flush=True)
print("-" * 75, flush=True)

best_rows = []
for _, r in res_df.iterrows():
    if r["Scope"] in SEP_SCOPES:
        print(flush=True)
    fs   = r["Best_FeatSet"]
    nval = r[f"{fs}_n"]
    nte  = r[f"{fs}_n_test"]
    rmse = r[f"{fs}_RMSE"]
    mv   = r[f"{fs}_model"]
    print(
        f"{r['Scope']:<16} {int(nval):>8,} {int(nte):>7,}  "
        f"{fs:<16} {r['Best_R2']:>7.3f} {rmse:>8.1f}  {mv}",
        flush=True,
    )
    best_rows.append(dict(
        Scope        = r["Scope"],
        ScopeType    = r["ScopeType"],
        Best_FeatSet = fs,
        Best_R2      = r["Best_R2"],
        Best_RMSE    = rmse,
        n_valid      = nval,
        n_test       = nte,
        XGB_variant  = mv,
        Features     = ", ".join(FS.get(fs, [])),
    ))

pd.DataFrame(best_rows).to_csv(OUT / "model_comparison_best.csv", index=False)
print(f"\nSaved -> {OUT / 'model_comparison_best.csv'}", flush=True)
