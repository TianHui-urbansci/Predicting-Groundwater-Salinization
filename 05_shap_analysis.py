"""
05_shap_analysis.py
===================
SHAP (SHapley Additive exPlanations) analysis for the best XGBoost
chloride-prediction model per US state.

Three plot types are produced for each state
--------------------------------------------
  1. Feature Importance   – mean |SHAP| bar chart (ranked, horizontal)
  2. Feature Effect       – beeswarm summary plot showing the direction and
                            magnitude of each feature's effect on log-Chloride
                            predictions (positive SHAP = higher Cl predicted)
  3. Dependence Plots     – one scatter plot per top-4 feature showing how
                            SHAP value varies with feature value, coloured by
                            the feature that most interacts with it

Workflow
--------
  1. Load the cleaned dataset (daily_chloride_final_20260620.csv)
  2. For each state, derive the best feature set from model_comparison_best.csv
     (falls back to "Static" if that file is absent)
  3. Drop NA on target + numeric features, apply 70/15/15 chronological split
  4. Retrain the best XGBoost variant on train+validation combined, evaluate on test
  5. Compute SHAP values on the TEST set using shap.TreeExplainer
  6. Save plots to  ml_results/shap/<STATE>/

Outputs (one sub-folder per state under ml_results/shap/)
----------------------------------------------------------
  <STATE>_importance.png   -- mean |SHAP| bar chart
  <STATE>_beeswarm.png     -- beeswarm summary (effect direction)
  <STATE>_dependence_<feat>.png  -- dependence plot for each top-4 feature
  shap_values_<STATE>.csv  -- raw SHAP values on test set (for reproducibility)

Usage
-----
  python 05_shap_analysis.py
  python 05_shap_analysis.py --states NJ FL GA   # run specific states only
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA = Path(__file__).parent / "ComprehensiveData"
OUT  = DATA / "ml_results" / "shap"
OUT.mkdir(parents=True, exist_ok=True)

TARGET    = "log_Chloride"
SEED      = 42
MIN_TRAIN = 50
MIN_TEST  = 20

# ── Feature-set definitions (must match 04_model_comparison.py) ───────────────
STATIC = ["Surf_Ele", "WellDepthM", "SLR0_DIST", "Aquifer", "Hydro_DIST"]
FS = {
    "Static":        STATIC,
    "Withdrawal":    STATIC + ["WaterEle_m"],
    "Storm":         STATIC + ["daily_precip", "7d_Before_prec", "1m_Before_prec"],
    "Coastal":       ["Ele_MAX", "Ele_MHHW", "WellDepthM", "SLR0_DIST",
                      "Aquifer", "Hydro_DIST"],
    "Paleo":         STATIC + ["Salt_DIST"],
    "Coastal+Paleo": ["Ele_MAX", "Ele_MHHW", "WellDepthM", "SLR0_DIST",
                      "Aquifer", "Hydro_DIST", "Salt_DIST"],
}
GULF_SCOPES = {"AL", "MS", "LA", "TX"}

# Readable feature labels for plots
FEAT_LABELS = {
    "Surf_Ele":       "Surface Elevation (m)",
    "WellDepthM":     "Well Depth (m)",
    "SLR0_DIST":      "Distance to Shoreline (m)",
    "Aquifer":        "Aquifer System",
    "Hydro_DIST":     "Distance to Surface Water (m)",
    "WaterEle_m":     "Water-Table Elevation (m asl)",
    "daily_precip":   "Daily Precipitation (mm)",
    "7d_Before_prec": "7-Day Prior Precipitation (mm)",
    "1m_Before_prec": "30-Day Prior Precipitation (mm)",
    "Ele_MAX":        "Elevation above Daily Tidal Max (m)",
    "Ele_MHHW":       "Elevation above Daily Mean Tide Level (m)",
    "Salt_DIST":      "Distance to Paleo-Saltwater (m)",
}

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

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi":   150,
})
STATE_COLORS = {
    "NJ": "#1f77b4", "PA": "#aec7e8", "DE": "#ffbb78", "MD": "#ff7f0e",
    "VA": "#2ca02c", "SC": "#98df8a", "NC": "#d62728", "GA": "#ff9896",
    "FL": "#9467bd", "AL": "#c5b0d5", "MS": "#8c564b", "LA": "#c49c94",
    "TX": "#e377c2",
}

# ── Load data ─────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print("Loading dataset ...", flush=True)
    df = pd.read_csv(
        DATA / "daily_chloride_final_20260620.csv",
        dtype={"SiteNo": str}, low_memory=False,
    )
    _STR = {"SiteNo", "Date", "State", "Aquifer", "geometry"}
    for col in df.columns:
        if col not in _STR:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["Chloride"] > 0].copy()
    df["log_Chloride"] = np.log1p(df["Chloride"])
    df["WaterEle_m"]   = df["Surf_Ele"] - df["WaterLevel_m"]
    df["Date_parsed"]  = pd.to_datetime(df["Date"], errors="coerce")
    df["Date_ord"]     = df["Date_parsed"].astype(np.int64)
    print(f"  {len(df):,} rows loaded", flush=True)
    return df


def load_best_featsets() -> dict:
    """
    Read model_comparison_best.csv to get best feature set per state.
    Falls back to 'Static' if file not found.
    """
    best_csv = DATA / "ml_results" / "model_comparison_best.csv"
    if not best_csv.exists():
        print(f"  [WARN] {best_csv.name} not found – defaulting to 'Static' for all states",
              flush=True)
        all_scopes = ["NJ","PA","DE","MD","VA","SC","NC","GA","FL","AL","MS","LA","TX"]
        return {s: "Static" for s in all_scopes}
    bdf = pd.read_csv(best_csv)
    return dict(zip(bdf["Scope"], bdf["Best_FeatSet"]))


# ── Split helper ──────────────────────────────────────────────────────────────
def split_703(scope_df: pd.DataFrame):
    tr, va, te = [], [], []
    for _, grp in scope_df.groupby("SiteNo", sort=False):
        g = grp.sort_values("Date_ord")
        n = len(g)
        if n == 1:
            tr.append(g)
        elif n == 2:
            tr.append(g.iloc[:1]); te.append(g.iloc[1:])
        else:
            n_tr = max(1, int(n * 0.70))
            n_va = max(1, int(n * 0.15))
            tr.append(g.iloc[:n_tr])
            va.append(g.iloc[n_tr : n_tr + n_va])
            te.append(g.iloc[n_tr + n_va :])
    cat = lambda lst: pd.concat(lst).drop(columns="Date_ord") if lst else pd.DataFrame()
    return cat(tr), cat(va), cat(te)


# ── Train final model (train+val combined), evaluate on test ──────────────────
def train_and_shap(state: str, df: pd.DataFrame, features: list, xgb_variant: str):
    """
    Prepare data, train best XGBoost variant, compute SHAP values on test set.
    Returns (model, encoder, X_test_df, shap_values, test_r2, feature_names).
    """
    sub_raw = df[df["State"] == state].copy()
    num_feats = [f for f in features if f in sub_raw.columns and f != "Aquifer"]
    sub = sub_raw.dropna(subset=[TARGET] + num_feats).copy()

    if len(sub) < MIN_TRAIN + MIN_TEST:
        return None

    tr, va, te = split_703(sub)
    if len(te) < MIN_TEST:
        return None

    # Ordinal-encode Aquifer on train; apply to val / test
    enc = None
    avail = [f for f in features if f in sub.columns]
    tr2 = tr[avail + [TARGET]].copy()
    va2 = va[avail + [TARGET]].copy() if len(va) else pd.DataFrame()
    te2 = te[avail + [TARGET]].copy()

    if "Aquifer" in avail:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        tr2["Aquifer"] = enc.fit_transform(tr2[["Aquifer"]]).ravel().astype(float)
        if len(va2): va2["Aquifer"] = enc.transform(va2[["Aquifer"]]).ravel().astype(float)
        te2["Aquifer"] = enc.transform(te2[["Aquifer"]]).ravel().astype(float)

    # Combine train + val for final model training
    tr_final = pd.concat([tr2, va2]) if len(va2) else tr2

    X_tr = tr_final[avail].values
    X_va = va2[avail].values if len(va2) else tr2[avail].values  # early-stop reference
    X_te = te2[avail].values
    y_tr = tr_final[TARGET].values
    y_va = va2[TARGET].values if len(va2) else tr2[TARGET].values
    y_te = te2[TARGET].values

    params = XGB_VARIANTS[xgb_variant]
    mdl = xgb.XGBRegressor(**params)
    mdl.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    y_pred = np.expm1(np.clip(mdl.predict(X_te), 0, None))
    y_true = np.expm1(y_te)
    test_r2 = r2_score(y_true, y_pred)

    # SHAP values on test set
    explainer  = shap.TreeExplainer(mdl)
    shap_vals  = explainer.shap_values(X_te)           # shape (n_test, n_feats)
    X_test_df  = pd.DataFrame(X_te, columns=avail)

    return dict(
        model       = mdl,
        encoder     = enc,
        X_test      = X_test_df,
        shap_values = shap_vals,
        test_r2     = test_r2,
        features    = avail,
        n_train     = len(tr_final),
        n_test      = len(te2),
    )


# ── Plot 1: Feature Importance (mean |SHAP|) ──────────────────────────────────
def plot_importance(state, result, outdir):
    shv   = result["shap_values"]        # (n_test, n_feats)
    feats = result["features"]
    mean_abs = np.abs(shv).mean(axis=0)
    order = np.argsort(mean_abs)         # ascending → bottom-to-top

    labels = [FEAT_LABELS.get(feats[i], feats[i]) for i in order]
    vals   = mean_abs[order]
    color  = STATE_COLORS.get(state, "#4c72b0")

    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.45 * len(feats))))
    bars = ax.barh(range(len(feats)), vals, color=color, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean |SHAP value|  (impact on log-Chloride prediction)")
    ax.set_title(f"{state} – Feature Importance\n"
                 f"(test R² = {result['test_r2']:.3f}, "
                 f"n_test = {result['n_test']:,})")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = outdir / f"{state}_importance.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"    Saved {path.name}", flush=True)


# ── Plot 2: Beeswarm / Summary (Feature Effect Direction) ─────────────────────
def plot_beeswarm(state, result, outdir):
    shv    = result["shap_values"]
    feats  = result["features"]
    X_test = result["X_test"]
    labels = [FEAT_LABELS.get(f, f) for f in feats]

    # shap.summary_plot (beeswarm mode) supports matplotlib and plot_size kwarg
    fig_h  = max(4.0, 0.55 * len(feats))
    plt.figure(figsize=(9, fig_h))
    shap.summary_plot(
        shv,
        X_test.values,
        feature_names=labels,
        max_display=len(feats),
        plot_type="dot",          # beeswarm / dot style
        show=False,
        plot_size=None,           # let us control figure size
    )
    plt.title(
        f"{state} – Feature Effect Direction\n"
        f"(test R² = {result['test_r2']:.3f},  n_test = {result['n_test']:,})",
        fontsize=12, pad=10,
    )
    plt.xlabel("SHAP value  (positive → higher Chloride predicted)", fontsize=10)
    plt.tight_layout()
    path = outdir / f"{state}_beeswarm.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"    Saved {path.name}", flush=True)


# ── Plot 3: Dependence Plots (top-4 features) ────────────────────────────────
def plot_dependence(state, result, outdir, n_top=4):
    shv    = result["shap_values"]
    feats  = result["features"]
    X_test = result["X_test"]

    mean_abs = np.abs(shv).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:n_top]

    for rank, fi in enumerate(top_idx, 1):
        feat_name  = feats[fi]
        feat_label = FEAT_LABELS.get(feat_name, feat_name)

        # Auto-select interaction feature (highest |SHAP| correlation)
        corrs = []
        for j, other in enumerate(feats):
            if j == fi: continue
            xj = X_test.iloc[:, j].values
            if np.std(xj) > 0:
                corrs.append((j, abs(np.corrcoef(xj, shv[:, fi])[0, 1])))
        interact_idx = max(corrs, key=lambda x: x[1])[0] if corrs else fi

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        sc = ax.scatter(
            X_test.iloc[:, fi],
            shv[:, fi],
            c=X_test.iloc[:, interact_idx],
            cmap="coolwarm",
            alpha=0.6,
            s=18,
            edgecolors="none",
        )
        cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.03)
        cb.set_label(FEAT_LABELS.get(feats[interact_idx], feats[interact_idx]),
                     fontsize=9)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_xlabel(feat_label)
        ax.set_ylabel(f"SHAP value for {feat_label}")
        ax.set_title(f"{state} – SHAP Dependence: {feat_label}\n"
                     f"(rank #{rank} by mean |SHAP|)", pad=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        safe_name = feat_name.replace("+", "plus").replace(" ", "_")
        path = outdir / f"{state}_dependence_{safe_name}.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close("all")
        print(f"    Saved {path.name}", flush=True)


# ── Per-state driver ──────────────────────────────────────────────────────────
def run_state(state, df, best_featsets):
    print(f"\n{'='*60}", flush=True)
    print(f"  STATE: {state}", flush=True)
    print(f"{'='*60}", flush=True)

    outdir = OUT / state
    outdir.mkdir(exist_ok=True)

    # Select feature set
    fs_name  = best_featsets.get(state, "Static")
    # Override to State-eligible feature sets
    if fs_name in ("Paleo", "Coastal+Paleo") and state not in GULF_SCOPES:
        fs_name = "Coastal"
    features = FS[fs_name]
    print(f"  Feature set : {fs_name}  ({', '.join(features)})", flush=True)

    # Try best variant first; if it underperforms, try the other
    best_variant  = "base"
    best_csv      = DATA / "ml_results" / "model_comparison_best.csv"
    if best_csv.exists():
        bdf = pd.read_csv(best_csv)
        row = bdf[bdf["Scope"] == state]
        if not row.empty and str(row.iloc[0]["XGB_variant"]) in XGB_VARIANTS:
            best_variant = str(row.iloc[0]["XGB_variant"])

    result = train_and_shap(state, df, features, best_variant)
    if result is None:
        print(f"  [SKIP] Insufficient data for {state}", flush=True)
        return None

    print(f"  XGB variant : {best_variant}", flush=True)
    print(f"  n_train     : {result['n_train']:,}", flush=True)
    print(f"  n_test      : {result['n_test']:,}", flush=True)
    print(f"  Test R²     : {result['test_r2']:.4f}", flush=True)

    # Save raw SHAP values
    shap_df = pd.DataFrame(result["shap_values"], columns=result["features"])
    shap_df.to_csv(outdir / f"shap_values_{state}.csv", index=False)
    print(f"  Saved shap_values_{state}.csv", flush=True)

    # Save mean |SHAP| summary
    mean_abs = np.abs(result["shap_values"]).mean(axis=0)
    imp_df   = pd.DataFrame({
        "Feature":   result["features"],
        "MeanAbsSHAP": mean_abs,
    }).sort_values("MeanAbsSHAP", ascending=False)
    imp_df.to_csv(outdir / f"shap_importance_{state}.csv", index=False)

    # Plots
    plot_importance(state, result, outdir)
    plot_beeswarm(state, result, outdir)
    plot_dependence(state, result, outdir, n_top=4)

    return dict(State=state, FeatSet=fs_name, XGB=best_variant,
                Test_R2=round(result["test_r2"], 4),
                n_train=result["n_train"], n_test=result["n_test"])


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", nargs="+",
                        default=["NJ","PA","DE","MD","VA",
                                 "SC","NC","GA","FL",
                                 "AL","MS","LA","TX"],
                        help="States to process (default: all 13)")
    args = parser.parse_args()

    df              = load_data()
    best_featsets   = load_best_featsets()

    summary_rows = []
    for state in args.states:
        row = run_state(state, df, best_featsets)
        if row:
            summary_rows.append(row)

    # ── Summary table ─────────────────────────────────────────────────────────
    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(OUT / "shap_summary.csv", index=False)
        print("\n" + "=" * 65, flush=True)
        print("SHAP ANALYSIS SUMMARY", flush=True)
        print("=" * 65, flush=True)
        print(f"\n{'State':<6} {'FeatSet':<16} {'XGB':<5} {'n_train':>8} "
              f"{'n_test':>7} {'Test_R²':>8}", flush=True)
        print("-" * 55, flush=True)
        for _, r in summary.iterrows():
            print(f"{r['State']:<6} {r['FeatSet']:<16} {r['XGB']:<5} "
                  f"{int(r['n_train']):>8,} {int(r['n_test']):>7,} "
                  f"{r['Test_R2']:>8.4f}", flush=True)
        print(f"\nAll plots  -> {OUT}/", flush=True)
        print(f"Summary    -> {OUT / 'shap_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
