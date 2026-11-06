"""Shared settings and helpers for the analysis scripts (01-08, figures, supplementary).

Every model in the paper uses the same protocol, defined once here:
  * target          log1p(Chloride), scored as R2_orig after expm1 (predictions clipped
                    at 0 first) and as R2_log on the transformed scale
  * split           chronological 70/15/15 within each well (SiteNo)
  * learner         XGBoost, max_depth 6 or 8, early stopping on the validation part
  * Aquifer         ordinal-encoded on the training part (unseen categories -> -1)

The input is the published dataset (Zenodo). Point the scripts at it with
    python 03_state_models.py --data path/to/Coastal_Groundwater_Chloride_AtlanticGulf_1906_2023.csv
or by setting the SALINITY_DATA environment variable. By default it is read from
./data/ next to this file.
"""
import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_NAME = "Coastal_Groundwater_Chloride_AtlanticGulf_1906_2023.csv"
DEFAULT_DATA = Path(os.environ.get("SALINITY_DATA", ROOT / "data" / DATA_NAME))
RESULTS = Path(os.environ.get("SALINITY_RESULTS", ROOT / "results"))
FIGURES = Path(os.environ.get("SALINITY_FIGURES", ROOT / "figures" / "output"))

# ── Study design ───────────────────────────────────────────────────────────────
STATES = ["NJ", "DE", "MD", "VA", "SC", "NC", "GA", "FL", "AL", "MS", "LA", "TX"]
REGIONS = {"Mid-Atlantic": ["NJ", "DE", "MD", "VA"],
           "South Atlantic": ["SC", "NC", "GA", "FL"],
           "Gulf": ["AL", "MS", "LA", "TX"]}
REGION_OF = {s: r for r, ss in REGIONS.items() for s in ss}
# Pennsylvania (85 observations) is too sparse to model on its own but is pooled into
# the Mid-Atlantic scope of Tables 3 and 4.
POOLED = {"Mid-Atlantic": ["NJ", "PA", "DE", "MD", "VA"],
          "South Atlantic": REGIONS["South Atlantic"], "Gulf": REGIONS["Gulf"]}
SCOPES = ["NJ", "DE", "MD", "VA", "Mid-Atlantic", "SC", "NC", "GA", "FL",
          "South Atlantic", "AL", "MS", "LA", "TX", "Gulf"]
GULF = {"AL", "MS", "LA", "TX"}
EPA_CL = 250.0                              # U.S. EPA secondary standard, mg/L

# Analysis population (Section 2.1), as in the published dataset. Applied again on loading so the
# modelling scripts state the population they use and cannot run on the full compilation.
MAX_SLR0_DIST = 20_000.0                    # m from the present-day (0 ft) inundation extent
HALITE = 360_000.0                          # mg/L; higher values are recording errors
# USGS principal aquifers of the Atlantic and Gulf Coastal Plain (USGS, 2003). Wells in
# any other unit are excluded: the Appalachian Highlands bedrock aquifers landward of the
# Fall Line (Piedmont and Blue Ridge, Valley and Ridge, Early Mesozoic basin,
# Pennsylvanian, Mississippian, New York and New England carbonate-rock), and records
# assigned to "Other rocks", "Other aquifers" or no aquifer. Texas coastal uplands and
# Edwards-Trinity have no wells within 20 km, so 12 of these 14 occur in the dataset.
COASTAL_PLAIN_AQUIFERS = {
    "Northern Atlantic Coastal Plain aquifer system",
    "Southeastern Coastal Plain aquifer system",
    "Floridan aquifer system",
    "Biscayne aquifer",
    "Castle Hayne aquifer",
    "Surficial aquifer system",
    "Intermediate aquifer system",
    "Coastal lowlands aquifer system",
    "Texas coastal uplands aquifer system",
    "Mississippi embayment aquifer system",
    "Mississippi River Valley alluvial aquifer",
    "Edwards-Trinity aquifer system",
    "Sand and gravel aquifers (glaciated regions)",
    "Alluvial aquifers",
}


def in_population(df):
    """Rows in the analysis population: 0 < Chloride <= HALITE, a Coastal Plain aquifer
    (COASTAL_PLAIN_AQUIFERS) and SLR0_DIST <= 20 km."""
    cl = pd.to_numeric(df["Chloride"], errors="coerce")
    dist = pd.to_numeric(df["SLR0_DIST"], errors="coerce")
    return ((cl > 0) & (cl <= HALITE) & df["Aquifer"].isin(COASTAL_PLAIN_AQUIFERS)
            & (dist <= MAX_SLR0_DIST))

# ── Predictors (Section 2.4) ───────────────────────────────────────────────────
CORE = ["WellDepthM", "SLR0_DIST", "Hydro_DIST", "Aquifer"]
BASE = ["Surf_Ele", "WellDepthM", "SLR0_DIST", "Aquifer", "Hydro_DIST"]
TIDE = "Ele_abv_MWL_1y"
PRECIP = ["daily_precip", "7d_Before_prec", "1m_Before_prec"]
SPECS = {                                    # the five specifications of Table 3
    "Base": BASE,
    "GWE":  BASE + ["WaterEle_m"],
    "PREC": BASE + PRECIP,
    "SALT": BASE + ["Salt_DIST"],            # Gulf states only
    "TIDE": [TIDE, "WellDepthM", "SLR0_DIST", "Aquifer", "Hydro_DIST"],
}

# ── Projection models (Section 3.4.1; evidence produced by step 6) ────────────
# Elevation term of each state's projection model: tide-referenced where it covered at
# least 90% of wells and either showed no matched loss beyond run-to-run noise or was
# the only specification above R2_orig 0.7 (Texas); land-surface
# elsewhere. Gulf models also carry Salt_DIST.
PROJECTION_ELEV = {s: (TIDE if s in {"DE", "VA", "GA", "FL", "TX"} else "Surf_Ele")
                   for s in STATES}
# States whose observations are concentrated on a few heavily sampled wells; fitted on
# one record per well (well medians), otherwise their distance response inverts.
WELL_LEVEL = {"SC", "LA"}


def projection_features(state):
    return [PROJECTION_ELEV[state]] + CORE + (["Salt_DIST"] if state in GULF else [])


# ── Learner ────────────────────────────────────────────────────────────────────
TARGET = "log_Chloride"
MIN_TRAIN, MIN_TEST = 50, 20
SEEDS5 = [42, 7, 101, 2026, 31337]
SEEDS10 = SEEDS5 + [13, 777, 20261005, 555, 88]
PAR = dict(n_estimators=1000, learning_rate=0.05, subsample=0.8, colsample_bytree=1.0,
           reg_alpha=0.1, reg_lambda=1.0, tree_method="hist", early_stopping_rounds=30,
           n_jobs=-1, verbosity=0)
DEPTHS = [dict(PAR, max_depth=6, min_child_weight=5),
          dict(PAR, max_depth=8, min_child_weight=3)]

_TEXT = {"SiteNo", "Date", "State", "Aquifer", "Confine_class"}


def data_path():
    """--data on the command line, else SALINITY_DATA, else ./data/<DATA_NAME>."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    a, _ = p.parse_known_args()
    if not a.data.exists():
        sys.exit(f"dataset not found: {a.data}\n"
                 f"download {DATA_NAME} from Zenodo and pass --data <path>")
    return a.data


def load(path=None, states=STATES):
    """Read the analysis dataset, keep the analysis population (in_population) in
    `states`, coerce numeric columns and add the target."""
    path = path or data_path()
    df = pd.read_csv(path, dtype={"SiteNo": str}, low_memory=False)
    for c in df.columns:
        if c not in _TEXT:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = in_population(df)
    if (~keep).any():
        print(f"{path.name}: {(~keep).sum():,} observations outside the analysis "
              f"population (Coastal Plain aquifers within 20 km) dropped")
    df = df[keep & df["State"].isin(states)].copy()
    df[TARGET] = np.log1p(df["Chloride"])
    df["Date_ord"] = pd.to_datetime(df["Date"], errors="coerce").astype("int64")
    print(f"{path.name}: {len(df):,} observations, {df['SiteNo'].nunique():,} wells")
    return df


def out(name):
    RESULTS.mkdir(exist_ok=True)
    return RESULTS / name


def split_703(g):
    """Chronological 70/15/15 split within each well.

    One-sample wells go to training; two-sample wells contribute their later sample to
    the test set; otherwise the first 70% train, the next 15% validate, the rest test.
    """
    tr, va, te = [], [], []
    for _, s in g.groupby("SiteNo", sort=False):
        s = s.sort_values("Date_ord"); n = len(s)
        if n == 1:
            tr.append(s)
        elif n == 2:
            tr.append(s.iloc[:1]); te.append(s.iloc[1:])
        else:
            a = max(1, int(n * .70)); b = max(1, int(n * .15))
            tr.append(s.iloc[:a]); va.append(s.iloc[a:a + b]); te.append(s.iloc[a + b:])
    cat = lambda L: pd.concat(L) if L else g.iloc[0:0]
    return cat(tr), cat(va), cat(te)


def encode(tr, va, te, feats):
    """Copy the three parts and ordinal-encode Aquifer on the training part."""
    tr, va, te = tr.copy(), va.copy(), te.copy()
    if "Aquifer" in feats:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        tr["Aquifer"] = enc.fit_transform(tr[["Aquifer"]]).ravel().astype(float)
        for d in (va, te):
            if len(d):
                d["Aquifer"] = enc.transform(d[["Aquifer"]]).ravel().astype(float)
    return tr, va, te


def r2_orig(y_log, p_log):
    return r2_score(np.expm1(y_log), np.expm1(np.clip(p_log, 0, None)))


def r2_log(y_log, p_log):
    return r2_score(y_log, p_log)


def scope_rows(df, scope):
    """One state, or a pooled region (POOLED), from a frame loaded with PA included."""
    return df[df["State"].isin(POOLED.get(scope, [scope]))]


def complete(d, feats):
    """Rows complete on the target, the well identifier, the date and `feats`."""
    return d[d[list(feats) + [TARGET, "SiteNo", "Date_ord"]].notna().all(axis=1)]
