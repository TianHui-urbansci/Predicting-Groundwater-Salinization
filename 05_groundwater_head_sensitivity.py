"""Step 5 - Sensitivity to groundwater head (Sections 2.5.3 and 3.3.3, Tables 5 and 6).

Sample: the 43,839 observations at 4,215 wells that carry a water-level measurement.
Six specifications are fitted on identical rows and one chronological split:

  GH1  Base-  = WellDepthM, SLR0_DIST, Hydro_DIST, Aquifer
  GH2  Base- + Surf_Ele
  GH3  Base- + WaterLevel_m             depth to water, positive downward
  GH4  Base- + WaterEle_m               water-level elevation = Surf_Ele - WaterLevel_m
  GH5  Base- + Surf_Ele + WaterLevel_m
  GH6  Base- + Surf_Ele + WaterEle_m

Strata are all wells and each USGS principal aquifer with at least 150 observations and
40 wells. Differences in R2_orig get 95% percentile intervals from 1,000 paired bootstrap
resamples of TEST WELLS (not rows, because repeat samples at a well are not
independent); both specifications are rescored on the same resampled wells without
refitting.

Output: results/table5_head_specifications.csv, results/table6_head_bootstrap.csv
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import wilcoxon
from sklearn.metrics import r2_score

from common import (DEPTHS, MIN_TEST, MIN_TRAIN, STATES, TARGET, encode, load, out,
                    split_703)

MIN_TEST_WELLS, NBOOT, BOOT_SEED = 10, 1000, 20261003
# one canonical column order; every specification keeps its predictors in this order
CANON = ["WellDepthM", "SLR0_DIST", "Hydro_DIST", "Aquifer",
         "Surf_Ele", "WaterLevel_m", "WaterEle_m"]
BASEM = {"WellDepthM", "SLR0_DIST", "Hydro_DIST", "Aquifer"}
GH = {"GH1": BASEM,
      "GH2": BASEM | {"Surf_Ele"},
      "GH3": BASEM | {"WaterLevel_m"},
      "GH4": BASEM | {"WaterEle_m"},
      "GH5": BASEM | {"Surf_Ele", "WaterLevel_m"},
      "GH6": BASEM | {"Surf_Ele", "WaterEle_m"}}
GH = {k: [c for c in CANON if c in v] for k, v in GH.items()}
CONTRASTS = [("Topography alone", "GH2", "GH1"),
             ("Head alone (depth to water)", "GH3", "GH1"),
             ("Head alone (water-level elevation)", "GH4", "GH1"),
             ("Head given topography (depth to water)", "GH5", "GH2"),
             ("Head given topography (water-level elevation)", "GH6", "GH2"),
             ("Topography vs head (depth to water)", "GH2", "GH3"),
             ("Topography vs head (water-level elevation)", "GH2", "GH4")]


def fit_predict(tr, va, te, feats):
    """Better of depth 6 and 8 by test R2_orig: (R2_orig, R2_log, predictions in mg/L)."""
    if len(tr) < MIN_TRAIN or len(te) < MIN_TEST or te["SiteNo"].nunique() < MIN_TEST_WELLS:
        return None
    tr, va, te = encode(tr, va, te, feats)
    ev = [(va[feats].values, va[TARGET].values)] if len(va) else \
         [(tr[feats].values, tr[TARGET].values)]
    y = te[TARGET].values
    best = None
    for par in DEPTHS:
        m = xgb.XGBRegressor(**dict(par, random_state=42))
        m.fit(tr[feats].values, tr[TARGET].values, eval_set=ev, verbose=False)
        pl = m.predict(te[feats].values)
        p = np.expm1(np.clip(pl, 0, None))
        s = r2_score(np.expm1(y), p)
        if best is None or s > best[0]:
            best = (s, r2_score(y, pl), p)
    return best


def boot_ci(obs, pa, pb, wells, rng):
    """Paired percentile interval for R2(a) - R2(b), resampling test wells."""
    uw = np.unique(wells)
    by_well = {w: np.where(wells == w)[0] for w in uw}
    d = []
    for _ in range(NBOOT):
        idx = np.concatenate([by_well[w] for w in rng.choice(uw, size=len(uw))])
        y = obs[idx]
        if np.ptp(y) > 0:
            d.append(r2_score(y, pa[idx]) - r2_score(y, pb[idx]))
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


# PA (85 records) is loaded in addition to STATES because this analysis pools wells across
# states: any PA well with a water-level measurement enters the "All wells" and aquifer strata.
df = load(states=STATES + ["PA"])
H = df[df[sorted(set(CANON)) + [TARGET]].notna().all(axis=1)].copy()
print(f"head sample: {len(H):,} observations, {H['SiteNo'].nunique():,} wells\n")

STRATA = [("All wells", H)]
for aq, s in H.groupby("Aquifer"):
    if len(s) >= 150 and s["SiteNo"].nunique() >= 40:
        STRATA.append((str(aq), s))

rng = np.random.default_rng(BOOT_SEED)
t5, t6 = [], []
for name, S in STRATA:
    tr, va, te = split_703(S)
    fits = {k: fit_predict(tr, va, te, f) for k, f in GH.items()}
    if any(v is None for v in fits.values()):
        print(f"  {name:46s} not estimable"); continue
    t5.append(dict(Stratum=name, n=len(S), wells=S["SiteNo"].nunique(),
                   wells_test=te["SiteNo"].nunique(),
                   r_head_topo=round(S["WaterEle_m"].corr(S["Surf_Ele"]), 3),
                   median_depth_to_water=round(S["WaterLevel_m"].median(), 2),
                   **{k: round(v[0], 4) for k, v in fits.items()},
                   **{k + "_log": round(v[1], 4) for k, v in fits.items()}))
    obs, wells = np.expm1(te[TARGET].values), te["SiteNo"].values
    for lab, a, b in CONTRASTS:
        lo, hi = boot_ci(obs, fits[a][2], fits[b][2], wells, rng)
        delta = fits[a][0] - fits[b][0]
        t6.append(dict(Stratum=name, Contrast=lab, arm=a, ref=b,
                       wells_test=te["SiteNo"].nunique(), delta=round(delta, 4),
                       lo=round(lo, 4), hi=round(hi, 4),
                       excludes_zero=bool(lo > 0 or hi < 0),
                       cell=f"{delta:+.3f} [{lo:+.3f}, {hi:+.3f}]"))
    print(f"  {name[:44]:46s} " + " ".join(f"{k}={v[0]:.3f}" for k, v in fits.items()))

T5, T6 = pd.DataFrame(t5), pd.DataFrame(t6)
T5.to_csv(out("table5_head_specifications.csv"), index=False)
T6.to_csv(out("table6_head_bootstrap.csv"), index=False)

for lab, a, b in CONTRASTS:
    k = T6[(T6.Contrast == lab) & (T6.Stratum != "All wells")]
    p = T6[(T6.Contrast == lab) & (T6.Stratum == "All wells")].iloc[0]
    print(f"\n{lab} ({a} - {b}): all wells {p.cell}")
    for _, r in k.iterrows():
        print(f"  {r.Stratum[:40]:42s}{r.cell}{'  *' if r.excludes_zero else ''}")
    if len(k) >= 6:
        w, pv = wilcoxon(k.delta.values)
        print(f"  Wilcoxon signed-rank across {len(k)} aquifers: W = {w:.1f}, p = {pv:.3f}")
