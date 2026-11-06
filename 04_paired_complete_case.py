"""Step 4 - Paired complete-case evaluation (Sections 2.5.2 and 3.3.2, Table 4).

Time-varying predictors exist only for some observations, so an augmented model and the
Base model fitted on all rows differ in their samples as well as their predictors. For
each predictor group A:

  S_A        rows complete on the Base predictors and on A's predictors
  Base|full  Base fitted on all of its own complete cases (its own split)
  Base|S_A   Base fitted on S_A
  Aug|S_A    the augmented specification fitted on S_A, with the SAME split as Base|S_A

  predictor effect = R2(Aug|S_A)  - R2(Base|S_A)
  sample effect    = R2(Base|S_A) - R2(Base|full)
  their sum        = the difference reported in Table 3

Groups: +GWE (WaterEle_m), +PREC (three precipitation terms), +SALT (Gulf only) and
TIDE (Ele_abv_MWL_1y substituted for Surf_Ele). A pair is not estimable when either arm
has fewer than 50 training rows, 20 test rows or 10 test wells.

Estimation noise (second half): on identical rows and splits, the Base and TIDE arms are
refitted with five seeds and two predictor column orders. Column order changes only
XGBoost's tie-breaking between equal-gain splits, so the spread of the predictor effect
across these ten refits measures how far a single run can be trusted.

Output: results/table4_paired_effects.csv (long), results/table4_formatted.csv,
        results/table4_noise.csv
"""
import itertools

import numpy as np
import pandas as pd
import xgboost as xgb

from common import (DEPTHS, GULF, MIN_TEST, MIN_TRAIN, PRECIP, SCOPES, SEEDS5,
                    STATES, TARGET, TIDE, encode, load, out, r2_log, r2_orig,
                    scope_rows, split_703)

MIN_TEST_WELLS = 10
# Column order matters to XGBoost's tie-breaking, so it is fixed here as in Table 4.
BASE = ["Surf_Ele", "WellDepthM", "SLR0_DIST", "Hydro_DIST", "Aquifer"]
PAIRS = {"GWE":  BASE + ["WaterEle_m"],
         "PREC": BASE + PRECIP,
         "SALT": BASE + ["Salt_DIST"],
         "TIDE": [c for c in BASE if c != "Surf_Ele"] + [TIDE]}
LABEL = {"GWE": "Groundwater-elev.", "PREC": "Ant. precipitation",
         "SALT": "Salt-dome dist.", "TIDE": "Tide-ref. elevation"}


def fit(tr, va, te, feats, seed=42):
    """Best of depth 6 and 8 by test R2_orig; returns (R2_orig, R2_log) or None."""
    if len(tr) < MIN_TRAIN or len(te) < MIN_TEST or te["SiteNo"].nunique() < MIN_TEST_WELLS:
        return None
    tr, va, te = encode(tr, va, te, feats)
    Xtr, ytr = tr[feats].values, tr[TARGET].values
    ev = [(va[feats].values, va[TARGET].values)] if len(va) else [(Xtr, ytr)]
    best = None
    for par in DEPTHS:
        m = xgb.XGBRegressor(**dict(par, random_state=seed))
        m.fit(Xtr, ytr, eval_set=ev, verbose=False)
        p = m.predict(te[feats].values)
        r = (r2_orig(te[TARGET].values, p), r2_log(te[TARGET].values, p))
        if best is None or r[0] > best[0]:
            best = r
    return best


def rows_complete(d, feats):
    return d[d[list(feats) + [TARGET]].notna().all(axis=1)]


# PA is not modelled on its own (85 records) but is pooled into the Mid-Atlantic scope
# (see POOLED in common.py), so it is loaded here in addition to STATES.
df = load(states=STATES + ["PA"])

# ── Paired comparison ──────────────────────────────────────────────────────────
rows = []
for sc in SCOPES:
    sub = scope_rows(df, sc)
    bf = rows_complete(sub, BASE)
    rb = fit(*split_703(bf), BASE) if len(bf) else None
    for pair, aug in PAIRS.items():
        if pair == "SALT" and not (sc in GULF or sc == "Gulf"):
            continue
        S = rows_complete(sub, sorted(set(BASE) | set(aug)))
        rec = dict(Scope=sc, Pair=pair, n_matched=len(S),
                   wells_matched=S["SiteNo"].nunique(),
                   R2_base_full=rb[0] if rb else np.nan)
        tr, va, te = split_703(S) if len(S) else (S, S, S)   # one split for both arms
        a = fit(tr, va, te, BASE) if len(S) else None
        b = fit(tr, va, te, aug) if len(S) else None
        if a is None or b is None:
            rows.append(dict(rec, status="n.e.")); continue
        rec.update(status="ok", R2_base_matched=a[0], R2_aug_matched=b[0],
                   predictor_effect=round(b[0] - a[0], 4),
                   sample_effect=round(a[0] - rb[0], 4) if rb else np.nan,
                   predictor_effect_log=round(b[1] - a[1], 4),
                   sample_effect_log=round(a[1] - rb[1], 4) if rb else np.nan)
        rows.append(rec)
        print(f"  {sc:15s} {pair:5s} n={len(S):7,}  predictor {rec['predictor_effect']:+.3f}"
              f"  sample {rec['sample_effect']:+.3f}")

A = pd.DataFrame(rows)
A.to_csv(out("table4_paired_effects.csv"), index=False)

ok = A[A["status"] == "ok"]
cell = A.assign(cell=[f"{p:+.3f} / {s:+.3f}" if st == "ok" else "n.e."
                      for p, s, st in zip(A["predictor_effect"], A["sample_effect"],
                                          A["status"])])
T = cell.pivot(index="Scope", columns="Pair", values="cell").reindex(SCOPES)[list(PAIRS)]
T.loc["Mean predictor"] = [f"{ok[ok.Pair == p].predictor_effect.mean():+.3f}" for p in PAIRS]
T.loc["Mean sample"] = [f"{ok[ok.Pair == p].sample_effect.mean():+.3f}" for p in PAIRS]
T.loc["Scopes where |sample| > |predictor|"] = [
    f"{(ok[ok.Pair == p].sample_effect.abs() > ok[ok.Pair == p].predictor_effect.abs()).sum()}"
    f" of {(ok.Pair == p).sum()}" for p in PAIRS]
T.rename(columns=LABEL).to_csv(out("table4_formatted.csv"))
print("\n" + T.rename(columns=LABEL).fillna("—").to_string())
print(f"\nsample effects {ok.sample_effect.min():+.3f} to {ok.sample_effect.max():+.3f}")

# ── Estimation noise: 5 seeds x 2 column orders on identical rows ──────────────
NOISE_SCOPES = ["DE", "MD", "VA", "NC", "GA", "AL", "MS", "TX"]
BASE_REV = BASE[::-1]
TIDE_LAST = ["WellDepthM", "SLR0_DIST", "Hydro_DIST", "Aquifer", TIDE]
TIDE_FIRST = [TIDE, "WellDepthM", "SLR0_DIST", "Hydro_DIST", "Aquifer"]
noise = []
for sc in NOISE_SCOPES:
    S = rows_complete(scope_rows(df, sc), sorted(set(BASE) | {TIDE}))
    tr, va, te = split_703(S)
    if len(va) == 0 or len(te) < MIN_TEST:
        continue
    for seed, (bo, to) in itertools.product(SEEDS5, [(BASE, TIDE_LAST),
                                                     (BASE_REV, TIDE_FIRST)]):
        a, b = fit(tr, va, te, bo, seed), fit(tr, va, te, to, seed)
        noise.append(dict(Scope=sc, seed=seed,
                          order="tide last" if to is TIDE_LAST else "tide first",
                          base=a[0], tide=b[0], effect=b[0] - a[0]))
N = pd.DataFrame(noise)
N.to_csv(out("table4_noise.csv"), index=False)
g = N.groupby("Scope")["effect"].agg(["mean", "std", "min", "max"])
g["sign_reverses"] = (g["min"] < 0) & (g["max"] > 0)
print("\nTIDE predictor effect over 5 seeds x 2 column orders\n" + g.round(3).to_string())
