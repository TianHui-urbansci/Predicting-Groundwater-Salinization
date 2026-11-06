"""Step 6 - Projection model selection (Sections 2.6.1 and 3.4.1).

Part A  Configuration search. In each state, 12 predictor configurations (24 in the Gulf)
        are fitted with three seeds: core predictors (WellDepthM, SLR0_DIST, Hydro_DIST,
        Aquifer) plus an elevation term (land-surface, tide-referenced, or both) and
        optionally groundwater elevation, antecedent precipitation (1d_Before_prec) and
        salt-dome distance (Gulf). Tree depth is chosen on the VALIDATION partition and
        the test R2_orig is reported, together with the matched effect over the
        land-surface baseline fitted on the same rows. This search found no reliable
        multi-predictor optimum, which motivated Part B.

Part B  Two specifications per state, ten seeds, depth chosen on validation:
          land-surface     Surf_Ele       + core [+ Salt_DIST in the Gulf]
          tide-referenced  Ele_abv_MWL_1y + core [+ Salt_DIST in the Gulf]
        Each is scored on its own complete cases (R2_orig, well coverage), and the
        tide-referenced specification is also compared with the land-surface one on
        identical rows and splits (matched effect).

Choice  Tide-referenced elevation is used where it covers at least 90% of the state's
        wells and either does not lose to land-surface elevation on matched rows by more
        than the run-to-run variation (one SD over ten seeds), or is the only one of the
        two above the accuracy threshold (R2_orig > 0.7); land-surface elevation is used
        elsewhere. The resulting choice is fixed in common.PROJECTION_ELEV, which steps
        7 and 8 use; this script reports whether its rule reproduces that table.
        Whether a state's model is then used for projection (accuracy, interpretable
        response direction) is decided in step 8, after SHAP (step 7).

Output: results/projection_config_search.csv, results/projection_spec_comparison.csv,
        results/projection_model_choice.csv
"""
import itertools

import numpy as np
import pandas as pd
import xgboost as xgb

from common import (CORE, DEPTHS, GULF, MIN_TEST, MIN_TRAIN, PROJECTION_ELEV, SEEDS10,
                    STATES, TARGET, TIDE, encode, load, out, r2_orig, split_703)

MIN_TEST_WELLS = 10
COVER_MIN, R2_MIN = 0.90, 0.70
SEARCH_SEEDS = [42, 7, 2026]
PRECIP_TERM = "1d_Before_prec"
ELEV = {"surf": ["Surf_Ele"], "tide": [TIDE], "surf+tide": ["Surf_Ele", TIDE]}


def score(tr, va, te, feats, seed):
    """Depth chosen on validation R2_orig; returns (test R2_orig, validation R2_orig)."""
    tr, va, te = encode(tr, va, te, feats)
    best = None
    for par in DEPTHS:
        m = xgb.XGBRegressor(**dict(par, random_state=seed))
        m.fit(tr[feats].values, tr[TARGET].values,
              eval_set=[(va[feats].values, va[TARGET].values)], verbose=False)
        rv = r2_orig(va[TARGET].values, m.predict(va[feats].values))
        rt = r2_orig(te[TARGET].values, m.predict(te[feats].values))
        if best is None or rv > best[1]:
            best = (rt, rv)
    return best


def usable(tr, va, te):
    return (len(tr) >= MIN_TRAIN and len(te) >= MIN_TEST and len(va) > 0
            and te["SiteNo"].nunique() >= MIN_TEST_WELLS)


def rows_complete(d, feats):
    return d[d[list(feats) + [TARGET]].notna().all(axis=1)]


df = load()
has_precip = PRECIP_TERM in df.columns
if not has_precip:
    print(f"note: {PRECIP_TERM} not in the dataset; precipitation configurations skipped")

# ── Part A: configuration search ───────────────────────────────────────────────
rows = []
for st in STATES:
    sub = df[df["State"] == st]
    allw = sub["SiteNo"].nunique()
    for (ename, elev), head, prec, salt in itertools.product(
            ELEV.items(), [False, True], [False, True] if has_precip else [False],
            [False, True] if st in GULF else [False]):
        feats = (CORE + elev + (["WaterEle_m"] if head else [])
                 + ([PRECIP_TERM] if prec else []) + (["Salt_DIST"] if salt else []))
        label = "+".join([ename] + [n for n, f in (("head", head), ("precip", prec),
                                                     ("salt", salt)) if f])
        S = rows_complete(sub, feats)
        tr, va, te = split_703(S) if len(S) else (S, S, S)
        if not len(S) or not usable(tr, va, te):
            continue
        ts, vs, eff = [], [], []
        for sd in SEARCH_SEEDS:
            rt, rv = score(tr, va, te, feats, sd)
            rb, _ = score(tr, va, te, CORE + ["Surf_Ele"], sd)
            ts.append(rt); vs.append(rv); eff.append(rt - rb)
        rows.append(dict(State=st, Config=label, features="|".join(feats),
                         wells=S["SiteNo"].nunique(),
                         well_coverage=round(S["SiteNo"].nunique() / allw, 4), n=len(S),
                         R2_test_mean=round(np.mean(ts), 4),
                         R2_test_sd=round(np.std(ts, ddof=1), 4),
                         R2_val_mean=round(np.mean(vs), 4),
                         matched_effect=round(np.mean(eff), 4)))
        print(f"  {st} {label:24s} cover {rows[-1]['well_coverage']:6.1%}  "
              f"test {rows[-1]['R2_test_mean']:+.3f}  val {rows[-1]['R2_val_mean']:+.3f}  "
              f"effect {rows[-1]['matched_effect']:+.3f}")
A = pd.DataFrame(rows)
A.to_csv(out("projection_config_search.csv"), index=False)
best_val = A.loc[A.groupby("State")["R2_val_mean"].idxmax()].set_index("State")
best_test = A.loc[A.groupby("State")["R2_test_mean"].idxmax()].set_index("State")
agree = (best_val["Config"] == best_test["Config"]).sum()
print(f"\nvalidation-selected configuration = test-best configuration in {agree} of "
      f"{len(best_val)} states")

# ── Part B: land-surface vs tide-referenced ────────────────────────────────────
rows = []
for st in STATES:
    sub = df[df["State"] == st]
    allw = sub["SiteNo"].nunique()
    salt = ["Salt_DIST"] if st in GULF else []
    land, tide = ["Surf_Ele"] + CORE + salt, [TIDE] + CORE + salt
    rec = dict(State=st, wells_all=allw)
    for name, feats in (("land", land), ("tide", tide)):
        S = rows_complete(sub, feats)
        tr, va, te = split_703(S)
        vals = [score(tr, va, te, feats, sd)[0] for sd in SEEDS10] if usable(tr, va, te) \
            else [np.nan]
        rec.update({f"{name}_wells": S["SiteNo"].nunique(),
                    f"{name}_coverage": round(S["SiteNo"].nunique() / allw, 4),
                    f"{name}_R2": round(np.mean(vals), 4),
                    f"{name}_R2_sd": round(np.std(vals, ddof=1), 4) if len(vals) > 1 else np.nan})
    S = rows_complete(sub, sorted(set(land) | set(tide)))       # matched comparison
    tr, va, te = split_703(S)
    if usable(tr, va, te):
        e = [score(tr, va, te, tide, sd)[0] - score(tr, va, te, land, sd)[0] for sd in SEEDS10]
        rec.update(matched_effect=round(np.mean(e), 4), matched_effect_sd=round(np.std(e, ddof=1), 4))
    else:
        rec.update(matched_effect=np.nan, matched_effect_sd=np.nan)
    rows.append(rec)
    print(f"  {st}  land {rec['land_R2']:+.3f} ({rec['land_coverage']:.0%})  "
          f"tide {rec['tide_R2']:+.3f} ({rec['tide_coverage']:.0%})  "
          f"matched {rec['matched_effect']:+.3f}")
B = pd.DataFrame(rows)
B.to_csv(out("projection_spec_comparison.csv"), index=False)

# ── Choice ─────────────────────────────────────────────────────────────────────
def choose(r):
    if r.tide_coverage < COVER_MIN:
        return "land", f"tide-referenced covers {r.tide_coverage:.0%} of wells"
    if r.matched_effect >= -r.matched_effect_sd:
        return "tide", f"matched effect {r.matched_effect:+.3f} (sd {r.matched_effect_sd:.3f})"
    if r.land_R2 <= R2_MIN < r.tide_R2:
        return "tide", f"only tide-referenced exceeds R2 {R2_MIN} ({r.tide_R2:.3f})"
    return "land", f"matched loss {r.matched_effect:+.3f} beyond sd {r.matched_effect_sd:.3f}"


C = B[["State", "land_R2", "tide_R2", "tide_coverage", "matched_effect"]].copy()
C[["elevation", "reason"]] = B.apply(lambda r: pd.Series(choose(r)), axis=1)
C["used"] = C["State"].map(PROJECTION_ELEV).map({"Surf_Ele": "land", TIDE: "tide"})
C.to_csv(out("projection_model_choice.csv"), index=False)
print("\n" + C.to_string(index=False))
diff = C[C.elevation != C.used]
print("\nrule reproduces common.PROJECTION_ELEV in all states" if diff.empty else
      f"\nrule differs from common.PROJECTION_ELEV in: {', '.join(diff.State)}")
