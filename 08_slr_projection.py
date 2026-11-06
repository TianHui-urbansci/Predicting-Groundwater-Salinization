"""Step 8 - Exploratory projection under sea-level rise (Sections 2.6 and 3.4, Table 7,
Figures 7 and 8).

Scenarios, rounded to NOAA's 1-ft inundation extents:
                      SSP2-4.5   SSP5-8.5
  NJ ... AL, MS        +4 ft      +7 ft      (Mid-Atlantic, South Atlantic, Eastern Gulf)
  LA, TX               +5 ft      +9 ft      (Western Gulf)

Eligibility (Section 2.6.1). A state's projection model (common.projection_features) is
used only if its test R2_orig exceeds 0.7 and the predictors a scenario changes respond
in the interpretable direction, i.e. their SHAP direction from step 7 is negative
(SLR0_DIST, and Ele_abv_MWL_1y in tide-referenced models). States that fail report
inundation only; their fresh-to-saline cells are not estimated (not zero).

Per scenario, only the predictors sea-level rise changes are updated: SLR0_DIST becomes
the scenario distance, and in tide-referenced models Ele_abv_MWL_1y is lowered by the
rise. Each well is one record (well medians); predictions are averaged over five seeds.

  Case 1  inundated      inside the scenario extent (distance 0) or land surface at or
                         below the scenario rise; excluded from chloride prediction
  Case 2  extrapolation  scenario distance below the 5th percentile of training
                         SLR0_DIST, or lowered Ele_abv_MWL_1y below its training minimum;
                         predicted but flagged
  Case 3  calibrated     all other wells

A fresh-to-saline transition is a well outside Case 1 with observed median chloride at
or below 250 mg/L and seed-mean projected chloride above it. For tide-referenced models
the transitions are also recounted with the distance change alone (the elevation term
held at present-day values), which separates the two pathways.

Output: results/table7_slr_projection.csv, results/table7_regional.csv,
        results/projection_eligibility.csv, results/figure_wells_SSP245.csv,
        results/figure_wells_SSP585.csv
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder

from common import (DEPTHS, EPA_CL, PROJECTION_ELEV, REGION_OF, RESULTS, SEEDS5, STATES,
                    TARGET, TIDE, WELL_LEVEL, load, out, projection_features, r2_orig,
                    split_703)

FT_M = 0.3048
R2_MIN = 0.70
SCENARIOS = {s: ((5, 9) if s in {"LA", "TX"} else (4, 7)) for s in STATES}
SSP = {0: "SSP245", 1: "SSP585"}


def collapse(d, feats):
    num = [c for c in feats if c != "Aquifer"] + [TARGET]
    g = d.groupby("SiteNo")
    w = g[num].median()
    w["Aquifer"] = g["Aquifer"].first()
    return w.reset_index()


shap_dir = pd.read_csv(RESULTS / "shap_importance.csv")    # written by step 7
df = load()
rows, elig, wells_out = [], [], {0: [], 1: []}
for st in STATES:
    feats, elev = projection_features(st), PROJECTION_ELEV[st]
    s = df[df["State"] == st]
    scen_cols = [f"SLR_{x}_DIST" for x in SCENARIOS[st]]
    panel = (s.groupby("SiteNo")
               .agg(obs_Cl=("Chloride", "median"), lon=("lon", "first"), lat=("lat", "first"),
                    **{c: (c, "median") for c in
                       dict.fromkeys([c for c in feats if c != "Aquifer"] + ["Surf_Ele"])},
                    Aquifer=("Aquifer", "first"), **{c: (c, "first") for c in scen_cols})
               .reset_index())
    modelable = panel[feats].notna().all(axis=1)

    # fit five seeds on the state's complete cases; predict the modelable panel
    S = s[s[feats + [TARGET]].notna().all(axis=1)]
    tr, va, te = split_703(S)
    if st in WELL_LEVEL:
        tr, va, te = (collapse(x, feats) for x in (tr, va, te))
    tr, va, te, P = tr.copy(), va.copy(), te.copy(), panel[modelable].copy()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    tr["Aquifer"] = enc.fit_transform(tr[["Aquifer"]]).ravel().astype(float)
    for d in (va, te, P):
        d["Aquifer"] = enc.transform(d[["Aquifer"]]).ravel().astype(float)
    d5 = np.percentile(tr["SLR0_DIST"], 5)
    tide_min = tr[TIDE].min() if elev == TIDE else np.nan
    r2s, base, scen, dist_only = [], [], {x: [] for x in SCENARIOS[st]}, {x: [] for x in SCENARIOS[st]}
    pred = lambda m, X: np.expm1(np.clip(m.predict(X[feats].values), 0, None))
    for sd in SEEDS5:
        best = None
        for par in DEPTHS:
            m = xgb.XGBRegressor(**dict(par, random_state=sd))
            m.fit(tr[feats].values, tr[TARGET].values,
                  eval_set=[(va[feats].values, va[TARGET].values)], verbose=False)
            rv = r2_orig(va[TARGET].values, m.predict(va[feats].values))
            if best is None or rv > best[0]:
                best = (rv, m)
        m = best[1]
        r2s.append(r2_orig(te[TARGET].values, m.predict(te[feats].values)))
        base.append(pred(m, P))
        for x in SCENARIOS[st]:
            Q = P.copy()
            Q["SLR0_DIST"] = Q[f"SLR_{x}_DIST"]
            dist_only[x].append(pred(m, Q))
            if elev == TIDE:
                Q[TIDE] = Q[TIDE] - x * FT_M
            scen[x].append(pred(m, Q))
    r2 = float(np.mean(r2s))

    # eligibility: accuracy and response direction of the scenario-changed predictors
    k = shap_dir[(shap_dir.State == st) & shap_dir.predictor.isin(["SLR0_DIST", TIDE])]
    wrong = k[k.direction_rho > 0]
    has_model = bool(r2 > R2_MIN and k.direction_rho.notna().any() and wrong.empty)
    elig.append(dict(State=st, elevation=elev, R2_orig=round(r2, 3),
                     accuracy=r2 > R2_MIN, direction=wrong.empty, projected=has_model,
                     reason="; ".join(([f"R2 {r2:.3f} <= {R2_MIN}"] if r2 <= R2_MIN else [])
                                      + [f"{p} rho {v:+.3f}" for p, v in
                                         zip(wrong.predictor, wrong.direction_rho)])))

    b_mean = np.mean(base, axis=0)
    total, existing = len(panel), int((panel.obs_Cl > EPA_CL).sum())
    for i, x in enumerate(SCENARIOS[st]):
        col = f"SLR_{x}_DIST"
        inund = ((panel[col] <= 1e-6) & panel[col].notna()) | \
                (panel["Surf_Ele"].notna() & (panel["Surf_Ele"] <= x * FT_M))
        rec = dict(State=st, region=REGION_OF[st], model="yes" if has_model else "no",
                   R2_orig=round(r2, 3) if has_model else np.nan, scenario=f"{x} ft",
                   rise_m=round(x * FT_M, 2), total_wells=total, existing_saline=existing,
                   inundated=int(inund.sum()))
        not_in = ~inund[modelable].values
        p_mean, p_dist = np.mean(scen[x], axis=0), np.mean(dist_only[x], axis=0)
        fresh = not_in & (P["obs_Cl"].values <= EPA_CL)
        f2s = fresh & (p_mean > EPA_CL)
        if has_model:
            ext = P[col].values < d5
            if elev == TIDE:
                ext |= (P[TIDE].values - x * FT_M) < tide_min
            good = not_in & (b_mean > 1e-6)
            ch = 100 * (p_mean[good] - b_mean[good]) / b_mean[good]
            rec.update(case2_extrapolation=int((ext & not_in).sum()),
                       case3_calibrated=int((~ext & not_in).sum()),
                       fresh_to_saline=int(f2s.sum()),
                       fresh_to_saline_distance_only=(int((fresh & (p_dist > EPA_CL)).sum())
                                                      if elev == TIDE else np.nan),
                       median_pct_change=round(np.median(ch), 2),
                       pct_wells_rising=round(100 * np.mean(p_mean[good] > b_mean[good]), 1))
        new = rec["inundated"] + rec.get("fresh_to_saline", 0)
        rec.update(total_newly_at_risk=new, pct_of_wells=round(100 * new / total, 1))
        rows.append(rec)

        w = panel[["SiteNo", "lon", "lat", "obs_Cl"]].assign(
            State=st, scenario_ft=x, has_model=has_model,
            case=np.where(inund, 1, 3), pred_Cl=np.nan, fresh_to_saline=False)
        idx = np.where(modelable)[0]
        if has_model:
            w.loc[w.index[idx], "pred_Cl"] = p_mean
            w.loc[w.index[idx], "fresh_to_saline"] = f2s
            w.loc[w.index[idx[ext & not_in]], "case"] = 2
        wells_out[i].append(w)
        print(f"  {st} +{x} ft  inundated {rec['inundated']:5,}  fresh->saline "
              + (f"{rec['fresh_to_saline']:4d}" if has_model else " n/a")
              + f"  newly at risk {new:5,} ({rec['pct_of_wells']}%)")

T, E = pd.DataFrame(rows), pd.DataFrame(elig)
T.to_csv(out("table7_slr_projection.csv"), index=False)
E.to_csv(out("projection_eligibility.csv"), index=False)
for i, ssp in SSP.items():
    pd.concat(wells_out[i]).to_csv(out(f"figure_wells_{ssp}.csv"), index=False)

# ── totals and regional aggregates ─────────────────────────────────────────────
T["tier"] = np.where(T.scenario.isin(["4 ft", "5 ft"]), "SSP2-4.5", "SSP5-8.5")
reg = []
for (r, tier), g in pd.concat([T, T.assign(region="Total")]).groupby(["region", "tier"]):
    f2s = g.loc[g.model == "yes", "fresh_to_saline"].sum()
    new = g.inundated.sum() + f2s
    reg.append(dict(region=r, tier=tier, wells=g.total_wells.sum(),
                    existing_saline=g.existing_saline.sum(), inundated=g.inundated.sum(),
                    fresh_to_saline=int(f2s), newly_at_risk=int(new),
                    pct_of_wells=round(100 * new / g.total_wells.sum(), 1),
                    pct_inundation=round(100 * g.inundated.sum() / new, 1),
                    pct_of_existing_saline=round(100 * new / g.existing_saline.sum(), 0)))
R = pd.DataFrame(reg)
R.to_csv(out("table7_regional.csv"), index=False)
print("\n" + E.to_string(index=False))
print("\n" + R.to_string(index=False))
