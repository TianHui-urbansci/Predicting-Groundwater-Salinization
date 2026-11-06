"""Step 3 - Base and alternative specifications by state and region (Section 3.3.1, Table 3).

  Base  Surf_Ele, WellDepthM, SLR0_DIST, Aquifer, Hydro_DIST
  GWE   Base + WaterEle_m
  PREC  Base + daily_precip, 7d_Before_prec, 1m_Before_prec
  SALT  Base + Salt_DIST                                      Gulf scopes only
  TIDE  Ele_abv_MWL_1y in place of Surf_Ele

Each specification is fitted on its own complete cases (rows with every one of its
predictors), with a chronological 70/15/15 split per well. XGBoost is fitted at depth 6
and depth 8 and the better test R2_orig is reported. Because each specification uses a
different subset of observations, these numbers mix predictor and sample effects; step 4
separates the two.

Output: results/table3_state_models.csv  (long), results/table3_formatted.csv (n / R2_orig)
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from common import (DEPTHS, GULF, MIN_TEST, MIN_TRAIN, SCOPES, SPECS, STATES,
                    TARGET, complete, encode, load, out, r2_log, r2_orig, scope_rows,
                    split_703)


def fit(tr, va, te, feats):
    """Best of depth 6 and 8 by test R2_orig; returns (R2_orig, R2_log)."""
    if len(tr) < MIN_TRAIN or len(te) < MIN_TEST:
        return None
    tr, va, te = encode(tr, va, te, feats)
    Xtr, ytr = tr[feats].values, tr[TARGET].values
    ev = [(va[feats].values, va[TARGET].values)] if len(va) else [(Xtr, ytr)]
    best = None
    for par in DEPTHS:
        m = xgb.XGBRegressor(**dict(par, random_state=42))
        m.fit(Xtr, ytr, eval_set=ev, verbose=False)
        p = m.predict(te[feats].values)
        r = (r2_orig(te[TARGET].values, p), r2_log(te[TARGET].values, p))
        if best is None or r[0] > best[0]:
            best = r
    return best


# PA is not modelled on its own (85 records) but is pooled into the Mid-Atlantic scope
# (see POOLED in common.py), so it is loaded here in addition to STATES.
df = load(states=STATES + ["PA"])
rows = []
for sc in SCOPES:
    sub = scope_rows(df, sc)
    gulf = sc in GULF or sc == "Gulf"
    for spec, feats in SPECS.items():
        if spec == "SALT" and not gulf:
            continue
        cc = complete(sub, feats)
        r = fit(*split_703(cc), feats) if len(cc) else None
        rows.append(dict(Scope=sc, Spec=spec, n=len(cc), wells=cc["SiteNo"].nunique(),
                         R2_orig=round(r[0], 4) if r else np.nan,
                         R2_log=round(r[1], 4) if r else np.nan))
        print(f"  {sc:15s} {spec:5s} n={len(cc):7,} "
              + (f"R2_orig={r[0]:7.3f} R2_log={r[1]:6.3f}" if r else "not estimable"))

R = pd.DataFrame(rows)
R.to_csv(out("table3_state_models.csv"), index=False)

cell = R.assign(cell=[f"{n:,} / {r:.3f}" if pd.notna(r) else f"{n:,} / —"
                      for n, r in zip(R["n"], R["R2_orig"])])
T = cell.pivot(index="Scope", columns="Spec", values="cell").reindex(SCOPES)[list(SPECS)]
T.to_csv(out("table3_formatted.csv"))
print("\n" + T.fillna("—").to_string())
