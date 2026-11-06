"""Step 7 - Predictor attribution and response direction (Sections 2.5.4 and 3.3.4, Figure 6).

For each state's projection model (common.projection_features), fitted with five seeds
and tree depth chosen on validation:

  attribution   SHAP values (TreeExplainer, log1p units) over the projection panel: one
                record per well (well medians) for every well carrying the model's
                predictors. A predictor's share is its mean |SHAP| divided by the sum
                over all predictors, averaged over seeds.
  direction     Spearman rho between a predictor's values and its SHAP values; negative
                means predicted chloride falls as the predictor rises. Not reported for
                Aquifer, whose ordinal codes have no physical order.
  observed      Spearman rho between each well's median predictor value and its median
                chloride, to check whether the modelled direction is also in the data;
                for well depth, also the partial correlation controlling for aquifer.

SC and LA are fitted on one record per well (common.WELL_LEVEL). R2_orig is scored on
the held-out test partition. SHAP describes how the fitted models use the predictors,
not causal effects, and correlated predictors may share attribution.

Output: results/shap_importance.csv, results/shap_dependence.csv,
        results/observed_correlations.csv
"""
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.preprocessing import OrdinalEncoder

from common import (DEPTHS, SEEDS5, STATES, TARGET, WELL_LEVEL, load, out,
                    projection_features, r2_orig, split_703)

EXPECT_NEG = {"SLR0_DIST", "Ele_abv_MWL_1y", "Surf_Ele", "Salt_DIST"}
NO_DIRECTION = {"Aquifer"}


def collapse(d, feats):
    """One row per well: medians of the numeric predictors and target, first aquifer."""
    num = [c for c in feats if c != "Aquifer"] + [TARGET]
    g = d.groupby("SiteNo")
    w = g[num].median()
    w["Aquifer"] = g["Aquifer"].first()
    return w.reset_index()


def partial_spearman(x, y, groups):
    """Spearman correlation of x and y after removing group means of their ranks."""
    rx, ry = x.rank(), y.rank()
    return spearmanr(rx - rx.groupby(groups).transform("mean"),
                     ry - ry.groupby(groups).transform("mean"))


df = load()
imp, dep, obs = [], [], []
for st in STATES:
    feats = projection_features(st)
    sub = df[df["State"] == st]
    S = sub[sub[feats + [TARGET]].notna().all(axis=1)]
    tr, va, te = split_703(S)
    if st in WELL_LEVEL:
        tr, va, te = (collapse(x, feats) for x in (tr, va, te))
    tr, va, te = tr.copy(), va.copy(), te.copy()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    tr["Aquifer"] = enc.fit_transform(tr[["Aquifer"]]).ravel().astype(float)
    for d in (va, te):
        d["Aquifer"] = enc.transform(d[["Aquifer"]]).ravel().astype(float)

    # attribution domain: one record per well, every well the model can score
    panel = (sub.groupby("SiteNo")
                .agg(**{c: (c, "median") for c in feats if c != "Aquifer"},
                     Aquifer=("Aquifer", "first")).reset_index())
    panel = panel[panel[feats].notna().all(axis=1)].copy()
    panel["Aquifer"] = enc.transform(panel[["Aquifer"]]).ravel().astype(float)
    X = panel[feats].values

    absmean, direction, r2s, stack = [], [], [], []
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
        sv = shap.TreeExplainer(m).shap_values(X)
        stack.append(sv)
        absmean.append(np.abs(sv).mean(axis=0))
        direction.append([np.nan if f in NO_DIRECTION else spearmanr(X[:, j], sv[:, j]).statistic
                          for j, f in enumerate(feats)])
    A, D = np.array(absmean), np.array(direction)
    for j, f in enumerate(feats):
        rho = np.nan if f in NO_DIRECTION else D[:, j].mean()
        imp.append(dict(State=st, spec="+".join(feats), predictor=f,
                        n_test=len(te), wells_test=te["SiteNo"].nunique(),
                        n_attribution=len(panel), R2_test_mean=round(np.mean(r2s), 4),
                        mean_abs_shap=round(A[:, j].mean(), 4),
                        shap_share=round(A[:, j].mean() / A.mean(axis=0).sum(), 4),
                        direction_rho=round(rho, 3) if pd.notna(rho) else np.nan,
                        direction_rho_sd=(round(D[:, j].std(ddof=1), 3)
                                          if f not in NO_DIRECTION else np.nan),
                        expected="negative" if f in EXPECT_NEG else
                                 "n/a" if f in NO_DIRECTION else "none stated"))

    # binned response to the elevation term and to the predictor a scenario changes
    SV = np.mean(np.stack(stack), axis=0)
    for f in (feats[0], "SLR0_DIST"):
        j = feats.index(f)
        x = X[:, j]
        qs = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
        if len(qs) < 3:
            continue
        b = np.clip(np.digitize(x, qs[1:-1]), 0, len(qs) - 2)
        for k in range(len(qs) - 1):
            msk = b == k
            if msk.sum() >= 5:
                dep.append(dict(State=st, predictor=f, bin=k + 1,
                                x_median=round(np.median(x[msk]), 3), n=int(msk.sum()),
                                mean_shap=round(SV[msk, j].mean(), 4)))

    # observed well-median associations
    w = sub.groupby("SiteNo").agg(**{c: (c, "median") for c in feats if c != "Aquifer"},
                                  Aquifer=("Aquifer", "first"), Cl=("Chloride", "median"))
    for f in [c for c in feats if c != "Aquifer"]:
        k = w[[f, "Cl", "Aquifer"]].dropna()
        r = spearmanr(k[f], k["Cl"])
        rec = dict(State=st, predictor=f, wells=len(k), rho=round(r.statistic, 3),
                   p=r.pvalue)
        if f == "WellDepthM":
            pr = partial_spearman(k[f], k["Cl"], k["Aquifer"])
            rec.update(partial_rho_aquifer=round(pr.statistic, 3), partial_p=pr.pvalue)
        obs.append(rec)
    print(f"  {st}  R2_orig={np.mean(r2s):.3f}  " + "  ".join(
        f"{f[:10]}={A[:, j].mean() / A.mean(axis=0).sum():.0%}"
        + ("" if f in NO_DIRECTION else f"({D[:, j].mean():+.2f})")
        for j, f in enumerate(feats)))

I, P, O = pd.DataFrame(imp), pd.DataFrame(dep), pd.DataFrame(obs)
I.to_csv(out("shap_importance.csv"), index=False)
P.to_csv(out("shap_dependence.csv"), index=False)
O.to_csv(out("observed_correlations.csv"), index=False)

elev = I[I.predictor.isin(["Surf_Ele", "Ele_abv_MWL_1y"])]
print(f"\nelevation: rho {elev.direction_rho.min():+.3f} to {elev.direction_rho.max():+.3f}, "
      f"share {elev.shap_share.min():.0%} to {elev.shap_share.max():.0%}")
oe = O[O.predictor.isin(["Surf_Ele", "Ele_abv_MWL_1y"])]
print(f"observed elevation-chloride rho {oe.rho.min():+.3f} to {oe.rho.max():+.3f}")
bad = I[(I.predictor == "SLR0_DIST") & (I.direction_rho > 0)]
print(f"SLR0_DIST responds positively in: {', '.join(bad.State) or 'none'}")
