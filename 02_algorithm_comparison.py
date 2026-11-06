"""Step 2 - Algorithm selection (Sections 2.4 and 3.3.1).

Random Forest, XGBoost, Extra Trees and an ANN (multilayer perceptron) are fitted on the
Base predictors, first pooled across the 12 states and then within each state, using the
same chronological 70/15/15 split. XGBoost was adopted because it gave the highest test
R2_orig in most scopes, i.e. it best reproduced high chloride after back-transformation.

The ANN uses standardised inputs; the tree models do not need scaling.

Output: results/algorithm_comparison.csv
"""
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from common import (BASE, MIN_TEST, MIN_TRAIN, PAR, STATES, TARGET, complete, encode,
                    load, out, r2_log, r2_orig, split_703)


def rf():
    return RandomForestRegressor(n_estimators=500, min_samples_leaf=5, n_jobs=-1,
                                 random_state=42)


def et():
    return ExtraTreesRegressor(n_estimators=500, min_samples_leaf=5, n_jobs=-1,
                               random_state=42)


def ann():
    return MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu",
                        solver="adam", learning_rate_init=0.001, max_iter=500,
                        early_stopping=True, validation_fraction=0.1,
                        n_iter_no_change=20, random_state=42)


ALGOS = {"RandomForest": rf, "XGBoost": None, "ExtraTrees": et, "ANN": ann}


def run(scope, d):
    d = complete(d, BASE)
    tr, va, te = encode(*split_703(d), BASE)
    if len(tr) < MIN_TRAIN or len(te) < MIN_TEST:
        return []
    Xtr, ytr, Xte, yte = tr[BASE].values, tr[TARGET].values, te[BASE].values, te[TARGET].values
    rows = []
    for name, make in ALGOS.items():
        t0 = time.time()
        if name == "XGBoost":
            m = xgb.XGBRegressor(**dict(PAR, max_depth=6, min_child_weight=5,
                                        random_state=42))
            ev = [(va[BASE].values, va[TARGET].values)] if len(va) else [(Xtr, ytr)]
            m.fit(Xtr, ytr, eval_set=ev, verbose=False)
            p = m.predict(Xte)
        elif name == "ANN":
            sc = StandardScaler().fit(Xtr)
            p = make().fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))
        else:
            p = make().fit(Xtr, ytr).predict(Xte)
        rmse = np.sqrt(mean_squared_error(np.expm1(yte), np.expm1(np.clip(p, 0, None))))
        rows.append(dict(Scope=scope, Algorithm=name, n_train=len(tr), n_test=len(te),
                         R2_log=round(r2_log(yte, p), 4), R2_orig=round(r2_orig(yte, p), 4),
                         RMSE_orig=round(rmse, 2)))
        print(f"  {scope:8s} {name:13s} R2_log={rows[-1]['R2_log']:7.3f} "
              f"R2_orig={rows[-1]['R2_orig']:7.3f}  [{time.time() - t0:.0f}s]")
    return rows


df = load()
rows = run("Pooled", df)
for st in STATES:
    rows += run(st, df[df["State"] == st])

R = pd.DataFrame(rows)
R.to_csv(out("algorithm_comparison.csv"), index=False)

wins = (R[R["Scope"] != "Pooled"].sort_values("R2_orig", ascending=False)
          .groupby("Scope").head(1)["Algorithm"].value_counts())
print("\npooled:\n" + R[R["Scope"] == "Pooled"][["Algorithm", "R2_log", "R2_orig"]]
      .to_string(index=False))
print(f"\nhighest R2_orig by state:\n{wins.to_string()}")
