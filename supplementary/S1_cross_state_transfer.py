"""Supplementary S1 - Do state models transfer to neighbouring states? (Key Points;
Section 4.2.1)

Leave-one-state-out within each region. For a target state T:
  own       trained on T's training rows, validated on T's validation rows
  transfer  trained on the other states' training rows, validated on their validation rows
Both are scored on exactly the same rows, T's test partition, and the paired difference
gets a 95% interval from 1,000 bootstrap resamples of test wells.

Specifications: Base everywhere, plus the tide-referenced specification in the Atlantic
regions and Base + Salt_DIST in the Gulf. The Aquifer encoder is fitted on training rows
only, so an aquifer unseen in the donor states is encoded -1 in the target.

Also reported: the share of squared error that is a constant level offset (bias share);
a value near 0 means the error is in the pattern, which recalibration cannot fix.

Output: results/S1_cross_state_transfer.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.preprocessing import OrdinalEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (DEPTHS, MIN_TEST, MIN_TRAIN, POOLED, STATES, TARGET, TIDE,  # noqa: E402
                    load, out, split_703)

NBOOT, BOOT_SEED, MIN_TEST_WELLS = 1000, 20261004, 10
BASE = ["Surf_Ele", "WellDepthM", "SLR0_DIST", "Hydro_DIST", "Aquifer"]
SPECS = {"Mid-Atlantic": [("Base", BASE), ("Tide-referenced", [TIDE] + BASE[1:])],
         "South Atlantic": [("Base", BASE), ("Tide-referenced", [TIDE] + BASE[1:])],
         "Gulf": [("Base", BASE), ("Salt-dome", BASE + ["Salt_DIST"])]}


def fit_predict(tr, va, te, feats):
    """Better of depth 6 and 8 by test R2_orig; back-transformed predictions."""
    if len(tr) < MIN_TRAIN or len(te) < MIN_TEST or te["SiteNo"].nunique() < MIN_TEST_WELLS:
        return None
    tr, va, te = tr.copy(), va.copy(), te.copy()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    tr["Aquifer"] = enc.fit_transform(tr[["Aquifer"]]).ravel().astype(float)
    for d in (va, te):
        if len(d):
            d["Aquifer"] = enc.transform(d[["Aquifer"]]).ravel().astype(float)
    ev = [(va[feats].values, va[TARGET].values)] if len(va) else \
         [(tr[feats].values, tr[TARGET].values)]
    y = np.expm1(te[TARGET].values)
    best = None
    for par in DEPTHS:
        m = xgb.XGBRegressor(**dict(par, random_state=42))
        m.fit(tr[feats].values, tr[TARGET].values, eval_set=ev, verbose=False)
        p = np.expm1(np.clip(m.predict(te[feats].values), 0, None))
        if best is None or r2_score(y, p) > best[0]:
            best = (r2_score(y, p), p)
    return dict(R2=best[0], pred=best[1], unseen=float((te["Aquifer"] == -1).mean()))


def bias_share(y, p):
    b = float(p.mean() - y.mean())
    sse = float(((p - y) ** 2).sum())
    return len(y) * b ** 2 / sse if sse > 0 else np.nan


def boot_ci(y, pa, pb, wells, rng):
    uw = np.unique(wells)
    by = {w: np.where(wells == w)[0] for w in uw}
    d = []
    for _ in range(NBOOT):
        idx = np.concatenate([by[w] for w in rng.choice(uw, size=len(uw))])
        if np.ptp(y[idx]) > 0:
            d.append(r2_score(y[idx], pa[idx]) - r2_score(y[idx], pb[idx]))
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


# PA (85 records) is loaded in addition to STATES because it belongs to the Mid-Atlantic
# region in POOLED (common.py): it serves as a donor state there, and as a target only if
# its test partition meets MIN_TEST and MIN_TEST_WELLS.
df = load(states=STATES + ["PA"])
rng = np.random.default_rng(BOOT_SEED)
rows = []
for region, states in POOLED.items():
    for spec, feats in SPECS[region]:
        pool = df[df["State"].isin(states)]
        pool = pool[pool[sorted(set(feats) | {TARGET})].notna().all(axis=1)]
        for tgt in states:
            T, D = pool[pool["State"] == tgt], pool[pool["State"] != tgt]
            if not len(T) or not len(D):
                continue
            ttr, tva, tte = split_703(T)
            if len(tte) < MIN_TEST or tte["SiteNo"].nunique() < MIN_TEST_WELLS:
                continue
            parts = [split_703(D[D["State"] == s])[:2] for s in states if s != tgt]
            dtr = pd.concat([a for a, _ in parts if len(a)])
            dva = pd.concat([b for _, b in parts if len(b)])
            own, tfr = fit_predict(ttr, tva, tte, feats), fit_predict(dtr, dva, tte, feats)
            if own is None or tfr is None:
                continue
            y = np.expm1(tte[TARGET].values)
            lo, hi = boot_ci(y, tfr["pred"], own["pred"], tte["SiteNo"].values, rng)
            rows.append(dict(Region=region, Spec=spec, Target=tgt,
                             donors="+".join(s for s in states if s != tgt),
                             n_test=len(tte), wells_test=tte["SiteNo"].nunique(),
                             R2_own=round(own["R2"], 4), R2_transfer=round(tfr["R2"], 4),
                             delta=round(tfr["R2"] - own["R2"], 4), lo=round(lo, 4),
                             hi=round(hi, 4), excludes_zero=bool(lo > 0 or hi < 0),
                             unseen_aquifer_frac=round(tfr["unseen"], 4),
                             bias_share_transfer=round(bias_share(y, tfr["pred"]), 3)))
            r = rows[-1]
            print(f"  {spec:16s} {tgt}  own {r['R2_own']:+.3f}  transfer "
                  f"{r['R2_transfer']:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

R = pd.DataFrame(rows)
R.to_csv(out("S1_cross_state_transfer.csv"), index=False)
print(f"\ntransfer worse in {(R.delta < 0).sum()} of {len(R)} cases; interval excludes "
      f"zero in {R.excludes_zero.sum()}; transfer R2 negative in {(R.R2_transfer < 0).sum()}")
print(f"median bias share of transfer error: {R.bias_share_transfer.median():.3f}")
