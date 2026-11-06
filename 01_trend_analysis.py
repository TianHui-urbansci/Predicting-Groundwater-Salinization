"""Step 1 - Long-term chloride trends (Sections 2.2 and 3.1, Figure 4).

Mann-Kendall test and Theil-Sen slope on ANNUAL MEAN chloride at each well, for wells
with at least five years of record (3,752 wells in the 12 analysed states).

  S        Mann-Kendall statistic with tie-corrected variance and continuity correction
  p        two-sided normal approximation
  tau      S / (n(n-1)/2)
  slope    Theil-Sen slope in mg/L per year; pairs are spaced by calendar year, not by
           index, because annual series have gaps
  trend    increasing / decreasing when p < 0.05, otherwise no trend

Equivalent to pymannkendall.original_test.

Output: results/mk_results.csv, results/mk_state_summary.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from common import STATES, data_path, in_population, out

MIN_YEARS = 5


def mann_kendall(x, yr):
    n = len(x)
    s = int(sum(np.sign(x[i + 1:] - x[i]).sum() for i in range(n - 1)))
    _, cnt = np.unique(x, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - (cnt * (cnt - 1) * (2 * cnt + 5)).sum()) / 18.0
    z = (s - 1) / np.sqrt(var) if s > 0 else (s + 1) / np.sqrt(var) if s < 0 else 0.0
    p = 2 * (1 - norm.cdf(abs(z)))
    tau = s / (n * (n - 1) / 2)
    sl = [(x[j] - x[i]) / (yr[j] - yr[i]) for i in range(n - 1) for j in range(i + 1, n)]
    return s, tau, p, float(np.median(sl))


df = pd.read_csv(data_path(), dtype={"SiteNo": str},
                 usecols=["SiteNo", "State", "Date", "Chloride", "Aquifer", "SLR0_DIST"])
df = df[in_population(df) & df["State"].isin(STATES)]
df["year"] = pd.to_datetime(df["Date"]).dt.year
print(f"{len(df):,} observations, {df['SiteNo'].nunique():,} wells")

ann = (df.groupby(["SiteNo", "State", "year"], as_index=False)["Chloride"].mean()
         .sort_values(["SiteNo", "year"]))
stat = df.groupby("SiteNo")["Chloride"].agg(cl_mean="mean", cl_median="median")

rows = []
for (site, st), g in ann.groupby(["SiteNo", "State"], sort=False):
    if len(g) < MIN_YEARS:
        continue
    s, tau, p, slope = mann_kendall(g["Chloride"].to_numpy(float), g["year"].to_numpy(float))
    rows.append(dict(SiteNo=site, State=st, n_years=len(g),
                     year_min=int(g["year"].min()), year_max=int(g["year"].max()),
                     tau=tau, p_value=p, slope=slope,
                     trend=("increasing" if p < 0.05 and s > 0 else
                            "decreasing" if p < 0.05 and s < 0 else "no trend"),
                     cl_mean=stat.loc[site, "cl_mean"],
                     cl_median=stat.loc[site, "cl_median"]))
R = pd.DataFrame(rows)
R.to_csv(out("mk_results.csv"), index=False)

# State summary, as reported in Section 3.1
g = R.groupby("State")
S = pd.DataFrame({
    "wells": g.size(),
    "pct_positive_slope": g["slope"].apply(lambda v: (v > 0).mean() * 100),
    "pct_sig_increasing": g["trend"].apply(lambda v: (v == "increasing").mean() * 100),
    "pct_sig_decreasing": g["trend"].apply(lambda v: (v == "decreasing").mean() * 100),
    "median_slope_sig_increasing": R[R["trend"] == "increasing"].groupby("State")["slope"].median(),
    "pct_wells_above_250": g["cl_median"].apply(lambda v: (v > 250).mean() * 100),
}).reindex(STATES).round(1)
S.to_csv(out("mk_state_summary.csv"))

n = len(R)
print(f"\n{n:,} wells with >= {MIN_YEARS} annual values")
print(f"  positive Sen slope      {(R['slope'] > 0).sum():,} ({(R['slope'] > 0).mean():.1%})")
print(f"  significant increase    {(R['trend'] == 'increasing').sum():,} "
      f"({(R['trend'] == 'increasing').mean():.1%})")
print(f"\n{S.to_string()}")
