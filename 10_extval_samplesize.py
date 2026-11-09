# -*- coding: utf-8 -*-
"""Simulation-based external-validation sample size calculation (Snell 2021 /
Riley 2024 part 3 framework, refs 20-22). Bootstrap subsamples of the pooled
prospective external case-mix; logistic recalibration refitted per replicate.
Targets: SE(O/E) <= 0.10 and SE(calibration slope) <= 0.10."""
import numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(20261105)
ext = pd.read_csv('external_analysis.csv')
mdl = joblib.load('models/por_models.joblib')
FEATS = ['age', 'bmi', 'amh', 'base_fsh']
X = ext[FEATS].astype(float)
y = ext['por'].astype(int).values
p = mdl['random_forest'].predict_proba(X)[:, 1]
lp = np.log(p / (1 - p)).reshape(-1, 1)
n_all, ev_all = len(y), y.sum()
oe_all = y.sum() / p.sum()
print(f"pooled external n={n_all}, events={ev_all} ({ev_all/n_all:.1%}), pooled O/E={oe_all:.2f}")

def one_boot(idx):
    yb, lpb = y[idx], lp[idx]
    if yb.sum() < 5 or (1 - yb).sum() < 5:
        return None
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(lpb, yb)                       # logistic recalibration
    slope = float(lr.coef_[0][0])
    oe = yb.sum() / p[idx].sum()          # observed / expected events
    return slope, oe

rows = []
for n in [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1285]:
    slopes, oes, evs = [], [], []
    for r in range(1000):
        idx = rng.choice(n_all, size=n, replace=True)
        res = one_boot(idx)
        if res is None: continue
        slopes.append(res[0]); oes.append(res[1]); evs.append(y[idx].sum())
    rows.append(dict(n=n, mean_events=np.mean(evs), se_slope=np.std(slopes, ddof=1),
                     se_oe=np.std(oes, ddof=1)))
    print(f"n={n:5d}  mean events={np.mean(evs):6.1f}  "
          f"SE(slope)={np.std(slopes, ddof=1):.3f}  SE(O/E)={np.std(oes, ddof=1):.3f}")

df = pd.DataFrame(rows)
df.to_csv('sensitivity/results/extval_samplesize.csv', index=False)
ok = df[df.se_oe <= 0.10]
if len(ok):
    print(f"\nminimum n with SE(O/E)<=0.10: {int(ok.n.iloc[0])} (mean events {ok.mean_events.iloc[0]:.0f})")
ok2 = df[df.se_slope <= 0.10]
if len(ok2):
    print(f"minimum n with SE(slope)<=0.10: {int(ok2.n.iloc[0])} (mean events {ok2.mean_events.iloc[0]:.0f})")
else:
    print(f"SE(slope)<=0.10 not reached on grid; best {df.se_slope.min():.3f} at n={int(df.loc[df.se_slope.idxmax(),'n'])}... "
          f"(SE at n=1285: {df[df.n==1285].se_slope.iloc[0]:.3f})")
sl, oe = [], []
for r in range(2000):
    idx = rng.choice(n_all, size=n_all, replace=True)
    res = one_boot(idx)
    if res: sl.append(res[0]); oe.append(res[1])
print(f"achieved at pooled n={n_all}: SE(slope)={np.std(sl, ddof=1):.3f}, SE(O/E)={np.std(oe, ddof=1):.3f}")
