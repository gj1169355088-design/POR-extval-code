#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MI sensitivity analysis (reviewer point 1: AMH selective ordering / complete-case bias)
======================================================================================
Population : first eligible stimulated cycle per patient in the Shanghai EHR
             (28,420 eligible cycles -> 16,311 women, missing predictors RETAINED;
              complete-case primary analysis used 11,749 of them).
Imputation : MICE-style iterative Bayesian-ridge imputer, M = 10, posterior draws,
             imputation model includes the OUTCOME (por), protocol-class one-hots
             and calendar year as auxiliaries; imputed values clipped to plausible ranges.
Models     : identical to primary -- RF(150, depth 6, min leaf 10, seed 42) and
             LR L2 (C=1) on z-scored features (fixed scaling from the primary
             complete-case derivation distribution so coefficients pool on one scale).
Arms       : (A) MI, all 16,311 women;  (B) complete-case within the same population.
Evaluation : internal 80:20 hold-out (seed 42, stratified) + transported external
             (Zunyi / Jining, frozen per-imputation models, pooled via Rubin's rules).
Outputs    : sensitivity/results/mi_results.csv + mi_summary.md
"""
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
import statsmodels.api as sm

warnings.filterwarnings("ignore")
BASE = "."
FEATS = ["age", "bmi", "amh", "base_fsh"]
M = 10
SEED = 20261105


def num(s):
    try:
        return float(s)
    except Exception:
        return None


def protocol_group(sr):
    sr = str(sr)
    if "拮抗剂" in sr:
        return "antagonist"
    if any(k in sr for k in ["长方案", "短方案", "卵泡期长", "晚卵泡期", "黄体期促排", "超长"]):
        return "agonist"
    if "微刺激" in sr:
        return "mild"
    if "PPOS" in sr or "孕激素" in sr:
        return "ppos"
    return "other"


def calib_int_slope(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(lp, y)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def oe_ratio(y, p):
    exp_events = p.sum()
    return y.sum() / exp_events if exp_events > 0 else np.nan


def auc_boot(y, p, n_boot=500, seed=7):
    obs = roc_auc_score(y, p)
    rng = np.random.RandomState(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(roc_auc_score(yy, pp))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return obs, lo, hi


# ---------------- load ----------------
mi_pop = pd.read_csv(f"{BASE}/shanghai_eligible_first_withmissing.csv", encoding="utf-8-sig")
prim = pd.read_csv(f"{BASE}/shanghai_analysis.csv", encoding="utf-8-sig")
ext = pd.read_csv(f"{BASE}/external_analysis.csv", encoding="utf-8-sig")
ext = ext.rename(columns={"base_fsh": "base_fsh"}) if "base_fsh" in ext.columns else ext

# fixed z-scaling from primary complete-case derivation (coefficient pooling on one scale)
SCALE = {c: (float(prim[c].mean()), float(prim[c].std())) for c in FEATS}

# auxiliary design matrix
pg = mi_pop["stim_regimen"].map(protocol_group)
aux = pd.get_dummies(pg, prefix="pg").astype(float)
aux["year"] = pd.to_datetime(mi_pop["date"]).dt.year.astype(float)
aux["por"] = mi_pop["por"].astype(float)

imp_cols = FEATS + list(aux.columns)
Xraw = pd.concat([mi_pop[FEATS].astype(float), aux], axis=1)
y_all = mi_pop["por"].values.astype(int)

print(f"MI population: n={len(mi_pop)}, POR={y_all.mean():.3f}")
print("missing:", mi_pop[FEATS].isna().mean().round(3).to_dict())


def run_arm(X_imp, y, label):
    """Fit per-dataset; returns per-fit metrics dict list + LR coefs."""
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    res = {"internal": [], "external": [], "coef": [], "coef_se": []}
    rf = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=10,
                                random_state=42, n_jobs=-1)
    Xs = X_imp.copy()
    for c in FEATS:
        mu, sd = SCALE[c]
        Xs[c] = (X_imp[c] - mu) / sd
    rf.fit(Xs[FEATS].iloc[tr], y[tr])
    lr = make_lr().fit(Xs[FEATS].iloc[tr], y[tr])
    # LR coefficients (standardised scale) with proper within-imputation SEs (GLM)
    Xtr = Xs[FEATS].iloc[tr].copy()
    Xtr.insert(0, "const", 1.0)
    glm = sm.GLM(y[tr], Xtr.values, family=sm.families.Binomial()).fit()
    res["coef"].append(glm.params.copy())       # [const, 4 feats]
    res["coef_se"].append(glm.bse.copy())
    res["feat_names"] = FEATS
    # internal hold-out
    p_te = rf.predict_proba(Xs[FEATS].iloc[te])[:, 1]
    y_te = y[te]
    auc, lo, hi = auc_boot(y_te, p_te)
    ci, sl = calib_int_slope(y_te, p_te)
    res["internal"].append(dict(auc=auc, lo=lo, hi=hi, brier=brier_score_loss(y_te, p_te),
                                ci=ci, slope=sl))
    # external transported
    for center in ext["center"].unique():
        sub = ext[ext["center"] == center].reset_index(drop=True)
        ye = sub["por"].values.astype(int)
        Xe = sub[FEATS].astype(float).copy()
        for c in FEATS:
            mu, sd = SCALE[c]
            Xe[c] = (Xe[c] - mu) / sd
        pe = rf.predict_proba(Xe[FEATS])[:, 1]
        auc_e, lo_e, hi_e = auc_boot(ye, pe)
        ci_e, sl_e = calib_int_slope(ye, pe)
        res["external"].append(dict(center=center, auc=auc_e, lo=lo_e, hi=hi_e,
                                    brier=brier_score_loss(ye, pe),
                                    ci=ci_e, slope=sl_e, oe=oe_ratio(ye, pe)))
    return res


def make_lr():
    return LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000)


def rubin(est_list, se_list):
    """Rubin pooling: est = mean; total var = mean(within) + (1+1/M)*between."""
    est = np.mean(est_list)
    W = np.mean(np.square(se_list))
    B = np.var(est_list, ddof=1)
    Tv = W + (1 + 1 / len(est_list)) * B
    return est, np.sqrt(Tv)


# ---------------- arm B: complete-case within MI population ----------------
cc_mask = mi_pop[FEATS].notna().all(axis=1)
print(f"\nArm B complete-case within population: n={cc_mask.sum()}")

# ---------------- arm A: MICE, M imputations ----------------
rows_out = []
imp_internal, imp_ext, imp_coef, imp_coef_se, imp_auc_se = [], [], [], [], []
for m in range(M):
    it = IterativeImputer(estimator=None, max_iter=10, sample_posterior=True,
                          random_state=SEED + m, min_value=-1e6, max_value=1e6)
    Xi = pd.DataFrame(it.fit_transform(Xraw), columns=imp_cols)
    # clip imputed predictors to plausible ranges
    Xi["bmi"] = Xi["bmi"].clip(14, 50)
    Xi["amh"] = Xi["amh"].clip(0.01, 25)
    Xi["base_fsh"] = Xi["base_fsh"].clip(0.1, 40)
    Xi["age"] = Xi["age"].clip(18, 50)
    y = y_all
    res = run_arm(Xi, y, "MI")
    imp_internal.append(res["internal"][0])
    imp_ext.append(res["external"])
    imp_coef.append(res["coef"][0])
    imp_coef_se.append(res["coef_se"][0])
    # within-imputation AUC variance from bootstrap CI width
    it_auc = res["internal"][0]["auc"]
    it_se = (res["internal"][0]["hi"] - res["internal"][0]["lo"]) / 3.92
    imp_auc_se.append(it_se)
    print(f"  imputation {m+1}/{M}: internal AUC {it_auc:.3f} | "
          f"Zunyi {res['external'][0]['auc']:.3f} OE {res['external'][0]['oe']:.2f} | "
          f"Jining {res['external'][1]['auc']:.3f} OE {res['external'][1]['oe']:.2f}")

# ---- pool MI results (Rubin) ----
auc_est, auc_se = rubin([d["auc"] for d in imp_internal], imp_auc_se)
brier_m = np.mean([d["brier"] for d in imp_internal])
ci_m = np.mean([d["ci"] for d in imp_internal])
sl_m = np.mean([d["slope"] for d in imp_internal])
sl_sd = np.std([d["slope"] for d in imp_internal], ddof=1)

coefs = np.vstack(imp_coef)              # M x (1+4): [const, 4 feats], fixed standardised scale
coef_se = np.vstack(imp_coef_se)          # M x (1+4) within-imputation SEs
Qbar = coefs.mean(axis=0)
Ubar = np.square(coef_se).mean(axis=0)
B = coefs.var(axis=0, ddof=1)
T = Ubar + (1 + 1 / M) * B
or_mean = np.exp(Qbar[1:])
or_lo = np.exp(Qbar[1:] - 1.96 * np.sqrt(T[1:]))
or_hi = np.exp(Qbar[1:] + 1.96 * np.sqrt(T[1:]))
or_se = np.sqrt(T[1:])

ext_pool = {}
for k, center in enumerate(ext["center"].unique()):
    aucs = [row[k]["auc"] for row in imp_ext]
    ses = [(row[k]["hi"] - row[k]["lo"]) / 3.92 for row in imp_ext]
    est, se = rubin(aucs, ses)
    ext_pool[center] = dict(
        auc=est, se=se,
        oe_mean=np.mean([row[k]["oe"] for row in imp_ext]),
        oe_sd=np.std([row[k]["oe"] for row in imp_ext], ddof=1),
        slope_mean=np.mean([row[k]["slope"] for row in imp_ext]),
        ci_mean=np.mean([row[k]["ci"] for row in imp_ext]),
        brier_mean=np.mean([row[k]["brier"] for row in imp_ext]))

# ---- complete-case arm within population (single fit) ----
cc_run = run_arm(mi_pop.loc[cc_mask, FEATS].reset_index(drop=True), y_all[cc_mask.values], "CC")
cc_res = cc_run
cc_internal = cc_res["internal"][0]
cc_glm_se = cc_res["coef_se"][0]

# ---------------- report ----------------
prim_ors = dict(zip(FEATS, [None] * 4))
try:
    lr_ref = pd.read_csv(f"{BASE}/tables/lr_ors.csv")
    print("\nprimary OR reference:")
    print(lr_ref.to_string())
except Exception:
    pass

lines = ["# MI sensitivity analysis (AMH selective-ordering / complete-case bias)", "",
         f"## Population", f"- MI cohort: first eligible stimulated cycle per patient, n = {len(mi_pop)} "
         f"(POR {y_all.mean():.1%}); complete-case primary used 11,749 of these women.",
         f"- Missingness: " + ", ".join(f"{c} {mi_pop[c].isna().mean():.1%}" for c in ["bmi", "amh", "base_fsh"]),
         "", "## Method",
         f"- MICE-style iterative Bayesian-ridge imputation, M = {M}, posterior draws; "
         "imputation model includes the outcome, protocol class one-hots and calendar year; "
         "imputed values clipped to plausible ranges.",
         "- Models identical to primary (RF 150/6/10; LR L2 C=1 on z-scored features with "
         "fixed primary-derivation scaling). 80:20 split (seed 42).",
         "", "## Results (Rubin-pooled across M=10)", "",
         "### Internal hold-out",
         f"- MI: AUC {auc_est:.3f} (Rubin SE {auc_se:.3f}), Brier {brier_m:.3f}, "
         f"calibration intercept {ci_m:.2f}, slope {sl_m:.2f} (SD {sl_sd:.2f})",
         f"- Complete-case within population (n={int(cc_mask.sum())}): "
         f"AUC {cc_internal['auc']:.3f}, Brier {cc_internal['brier']:.3f}, "
         f"calibration intercept {cc_internal['ci']:.2f}, slope {cc_internal['slope']:.2f}",
         "", "### Transported external (frozen MI model)"]
for center, d in ext_pool.items():
    lines.append(f"- {center}: pooled AUC {d['auc']:.3f} (Rubin SE {d['se']:.3f}), "
                 f"O/E {d['oe_mean']:.2f} (SD {d['oe_sd']:.2f}), "
                 f"calibration intercept {d['ci_mean']:.2f}, slope {d['slope_mean']:.2f}, "
                 f"Brier {d['brier_mean']:.4f}")
cc_ext = {r["center"]: r for r in cc_res["external"]}
lines.append("")
lines.append("### Complete-case arm, transported external")
for center, r in cc_ext.items():
    lines.append(f"- {center}: AUC {r['auc']:.3f}, O/E {r['oe']:.2f}, "
                 f"calibration intercept {r['ci']:.2f}, slope {r['slope']:.2f}")

lines += ["", "### LR odds ratios per SD (standardised; Rubin-pooled across M=10)"]
for c, o, lo_, hi_ in zip(FEATS, or_mean, or_lo, or_hi):
    lines.append(f"- {c}: OR {o:.2f} ({lo_:.2f}-{hi_:.2f})")
cc_ors = np.exp(cc_res["coef"][0][1:])
cc_lo = np.exp(cc_res["coef"][0][1:] - 1.96 * cc_glm_se[1:])
cc_hi = np.exp(cc_res["coef"][0][1:] + 1.96 * cc_glm_se[1:])
lines.append("")
lines.append("### Complete-case arm within population: LR odds ratios per SD")
for c, o, lo_, hi_ in zip(FEATS, cc_ors, cc_lo, cc_hi):
    lines.append(f"- {c}: OR {o:.2f} ({lo_:.2f}-{hi_:.2f})")

txt = "\n".join(lines)
import os
os.makedirs(f"{BASE}/sensitivity/results", exist_ok=True)
open(f"{BASE}/sensitivity/results/mi_summary.md", "w").write(txt)
rows_out.append(dict(arm="MI", n=len(mi_pop), internal_auc=auc_est, internal_auc_se=auc_se,
                     internal_brier=brier_m, internal_calib_int=ci_m, internal_calib_slope=sl_m))
pd.DataFrame(rows_out).to_csv(f"{BASE}/sensitivity/results/mi_results.csv", index=False)
pd.DataFrame({"feature": FEATS, "OR_MI": or_mean, "lo_MI": or_lo, "hi_MI": or_hi,
              "OR_CC": cc_ors, "lo_CC": cc_lo, "hi_CC": cc_hi}).to_csv(
    f"{BASE}/sensitivity/results/mi_lr_ors.csv", index=False)
ext_rows = []
for center, d in ext_pool.items():
    ext_rows.append(dict(arm="MI", center=center, **d))
for center, r in cc_ext.items():
    ext_rows.append(dict(arm="CC_within_pop", center=center, auc=r["auc"], se=np.nan,
                         oe_mean=r["oe"], oe_sd=np.nan, slope_mean=r["slope"],
                         ci_mean=r["ci"], brier_mean=r["brier"]))
pd.DataFrame(ext_rows).to_csv(f"{BASE}/sensitivity/results/mi_external.csv", index=False)
print("\n" + txt)
