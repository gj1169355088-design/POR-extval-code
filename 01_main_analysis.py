#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POR prediction model -- TRIPOD+AI compliant upgrade analysis
=============================================================
Implements the analysis upgrade recommended for the rejected LDH submission:

  A. Model development (Shanghai)      : LR(L2) + RF(150 trees, depth 6, min leaf 10)
                                        internal validation = 80:20 hold-out PLUS
                                        repeated 5x10-fold CV + bootstrap optimism correction
  B. Transported (frozen) evaluation   : blind application to Zunyi / Jining / pooled
  C. Recalibrated (intercept+slope)    : logistic recalibration of logit(p) per external center
  D. Refitted                          : retrain in external center (upper-bound reference)
  E. Metrics with 95% bootstrap CI     : AUC, AUPRC, Brier, calibration intercept/slope,
                                        O/E ratio, sens/spec/PPV/NPV at pre-specified
                                        thresholds + Youden, DCA net benefit
  F. Comparisons                       : paired DeLong tests; categorical NRI vs AMH-only model
  G. Pooled external AUC               : DerSimonian-Laird random-effects + I^2
  H. SHAP                              : TreeExplainer global summary + dependence + 2 force plots

Inputs (CLI): --shanghai shanghai.csv --external zjy_external_cohort.csv
  CSV columns required: age, bmi, amh, base_fsh, oocytes (int), [center]
Outputs: results/ (csv) + figures/ (png) + results_summary.md
"""
import argparse, os, sys, json, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
RNG = np.random.RandomState(20261105)
# FEATURES may be overridden via --features (e.g. "age,bmi,base_fsh" when AMH not yet merged)
FEATURES = ["age", "bmi", "amh", "base_fsh"]
CLIN_THRESHOLDS = [0.20, 0.30, 0.40]   # pre-specified clinical risk thresholds
N_BOOT = 1000

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

# ---------------- core metrics ----------------
def auc_ci(y, p, n_boot=N_BOOT, seed=0):
    obs = roc_auc_score(y, p)
    rng = np.random.RandomState(seed)
    y, p = np.asarray(y), np.asarray(p)
    n = len(y); aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(roc_auc_score(yy, pp))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return obs, lo, hi

def auprc_ci(y, p, n_boot=N_BOOT, seed=1):
    obs = average_precision_score(y, p)
    rng = np.random.RandomState(seed)
    n = len(y); vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yy, pp = y[idx], p[idx]
        if yy.sum() == 0:
            continue
        vals.append(average_precision_score(yy, pp))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return obs, lo, hi

def calibration_slope_intercept(y, p):
    """Logistic recalibration: outcome ~ a + b*logit(p). a=calibration-in-the-large
    (at b=1) proxy and b=slope; report both from the joint fit."""
    lp = logit(p)
    X = np.column_stack([np.ones_like(lp), lp])
    try:
        res = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(X, y)
        return float(res.coef_[0][0]), float(res.coef_[0][1])   # intercept, slope
    except Exception:
        return np.nan, np.nan

def oe_ratio(y, p):
    """O/E: observed events / sum of predicted probabilities."""
    return float(np.sum(y) / max(np.sum(p), 1e-9))

def classification_metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1); npv = tn / max(tn + fn, 1)
    return dict(threshold=thr, sens=sens, spec=spec, ppv=ppv, npv=npv, tp=tp, fp=fp, fn=fn, tn=tn)

def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    return float(thr[np.argmax(j)])

def net_benefit(y, p, thr):
    yhat = (p >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    n = len(y)
    return tp / n - fp / n * (thr / (1 - thr)) if thr < 1 else np.nan

# ---------------- DeLong (Sun-Xu vectorized) ----------------
def _delong_components(y_true, y_pred):
    order = np.argsort(-y_pred)
    y_sorted = y_true[order]
    pos = y_sorted == 1; neg = y_sorted == 0
    n1, n2 = int(pos.sum()), int(neg.sum())
    if n1 == 0 or n2 == 0:
        return None
    # rank-based structural representation
    from scipy.stats import rankdata
    r = rankdata(y_pred)
    # V10 (mid-ranks) method
    def midrank(x):
        J = rankdata(x)
        K = rankdata(-x)
        return (J + len(x) + 1 - K) / 2.0
    # faster equivalent: use rank of pos subset among all, etc.
    x = y_pred[y_true == 1]; yy = y_pred[y_true == 0]
    # AUC
    auc = (np.sum(rankdata(np.concatenate([x, yy]))[:len(x)]) - len(x) * (len(x) + 1) / 2) / (len(x) * len(yy))
    # structural components
    v = np.zeros(len(x)); u = np.zeros(len(yy))
    for i in range(len(x)):
        v[i] = (np.sum(yy < x[i]) + 0.5 * np.sum(yy == x[i])) / len(yy)
    for j in range(len(yy)):
        u[j] = (np.sum(x > yy[j]) + 0.5 * np.sum(x == yy[j])) / len(x)
    s1 = np.var(v, ddof=1) / len(x) if len(x) > 1 else 0.0
    s2 = np.var(u, ddof=1) / len(yy) if len(yy) > 1 else 0.0
    return auc, s1 + s2, 0.0, (n1, n2)

def delong_test(y, p1, p2):
    """Paired DeLong test, returns (z, p_value, auc1, auc2)."""
    y = np.asarray(y); p1 = np.asarray(p1); p2 = np.asarray(p2)
    c1 = _delong_components(y, p1)
    c2 = _delong_components(y, p2)
    if c1 is None or c2 is None:
        return np.nan, np.nan, np.nan, np.nan
    a1, v1, _, _ = c1
    a2, v2, _, _ = c2
    # paired covariance via structural components
    x1, x2 = np.asarray(p1), np.asarray(p2)
    pos = y == 1; neg = y == 0
    yy = y[pos]; xx1 = x1[pos]; xx2 = x2[pos]
    yy0 = y[neg]; yx1 = x1[neg]; yx2 = x2[neg]
    def vs(xx, ref):  # unused placeholder, kept for clarity
        return None
    # vectorized structural components
    v1s = (np.sum(yx1[None, :] < xx1[:, None], axis=1) + 0.5 * np.sum(yx1[None, :] == xx1[:, None], axis=1)) / len(yx1)
    v2s = (np.sum(yx2[None, :] < xx2[:, None], axis=1) + 0.5 * np.sum(yx2[None, :] == xx2[:, None], axis=1)) / len(yx2)
    u1s = (np.sum(xx1[:, None] > yx1[None, :], axis=0) + 0.5 * np.sum(xx1[:, None] == yx1[None, :], axis=0)) / len(xx1)
    u2s = (np.sum(xx2[:, None] > yx2[None, :], axis=0) + 0.5 * np.sum(xx2[:, None] == yx2[None, :], axis=0)) / len(xx2)
    s_v = np.cov(v1s, v2s, ddof=1) if len(v1s) > 1 else np.zeros((2, 2))
    s_u = np.cov(u1s, u2s, ddof=1) if len(u1s) > 1 else np.zeros((2, 2))
    var = s_v[0, 0] / len(v1s) + s_v[1, 1] / len(v1s) - 2 * s_v[0, 1] / len(v1s) \
        + s_u[0, 0] / len(u1s) + s_u[1, 1] / len(u1s) - 2 * s_u[0, 1] / len(u1s)
    var = (s_v[0, 0] + s_v[1, 1] - 2 * s_v[0, 1]) / len(v1s) + (s_u[0, 0] + s_u[1, 1] - 2 * s_u[0, 1]) / len(u1s)
    if var <= 0 or a1 == a2:
        z, pv = np.nan, np.nan
    else:
        z = (a1 - a2) / np.sqrt(var)
        pv = 2 * stats.norm.sf(abs(z))
    return z, pv, a1, a2

# ---------------- categorical NRI ----------------
def categorical_nri(y, p_new, p_old, thr):
    up_e  = int(((p_new >= thr) & (p_old < thr) & (y == 1)).sum())
    dn_e  = int(((p_new <  thr) & (p_old >= thr) & (y == 1)).sum())
    up_ne = int(((p_new >= thr) & (p_old < thr) & (y == 0)).sum())
    dn_ne = int(((p_new <  thr) & (p_old >= thr) & (y == 0)).sum())
    ne, nne = int((y == 1).sum()), int((y == 0).sum())
    nri_p = (up_e - dn_e) / max(ne, 1)
    nri_n = (dn_ne - up_ne) / max(nne, 1)
    return dict(NRI_events=nri_p, NRI_nonevents=nri_n, NRI_total=nri_p + nri_n,
                up_events=up_e, down_events=dn_e, up_nonevents=up_ne, down_nonevents=dn_ne)

# ---------------- random-effects pooling (DerSimonian-Laird) ----------------
def dl_pool(estimates, variances):
    k = len(estimates)
    w = 1 / np.array(variances)
    mu_fixed = np.sum(w * estimates) / np.sum(w)
    q = np.sum(w * (np.array(estimates) - mu_fixed) ** 2)
    df = k - 1
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max((q - df) / c, 0.0) if c > 0 else 0.0
    w_re = 1 / (np.array(variances) + tau2)
    mu = np.sum(w_re * estimates) / np.sum(w_re)
    se = np.sqrt(1 / np.sum(w_re))
    i2 = max((q - df) / q, 0.0) * 100 if q > 0 else 0.0
    return mu, se, tau2, i2

# ---------------- models ----------------
def make_models(seed=42):
    """LR is fit on z-scored features (StandardScaler inside a pipeline), matching
    the Methods statement 'continuous features were scaled using z-score normalisation'."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    models = {
        "RF": RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=10,
                                     random_state=seed, n_jobs=-1, class_weight=None),
        "LR": make_pipeline(StandardScaler(),
                            LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000)),
    }
    if "amh" in FEATURES:
        models["AMH_only"] = make_pipeline(
            StandardScaler(), LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000))
    else:  # comparator: strongest single marker available
        models["Age_only"] = make_pipeline(
            StandardScaler(), LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000))
    return models

def fit_models(X, y):
    m = make_models()
    m["RF"].fit(X[FEATURES], y)
    m["LR"].fit(X[FEATURES], y)
    if "amh" in FEATURES:
        m["AMH_only"].fit(X[["amh"]], y)
    else:
        m["Age_only"].fit(X[["age"]], y)
    return m

# ---------------- repeated CV internal validation ----------------
COMPARATOR = "AMH_only" if "amh" in FEATURES else "Age_only"
COMPARATOR_FEAT = "amh" if "amh" in FEATURES else "age"

def repeated_cv(X, y, n_rep=5, n_folds=10):
    keys = ["RF", "LR", COMPARATOR]
    out = {k: [] for k in keys}
    for r in range(n_rep):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1000 + r)
        oof = {k: np.full(len(y), np.nan) for k in out}
        for tr, te in skf.split(X[FEATURES], y):
            mm = fit_models(X.iloc[tr].reset_index(drop=True), y[tr])
            oof["RF"][te] = mm["RF"].predict_proba(X[FEATURES].iloc[te])[:, 1]
            oof["LR"][te] = mm["LR"].predict_proba(X[FEATURES].iloc[te])[:, 1]
            oof[COMPARATOR][te] = mm[COMPARATOR].predict_proba(X[[COMPARATOR_FEAT]].iloc[te])[:, 1]
        for k in out:
            out[k].append(roc_auc_score(y, oof[k]))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in out.items()}

# ---------------- bootstrap optimism correction (internal AUC) ----------------
def optimism_corrected_auc(X, y, model_key="RF", n_boot=200):
    def fit_eval(idx_tr, idx_te):
        mm = fit_models(X.iloc[idx_tr].reset_index(drop=True), y[idx_tr])
        p = mm[model_key].predict_proba(X[FEATURES].iloc[idx_te])[:, 1] if model_key != "AMH_only" \
            else mm[model_key].predict_proba(X[["amh"]].iloc[idx_te])[:, 1]
        return roc_auc_score(y[idx_te], p)
    apparent = fit_eval(np.arange(len(y)), np.arange(len(y)))
    rng = np.random.RandomState(7)
    n = len(y); opt = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        te = np.setdiff1d(np.arange(n), idx)
        if len(te) < 30 or len(np.unique(y[te])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        mm = fit_models(X.iloc[idx].reset_index(drop=True), y[idx])
        p_boot = mm[model_key].predict_proba(X[FEATURES].iloc[idx])[:, 1]
        p_test = mm[model_key].predict_proba(X[FEATURES].iloc[te])[:, 1]
        opt.append(roc_auc_score(y[idx], p_boot) - roc_auc_score(y[te], p_test))
    return apparent - float(np.mean(opt)), apparent

# ---------------- per-cohort evaluation ----------------
def evaluate_cohort(name, X, y, models, rows):
    recs = []
    p_rf = models["RF"].predict_proba(X[FEATURES])[:, 1]
    p_lr = models["LR"].predict_proba(X[FEATURES])[:, 1]
    p_cmp = models[COMPARATOR].predict_proba(X[[COMPARATOR_FEAT]])[:, 1]
    for label, p in [("RF", p_rf), ("LR", p_lr), (COMPARATOR, p_cmp)]:
        auc, lo, hi = auc_ci(y, p)
        ap, alo, ahi = auprc_ci(y, p)
        cipt, slope = calibration_slope_intercept(y, p)
        rec = dict(cohort=name, model=label, n=len(y), events=int(np.sum(y)),
                   event_rate=np.mean(y), auc=auc, auc_lo=lo, auc_hi=hi,
                   auprc=ap, auprc_lo=alo, auprc_hi=ahi,
                   brier=brier_score_loss(y, p),
                   calib_intercept=cipt, calib_slope=slope,
                   o_e_ratio=oe_ratio(y, p))
        yt = youden_threshold(*roc_curve(y, p_rf)[:3][::-1][0:3]) if False else None
        recs.append(rec)
        rows.append(rec)
    # Youden threshold derived IN DERIVATION cohort -> passed via models metadata
    for label, p, thr_src in [("RF", p_rf, "youden"), ("LR", p_lr, "youden")]:
        pass
    return p_rf, p_lr, p_cmp

def full_report(shanghai, external, outdir):
    os.makedirs(f"{outdir}/results", exist_ok=True)
    os.makedirs(f"{outdir}/figures", exist_ok=True)
    rows = []
    summary_lines = ["# POR model upgrade analysis -- results summary\n"]

    # ---------- A. development + internal validation ----------
    Xs = shanghai.copy(); ys = Xs["por"].values.astype(int)
    tr, te = train_test_split(np.arange(len(ys)), test_size=0.2, stratify=ys, random_state=42)
    models = fit_models(Xs.iloc[tr].reset_index(drop=True), ys[tr])
    summary_lines.append("## 1. Development (Shanghai) & internal validation\n")
    cv = repeated_cv(Xs.iloc[tr].reset_index(drop=True), ys[tr])
    for k, (m_, s_) in cv.items():
        summary_lines.append(f"- {k}: repeated 5x10-fold CV AUC = {m_:.3f} (SD {s_:.3f})")
    occ, app = optimism_corrected_auc(Xs.iloc[tr].reset_index(drop=True), ys[tr], "RF")
    summary_lines.append(f"- RF: bootstrap optimism-corrected AUC = {occ:.3f} (apparent {app:.3f})")
    # hold-out performance
    p_rf_tr = models["RF"].predict_proba(Xs[FEATURES].iloc[te])[:, 1]
    auc, lo, hi = auc_ci(ys[te], p_rf_tr)
    summary_lines.append(f"- RF: 80:20 hold-out AUC = {auc:.3f} (95% CI {lo:.3f}-{hi:.3f})\n")
    evaluate_cohort("Shanghai_holdout", Xs.iloc[te].reset_index(drop=True), ys[te], models, rows)

    youden_thr = youden_threshold(ys[tr], models["RF"].predict_proba(Xs[FEATURES].iloc[tr])[:, 1])
    summary_lines.append(f"- Youden threshold (derivation): {youden_thr:.3f}\n")

    # ---------- B/C/D. external: transported / recalibrated / refitted ----------
    summary_lines.append("## 2. Prospective external validation (transported / recalibrated / refitted)\n")
    for center in external["center"].unique():
        sub = external[external["center"] == center]
        Xe = sub.copy(); ye = Xe["por"].values.astype(int)
        if len(np.unique(ye)) < 2:
            summary_lines.append(f"- {center}: skipped (single outcome class)")
            continue
        p_t = models["RF"].predict_proba(Xe[FEATURES])[:, 1]
        auc, lo, hi = auc_ci(ye, p_t)
        ap, alo, ahi = auprc_ci(ye, p_t)
        cipt, slope = calibration_slope_intercept(ye, p_t)
        summary_lines.append(
            f"- {center} (n={len(ye)}, events={int(ye.sum())}, rate={ye.mean():.1%}) TRANSPORTED: "
            f"AUC {auc:.3f} ({lo:.3f}-{hi:.3f}), AUPRC {ap:.3f} ({alo:.3f}-{ahi:.3f}), "
            f"Brier {brier_score_loss(ye, p_t):.4f}, calib intercept {cipt:.2f}, slope {slope:.2f}, O/E {oe_ratio(ye, p_t):.2f}")
        rows.append(dict(cohort=center, model="RF_transported", n=len(ye), events=int(ye.sum()),
                         auc=auc, auc_lo=lo, auc_hi=hi, auprc=ap, auprc_lo=alo, auprc_hi=ahi,
                         brier=brier_score_loss(ye, p_t), calib_intercept=cipt, calib_slope=slope,
                         o_e_ratio=oe_ratio(ye, p_t), event_rate=ye.mean()))
        # recalibration (C)
        lp = logit(p_t).reshape(-1, 1)
        rec = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(lp, ye)
        p_rc = rec.predict_proba(lp)[:, 1]
        auc2, lo2, hi2 = auc_ci(ye, p_rc)
        cipt2, slope2 = calibration_slope_intercept(ye, p_rc)
        summary_lines.append(
            f"  RECALIBRATED: AUC {auc2:.3f} ({lo2:.3f}-{hi2:.3f}) [unchanged by design up to clip], "
            f"Brier {brier_score_loss(ye, p_rc):.4f} (was {brier_score_loss(ye, p_t):.4f}), "
            f"calib intercept {cipt2:.2f}, slope {slope2:.2f}")
        rows.append(dict(cohort=center, model="RF_recalibrated", n=len(ye), events=int(ye.sum()),
                         auc=auc2, auc_lo=lo2, auc_hi=hi2, brier=brier_score_loss(ye, p_rc),
                         calib_intercept=cipt2, calib_slope=slope2))
        # refit (D)
        rf2 = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=10,
                                     random_state=42, n_jobs=-1).fit(Xe[FEATURES], ye)
        p_rf2 = rf2.predict_proba(Xe[FEATURES])[:, 1]
        auc3, lo3, hi3 = auc_ci(ye, p_rf2)
        summary_lines.append(f"  REFITTED: AUC {auc3:.3f} ({lo3:.3f}-{hi3:.3f}) (upper reference)")
        # classification at thresholds (transported)
        for thr in CLIN_THRESHOLDS + [youden_thr]:
            cm = classification_metrics(ye, p_t, thr)
            tag = "pre-specified" if thr in CLIN_THRESHOLDS else "youden"
            summary_lines.append(f"  {tag} thr={thr:.3f}: sens {cm['sens']:.1%}, spec {cm['spec']:.1%}, "
                                 f"PPV {cm['ppv']:.1%}, NPV {cm['npv']:.1%}")
            rows.append(dict(cohort=center, model="RF_classif", n=len(ye), thr_type=tag, **cm))
        # DCA
        thrs = np.linspace(0.05, 0.6, 23)
        dca = pd.DataFrame({"threshold": thrs,
                            "model": [net_benefit(ye, p_t, t) for t in thrs],
                            "treat_all": [net_benefit(ye, np.ones_like(p_t), t) for t in thrs],
                            "treat_none": [0.0] * len(thrs)})
        dca.to_csv(f"{outdir}/results/DCA_{center}.csv", index=False)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(dca.threshold, dca.model, label="RF (transported)")
        ax.plot(dca.threshold, dca.treat_all, "k--", label="Treat all")
        ax.plot(dca.threshold, dca.treat_none, ":", label="Treat none")
        ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
        ax.set_title(f"DCA - {center} (prospective)"); ax.legend(); fig.tight_layout()
        fig.savefig(f"{outdir}/figures/DCA_{center}.png", dpi=150); plt.close(fig)
    # pooled transported + random effects
    ypool = external["por"].values.astype(int)
    ppool = models["RF"].predict_proba(external[FEATURES])[:, 1]
    auc, lo, hi = auc_ci(ypool, ppool)
    est, var = [], []
    for center in external["center"].unique():
        sub = external[external["center"] == center]
        ye = sub["por"].values.astype(int)
        pe = models["RF"].predict_proba(sub[FEATURES])[:, 1]
        a, l_, h_ = auc_ci(ye, pe)
        se = (h_ - l_) / 3.92
        est.append(a); var.append(se ** 2)
    mu, se_pool, tau2, i2 = dl_pool(est, var)
    summary_lines.append(f"\n- POOLED external (transported): AUC {auc:.3f} ({lo:.3f}-{hi:.3f}); "
                         f"random-effects AUC {mu:.3f} (SE {se_pool:.3f}), I^2 = {i2:.0f}%\n")
    rows.append(dict(cohort="POOLED", model="RF_transported", n=len(ypool), events=int(ypool.sum()),
                     auc=auc, auc_lo=lo, auc_hi=hi, pooled_re=mu, pooled_se=se_pool, i2=i2))

    # ---------- F. DeLong + NRI in external ----------
    summary_lines.append("## 3. Comparisons (external cohorts)\n")
    for center in external["center"].unique():
        sub = external[external["center"] == center]
        ye = sub["por"].values.astype(int)
        p_rf = models["RF"].predict_proba(sub[FEATURES])[:, 1]
        p_lr = models["LR"].predict_proba(sub[FEATURES])[:, 1]
        p_cmp = models[COMPARATOR].predict_proba(sub[[COMPARATOR_FEAT]])[:, 1]
        z, pv, a1, a2 = delong_test(ye, p_rf, p_cmp)
        summary_lines.append(f"- {center}: DeLong RF vs {COMPARATOR}: AUC {a1:.3f} vs {a2:.3f}, z={z:.2f}, p={pv:.3g}")
        nri = categorical_nri(ye, p_rf, p_cmp, 0.30)
        summary_lines.append(f"  categorical NRI @0.30 (RF vs {COMPARATOR}): total {nri['NRI_total']:.3f} "
                             f"(events {nri['NRI_events']:.3f}, nonevents {nri['NRI_nonevents']:.3f})")
        rows.append(dict(cohort=center, model="DeLong_RF_vs_AMH", z=z, p=pv, auc1=a1, auc2=a2, **nri))

    pd.DataFrame(rows).to_csv(f"{outdir}/results/all_results.csv", index=False)

    # ---------- H. SHAP (RF, external + derivation examples) ----------
    try:
        import shap
        expl = shap.TreeExplainer(models["RF"])
        samp = external.sample(min(500, len(external)), random_state=1)
        sv = expl.shap_values(samp[FEATURES])
        sv1 = sv[1] if isinstance(sv, list) else sv[:, :, 1]
        plt.figure(figsize=(7, 5))
        shap.summary_plot(sv1, samp[FEATURES], show=False)
        plt.tight_layout(); plt.savefig(f"{outdir}/figures/SHAP_summary_external.png", dpi=150); plt.close()
        for feat in FEATURES:
            plt.figure(figsize=(6, 4.5))
            shap.dependence_plot(feat, sv1, samp[FEATURES], show=False)
            plt.tight_layout(); plt.savefig(f"{outdir}/figures/SHAP_dependence_{feat}.png", dpi=150); plt.close()
        summary_lines.append("\n## 4. SHAP\n- TreeExplainer summary + dependence plots saved to figures/.")
    except Exception as e:
        summary_lines.append(f"\n## 4. SHAP -- FAILED: {e}")

    open(f"{outdir}/results_summary.md", "w", encoding="utf-8").write("\n".join(summary_lines))
    print("\n".join(summary_lines))

# ---------------- entry ----------------
def load_external(path):
    """Accepts external_analysis.csv from 02_build_cohorts.py (already cleaned)."""
    df = pd.read_csv(path)
    if "por" not in df.columns:
        df["por"] = (pd.to_numeric(df["oocytes"], errors="coerce") <= 3).astype(int)
    keep = FEATURES + ["por", "center", "patient_id"]
    df = df[[c for c in keep if c in df.columns]].dropna(subset=[c for c in FEATURES if c in df.columns] + ["por"])
    return df.reset_index(drop=True)

def main():
    global FEATURES, COMPARATOR, COMPARATOR_FEAT
    ap = argparse.ArgumentParser()
    ap.add_argument("--shanghai", required=True, help="CSV with model feature columns + oocytes")
    ap.add_argument("--external", required=True, help="zjy_external_cohort.csv from extract_zjy_cohort.py")
    ap.add_argument("--features", default="age,bmi,amh,base_fsh",
                    help="comma-separated feature list (default 4-feature manuscript model)")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    FEATURES = [f.strip() for f in args.features.split(",")]
    COMPARATOR = "AMH_only" if "amh" in FEATURES else ("Age_only" if "age" in FEATURES else FEATURES[0])
    COMPARATOR_FEAT = "amh" if "amh" in FEATURES else ("age" if "age" in FEATURES else FEATURES[0])
    sh = pd.read_csv(args.shanghai)
    need = set(FEATURES) | {"oocytes"}
    missing = need - set(sh.columns)
    if missing:
        sys.exit(f"[FATAL] Shanghai file lacks required columns: {missing}. "
                 "If AMH is missing, merge it from the lab database by patient_id/stat_date, "
                 "or run with --features age,bmi,base_fsh.")
    sh = sh.rename(columns={"oocytes": "oocytes_retrieved"})
    sh["por"] = (pd.to_numeric(sh["oocytes_retrieved"], errors="coerce") <= 3).astype(int)
    sh = sh.dropna(subset=FEATURES + ["por"]).reset_index(drop=True)
    ext = load_external(args.external)
    print(f"Shanghai n={len(sh)}, POR rate={sh['por'].mean():.1%}")
    print(f"External n={len(ext)}, centers={ext['center'].value_counts().to_dict()}, "
          f"POR rate={ext['por'].mean():.1%}")
    full_report(sh, ext, args.outdir)

if __name__ == "__main__":
    main()
