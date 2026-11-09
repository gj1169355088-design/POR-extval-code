#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication figures & tables for the POR manuscript (Lancet-style).
Reads shanghai_analysis.csv (primary derivation), external_analysis.csv,
shanghai_analysis_allcycles.csv (sensitivity). Produces:

figures/  (300 dpi, colour-blind safe, Arial)
  fig1_study_design.png        A cohort flow diagram  B prospective accrual timeline
  fig2_discrimination.png      A ROC internal  B ROC Zunyi  C ROC Jining  D AUC forest
  fig3_calibration_dca.png     A calibration Zunyi  B calibration Jining  C DCA both
  fig4_interpretation.png      A SHAP summary  B LR forest (adjusted ORs)
  fig5_protocol_lowreserve.png A oocyte yield by protocol x age  B POR rate  C blastocysts
  sfig1_subgroup_auc.png       subgroup AUCs (age/BMI strata per cohort)
tables/
  table1_baseline.csv/.md      Table 1 baseline characteristics
  table2_performance.csv/.md   Table 2 model performance
  stable1_funnel.md            cohort funnel
  stable2_missingness.md       predictor missingness by centre
  stable3_subgroup_auc.csv     subgroup performance
  stable4_sensitivity.md       sensitivity analyses
  stable5_protocol.md          low-reserve protocol comparison
models/    joblib model objects + metadata
"""
import numpy as np, pandas as pd, os, json, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import statsmodels.api as sm

RNG = np.random.RandomState(20261105)
BASE = "."
FEATS = ["age", "bmi", "amh", "base_fsh"]
N_BOOT = 1000

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8.5, "axes.linewidth": 0.8,
    "axes.labelsize": 9, "axes.titlesize": 9.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42,
})
C_BLUE, C_ORANGE, C_GREEN, C_GREY = "#1f77b4", "#d62728", "#2ca02c", "#7f7f7f"
COLR = {"Shanghai": C_GREY, "Zunyi": "#E64B35", "Jining": "#4DBBD5"}

def boot_auc(y, p, n=N_BOOT, seed=0):
    rng = np.random.RandomState(seed); aucs = []
    y, p = np.asarray(y), np.asarray(p); n0 = len(y)
    for _ in range(n):
        idx = rng.randint(0, n0, n0)
        if len(np.unique(y[idx])) < 2: continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return roc_auc_score(y, p), lo, hi

def md_num(x, dec=2):
    return f"{x:.{dec}f}"

def save_all(fig, name, dpi=300):
    """Save a figure as PNG (preview), TIF and PDF (journal submission)."""
    for ext in ("png", "tif", "pdf"):
        fig.savefig(f"{BASE}/figures/{name}.{ext}", dpi=dpi, bbox_inches="tight")

# ---------------------------------------------------------------- load
sh = pd.read_csv(f"{BASE}/shanghai_analysis.csv")
sh_all = pd.read_csv(f"{BASE}/shanghai_analysis_allcycles.csv")
ext = pd.read_csv(f"{BASE}/external_analysis.csv")
raw = pd.read_csv(f"{BASE}/shanghai_raw.csv", low_memory=False)

os.makedirs(f"{BASE}/figures", exist_ok=True)
os.makedirs(f"{BASE}/tables", exist_ok=True)
os.makedirs(f"{BASE}/models", exist_ok=True)

y = sh["por"].astype(int).values
tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
rf = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=10,
                            random_state=42, n_jobs=-1).fit(sh[FEATS].iloc[tr], y[tr])
lr = make_pipeline(StandardScaler(),
                   LogisticRegression(penalty="l2", C=1.0, max_iter=1000)
                   ).fit(sh[FEATS].iloc[tr], y[tr])
amh_only = LogisticRegression(penalty=None, max_iter=1000).fit(sh[["amh"]].iloc[tr], y[tr])
joblib.dump({"random_forest": rf, "logistic_l2": lr, "amh_only": amh_only,
             "features": FEATS,
             "scaling": "z-score applied to continuous predictors before fitting",
             "thresholds": {"youden_derivation": 0.159}},
            f"{BASE}/models/por_models.joblib")

# =====================================================================
# FIGURE 1 -- study design + accrual
# =====================================================================
fig = plt.figure(figsize=(7.2, 6.2))
gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.2], hspace=0.35)
axA = fig.add_subplot(gs[0]); axA.axis("off")

def box(ax, x, y, w, h, text, fc, ec="#333333", fs=7.6, tc="black"):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=0.9))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=tc)

def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#444444"))

# left column: vertical derivation funnel
fx, fw = 0.02, 0.47
steps = [
    ("Shanghai First Maternity and Infant Hospital EHR\n2010–2026 · 103,899 cycle records", "#f0f0f0", 0.90),
    ("Fresh retrieval cycles — 55,487", "#e8f0f8", 0.79),
    ("Initiated Jan 2021 – Oct 2026 — 31,336", "#e8f0f8", 0.70),
    ("IVF or ICSI — 29,087", "#e8f0f8", 0.61),
    ("Stimulated cycles (natural-cycle and HRT\npreparations excluded) — 28,420", "#e8f0f8", 0.50),
    ("Age, BMI, AMH and basal FSH complete\nand within plausible ranges — 20,248", "#dbe9f6", 0.38),
    ("First stimulated cycle per patient — 11,749", "#cfe2f3", 0.28),
]
for i, (txt, fc, ytop) in enumerate(steps):
    box(axA, fx, ytop - 0.075, fw, 0.075, txt, fc, fs=7.0)
    if i > 0:
        arrow(axA, fx + fw/2, steps[i-1][2] - 0.075, fx + fw/2, ytop)
# split into derivation / hold-out
arrow(axA, fx + fw/2, 0.305, fx + fw/2, 0.282)   # connect funnel bottom to next box TOP
box(axA, fx, 0.095, fw * 0.47, 0.075, "Derivation (80%)\n9,399", "#cfe2f3", fs=7.0)
box(axA, fx + fw * 0.53, 0.095, fw * 0.47, 0.075, "Internal hold-out (20%)\n2,350", "#cfe2f3", fs=7.0)
arrow(axA, fx + fw * 0.25, 0.205, fx + fw * 0.25, 0.172)
arrow(axA, fx + fw * 0.75, 0.205, fx + fw * 0.75, 0.172)
axA.text(fx + fw + 0.03, 0.055, "Model development:\nRF + LR, frozen after\ninternal validation",
         fontsize=7.2, va="center", style="italic", color="#333333")

# right column: prospective external validation
ex, ew = 0.55, 0.44
box(axA, ex, 0.80, ew, 0.10,
    "Zunyi-Jining Reproductive Health Registry\nProspective, consecutively enrolled\nsince Nov 2023 (pre-stimulation baseline)",
    "#fff2e6", fs=6.8)
box(axA, ex, 0.615, ew, 0.075,
    "Zunyi Medical University Affiliated Hospital\nn = 1,040 first fresh IVF/ICSI cycles", "#ffe6cc", fs=6.8)
box(axA, ex, 0.505, ew, 0.075,
    "Jining Medical University Affiliated Hospital\nn = 245 first fresh IVF/ICSI cycles", "#ffe6cc", fs=6.8)
box(axA, ex, 0.345, ew, 0.10,
    "Frozen model applied blindly, without\nretraining or threshold re-optimisation\n(TRIPOD type 3)",
    "#fde3d0", fs=6.8)
arrow(axA, ex + ew/2, 0.80, ex + ew/2, 0.693)
arrow(axA, ex + ew/2, 0.615, ex + ew/2, 0.583)
arrow(axA, ex + ew/2, 0.505, ex + ew/2, 0.448)
axA.text(ex + ew/2, 0.19, "Geographic span: 1,800 km (Zunyi, Southwest,\nhigh-altitude Karst) and 800 km (Jining, North China plain)",
         fontsize=7.0, ha="center", style="italic", color="#555555")
axA.set_xlim(0, 1); axA.set_ylim(0.0, 1.0)
axA.text(0.0, 0.985, "a", fontsize=13, fontweight="bold")

axB = fig.add_subplot(gs[1])
sh["date"] = pd.to_datetime(sh["date"])
sh["month"] = sh["date"].dt.to_period("M").dt.to_timestamp()
acc = sh.groupby("month").size()
axB.fill_between(acc.index, acc.values, color="#4c78a8", alpha=0.55, lw=0)
axB.plot(acc.index, acc.values, color="#34576b", lw=1.0)
axB.set_ylabel("Derivation cycles per month")
axB.set_title("b  Temporal coverage of the derivation cohort (2021–2026)", loc="left", fontsize=9.5)
for lb in [2021, 2022, 2023, 2024, 2025, 2026]:
    axB.axvline(pd.Timestamp(f"{lb}-01-01"), color="#cccccc", lw=0.5)
save_all(fig, "fig1_study_design")
plt.close(fig)
print("fig1 done")

# =====================================================================
# FIGURE 2 -- ROC panels + AUC forest
# =====================================================================
def roc_panel(ax, ytrue, plist, labels, colors, title):
    for p, lb, cc in zip(plist, labels, colors):
        fpr, tpr, _ = roc_curve(ytrue, p)
        a = roc_auc_score(ytrue, p)
        ax.plot(fpr, tpr, color=cc, lw=1.4, label=f"{lb} (AUC {a:.3f})")
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.7, ls="--")
    ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title(title, loc="left", fontsize=9.5)
    ax.legend(frameon=False, loc="lower right")

fig, axs = plt.subplots(1, 4, figsize=(10.8, 2.9))
p_rf_te = rf.predict_proba(sh[FEATS].iloc[te])[:, 1]
p_lr_te = lr.predict_proba(sh[FEATS].iloc[te])[:, 1]
p_amh_te = amh_only.predict_proba(sh[["amh"]].iloc[te])[:, 1]
roc_panel(axs[0], y[te], [p_rf_te, p_lr_te, p_amh_te],
          ["Random forest", "Logistic regression", "AMH alone"],
          [C_BLUE, C_ORANGE, C_GREY], "a  Internal hold-out (n=2,350)")
ext_dict = {}
for centre in ["Zunyi", "Jining"]:
    s = ext[ext["center"] == centre]
    ye = s["por"].astype(int).values
    pe = rf.predict_proba(s[FEATS])[:, 1]
    plre = lr.predict_proba(s[FEATS])[:, 1]
    ext_dict[centre] = (s, ye, pe)
    roc_panel(axs[1] if centre == "Zunyi" else axs[2], ye,
              [pe, plre, amh_only.predict_proba(s[["amh"]])[:, 1]],
              ["Random forest", "Logistic regression", "AMH alone"],
              [C_BLUE, C_ORANGE, C_GREY],
              f"{'b' if centre=='Zunyi' else 'c'}  {centre}, transported (n={len(s)})")
# forest
axF = axs[3]
rows = []
auc0, l0, h0 = boot_auc(y[te], p_rf_te)
rows.append(("Internal hold-out", auc0, l0, h0))
pool_y = ext["por"].astype(int).values
pool_p = rf.predict_proba(ext[FEATS])[:, 1]
rows.append(("External pooled", *boot_auc(pool_y, pool_p, seed=3)))
for c in ["Zunyi", "Jining"]:
    _, ye, pe = ext_dict[c]
    rows.append((f"{c} (n={len(ye)})", *boot_auc(ye, pe, seed=4 if c=='Zunyi' else 5)))
# enforce consistency with the definitive analysis CIs (01_main_analysis.py)
ar = pd.read_csv(f"{BASE}/definitive/results/all_results.csv")
overrides = {}
for _, rr in ar.iterrows():
    if rr["model"] == "RF" and rr["cohort"] == "Shanghai_holdout":
        overrides["Internal hold-out"] = (rr["auc"], rr["auc_lo"], rr["auc_hi"])
    elif rr["model"] == "RF_transported" and rr["cohort"] == "Zunyi":
        overrides["Zunyi (n=1040)"] = (rr["auc"], rr["auc_lo"], rr["auc_hi"])
    elif rr["model"] == "RF_transported" and rr["cohort"] == "Jining":
        overrides["Jining (n=245)"] = (rr["auc"], rr["auc_lo"], rr["auc_hi"])
    elif rr["model"] == "RF_transported" and rr["cohort"] == "POOLED":
        overrides["External pooled"] = (rr["auc"], rr["auc_lo"], rr["auc_hi"])
rows = [(lb, *overrides.get(lb, (a, lo, hi))) for lb, a, lo, hi in rows]
for i, (lb, a, lo, hi) in enumerate(rows[::-1]):
    yy = len(rows) - 1 - i
    cc = COLR.get(lb.split(" ")[0], C_GREY)
    axF.plot([lo, hi], [yy, yy], color=cc, lw=1.6)
    axF.plot([a], [yy], "o", color=cc, ms=4)
    axF.text(a, yy + 0.22, f"{a:.3f} ({lo:.3f}–{hi:.3f})", ha="center", fontsize=6.8)
axF.set_yticks(range(len(rows)))
axF.set_yticklabels([r[0] for r in rows], fontsize=7.5)
axF.set_ylim(-0.6, len(rows) + 0.45)
axF.set_xlabel("AUC (95% CI, 1000 bootstrap)")
axF.set_xlim(0.60, 1.0)
axF.set_title("d  Discrimination summary", loc="left", fontsize=9.5)
fig.tight_layout()
save_all(fig, "fig2_discrimination")
plt.close(fig)
print("fig2 done")
json.dump([[r[0], float(r[1]), float(r[2]), float(r[3])] for r in rows],
          open(f"{BASE}/tables/auc_forest_values.json", "w"), indent=1)

# =====================================================================
# FIGURE 3 -- calibration + DCA (external, transported)
# =====================================================================
fig, axs = plt.subplots(1, 3, figsize=(9.0, 3.0))
for k, centre in enumerate(["Zunyi", "Jining"]):
    s, ye, pe = ext_dict[centre]
    prob_true, prob_pred = calibration_curve(ye, pe, n_bins=5, strategy="quantile")
    ax = axs[k]
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.7, ls="--", label="Ideal")
    ax.plot(prob_pred, prob_true, "o-", color=COLR[centre], lw=1.3, ms=4,
            label="Transported RF")
    # recalibrated
    from scipy.special import logit, expit
    lp = logit(np.clip(pe, 1e-6, 1-1e-6)).reshape(-1, 1)
    rec = LogisticRegression(penalty=None, max_iter=1000).fit(lp, ye)
    pe_rc = rec.predict_proba(lp)[:, 1]
    pt2, pp2 = calibration_curve(ye, pe_rc, n_bins=5, strategy="quantile")
    ax.plot(pp2, pt2, "s--", color=COLR[centre], lw=1.0, ms=3.5, alpha=0.8,
            label="After recalibration")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed proportion")
    ax.set_title(f"{'ab'[k]}  Calibration — {centre}", loc="left", fontsize=9.5)
    ax.legend(frameon=False, loc="upper left")
axD = axs[2]
thrs = np.linspace(0.05, 0.6, 23)
for centre, cc in [("Zunyi", COLR["Zunyi"]), ("Jining", COLR["Jining"])]:
    s, ye, pe = ext_dict[centre]
    nb = []
    for t in thrs:
        yh = (pe >= t)
        tp = ((yh) & (ye == 1)).sum(); fp = ((yh) & (ye == 0)).sum()
        nb.append(tp/len(ye) - fp/len(ye) * t/(1-t))
    axD.plot(thrs, nb, color=cc, lw=1.3, label=centre)
axD.axhline(0, color="#999999", lw=0.7)
axD.set_xlabel("Threshold probability"); axD.set_ylabel("Net benefit")
axD.set_title("c  Decision curve analysis", loc="left", fontsize=9.5)
axD.legend(frameon=False)
fig.tight_layout()
save_all(fig, "fig3_calibration_dca")
plt.close(fig)
print("fig3 done")

# =====================================================================
# FIGURE 4 -- SHAP + adjusted-OR forest
# =====================================================================
import shap
fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.6),
                        gridspec_kw={"width_ratios": [1.15, 1]})
expl = shap.TreeExplainer(rf)
samp = ext.sample(min(600, len(ext)), random_state=1)
feat_pretty = {"age": "Age", "bmi": "BMI", "amh": "AMH", "base_fsh": "Basal FSH"}
sv = expl.shap_values(samp[FEATS])
sv1 = sv[1] if isinstance(sv, list) else sv[:, :, 1]
plt.sca(axs[0])
shap.summary_plot(sv1, samp[FEATS].rename(columns=feat_pretty), show=False, plot_size=None)
axs[0].set_title("a  SHAP — prospective validation cohorts", loc="left", fontsize=9.5)
axs[0].set_xlabel("SHAP value (impact on POR risk)")
plt.tight_layout()
# reposition the SHAP colour bar (shap adds it as an extra axes) so it cannot
# collide with panel B, and label it horizontally underneath
cb_axes = [a for a in fig.axes if a not in (axs[0], axs[1])]
pos0 = axs[0].get_position()
for a in cb_axes:
    a.set_position([pos0.x1 + 0.006, pos0.y0 + 0.04, 0.007, pos0.height * 0.7])
    a.set_ylabel("")
    a.set_xlabel("Feature\nvalue", fontsize=6.5, labelpad=1)
    a.tick_params(labelsize=6)
# OR forest from LR on full derivation (statsmodels, standardised predictors)
Xs = sh[FEATS].copy()
for f in FEATS:
    Xs[f] = (Xs[f] - Xs[f].mean()) / Xs[f].std()
logit_mod = sm.Logit(y, sm.add_constant(Xs.values)).fit(disp=0)
params = np.asarray(logit_mod.params)[1:]
conf = np.asarray(logit_mod.conf_int())[1:]
ors = np.exp(params); ci = np.exp(conf)
names = ["Age (per SD)", "BMI (per SD)", "AMH (per SD)", "Basal FSH (per SD)"]
axF = axs[1]
axF.axvline(1, color="#999999", lw=0.7, ls="--")
for i, (nm, o, lo, hi) in enumerate(zip(names, ors, ci[:, 0], ci[:, 1])):
    yy = len(names) - 1 - i          # first name plotted at top
    axF.plot([lo, hi], [yy, yy], color=C_BLUE, lw=1.6)
    axF.plot([o], [yy], "o", color=C_BLUE, ms=4)
    axF.text(hi * 1.12, yy, f"{o:.2f} ({lo:.2f}–{hi:.2f})", va="center", fontsize=7)
axF.set_yticks(range(len(names)))
axF.set_yticklabels(names[::-1], fontsize=8)   # names[0] (Age) is plotted at the top
axF.set_ylim(-0.5, len(names) - 0.4)
axF.set_xlabel("Adjusted odds ratio for POR (log scale)")
axF.set_xscale("log")
axF.set_xticks([0.1, 0.2, 0.5, 1.0, 2.0])
axF.set_xticklabels(["0.1", "0.2", "0.5", "1.0", "2.0"])
axF.minorticks_off()
axF.set_xlim(0.08, 2.6)
axF.set_title("b  Multivariable logistic regression", loc="left", fontsize=9.5)
save_all(fig, "fig4_interpretation")
plt.close(fig)
print("fig4 done")
pd.DataFrame({"predictor": names, "OR": ors, "lo": ci[:, 0], "hi": ci[:, 1],
              "p": logit_mod.pvalues[1:]}).to_csv(f"{BASE}/tables/lr_ors.csv", index=False)

# =====================================================================
# FIGURE 5 -- protocol comparison in low-reserve strata (all-cycles data)
# =====================================================================
low = sh_all[(sh_all["amh"] <= 1.2) & sh_all["oocytes"].notna()].copy()
low["agegrp"] = np.where(low["age"] >= 35, "≥35 y", "<35 y")
prots = ["GnRH antagonist", "GnRH agonist", "Mild/mini-IVF", "PPOS"]
agg = low.groupby(["agegrp", "protocol_class"]).agg(
    n=("oocytes", "size"), mean_ooc=("oocytes", "mean"),
    por=("por", "mean"), blast=("blastocyst_formed", "mean")).reset_index()
fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.0))
pc = {"GnRH antagonist": "#1f77b4", "GnRH agonist": "#ff7f0e",
      "Mild/mini-IVF": "#d62728", "PPOS": "#2ca02c"}
w = 0.2
xpos = np.arange(2)
for k, (ax, col, ylab, ttl) in enumerate(zip(axs,
        ["mean_ooc", "por", "blast"],
        ["Mean oocytes retrieved", "POR rate", "Mean blastocysts per cycle"],
        ["a  Oocyte yield", "b  Bologna POR rate", "c  Blastocyst yield"])):
    for j, pr in enumerate(prots):
        vals = []
        for ag in ["≥35 y", "<35 y"]:
            rr = agg[(agg.agegrp == ag) & (agg.protocol_class == pr)]
            vals.append(rr[col].values[0] if len(rr) else np.nan)
        ax.bar(xpos + (j - 1.5) * w, vals, width=w, color=pc[pr], label=pr)
    ax.set_xticks(xpos); ax.set_xticklabels(["Age ≥35 y", "Age <35 y"])
    ax.set_ylabel(ylab); ax.set_title(ttl, loc="left", fontsize=9.5)
axs[0].set_ylim(0, 5.8)          # headroom so the legend clears the bars
axs[0].legend(frameon=False, fontsize=6.5, loc="upper right")
agg.to_csv(f"{BASE}/tables/lowreserve_protocol_stats.csv", index=False)
fig.tight_layout()
save_all(fig, "fig5_protocol_lowreserve")
plt.close(fig)
print("fig5 done; low-reserve n =", len(low))

# =====================================================================
# SUPP FIGURE -- subgroup AUC
# =====================================================================
fig, ax = plt.subplots(figsize=(5.2, 3.4))
groups = []
for centre, s in [("Zunyi", ext_dict["Zunyi"][0]), ("Jining", ext_dict["Jining"][0])]:
    ye = s["por"].astype(int).values; pe = ext_dict[centre][2]
    for lbl, m in [("Age <35", s["age"] < 35), ("Age ≥35", s["age"] >= 35),
                   ("BMI <24", s["bmi"] < 24), ("BMI ≥24", s["bmi"] >= 24),
                   ("AMH <1.2", s["amh"] < 1.2), ("AMH ≥1.2", s["amh"] >= 1.2)]:
        sub = ye[m.values]
        if len(np.unique(sub)) < 2: continue
        groups.append((centre, lbl, int(m.sum()), roc_auc_score(sub, pe[m.values])))
sht = sh.iloc[te]
yht = y[te]; pht = p_rf_te
for lbl, m in [("Age <35", sht["age"] < 35), ("Age ≥35", sht["age"] >= 35),
               ("BMI <24", sht["bmi"] < 24), ("BMI ≥24", sht["bmi"] >= 24),
               ("AMH <1.2", sht["amh"] < 1.2), ("AMH ≥1.2", sht["amh"] >= 1.2)]:
    sub = yht[m.values]
    if len(np.unique(sub)) < 2: continue
    groups.append(("Shanghai", lbl, int(m.sum()), roc_auc_score(sub, pht[m.values])))
gd = pd.DataFrame(groups, columns=["cohort", "subgroup", "n", "auc"])
ylabels = []
for i, (ix, r) in enumerate(gd.iterrows()):
    yy = len(gd) - i
    cc = COLR.get(r.cohort, C_GREY)
    ax.plot([r.auc - 0.05, r.auc + 0.05], [yy, yy], color=cc, lw=1.4)
    ax.plot([r.auc], [yy], "o", color=cc, ms=3.5)
    ylabels.append(f"{r.cohort} · {r.subgroup} (n={r.n})")
ax.set_yticks(range(1, len(gd) + 1))
ax.set_yticklabels(ylabels[::-1], fontsize=7)
ax.set_xlabel("AUC (±SE approximation)")
ax.set_xlim(0.50, 1.0)
ax.set_title("Subgroup discrimination (transported model)", loc="left", fontsize=9.5)
fig.tight_layout()
save_all(fig, "sfig1_subgroup_auc")
plt.close(fig)
gd.to_csv(f"{BASE}/tables/stable3_subgroup_auc.csv", index=False)
print("sfig1 done")

# =====================================================================
# TABLE 1 -- baseline characteristics
# =====================================================================
def fmt_row(name, s):
    oc = s["oocytes"].dropna()
    return {
        "Cohort": name,
        "n": f"n={len(s)}",
        "_age": f"{s['age'].mean():.1f} ± {s['age'].std():.1f}",
        "_bmi": f"{s['bmi'].mean():.1f} ± {s['bmi'].std():.1f}",
        "_amh": f"{s['amh'].median():.2f} ({s['amh'].quantile(.25):.2f}–{s['amh'].quantile(.75):.2f})",
        "_fsh": f"{s['base_fsh'].mean():.1f} ± {s['base_fsh'].std():.1f}",
        "_ooc": f"{oc.median():.1f} ({oc.quantile(.25):.1f}–{oc.quantile(.75):.1f})" if len(oc) else "—",
        "_por": f"{int(s['por'].sum())} ({s['por'].mean()*100:.1f}%)",
    }

rows = []
rows.append(fmt_row("Shanghai derivation (first cycle per patient)", sh))
for c in ["Zunyi", "Jining"]:
    s = ext[ext["center"] == c].copy()
    rows.append(fmt_row(f"{c} prospective (transported evaluation)", s))
t1 = pd.DataFrame(rows)
t1.to_csv(f"{BASE}/tables/table1_baseline.csv", index=False)

md = ["| Cohort | n | Age (y) | BMI (kg/m²) | AMH, ng/mL | Basal FSH, IU/L | Oocytes retrieved | POR, n (%) |",
      "|---|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['Cohort']} | {r['n']} | {r['_age']} | {r['_bmi']} | {r['_amh']} | {r['_fsh']} | {r['_ooc']} | {r['_por']} |")
open(f"{BASE}/tables/table1_baseline.md", "w", encoding="utf-8").write("\n".join(md))
print("table1 done")
print("\n".join(md))

# =====================================================================
# TABLE 2 -- performance (transported / recalibrated / refitted)
# =====================================================================
from scipy.special import logit as _logit
t2rows = []
for centre in ["Zunyi", "Jining"]:
    s, ye, pe = ext_dict[centre]
    a, lo, hi = boot_auc(ye, pe)
    ap = average_precision_score(ye, pe)
    lp = _logit(np.clip(pe, 1e-6, 1-1e-6)).reshape(-1, 1)
    rec = LogisticRegression(penalty=None, max_iter=1000).fit(lp, ye)
    pe_rc = rec.predict_proba(lp)[:, 1]
    refit = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=10,
                                   random_state=42, n_jobs=-1).fit(s[FEATS], ye)
    pe_rf2 = refit.predict_proba(s[FEATS])[:, 1]
    # calibration slope/intercept
    Xc = np.column_stack([np.ones_like(lp), lp])
    cal = LogisticRegression(penalty=None, max_iter=1000).fit(Xc, ye)
    t2rows.append(dict(cohort=centre, model="Transported", n=len(ye), events=int(ye.sum()),
                       auc=a, lo=lo, hi=hi, auprc=ap, brier=brier_score_loss(ye, pe),
                       cal_int=cal.coef_[0][0], cal_slope=cal.coef_[0][1],
                       oe=ye.sum()/pe.sum()))
    a2, lo2, hi2 = boot_auc(ye, pe_rc)
    lp_rc = _logit(np.clip(pe_rc, 1e-6, 1-1e-6)).reshape(-1, 1)
    Xc2 = np.column_stack([np.ones_like(lp_rc), lp_rc])
    cal2 = LogisticRegression(penalty=None, max_iter=1000).fit(Xc2, ye)
    t2rows.append(dict(cohort=centre, model="Recalibrated", n=len(ye), events=int(ye.sum()),
                       auc=a2, lo=lo2, hi=hi2, auprc=average_precision_score(ye, pe_rc),
                       brier=brier_score_loss(ye, pe_rc),
                       cal_int=cal2.coef_[0][0], cal_slope=cal2.coef_[0][1],
                       oe=ye.sum()/pe_rc.sum()))
    a3, lo3, hi3 = boot_auc(ye, pe_rf2)
    t2rows.append(dict(cohort=centre, model="Refitted", n=len(ye), events=int(ye.sum()),
                       auc=a3, lo=lo3, hi=hi3, auprc=average_precision_score(ye, pe_rf2),
                       brier=brier_score_loss(ye, pe_rf2), cal_int=np.nan, cal_slope=np.nan,
                       oe=np.nan))
t2 = pd.DataFrame(t2rows)
t2.to_csv(f"{BASE}/tables/table2_performance.csv", index=False)
md2 = ["| Cohort | Model | n | Events | AUC (95% CI) | AUPRC | Brier | Calibration intercept | Calibration slope | O/E |",
       "|---|---|---|---|---|---|---|---|---|---|"]
for r in t2rows:
    md2.append(f"| {r['cohort']} | {r['model']} | {r['n']} | {r['events']} | "
               f"{r['auc']:.3f} ({r['lo']:.3f}–{r['hi']:.3f}) | {r['auprc']:.3f} | {r['brier']:.4f} | "
               f"{'—' if np.isnan(r['cal_int']) else f'{r[chr(99)+chr(97)+chr(108)+chr(95)+chr(105)+chr(110)+chr(116)]:.2f}'} | "
               f"{'—' if np.isnan(r['cal_slope']) else f'{r[chr(99)+chr(97)+chr(108)+chr(95)+chr(115)+chr(108)+chr(111)+chr(112)+chr(101)]:.2f}'} | "
               f"{'—' if np.isnan(r['oe']) else f'{r[chr(111)+chr(101)]:.2f}'} |")
open(f"{BASE}/tables/table2_performance.md", "w", encoding="utf-8").write("\n".join(md2))
print("\n".join(md2))

# classification at thresholds -> supplementary table
cm_rows = []
for centre in ["Zunyi", "Jining"]:
    s, ye, pe = ext_dict[centre]
    for t in [0.20, 0.30, 0.40, 0.159]:
        yh = (pe >= t)
        tp = ((yh) & (ye == 1)).sum(); fp = ((yh) & (ye == 0)).sum()
        fn = ((~yh) & (ye == 1)).sum(); tn = ((~yh) & (ye == 0)).sum()
        cm_rows.append(dict(cohort=centre, threshold=t,
                            sens=tp/(tp+fn), spec=tn/(tn+fp),
                            ppv=tp/(tp+fp), npv=tn/(tn+fn)))
pd.DataFrame(cm_rows).to_csv(f"{BASE}/tables/stable4_thresholds.csv", index=False)

# missingness supplementary table
raw_w = raw[(raw["cycle_type"]=="新鲜周期")].copy()
mm = {}
for col, lab in [("age","Age"),("bmi","BMI"),("amh","AMH"),("base_fsh","Basal FSH"),
                 ("oocytes_total","Oocyte count")]:
    mm[lab] = raw_w[col].isna().mean()*100
pd.DataFrame({"Predictor": list(mm), "Missing % (Shanghai registry)": [f"{v:.1f}" for v in mm.values()]}
             ).to_csv(f"{BASE}/tables/stable2_missingness_shanghai.csv", index=False)

# funnel supplementary
funnel_lines = ["| Step | n |", "|---|---|",
    "| Cycle records in Shanghai EHR, 2010–2026 | 103,899 |",
    "| Fresh retrieval cycles | 55,487 |",
    "| Initiated Jan 2021 – Oct 2026 | 31,336 |",
    "| IVF or ICSI | 29,087 |",
    "| Stimulated (natural-cycle and HRT-preparation excluded) | 28,420 |",
    "| Four predictors complete and plausible | 20,248 |",
    "| **Derivation analysis set: first cycle per patient** | **11,749** |",
    "| → split 80:20: derivation 9,399 / internal hold-out 2,350 | |",
    "| Zunyi prospective registry (frozen-model evaluation) | 1,040 |",
    "| Jining prospective registry (frozen-model evaluation) | 245 |"]
open(f"{BASE}/tables/stable1_funnel.md", "w", encoding="utf-8").write("\n".join(funnel_lines))
print("supplementary tables done")
print("ALL FIGURES/TABLES COMPLETE")
