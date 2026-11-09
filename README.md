# Code for: Prospective external validation of a four-predictor model for poor ovarian response

This repository contains the complete analysis pipeline for the npj Digital Medicine
manuscript on transporting a four-predictor poor ovarian response (POR) model
(age, BMI, anti-Müllerian hormone, basal FSH) from a Shanghai EHR derivation cohort
to two prospective external validation cohorts (Zunyi, Jining).

## Pipeline (run in order)

| Script | Purpose |
|---|---|
| `code/extract_shanghai_raw.py` | Parse the raw Shanghai EHR cycle export |
| `code/extract_shanghai.py` | Assemble the Shanghai analysis tables (all cycles / cycles / analysis) |
| `code/extract_zjy_cohort.py` | Map and assemble the Zunyi–Jining prospective registry export |
| `code/02_build_cohorts.py` | Cohort definitions, eligibility, Bologna POR labelling |
| `code/03_build_definitive_cohorts.py` | Definitive derivation cohorts (first complete cycle per patient; n = 11,749) |
| `code/01_main_analysis.py` | Model development (random forest, L2 logistic, AMH-only), internal validation (repeated 5×10-fold CV, bootstrap optimism correction), hold-out evaluation |
| `code/04_figures_tables.py` | Transported / recalibrated / refitted external evaluation, figures and tables |
| `code/05_rebuild_eligible_withmissing.py` | Full eligible population rebuild (16,311 first eligible cycles) and funnel audit |
| `code/06_mi_sensitivity.py` | Multiple-imputation sensitivity analysis (MICE-style, M = 10, Rubin pooling) |
| `code/10_extval_samplesize.py` | Simulation-based external-validation sample size calculation (refs: Snell 2021; Riley 2024 part 3) |

`por_models.joblib` contains the trained scikit-learn objects
(random forest, L2-penalised logistic pipeline, AMH-only model, decision thresholds)
with `feature_names_in_` = [age, bmi, amh, base_fsh].

## Environment

Python 3.12; scikit-learn 1.5+ (development used 1.9.1), pandas, numpy, statsmodels 0.15,
joblib, matplotlib. All stochastic steps use fixed seeds (42 for the train–hold-out split;
20261105 for bootstrap / cross-validation / imputation pipelines) and reproduce every
number in the manuscript exactly.

## Data availability

The package intentionally contains **no patient-level data**. De-identified individual
participant data for the derivation dataset and the prospective registry cohorts will be
deposited in a public repository upon acceptance; in the interim, requests can be addressed
to the corresponding author.
