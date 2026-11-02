# 04 · ML Model Evaluation

How the saved ML models were evaluated, first by the ML stage itself (`ML/ML/src/cargomind_ml/evaluate_models.py`) and then by the independent evaluation stage (`ML/Evaluation/`). It covers what the results show and how far they can be trusted.

**Related documents:** split and dataset → [01_ML_Dataset_Engineering.md](01_ML_Dataset_Engineering.md). EDA → [02_ML_Exploratory_Data_Analysis.md](02_ML_Exploratory_Data_Analysis.md). Models and thresholds → [03_Machine_Learning_Development.md](03_Machine_Learning_Development.md).

**Sources inspected:**
- `ML/ML/results/**`, `ML/ML/reports/ML_model_report.md`
- `ML/Evaluation/src/cargomind_eval/*.py`, `ML/Evaluation/scripts/run_evaluation.py`
- `ML/Evaluation/metrics/*.csv`, `ML/Evaluation/validation/*`
- `ML/Evaluation/reports/{evaluation_report,model_comparison,limitations}.md`

**Rule used in this document:** every table states the set it comes from. Validation, CV and test numbers are never mixed in one column.

---

## A. Evaluation objectives

| Objective | Where |
|---|---|
| Out-of-time performance of the four classifiers vs a majority-class reference | ML stage: `evaluate_models.evaluate_supervised` |
| Uncertainty that respects dependence between shipments (shared port-days) | ML: `day_block_bootstrap`; Evaluation: `uncertainty.py` (paired, day and port×day blocks) |
| Per-class behaviour, especially the costly Critical → On-Time error | `classification_metrics`; Evaluation `supervised.per_class_table`, `confusion_outputs`, `boundary_analysis` |
| Probability quality | Evaluation `calibration.py` |
| Operational decision policies | Evaluation `operational.py` |
| Anomaly detectors as a separate task | ML `evaluate_anomaly`; Evaluation `anomaly.py` |
| Temporal stability, prevalence shift, feature drift, label availability | Evaluation `temporal.py` |
| Robustness / ablations | ML `ablations_and_sensitivity`; Evaluation `robustness.py`, `refits.ablation_predictions` |
| Technical verification of the saved artifacts | Evaluation `reproducibility.py`; 55 / 55 checks pass (`ML/Evaluation/validation/reproducibility_checks.md`) |

Principles of the independent evaluation (`evaluation_report.md` §2):
- **Saved models are loaded unchanged.** Nothing in `ML/ML/models/` or `ML/ML/results/` is written.
- **Metrics are re-implemented** from confusion matrices and rank statistics (`cargomind_eval.metrics`). All 234 stored values match to within 5e-8, except float32 log loss for XGBoost.
- **No test-set tuning.** Thresholds and recalibrators are fitted on pooled out-of-fold validation predictions (3 CV blocks, 2,905 shipments) and applied once to test.
- **Refits are labelled "evaluation-only".** They clone the saved pipelines and are used only where an output was missing. They reproduce the stored CV scores and ablation results exactly.

Reproduce with `.venv/bin/python ML/Evaluation/scripts/run_evaluation.py`. It takes about 2 minutes (`run_info.json`: 127 s) and records the package versions: Python 3.9.6, scikit-learn 1.6.1, XGBoost 2.1.4, LightGBM 4.6.0, SHAP 0.46.0.

---

## B. Split verification

Source: `ML/Evaluation/validation/split_verification.json` (`temporal.verify_split`).

| Check | Result |
|---|---|
| Ordering train < purge < test | true |
| Observed gap, last train departure → first test departure | **73 h** (≥ 72 h required) |
| Shipment-ID overlap train/test, purge/any | 0 / 0 |
| Test fraction of observed shipments | 0.2505 |
| Split rebuilt from `ML/Dataset/Processed` = stored `splits.csv` | true |
| Modelling table rebuilt = stored `modelling_table.pkl` | true (`reproducibility_checks.md`) |
| CV fold indices inside the training period; gap per fold | true; 73 h each |
| Test data passed to `RandomizedSearchCV` | no (only `ctx["train"]`; per-fold scores reproduced from training rows) |
| Training labels not yet arrived at test start | 2,038 of 6,452 (32 %) |
| Fold training labels not yet arrived at the fold's validation start | fold 0: 1,754 (61.8 %); fold 1: 1,944 (51.2 %); fold 2: 2,023 (42.0 %) |
| Test labels known by the end of the test period | 20.6 % (arrivals run to 2025-03-13) |

Split sizes and dates are in [01 · §G](01_ML_Dataset_Engineering.md#g-data-splitting-and-preprocessing).

---

## C. Evaluation metrics

| Metric | Definition / implementation | Why |
|---|---|---|
| Macro-F1 (primary) | mean of per-class F1 | treats the three classes equally; it was the tuning metric |
| Weighted F1, accuracy | sklearn | reported only alongside per-class results: the majority reference reaches 61.9 % accuracy while detecting no delays |
| Per-class precision / recall / F1 | `precision_recall_fscore_support` | Critical recall is the operational priority |
| ROC-AUC (OvR macro / weighted / per class) | `roc_auc_score(multi_class="ovr")`; Evaluation `metrics.auc_rank` | ranking quality, independent of the threshold |
| Log loss, multiclass Brier (0–2), ECE (top-label, classwise, 10 bins) | `metrics.log_loss`, `brier_multiclass`, `ece_top_label`, `ece_classwise` | probability quality |
| Ordinal MAE | mean \|predicted − true class\| on the 0/1/2 scale | the classes are ordinal |
| Critical → On-Time rate | `((pred==0)&(y==2)).sum() / (y==2).sum()` | the costliest error |
| Anomaly metrics | precision / recall / F1 of `score > threshold`; ROC-AUC and PR-AUC of the score; flag rate per class; Cliff's δ; Kruskal p | positive = delayed (At-Risk or Critical), or Critical only |
| Uncertainty | 1,000 cluster-bootstrap resamples (`BOOTSTRAP_REPS`), percentile 95 % intervals; **day blocks** (11 test days) primary, **origin-port × day** (110) secondary; paired for comparisons | rows that share a port-day are not independent |

---

## D. Model comparison

### D.1 Tuning (time-series CV, mean of 3 folds)

Source: `ML/ML/results/comparison/supervised_comparison_cv.csv`.

| Model | CV Macro-F1 | sd | CV Critical recall | CV At-Risk recall |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.633 | 0.015 | 0.692 | 0.648 |
| Random Forest | 0.583 | 0.007 | 0.493 | 0.604 |
| XGBoost | 0.621 | 0.017 | 0.554 | 0.627 |
| LightGBM | 0.625 | 0.016 | 0.577 | 0.641 |

### D.2 Validation block (CV fold 2, 970 shipments, 25.9 % Critical)

Source: `ML/ML/results/comparison/supervised_comparison_validation.csv`. Each model is refit on the fold's training rows.

| Model | Accuracy | Macro-F1 | Weighted F1 | Critical recall |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.632 | 0.638 | 0.637 | 0.705 |
| Random Forest | 0.587 | 0.574 | 0.585 | 0.363 |
| XGBoost | 0.629 | 0.628 | 0.632 | 0.554 |
| LightGBM | 0.641 | **0.642** | 0.645 | 0.602 |
| Majority-class reference | 0.386 | 0.186 | 0.215 | 0.000 |

### D.3 Test period, all shipments (n = 2,394)

Sources: `ML/ML/results/comparison/supervised_comparison_test.csv`, `ML/Evaluation/metrics/supervised_metrics.csv`, `supervised_metric_ci.csv`.

| Model | Accuracy | Macro Precision | Macro Recall | **Macro-F1** (95 % day CI) | Weighted F1 | **Critical recall** (95 % day CI) | Critical precision |
|---|---:|---:|---:|---|---:|---|---:|
| Logistic Regression | 0.729 | 0.650 | 0.660 | **0.654** (0.599–0.696) | 0.730 | **0.595** (0.444–0.706) | 0.557 |
| Random Forest | 0.706 | 0.640 | 0.618 | 0.625 (0.570–0.657) | 0.710 | 0.460 (0.278–0.577) | 0.569 |
| XGBoost | 0.709 | 0.645 | 0.648 | 0.644 (0.586–0.679) | 0.717 | 0.540 (0.343–0.662) | 0.563 |
| LightGBM | 0.710 | 0.649 | 0.634 | 0.636 (0.579–0.671) | 0.717 | 0.479 (0.286–0.611) | 0.575 |
| Majority-class reference | 0.619 | 0.206 | 0.333 | 0.255 (0.247–0.262) | 0.473 | 0.000 | 0.000 |

| Model (test) | At-Risk recall | ROC-AUC OvR macro | Log loss | Brier | Critical → On-Time |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.552 | **0.866** | **0.601** | **0.361** | 4.7 % |
| Random Forest | 0.605 | 0.849 | 0.700 | 0.413 | 7.4 % |
| XGBoost | 0.638 | 0.861 | 0.610 | 0.370 | 5.6 % |
| LightGBM | **0.652** | 0.858 | 0.616 | 0.374 | **3.7 %** |

### D.4 Test period, sea shipments only

Source: `ML/ML/results/comparison/supervised_comparison_test_sea.csv`.

| Model | Accuracy | Macro-F1 | Critical recall |
|---|---:|---:|---:|
| Logistic Regression | 0.714 | 0.650 | 0.595 |
| Random Forest | 0.689 | 0.619 | 0.460 |
| XGBoost | 0.692 | 0.637 | 0.540 |
| LightGBM | 0.694 | 0.631 | 0.479 |
| Majority-class reference | 0.597 | 0.249 | 0.000 |

Sea-only results are about 0.005 lower for every model. None of the 128 test air shipments is Critical, so Critical results describe sea freight only.

### D.5 Per-class results (test)

Source: `ML/Evaluation/metrics/per_class_metrics.csv`.

| Model | On-Time P / R / F1 | At-Risk P / R / F1 | Critical P / R / F1 | OvR ROC-AUC (OT / AR / C) |
|---|---|---|---|---|
| Logistic Regression | 0.839 / 0.832 / 0.836 | 0.553 / 0.552 / 0.552 | 0.557 / 0.595 / 0.575 | 0.885 / 0.793 / 0.921 |
| Random Forest | 0.837 / 0.789 / 0.812 | 0.513 / 0.605 / 0.555 | 0.569 / 0.460 / 0.509 | 0.867 / 0.762 / 0.919 |
| XGBoost | 0.858 / 0.767 / 0.810 | 0.515 / 0.638 / 0.570 | 0.563 / 0.540 / 0.551 | 0.878 / 0.785 / 0.920 |
| LightGBM | 0.860 / 0.770 / 0.813 | 0.512 / 0.652 / 0.573 | 0.575 / 0.479 / 0.523 | 0.876 / 0.781 / 0.919 |

**Critical class (test, 215 shipments).**

| Model | Caught | Missed | → On-Time | → At-Risk | False Critical alerts (from On-Time / At-Risk) |
|---|---:|---:|---:|---:|---|
| Logistic Regression | 128 | 87 | 10 | 77 | 102 (15 / 87) |
| Random Forest | 99 | 116 | 16 | 100 | 75 (11 / 64) |
| XGBoost | 116 | 99 | 12 | 87 | 90 (13 / 77) |
| LightGBM | 103 | 112 | 8 | 104 | 76 (10 / 66) |

**Interpretation:**
- All four models **rank** Critical shipments equally well (ROC-AUC 0.919–0.921), and their Critical precision is statistically the same (0.557–0.575, about ±0.14).
- LR's higher Critical recall comes from where its argmax decision falls, not from better ranking.
- At-Risk is the least separable class (OvR AUC 0.76–0.79). Errors cluster near the thresholds: LR's error rate is 48 % at 10–12 h, 61 % at 12–14 h and 55 % at 48–50 h, against 10 % at ≤ 0 h and 3 % at > 96 h (`metrics/boundary_analysis.csv`).
- LR makes only 25 On-Time ↔ Critical errors in 2,394 shipments.
- Collapsing to "delayed (> 12 h)" vs On-Time, LR alerts on 677 of 913 delayed shipments (74 %), and 677 of its 926 alerts are correct (73 %).

Confusion figures:
- `ML/ML/results/figures/confusion_matrices_{validation,test,test_sea}.png`
- `ML/Evaluation/figures/confusion_matrices/*.png`

---

## E. Operational threshold policies

Source: `ML/Evaluation/metrics/operational_policies.csv`, `figures/operational/critical_alert_tradeoff.png` (`cargomind_eval.operational`).

Policy family `decide(p, t)`: predict Critical if P(Critical) ≥ t; otherwise predict the more probable of On-Time and At-Risk. `t` is chosen on **pooled out-of-fold validation** predictions over a grid of 0.05–0.95 in steps of 0.01, then applied **once** to test.

| Model and policy | t | Val. Critical recall | Test Macro-F1 | Test Critical recall | Test Critical precision | Critical alerts / 1,000 | Missed Critical / 1,000 | False Critical / 1,000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LR, argmax (default) | — | 0.689 | 0.654 | 0.595 | 0.557 | 96 | 36 | 43 |
| LR, Critical recall ≥ 0.80 | 0.36 | 0.800 | 0.652 | 0.726 | 0.513 | 127 | 25 | 62 |
| LR, max macro-F1 | 0.56 | 0.614 | 0.666 | 0.498 | 0.699 | 64 | 45 | 19 |
| LR, min expected cost (assumed costs) | 0.22 | 0.917 | 0.577 | 0.800 | 0.359 | 200 | 18 | 128 |
| XGB, argmax | — | 0.552 | 0.644 | 0.540 | 0.563 | 86 | 41 | 38 |
| XGB, Critical recall ≥ 0.80 | 0.20 | 0.810 | 0.597 | 0.791 | 0.377 | 188 | 19 | 117 |
| XGB, max macro-F1 | 0.41 | 0.601 | 0.657 | 0.609 | 0.575 | 95 | 35 | 41 |
| XGB, min expected cost (assumed costs) | 0.06 | 0.953 | 0.425 | 0.953 | 0.197 | 435 | 4 | 350 |

Notes on the table:
- All values come from `operational_policies.csv`; per-1,000 values are rounded.
- The RF and LightGBM rows are in the same file.
- The other selected thresholds are RF 0.32 / 0.25 / 0.08 and LGBM 0.47 / 0.21 / 0.05 (max F1 / recall ≥ 0.80 / min cost). See [03 · §G](03_Machine_Learning_Development.md#g-probability-and-decision-thresholds).

The **min expected cost** policy uses an **assumed** cost matrix (`operational.COST`; rows are true classes, columns are predictions):

| True \ Predicted | On-Time | At-Risk | Critical |
|---|---:|---:|---:|
| On-Time | 0 | 0.5 | 1 |
| At-Risk | 3 | 0 | 0.5 |
| Critical | 10 | 4 | 0 |

It picks very low thresholds (0.05–0.22), which track the 25 % Critical prevalence of the validation period. On test these would raise 200–630 Critical alerts per 1,000 shipments. **It should not be adopted without real cost data.**

**Probability calibration** (`calibration.py`; test, from `metrics/calibration_metrics.csv`):

| Model | Log loss | Brier | ECE top-label | Mean P(Critical) vs observed |
|---|---:|---:|---:|---|
| Logistic Regression | 0.601 | 0.361 | 0.059 | 0.139 vs 0.090 |
| Random Forest | 0.700 | 0.413 | 0.134 | 0.164 vs 0.090 |
| XGBoost | 0.610 | 0.370 | 0.033 | 0.127 vs 0.090 |
| LightGBM | 0.616 | 0.374 | 0.032 | 0.126 vs 0.090 |

Two corrections were tested, both fitted without test data:
- `prior_correct(p, prior)` computes `q = p · (training class share × 3)` and renormalises.
- `vector_scaler` fits a multinomial logistic regression on log-probabilities, using the pooled out-of-fold validation predictions.

| Model | Correction | Test log loss | Test ECE top-label | Test Critical recall (argmax) |
|---|---|---:|---:|---:|
| LR | raw | 0.601 | 0.059 | 0.595 |
| LR | prior-corrected | 0.569 | 0.031 | 0.451 |
| LR | vector scaling | 0.568 | 0.028 | 0.470 |
| XGB | prior-corrected | 0.573 | 0.014 | 0.391 |
| XGB | vector scaling | 0.583 | 0.054 | 0.591 |

Prior correction improves log loss, Brier and ECE for all four models but lowers argmax Critical recall. This is why the evaluation recommends separating **display** (prior-corrected probabilities) from **decision** (an explicit P(Critical) threshold).

**Recommended operating policy** (evaluation report §15):
- Use LR with **t ≈ 0.36** on P(Critical). On test this gave 73 % Critical recall with 62 false Critical alerts per 1,000 shipments.
- Show prior-corrected probabilities on the dashboard.
- Show Isolation Forest as a separate "unusual port conditions" flag.

None of this is implemented in the serving adapter ([03 · §H](03_Machine_Learning_Development.md#h-inference-architecture)).

---

## F. Anomaly detection evaluation

Sources:
- `ML/ML/results/comparison/anomaly_comparison_*.csv`
- `ML/Evaluation/metrics/anomaly_*.csv`
- `ML/Evaluation/validation/anomaly_seed_stability.csv`
- `ML/Evaluation/figures/anomaly_distributions/`

Thresholds: IF 0.5609, OCSVM 75.87 ([03 · §E](03_Machine_Learning_Development.md#e-anomaly-detection)).

**Validation (fold-2 block; selection set).** Source: `models/*_config.json`.

| Detector | PR-AUC delayed | ROC-AUC delayed | PR-AUC Critical |
|---|---:|---:|---:|
| Isolation Forest | **0.764** | 0.673 | 0.556 |
| One-Class SVM | 0.755 | 0.653 | 0.490 |

**Test.**

| Test, positive = delayed (prevalence 0.381) | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |
|---|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 0.882 | 0.180 | 0.298 | 0.677 | 0.628 | 0.015 |
| One-Class SVM | 0.831 | 0.183 | 0.300 | 0.653 | 0.601 | 0.023 |

| Test, positive = Critical (prevalence 0.090) | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |
|---|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 0.452 | 0.391 | 0.419 | 0.792 | 0.451 | 0.047 |
| One-Class SVM | 0.448 | 0.419 | 0.433 | 0.764 | 0.432 | 0.051 |

**Flag rates by class (test).**

| Detector | Overall | On-Time | At-Risk | Critical | Share of flags that are Critical | Critical lift | Cliff's δ Critical vs On-Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 7.8 % | 1.5 % | 11.5 % | 39.1 % | 45 % | 5.0× | 0.650 |
| One-Class SVM | 8.4 % | 2.3 % | 11.0 % | 41.9 % | 45 % | 5.0× | 0.580 |

**Comparison and stability:**
- IF − OCSVM, paired on test:
  - delayed ROC-AUC **+0.024**, with an interval excluding 0 under both block schemes;
  - delayed PR-AUC +0.026 (day blocks −0.000 to +0.049; port-day blocks +0.004 to +0.049);
  - Critical PR-AUC +0.019, not distinguishable.
- Across 6 seeds or subsamples:
  - IF delayed PR-AUC 0.628–0.635, Spearman ≥ 0.989 with the saved model, flag-set Jaccard ≥ 0.83;
  - OCSVM 0.598–0.602, ≥ 0.997, ≥ 0.88.
- Fit time: IF 0.35 s, OCSVM 0.06 s on 3,000 rows. Model size: IF 4.3 MB, OCSVM 0.1 MB. Both are machine-specific measurements.

**Value beyond LR** (`metrics/anomaly_incremental_value.csv`):

| Group | Critical rate |
|---|---:|
| LR predicts Critical, IF flags | 71.7 % (n = 99) |
| LR predicts Critical, no flag | 43.5 % (n = 131) |
| LR predicts At-Risk, IF flags | 18.6 % |
| LR predicts At-Risk, no flag | 10.2 % |

- IF recovers only 13 of the 87 Critical shipments that LR misses (14.9 %).
- The pair-weighted AUC within deciles of LR's P(Critical) is 0.62.

**Reading:**
- A flag is a high-precision warning of extreme origin conditions, not a delay classifier.
- Recall is low by design. The threshold is a 5 % false-alarm budget, and many delays arise under conditions that look normal at the origin or at the destination or hub, which the detectors do not see.
- Validation anomaly metrics are partly in-sample: the final fit includes the validation block's On-Time rows, so the On-Time flag rate is 5.1 % on validation vs 1.5 % on test.

---

## G. Robustness and statistical analysis

### G.1 Ablations and sensitivity (test, tuned hyperparameters, refit only)

Source: `ML/ML/results/comparison/ablation_and_sensitivity.csv`.

**Test macro-F1:**

| Variant | LR | RF | XGB | LGBM |
|---|---:|---:|---:|---:|
| Full model (reference) | 0.654 | 0.625 | 0.644 | 0.636 |
| No port/carrier identity | 0.674 | 0.635 | 0.626 | 0.629 |
| No operational state (booking-time info only) | 0.360 | 0.431 | 0.427 | 0.433 |
| No historical delay rates | 0.665 | 0.609 | 0.645 | 0.648 |
| Train only on labels available at test start | 0.633 | 0.603 | 0.609 | 0.618 |
| Exclude shipments with filled inputs | 0.655 | 0.633 | 0.645 | 0.643 |
| Sea shipments only | 0.650 | 0.619 | 0.637 | 0.642 |
| Prediction 24 h before departure | 0.422 | 0.475 | 0.461 | 0.458 |

**Paired differences vs the full model.** Source: `ML/Evaluation/validation/robustness_analysis.csv`. Δ is on the same test shipments, with 95 % day-block CIs; **bold** means the interval excludes 0.

| Variant | Δ macro-F1 LR | RF | XGB | LGBM | Δ Critical recall LR | RF | XGB | LGBM |
|---|---|---|---|---|---|---|---|---|
| No operational state | **−0.294** | **−0.194** | **−0.216** | **−0.203** | −0.019 | −0.135 | −0.233 | −0.130 |
| 24 h before departure | **−0.232** | **−0.150** | **−0.182** | **−0.179** | +0.028 | **−0.219** | **−0.288** | **−0.242** |
| Labels known at test start only | **−0.022** | −0.023 | **−0.035** | −0.019 | −0.023 | **−0.172** | **−0.181** | **−0.116** |
| No port/carrier identity | **+0.020** | +0.010 | −0.017 | −0.007 | **+0.047** | +0.005 | −0.033 | +0.042 |
| No historical delay rates | **+0.011** | −0.016 | +0.001 | +0.012 | +0.009 | **−0.047** | −0.047 | +0.009 |
| Exclude filled inputs | +0.005 | +0.006 | −0.002 | +0.005 | +0.005 | 0.000 | −0.035 | **+0.060** |
| Sea only | −0.000 | −0.001 | −0.000 | +0.011 | 0.000 | −0.005 | −0.028 | **+0.065** |

**Findings:**
- **Operational state is essential.** Without it, macro-F1 falls to 0.36–0.43.
- **The cutoff matters.** At 24 h before departure, LR keeps Critical recall (0.623) only by over-flagging: its Critical precision falls from 0.557 to 0.162. This matches the EDA's signal-decay finding ([02 · §H](02_ML_Exploratory_Data_Analysis.md#h-integrated-shipment-level-analysis-notebook-08-sections-h1h8)).
- **Realistic label availability** costs the trees 0.12–0.18 Critical recall, but LR only 0.023. This is the strongest argument for LR.
- **Data-cleaning choices** (filled inputs, air) do not drive the results.
- **Slices** (`validation/slice_performance.csv`):
  - LR Critical recall is 0.83 with an active origin disruption and 0.47 without one.
  - Per-port Critical recall ranges from 0.00 (DEHAM, 6 Critical; INMAA, 2) to 0.89 (INNSA, 46), and most per-port counts are too small to interpret.

### G.2 Pairwise statistical comparison (test, paired cluster bootstrap)

Source: `ML/Evaluation/metrics/pairwise_comparisons.csv`. Day blocks are shown.

| Comparison | Macro-F1 Δ (95 % CI) | Critical recall Δ (95 % CI) | Log loss Δ | ROC-AUC macro Δ | Verdict |
|---|---|---|---|---|---|
| LR − RF | +0.029 (+0.000, +0.066)* | **+0.135** (+0.071, +0.220) | **−0.100** | **+0.017** | LR better on Critical recall and probabilities; macro-F1 borderline |
| LR − XGB | +0.011 (−0.013, +0.042) | +0.056 (−0.009, +0.151) | −0.009 (n.s.) | +0.005 (n.s.) | **not distinguishable** |
| LR − LGBM | +0.018 (−0.005, +0.049) | **+0.116** (+0.031, +0.211) | **−0.015** | **+0.008** | LR better on Critical recall and probabilities |
| XGB − LGBM | +0.007 (−0.014, +0.026) | +0.060 (+0.000, +0.103)† | **−0.006** | +0.003 (n.s.) | practically equivalent |

\* Port-day blocks: −0.004 to +0.066. † With port-day blocks the interval excludes 0.

Cautions:
- Only 11 test days are available, and percentile bootstraps with few clusters under-cover.
- Port-day blocks assume ports are independent within a day, which is optimistic.
- **Differences below about 0.02 macro-F1 should be treated as ties.**

### G.3 Temporal stability and drift

Sources: `metrics/temporal_performance.csv`, `class_prevalence_by_day.csv`, `critical_precision_vs_prevalence.csv`, `feature_drift_train_vs_test.csv`.

**CV folds vs test:**

| Model | CV macro-F1 (fold 0 / 1 / 2) | Test macro-F1 | CV Critical recall (fold 0 / 1 / 2) | Test Critical recall |
|---|---|---:|---|---:|
| LR | 0.649 / 0.613 / 0.638 | 0.654 | 0.762 / 0.607 / 0.705 | 0.595 |
| XGB | 0.636 / 0.598 / 0.628 | 0.644 | 0.605 / 0.504 / 0.554 | 0.540 |
| LGBM | 0.627 / 0.604 / 0.642 | 0.636 | 0.641 / 0.488 / 0.602 | 0.479 |
| RF | 0.586 / 0.589 / 0.574 | 0.625 | 0.704 / 0.413 / 0.363 | 0.460 |

**Prevalence shift:**
- Daily Critical share ranges from 3 % to 43 % in training. It is 23–26 % in the CV blocks, 15.7 % on test days 1–3 and 5–10 % on later test days.
- Holding each model's test recall and FPR fixed, LR's Critical precision would be 0.727 at training prevalence, 0.816 at validation prevalence, 0.557 at test prevalence and 0.401 at 5 %.
- Daily Critical ROC-AUC stays between 0.83 and 0.97, but recall falls on low-prevalence days. On those days the Critical shipments are less extreme.

**Feature drift (PSI, train vs test, `temporal.psi`, 10 quantile bins):**

| Feature | PSI | Train mean | Test mean |
|---|---:|---:|---:|
| `hist_carrier_mean_delay_h` | 3.99 | 9.91 | 15.34 |
| `hist_carrier_pct_delayed` | 3.11 | 30.3 | 40.7 |
| `hist_destination_port_id_pct_delayed` | 2.82 | 29.8 | 40.8 |
| `hist_origin_port_id_pct_delayed` | 2.51 | 29.5 | 40.7 |
| `hol_h_days_in_voyage_sched` | 1.31 | 0.86 | 0.59 |
| `h_util_mean24` | 0.55 | 73.7 | 73.1 |
| `wx_h_gust_max24` | 0.32 | 39.9 | 37.0 |

The history drift is a **right-censoring** artefact: early in the dataset, "already arrived" shipments are mostly fast ones. Origin and destination operational-state features are stable (PSI < 0.3).

---

## H. Feature importance and explainability

Methods (`ML/ML/src/cargomind_ml/feature_importance.py`; permutation and SHAP use the test period):
- LR coefficients (log-odds per SD);
- RF impurity;
- XGBoost and LightGBM total gain;
- per-feature permutation importance (macro-F1 drop, `N_REPEATS = 5`);
- **group permutation**, which shuffles all columns of a group together and so handles correlated features;
- SHAP `TreeExplainer` for XGBoost and LightGBM.

One-hot levels and missing-value indicators are summed back to their 51 source columns. **None of this is causal.**

**Group permutation, drop in test macro-F1.** Source: `ML/ML/results/feature_importance/group_permutation_importance.csv` and ML report §5.

| Group | LR | RF | XGB | LGBM |
|---|---:|---:|---:|---:|
| Origin weather | 0.058 | 0.045 | 0.070 | 0.066 |
| Destination state at departure | 0.063 | 0.036 | 0.069 | 0.063 |
| Shipment attributes (mostly `shipping_mode`) | 0.058 | 0.028 | 0.037 | 0.035 |
| Origin disruptions | **0.079** | 0.011 | 0.019 | 0.011 |
| Origin congestion | 0.051 | 0.008 | 0.021 | 0.019 |
| Hub state | 0.031 | 0.005 | 0.026 | 0.023 |
| Anchor queue | 0.028 | 0.012 | 0.022 | 0.020 |
| Carrier | 0.020 | ≈ 0 | 0.002 | 0.006 |
| Calendar | 0.012 | ≈ 0 | 0.008 | 0.008 |
| Container dwell | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 |
| Historical delay rates | ≈ 0 | 0.017 | ≈ 0 | ≈ 0 |
| Route identity | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 |

For LR, permuting weather drops Critical recall by 0.130 and permuting disruptions drops it by 0.158 (same CSV).

**Other read-outs (ML report §5):**
- **SHAP (XGBoost, Critical class), mean |SHAP|:**

  | Feature | Mean \|SHAP\| |
  |---|---:|
  | `dis_d_active_any` | 0.58 |
  | `wx_o_gust_max24` | 0.43 |
  | `d_util_mean24` | 0.29 |
  | `o_anchor_mean24` | 0.21 |
  | origin holiday distance | 0.20 |
  | carrier | 0.17 |

  Figures: `ML/ML/results/figures/shap_critical_{xgboost,lightgbm}.png`.
- **LR coefficients** (`logistic_regression_coefficients.csv`):
  - air → On-Time +3.4;
  - destination disruption → Critical +0.89;
  - origin max gust → Critical +0.84;
  - Coral Bay Logistics → Critical +0.41;
  - Oceanic Lines → On-Time +0.86;
  - active origin weather event → Critical +0.50.
- **Gain and impurity disagree with permutation.** Trees spend 10–20 % of their gain on `hist_*` and utilisation, but permuting these barely changes predictions (`importance_share_by_group.csv`). This is the known split-importance bias, so permutation and SHAP are the more reliable read-outs.
- **The models learn conditions, not identities.** Removing identity changes little (§G.1), while removing operational state collapses performance.

**Explainability caveat.** Destination state at departure is a top-2 group for every model, even though the EDA found it weak on its own ([02 · §D](02_ML_Exploratory_Data_Analysis.md#d-port-operations-analysis-notebook-04-sections-d1d5)). The generator puts destination conditions at the departure hour directly into the delay formula ([01 · §B.5](01_ML_Dataset_Engineering.md#b5-shipments-and-the-delay-formula-section-8-lines-295366)). The feature is therefore not a timing leak, but it is a synthetic artefact that is unlikely to transfer to real data.

---

## I. Limitations and final verdict

**Limitations.** The full list of 25 items is in `ML/Evaluation/reports/limitations.md`. The main ones:
1. **Synthetic data.** The strongest signals are generator artefacts, and no metric here is evidence of real-world performance.
2. **One short test window:** 11 departure days, 215 Critical shipments, no Critical air shipments and no rolling-origin backtest.
3. **Labels used before they would be known** in CV and training (42–62 % per fold; 32 % at the test start), so CV is optimistic.
4. **Right-censored history features** (PSI 2.5–4.0). Removing them improves LR.
5. **Non-stationary prevalence.** Precision, F1, calibration and validation-chosen thresholds are period-specific.
6. **Balanced class weights distort the probabilities**, and RF's probabilities are poorly scaled (ECE 0.134).
7. **At-Risk is inherently noisy.** It is a 36 h band between two hard cut-offs.
8. **Detection is weak in quiet periods.** LR Critical recall is 0.47 without an active origin disruption.
9. **The anomaly detectors** have no ground truth, see only the origin port, have partly in-sample validation metrics and use a fixed-percentile threshold.
10. **Approximate intervals and an assumed cost matrix.** Few bootstrap clusters, and the costs are illustrative.

**Final verdict** (evaluation report §1, §13, §15; ML report §8):

| Question | Answer | Strength of evidence |
|---|---|---|
| Best overall classifier | **Logistic Regression** (test macro-F1 0.654; CV rule) | point estimate only vs XGB / LGBM; borderline vs RF |
| Best Critical detector | **Logistic Regression** (test Critical recall 0.595 at equal precision) | significant vs RF and LGBM, not vs XGB; reflects the default operating point, not better ranking |
| Probability quality | **Logistic Regression** (lowest log loss and Brier) | significant vs RF and LGBM |
| Robustness to label availability | **Logistic Regression** | trees lose 0.12–0.18 Critical recall |
| Most interpretable | **Logistic Regression** | — |
| Best anomaly detector | **Isolation Forest** | ROC-AUC significant; PR-AUC borderline |
| Prototype choice | **Logistic Regression** with a P(Critical) threshold of about 0.36, prior-corrected probabilities for display, Isolation Forest as a separate flag, and XGBoost as challenger | — |

The ML stage's selections are **confirmed in direction**, with two qualifications: LR's lead over XGBoost lies within statistical uncertainty, and "best Critical detector" reflects LR's operating point rather than better ranking.

**Before real-world deployment,** the evaluation requires:
- training and validating on real data;
- label-aware validation with rolling origins;
- censoring-aware or no history features;
- monitoring and recalibration for prevalence shift;
- thresholds set from real costs;
- longer and more varied evaluation periods;
- dropping or regularising LR's identity one-hots;
- validating the anomaly detectors against labelled port-incident data.
