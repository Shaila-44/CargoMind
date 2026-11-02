# CargoMind - Independent Model Evaluation

Scope: the six models saved by the ML stage in `ML/ML/models/`: four supervised `delay_status` classifiers and two anomaly detectors. Prediction is made at `departure_ts − 1 h`. The data is synthetic.

Reproduce with: `.venv/bin/python ML/Evaluation/scripts/run_evaluation.py` (about 2 minutes). Every number in this report comes from a file under `ML/Evaluation/`; the file is named next to each table. The generated tables are in [model_comparison.md](model_comparison.md), and the known limitations are in [limitations.md](limitations.md).

---

## 1. Executive summary

- **The artifacts are sound.** 55 of 55 technical checks pass ([reproducibility_checks.md](../validation/reproducibility_checks.md)):
  - The saved models reproduce the stored test probabilities exactly.
  - Refitting from the saved configurations reproduces both the models and the per-fold CV scores bit-for-bit.
  - The point-in-time history features match a from-scratch recomputation.
  - No outcome column reaches inference.
  - The split, the purge gap (73 h observed) and the "no test data in tuning" rule are all confirmed.
- **The supervised ranking is only partly confirmed.**
  - Logistic Regression has the best point estimate on every headline metric. Test macro-F1: LR 0.654, XGB 0.644, LGBM 0.636, RF 0.625.
  - Macro-F1 does **not** statistically separate LR from XGBoost or LightGBM. Paired day-block 95 % intervals for LR − XGB: −0.013 to +0.042; for LR − LGBM: −0.005 to +0.049.
  - LR's advantage over Random Forest is borderline: +0.000 to +0.066 with day blocks, and −0.004 to +0.066 with port-day blocks.
  - What *is* clear:
    - LR has higher Critical recall than RF (+0.135) and LightGBM (+0.116), with the same Critical precision.
    - LR has better probabilities than RF and LightGBM (lower log loss).
    - LR is far more robust when trained only on labels that would really be available (section 10).
- **Critical detection is moderate.**
  - With its default decision, LR catches 128 of 215 Critical shipments (59.5 %) and misses 87.
  - Only 10 of those 87 are predicted On-Time; 77 are flagged At-Risk, i.e. the alert is raised but the severity is underestimated.
  - Of the 230 Critical alerts LR raises, 102 are false. 87 of those 102 are At-Risk shipments that were still 12–48 h late.
  - All four models rank Critical shipments almost equally well (Critical ROC-AUC 0.919–0.921). The recall differences come from *where the default decision falls*, not from ranking ability.
- **The probabilities are usable for ranking but miscalibrated in absolute terms.**
  - Balanced class weights plus the drop in Critical prevalence from training (17 %) to test (9 %) mean every model over-predicts Critical: mean P(Critical) is 0.13–0.16 against 9 % observed.
  - A prior correction that needs no test data improves log loss, Brier score and ECE for all four models.
- **The anomaly detectors add some information beyond the classifier, but not much.**
  - Isolation Forest beats One-Class SVM on delayed ROC-AUC (+0.024, interval excludes 0). Its PR-AUC advantage is borderline. Both are stable across seeds.
  - Flags are concentrated in Critical shipments: 39 % of Critical vs 1.5 % of On-Time shipments are flagged (5× lift).
  - The detector recovers only 13 of the 87 Critical shipments that LR misses (14.9 %).
- **The main risks come from the data, not the models.**
  - Class prevalence moves strongly over time.
  - Validation folds train on labels that would not yet be known (42–62 % of each fold's training labels arrive after the fold's validation start).
  - The historical-delay features drift heavily (PSI 2.5–4.0) because of right-censoring. Removing them *improves* LR.
  - Destination-port state measured at departure is a top-2 permutation group for every model. This is a synthetic-generator artefact.
  - Nothing here supports a real-world performance claim.

**Recommendation:** use Logistic Regression for the prototype, with an explicit P(Critical) threshold chosen on validation data (not argmax), prior-corrected probabilities for display, and Isolation Forest as a secondary "unusual port conditions" signal. XGBoost is a statistically indistinguishable challenger.

## 2. Evaluation methodology

| Principle | How it was applied |
|---|---|
| Use the saved artifacts | All test-period results come from `ML/ML/models/*.joblib`, loaded unchanged. Nothing in `ML/ML/models/` or `ML/ML/results/` was written. |
| Independent recomputation | Metrics are re-implemented from confusion matrices and rank statistics (`metrics.py`) and cross-checked against the stored values. All 234 match (≤ 5e-8; the only non-zero differences are float32 log loss for XGBoost). |
| Clearly labelled refits | Refits use cloned saved pipelines with identical hyperparameters, and only where an output was missing (see below). They are labelled "evaluation-only" everywhere. |
| No test-set tuning | Decision thresholds and recalibrators are fitted on pooled out-of-fold validation predictions (3 CV blocks, 2,905 shipments) and applied to the test period once. |
| Dependence-aware uncertainty | Cluster bootstrap with 1,000 resamples. Primary scheme: whole departure days (as in the ML stage; **only 11 test days**). Secondary: origin-port × day (110 clusters). Model comparisons are paired. |
| Two separate problems | Classifiers and anomaly detectors are evaluated and ranked separately. |

**Artifacts that were missing, and how each was handled:**

| Missing artifact | Needed for | Resolution |
|---|---|---|
| Validation-period probabilities (the ML stage stored test predictions only) | Calibration on validation, threshold choice | Refit each saved configuration on each CV fold's training rows. The refits reproduce the stored per-fold CV scores exactly (`refits.py`). |
| Row-level predictions of the ablation / sensitivity variants (only summary metrics were stored) | Confidence intervals on the ablation deltas | Re-ran the same variant definitions. All 32 model-variant results match the stored table exactly. |
| Anomaly fit cost and seed stability | Cost / stability comparison | 6 evaluation-only refits per detector. |

## 3. Dataset and split

Source: `validation/split_verification.json`.

| | Train | Purge (unused) | Test |
|---|---|---|---|
| Shipments | 6,452 | 709 | 2,394 |
| Departures (UTC) | 2025-01-02 → 2025-01-29 11:00 | 2025-01-29 12:00 → 2025-02-01 11:00 | 2025-02-01 12:00 → 2025-02-11 15:00 |
| Critical share | 17.3 % | 13.1 % | 9.0 % |

**Verified:**
- The ordering is strictly train < purge < test.
- The observed gap from the last training departure to the first test departure is 73 h (≥ 72 h).
- No shipment ID is shared between sets.
- The split rebuilt from `ML/Dataset/Processed` equals the stored `splits.csv`.
- The modelling table rebuilt from the same source equals the stored `modelling_table.pkl`.

**Validation design.** There are 3 expanding-window folds, each with a 73 h gap. Every fold index lies inside the training period. The last fold's block (2025-01-25 → 01-29, 970 shipments, 25.9 % Critical) is the "validation set". The test period was never passed to `RandomizedSearchCV` (the code path uses `ctx["train"]` only, and the per-fold scores are reproduced from training rows alone).

**Air shipments.** 128 test shipments are air freight, and **none** of them is Critical, so Critical-class results describe sea freight only.

## 4. Supervised model performance (test period)

Sources: `metrics/supervised_metrics.csv`, `metrics/supervised_metric_ci.csv` (day blocks).

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Critical Recall |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.729 | 0.650 | 0.660 | **0.654** | **0.730** | **0.595** |
| Random Forest | 0.706 | 0.640 | 0.618 | 0.625 | 0.710 | 0.460 |
| XGBoost | 0.709 | 0.645 | 0.648 | 0.644 | 0.717 | 0.540 |
| LightGBM | 0.710 | 0.649 | 0.634 | 0.636 | 0.717 | 0.479 |
| Majority-class reference | 0.619 | 0.206 | 0.333 | 0.255 | 0.473 | 0.000 |

| Model | Macro-F1 95 % CI | Critical recall 95 % CI | ROC-AUC (OvR macro) | Log loss | Brier (multiclass) |
|---|---|---|---|---|---|
| Logistic Regression | 0.599–0.696 | 0.444–0.706 | 0.866 | 0.601 | 0.361 |
| Random Forest | 0.570–0.657 | 0.278–0.577 | 0.849 | 0.700 | 0.413 |
| XGBoost | 0.586–0.679 | 0.343–0.662 | 0.861 | 0.610 | 0.370 |
| LightGBM | 0.579–0.671 | 0.286–0.611 | 0.858 | 0.616 | 0.374 |

- Accuracy is reported only alongside per-class results.
- The majority-class reference reaches 61.9 % accuracy while detecting no delays at all. Accuracy is therefore not a meaningful headline here.

## 5. Per-class performance

Source: `metrics/per_class_metrics.csv`, test period.

| Model | On-Time P / R / F1 | At-Risk P / R / F1 | Critical P / R / F1 | OvR ROC-AUC (OT / AR / C) |
|---|---|---|---|---|
| Logistic Regression | 0.839 / 0.832 / 0.836 | 0.553 / 0.552 / 0.552 | 0.557 / 0.595 / 0.575 | 0.885 / 0.793 / 0.921 |
| Random Forest | 0.837 / 0.789 / 0.812 | 0.513 / 0.605 / 0.555 | 0.569 / 0.460 / 0.509 | 0.867 / 0.762 / 0.919 |
| XGBoost | 0.858 / 0.767 / 0.810 | 0.515 / 0.638 / 0.570 | 0.563 / 0.540 / 0.551 | 0.878 / 0.785 / 0.920 |
| LightGBM | 0.860 / 0.770 / 0.813 | 0.512 / 0.652 / 0.573 | 0.575 / 0.479 / 0.523 | 0.876 / 0.781 / 0.919 |

### Critical class

| Model | Critical (n) | Caught | Missed | → On-Time | → At-Risk | False Critical alerts (On-Time / At-Risk) |
|---|---|---|---|---|---|---|
| Logistic Regression | 215 | 128 | 87 | 10 | 77 | 102 (15 / 87) |
| Random Forest | 215 | 99 | 116 | 16 | 100 | 75 (11 / 64) |
| XGBoost | 215 | 116 | 99 | 12 | 87 | 90 (13 / 77) |
| LightGBM | 215 | 103 | 112 | 8 | 104 | 76 (10 / 66) |

- Critical precision is statistically the same for all four models: 0.557–0.575, each with a day-block interval of roughly ±0.14.
- Critical ROC-AUC is also the same (0.919–0.921).
- LR therefore buys its higher recall at no loss of precision. Its default decision simply sits at a better operating point; it does not rank Critical shipments better.
- The most dangerous error, Critical → On-Time, is rare for every model (8–16 shipments, 3.7–7.4 % of Critical).

### At-Risk class: why it is hard

At-Risk has the lowest separability for every model (OvR ROC-AUC 0.76–0.79, against 0.87–0.92 for the other classes). Errors cluster around the class thresholds (`metrics/boundary_analysis.csv`, `figures/confusion_matrices/prediction_mix_by_realised_delay_test.png`):

- **Error rate by realised delay (LR):**

  | Realised delay | Error rate |
  |---|---|
  | ≤ 0 h | 10 % |
  | 10–12 h | 48 % |
  | 12–14 h | 61 % |
  | 48–50 h | 55 % |
  | 50–54 h | 57 % |
  | > 96 h | 3 % |

- **Distance from the nearest threshold.** For misclassified shipments the median distance to the nearest threshold (12 h or 48 h) is 7.8–12.2 h, against 9.4–24.7 h for correctly classified ones. For Critical, errors sit a median 7.8–10.4 h from 48 h, while correct predictions sit 21.5–24.7 h away.
- **Interpretation:**
  - The model learns a *continuous* delay risk, and the classes are cut from it at 12 h and 48 h.
  - At-Risk is a 36-hour band squeezed between two thresholds, so a large part of it is "near a threshold".
  - The synthetic delay is a sum of operational terms plus noise, so shipments a few hours either side of a cut-off are close to indistinguishable before departure.
  - Many At-Risk "errors" are therefore near-misses rather than gross mistakes. The models seldom skip a class: LR makes only 25 On-Time↔Critical errors in 2,394 shipments.

## 6. Confusion matrix interpretation

Figures: `figures/confusion_matrices/confusion_{validation,test,test_sea}.png`. Counts and meanings are in `metrics/confusion_analysis.csv`.

| Error | LR count (test) | Operational meaning |
|---|---|---|
| Critical → On-Time | 10 | Missed warning: a > 48 h delay gets no alert. The costliest error. Rare. |
| Critical → At-Risk | 77 | Severity underestimated: an alert exists but at the lower tier, so the response may be too weak. The dominant Critical error for every model (77–104). |
| At-Risk → On-Time | 226 | Missed intervention window for a 12–48 h delay. The largest absolute error, concentrated at 12–18 h. |
| At-Risk → Critical | 87 | Over-escalation: mitigation cost is spent on a shipment that *is* late, just less so. |
| On-Time → At-Risk | 234 | Unnecessary low-tier alert. Cheap individually, but it erodes trust if frequent: about 1 in 6 On-Time shipments. |
| On-Time → Critical | 15 | False high-severity alarm on a shipment that is actually fine. 1.0 % of On-Time shipments. |

Collapsed to "any delay > 12 h" vs On-Time (derived from LR's test confusion matrix):

- LR alerts 677 of the 913 delayed shipments (74 %).
- 677 of its 926 alerts are correct (73 %).

## 7. Probability calibration

Sources: `metrics/calibration_metrics.csv`, `metrics/reliability_bins_test.csv`, `figures/calibration/`.

| Model (test) | Log loss | Brier (multiclass, 0–2) | ECE top-label | Mean P(Critical) vs observed | Accuracy when confidence is 0.75–0.85 |
|---|---|---|---|---|---|
| Logistic Regression | 0.601 | 0.361 | 0.059 | 0.139 vs 0.090 | 0.901 (n = 335) |
| Random Forest | 0.700 | 0.413 | 0.134 | 0.164 vs 0.090 | 0.989 (n = 264) |
| XGBoost | 0.610 | 0.370 | 0.033 | 0.127 vs 0.090 | 0.875 (n = 344) |
| LightGBM | 0.616 | 0.374 | 0.032 | 0.126 vs 0.090 | 0.888 (n = 313) |

**Findings:**

1. **Systematic Critical over-prediction and On-Time under-prediction on the test period.**
   - Balanced class weights train toward a uniform prior. Test Critical prevalence (9 %) is half the training level and a third of the validation level.
   - Reliability curves for P(Critical) sit below the diagonal, and those for P(On-Time) sit above it.
2. **Under-confidence on the top label.**
   - A predicted 80 % (0.75–0.85 band) is correct 90 % of the time for LR on test, but 79 % on validation.
   - In other words, "80 %" means different things in different periods. The probabilities are **not** stable absolute frequencies.
3. **Random Forest has the worst probabilities.**
   - It has the worst log loss, Brier score and ECE.
   - It is strongly under-confident: 0.75–0.85 confidence corresponds to 99 % accuracy.
   - Its median P(Critical) for true-Critical shipments is only 0.39, so it is a poor risk score.
4. **Separation is good for LR, XGB and LGBM.**
   - LR: median P(Critical) is 0.557 for true Critical vs 0.019 for true On-Time.
   - XGB: 0.496 vs 0.021.
5. **Post-hoc corrections (fitted without test data):**

   | Model | Correction | Log loss | ECE top-label | Critical recall (argmax) |
   |---|---|---|---|---|
   | LR | raw | 0.601 | 0.059 | 0.595 |
   | LR | prior-corrected | 0.569 | 0.031 | 0.451 |
   | LR | vector scaling on validation | 0.568 | 0.028 | 0.470 |
   | XGB | prior-corrected | 0.573 | 0.014 | 0.391 |
   | XGB | vector scaling on validation | 0.583 | 0.054 | 0.591 |

   - The prior correction improves log loss, Brier and ECE for all four models, but lowers argmax Critical recall.
   - Vector scaling fitted on validation helps LR and RF but makes the top-label ECE of XGB and LightGBM *worse*: it learns the 25 %-Critical validation prior, which does not hold on test.

**Verdict.**
- The probabilities are suitable as a **ranking / relative risk score**. Critical ROC-AUC is about 0.92 and stable across days (section 9).
- For a dashboard, show **prior-corrected** probabilities and label them as model estimates.
- Make alert decisions with an explicit threshold (section 12), not by reading a calibrated percentage.
- Any absolute calibration has to be monitored and refreshed as prevalence moves.

## 8. Anomaly detection evaluation

Sources: `metrics/anomaly_*.csv`, `validation/anomaly_seed_stability.csv`, `figures/anomaly_distributions/`.

The detectors score how unusual the origin port's operating state is: port-relative congestion, dwell, anchor queue, weather and disruptions. They were fitted on training On-Time shipments, and the flag threshold is the 95th percentile of training On-Time scores. Shipment labels are used here only to measure association.

| Test period | Precision | Recall | F1 | ROC-AUC | PR-AUC | False-positive rate |
|---|---|---|---|---|---|---|
| **Positive = delayed (At-Risk or Critical); PR-AUC baseline 0.381** | | | | | | |
| Isolation Forest | 0.882 | 0.180 | 0.298 | 0.677 | 0.628 | 0.015 |
| One-Class SVM | 0.831 | 0.183 | 0.300 | 0.653 | 0.601 | 0.023 |
| **Positive = Critical only; PR-AUC baseline 0.090** | | | | | | |
| Isolation Forest | 0.452 | 0.391 | 0.419 | 0.792 | 0.451 | 0.047 |
| One-Class SVM | 0.448 | 0.419 | 0.433 | 0.764 | 0.432 | 0.051 |

**Concentration across classes.**

| Detector | Flag rate (overall) | On-Time | At-Risk | Critical | Flags that are Critical | Critical lift |
|---|---|---|---|---|---|---|
| Isolation Forest | 7.8 % | 1.5 % | 11.5 % | 39.1 % | 45 % | 5.0× |
| One-Class SVM | 8.4 % | 2.3 % | 11.0 % | 41.9 % | 45 % | 5.0× |

- Critical makes up 9 % of the test period but 45 % of the flags. The score distributions shift monotonically from On-Time to At-Risk to Critical (survival curves in `anomaly_scores_test.png`).

**Why precision is high but recall is low:**
1. The threshold is deliberately strict: a 5 % false-alarm budget on normal training shipments.
2. A flag requires an *extreme* origin state. Many delays arise from moderate combinations of conditions or from destination or hub causes, which are not detector inputs. They look normal at the origin and are never flagged.
3. When the origin *is* extreme (major disruption, saturated port, storm), a delay is very likely. That produces high precision.

**Isolation Forest vs One-Class SVM** (paired test differences):

| Criterion | Result |
|---|---|
| Detection quality | IF delayed ROC-AUC +0.024; interval excludes 0 under both block schemes. Delayed PR-AUC +0.026 (day blocks −0.000 to +0.049; port-day +0.004 to +0.049). Critical PR-AUC +0.019, not distinguishable. |
| False alarms | IF On-Time flag rate is lower (1.5 % vs 2.3 %), but the paired interval includes 0. |
| Compute | Both negligible at this scale. Fit: IF 0.35 s, OCSVM 0.06 s (on a 3,000-row subsample). Scoring: about 20–23 ms per 1,000 rows. Model files: IF 4.3 MB, OCSVM 0.1 MB. OCSVM fit cost grows super-linearly with rows, so it will scale worse. |
| Stability | Across 6 seeds / subsamples: IF delayed PR-AUC 0.628–0.635, Spearman correlation with the saved model ≥ 0.989, flag-set Jaccard ≥ 0.83. OCSVM 0.598–0.602, ≥ 0.997, ≥ 0.88. Both stable. |
| Practical use | Isolation Forest: slightly better discrimination, no kernel or scaling choices, and it scales linearly. |

**Value beyond the classifier** (`metrics/anomaly_incremental_value.csv`):

| Isolation Forest flag | Critical rate |
|---|---|
| LR predicts Critical, flagged | 71.7 % (n = 99) |
| LR predicts Critical, not flagged | 43.5 % (n = 131) |
| LR predicts At-Risk, flagged | 18.6 % |
| LR predicts At-Risk, not flagged | 10.2 % |

- The flag is a useful **confidence modifier** on top of LR's prediction.
- It rescues few misses: 13 of the 87 Critical shipments LR misses (14.9 %) are flagged.
- Within deciles of LR's P(Critical), the anomaly score separates Critical from non-Critical with a pair-weighted AUC of 0.62. The information is real but modest.
- The 17 shipments LR calls On-Time that IF flags include no Critical shipments.

**Caveat.** The final detectors were fitted on *all* training On-Time rows, including the validation block. Validation-period anomaly metrics are therefore partly in-sample for the normal class: the On-Time flag rate is 5.1 % on validation vs 1.5 % on test.

## 9. Temporal validation

Sources: `metrics/temporal_performance.csv`, `metrics/class_prevalence_by_day.csv`, `figures/temporal_performance/`.

**Prevalence shift.** Daily Critical share ranges from about 3 % to 43 % in training. The validation folds sit in a high-delay episode (23–26 % Critical). Within the test period, Critical share is 15.7 % in days 1–3. After that, daily shares are 5–10 %, except 12.5 % on the final, partial day.

**Train / CV vs test:**

| Model | CV macro-F1 (fold 0 / 1 / 2) | Test macro-F1 | CV Critical recall (fold 0 / 1 / 2) | Test Critical recall |
|---|---|---|---|---|
| LR | 0.649 / 0.613 / 0.638 | 0.654 | 0.762 / 0.607 / 0.705 | 0.595 |
| XGB | 0.636 / 0.598 / 0.628 | 0.644 | 0.605 / 0.504 / 0.554 | 0.540 |
| LGBM | 0.627 / 0.604 / 0.642 | 0.636 | 0.641 / 0.488 / 0.602 | 0.479 |
| RF | 0.586 / 0.589 / 0.574 | 0.625 | 0.704 / 0.413 / 0.363 | 0.460 |

**How the prevalence shift changes the interpretation:**
- **Macro-F1 is fairly stable across periods**, but this hides offsetting changes.
- **Critical precision depends on prevalence.** Holding each model's test recall and false-positive rate fixed, LR's Critical precision would be 0.727 at the training prevalence (17.3 %), 0.816 at the validation prevalence, 0.557 at the test prevalence and 0.401 at 5 % (`metrics/critical_precision_vs_prevalence.csv`). The lower test precision mostly reflects rarer Critical events, not a worse model.
- **Critical discrimination is stable.** Daily Critical ROC-AUC stays between 0.83 and 0.97 for every model.
- **Critical recall still falls on low-prevalence days.** LR reaches 0.78 in days 1–3 but 0.33–0.55 on days 4–10 (5–10 % Critical); the tree models fall as low as 0.09 on 02-09. In calm periods the few Critical shipments are less extreme (closer to 48 h) and harder to call. This is case-mix change rather than broken ranking.
- **The prior shift also explains the calibration findings** in section 7.

**Label availability.**
- In each tuning fold, 42–62 % of the fold's training labels belong to shipments that had not yet arrived when that fold's validation period began.
- At the test start, 2,038 of 6,452 training labels (32 %) were not yet known.
- Only 20.6 % of test labels are known by the end of the test period; test arrivals run to 2025-03-13.
- The stored results are therefore an **offline retrospective** evaluation. In a realistic retraining run, only arrived shipments are labelled, and those are biased toward short delays because long-delay shipments arrive last. Section 10 quantifies the effect.

**Feature drift** (`metrics/feature_drift_train_vs_test.csv`):
- The historical-delay-rate features drift heavily (PSI 2.5–4.0). For example, `hist_carrier_pct_delayed` averages 30 % in training and 41 % in test.
- This is a right-censoring artefact. Early in the dataset, "shipments already arrived" are mostly fast ones, so historical rates are biased low and only stabilise later.
- The relationship to the outcome also weakens: the mean of `hist_carrier_pct_delayed` among Critical shipments, minus its mean among all shipments, is +4.6 pp in training but +1.3 pp in test.
- Origin and destination operational-state features are stable (PSI < 0.3).
- Hub-state features drift moderately (`h_util_mean24` 0.55, `wx_h_gust_max24` 0.32). They are missing for the 62–63 % of shipments on direct routes.
- `hol_h_days_in_voyage_sched` drifts strongly (1.31) because of where holidays fall in the calendar.

## 10. Robustness and sensitivity

Source: `validation/robustness_analysis.csv` and `figures/robustness/ablation_deltas.png`. Values are paired differences vs the full model on the same test shipments, with 95 % day-block intervals. **Bold** means the interval excludes 0.

| Variant | Δ macro-F1: LR | RF | XGB | LGBM | Δ Critical recall: LR | RF | XGB | LGBM |
|---|---|---|---|---|---|---|---|---|
| No operational state (booking-time info only) | **−0.294** | **−0.194** | **−0.216** | **−0.203** | −0.019 | −0.135 | −0.233 | −0.130 |
| Prediction 24 h before departure | **−0.232** | **−0.150** | **−0.182** | **−0.179** | +0.028 | **−0.219** | **−0.288** | **−0.242** |
| Train only on labels known at test start | **−0.022** | −0.023 | **−0.035** | −0.019 | −0.023 | **−0.172** | **−0.181** | **−0.116** |
| No port / carrier identity | **+0.020** | +0.010 | −0.017 | −0.007 | **+0.047** | +0.005 | −0.033 | +0.042 |
| No historical delay rates | **+0.011** | −0.016 | +0.001 | +0.012 | +0.009 | **−0.047** | −0.047 | +0.009 |
| Exclude shipments with filled inputs | +0.005 | +0.006 | −0.002 | +0.005 | +0.005 | 0.000 | −0.035 | **+0.060** |
| Sea shipments only | −0.000 | −0.001 | −0.000 | +0.011 | 0.000 | −0.005 | −0.028 | **+0.065** |

**Conclusions that hold across models:**
1. **Operational state is essential.** Without the port, weather and disruption state, macro-F1 falls to 0.36–0.43.
2. **The 1-hour horizon matters.**
   - At 24 hours before departure, macro-F1 falls to 0.42–0.48.
   - LR keeps its Critical recall (0.623) only by over-flagging: Critical precision collapses from 0.557 to 0.162.
3. **Realistic label availability hurts the trees' Critical recall badly** (−0.12 to −0.18) but LR's only slightly (−0.023, interval includes 0). This is the strongest evidence for LR as the deployable choice.
4. **Data-cleaning choices do not drive the results.** Excluding rows with filled inputs or air freight changes little.
5. **LR does not need port or carrier identity.**
   - Removing identity *improves* LR (+0.020 macro-F1, +0.047 Critical recall), so the identity one-hots slightly overfit.
   - The historical-rate features also cost LR performance (+0.011 when removed), which is consistent with their censoring drift.

**Dependence on synthetic structure.**
- In the ML stage's group permutation importance, destination state at departure is a top-2 group for every model (macro-F1 drop 0.036–0.069). Weather and disruptions are the other leaders.
- The EDA found destination state weak on its own. Its importance here reflects how the generator was built (destination utilisation at the departure hour enters the delay formula). It is not a leak in timing, but it is unlikely to transfer to real data, where destination conditions at *arrival* (unknown at departure) would matter instead.
- Port slices (`validation/slice_performance.csv`) range from Critical recall 0.00 (DEHAM, 6 Critical; INMAA, 2) to 0.89 (INNSA, 46). Most ports have too few Critical shipments for a meaningful per-port estimate.
- With an active origin disruption, LR Critical recall is 0.83; without one it is 0.47. **The model catches disruption-driven Critical delays far better than "quiet" ones.**

**Not assumed:** none of these results says how the models would perform on real logistics data.

## 11. Statistical uncertainty

Source: `metrics/pairwise_comparisons.csv`. Day blocks shown; port-day blocks agree except where noted.

| Comparison | Macro-F1 Δ (95 % CI) | Critical recall Δ | Log loss Δ | ROC-AUC macro Δ | Verdict |
|---|---|---|---|---|---|
| LR − RF | +0.029 (+0.000, +0.066)* | **+0.135** (+0.071, +0.220) | **−0.100** | **+0.017** | LR better on Critical recall and probabilities; macro-F1 borderline |
| LR − XGB | +0.011 (−0.013, +0.042) | +0.056 (−0.009, +0.151) | −0.009 (n.s.) | +0.005 (n.s.) | **Not distinguishable** |
| LR − LGBM | +0.018 (−0.005, +0.049) | **+0.116** (+0.031, +0.211) | **−0.015** | **+0.008** | LR better on Critical recall and probabilities |
| XGB − LGBM | +0.007 (−0.014, +0.026) | +0.060 (+0.000, +0.103)† | **−0.006** | +0.003 (n.s.) | Practically equivalent |
| IF − OCSVM | delayed PR-AUC +0.026 (−0.000, +0.049)‡ | Critical recall −0.028 (n.s.) | — | delayed ROC-AUC **+0.024** | IF slightly better |

\* With port-day blocks: −0.004 to +0.066. † With port-day blocks the interval excludes 0. ‡ With port-day blocks: +0.004 to +0.049.

**Cautions:**
- The test period has only 11 departure days. Day-level intervals are therefore coarse and may be slightly too narrow, since percentile bootstraps with few clusters under-cover.
- Port-day blocks assume ports are independent within a day, which is optimistic because shared weather systems and hub congestion link them.
- Differences below about 0.02 macro-F1 should not be used to rank models.
- Single-model test intervals are wide: Critical recall spans about ±0.13 and Critical precision about ±0.14 for every model.

## 12. Operational interpretation

Source: `metrics/operational_policies.csv`; `figures/operational/critical_alert_tradeoff.png`. Thresholds were chosen on pooled out-of-fold validation data and applied once to the test period. Counts below are per 1,000 test shipments.

| Model and policy (t on P(Critical)) | Macro-F1 | Critical recall | Critical precision | Critical alerts | Missed Critical | Critical → On-Time | False Critical alerts |
|---|---|---|---|---|---|---|---|
| LR, argmax | 0.654 | 0.595 | 0.557 | 96 | 36 | 4.2 | 43 |
| LR, t = 0.36 (validation recall ≥ 0.80) | 0.652 | 0.726 | 0.513 | 127 | 25 | 4.2 | 62 |
| LR, t = 0.56 (validation max macro-F1) | 0.666 | 0.498 | 0.699 | 64 | 45 | 4.2 | 19 |
| XGB, argmax | 0.644 | 0.540 | 0.563 | 86 | 41 | 5.0 | 38 |
| XGB, t = 0.20 (validation recall ≥ 0.80) | 0.597 | 0.791 | 0.377 | 188 | 19 | 5.0 | 117 |
| XGB, t = 0.41 (validation max macro-F1) | 0.657 | 0.609 | 0.575 | 95 | 35 | 5.0 | 41 |

The "minimum expected cost" policy uses an *assumed* cost matrix (missing a Critical shipment costs 10× a false Critical alarm). On validation it picks very low thresholds (t = 0.05–0.22), so on test it would raise 200–630 Critical alerts per 1,000 shipments. This tracks the 25 % validation prevalence and should not be adopted without real cost data.

**Answers to the operator's questions:**

1. **Can the model identify a significant share of Critical shipments?** Yes, a majority but not most: about 60 % at the default decision and about 73 % with the validation-chosen recall-oriented threshold (LR). Disruption-driven Critical delays are caught far more often (0.83) than quiet-period ones (0.47).
2. **How many Critical shipments are missed?** 87 of 215 with LR argmax, about 36 per 1,000 shipments. Only 10 receive *no* alert; the other 77 are flagged At-Risk.
3. **How many unnecessary Critical alerts?** 102 of 230 Critical alerts (44 %) are wrong, but 87 of those are At-Risk shipments that are genuinely 12–48 h late. Pure false alarms on On-Time shipments: 15, about 6 per 1,000 shipments.
4. **Is At-Risk useful?** Partly. As a "watch" tier it is valuable: it absorbs 77 of the 87 missed Critical shipments, so the two alert tiers combined miss very few large delays. As a precise class it is weak (precision and recall about 0.55), because its 36-h band is crowded with near-threshold cases. A dashboard should show it as "elevated risk", not as a confident 12–48 h forecast.
5. **Are the probabilities suitable for a risk dashboard?** As a relative risk score, yes, for LR, XGB and LGBM; RF's probabilities are too compressed. Use prior-corrected values for display and re-check calibration whenever prevalence moves.
6. **Best balance of detection and false alarms?** LR. At the default decision it catches the most Critical shipments at equal precision. With a threshold (t ≈ 0.36), it reaches Critical recall 0.73 with 62 false Critical alerts per 1,000 shipments. XGBoost needs 117 false alerts per 1,000 for a similar recall (0.79), and the other tree models need similar numbers.
7. **Does the anomaly detector add information?** A little. It sharpens confidence in LR's Critical calls (71.7 % vs 43.5 % precision) but recovers few misses. Present it as a separate "unusual port conditions" indicator, not as a risk-score input.
8. **Which model suits the prototype?** Logistic Regression (section 15).

## 13. Model comparison

Full generated tables: [model_comparison.md](model_comparison.md).

| Criterion | Winner | Strength of evidence |
|---|---|---|
| Overall classification (macro-F1, chosen by the CV rule) | LR (0.654) | Point estimate only vs XGB / LGBM; borderline vs RF |
| Critical recall at the default decision | LR (0.595) | Significant vs RF and LGBM; not vs XGB |
| Critical ranking (ROC-AUC) | Tie (0.919–0.921) | — |
| Probability quality | LR (lowest log loss and Brier score) | Significant vs RF and LGBM; not vs XGB |
| Robustness to realistic label availability | LR | Trees lose 0.12–0.18 Critical recall |
| Interpretability | LR (L1-regularised linear model on standardised inputs) | — |
| Anomaly detection | Isolation Forest | ROC-AUC significant; PR-AUC borderline |

The ML stage's selections are **confirmed in direction**: LR best overall, LR best Critical detector, LR most interpretable, Isolation Forest best detector. The evaluation adds two qualifications:
- LR's lead over XGBoost lies within statistical uncertainty.
- "Best Critical detector" reflects LR's default operating point, not better ranking.

## 14. Limitations

Detailed list: [limitations.md](limitations.md). The most important points:
- The data is synthetic, and the strongest signals are generator artefacts (destination state at departure).
- There is only one test period: 11 days, 215 Critical shipments.
- Validation and tuning used labels that would not have been available in time.
- The historical-delay features are distorted by censoring.
- Large prevalence swings make precision and calibration period-specific.
- There are no air Critical shipments, so air-freight Critical performance is unknown.

## 15. Final recommendations

| Question | Recommendation |
|---|---|
| **Best overall classification model** | **Logistic Regression.** Highest macro-F1, weighted F1, accuracy and ROC-AUC on test, and selected by the pre-registered CV rule. Its lead over XGBoost is not statistically significant. |
| **Best Critical-shipment detector** | **Logistic Regression.** Highest Critical recall at equal precision, and the best recall/false-alarm trade-off under validation-chosen thresholds. It also stays robust when only available labels are used. All models rank Critical shipments equally well. |
| **Best anomaly detector** | **Isolation Forest.** Better delayed ROC-AUC and PR-AUC, fewer On-Time false alarms, stable across seeds, and linear scaling. |
| **Most interpretable model** | **Logistic Regression.** L1-regularised and linear in standardised inputs. Its coefficients still describe association, not causation. |
| **Most suitable for the CargoMind prototype** | **Logistic Regression** with: (a) a P(Critical) alert threshold chosen on validation data (t ≈ 0.36 gave 73 % Critical recall on test); (b) prior-corrected probabilities on the dashboard; (c) Isolation Forest shown as a separate "unusual port conditions" flag; (d) XGBoost kept as a challenger model. |

**Required before real-world deployment:**
1. **Train and validate on real shipment records.** Re-establish every finding, especially the reliance on destination state measured at departure.
2. **Make validation label-aware.** Tune and validate using only labels that had arrived by each fold's cutoff, and simulate periodic retraining (rolling origin) instead of a single split.
3. **Fix or replace the historical-delay features.** Use censoring-aware estimates (for example, only shipments whose full delay window has elapsed) or drop them; they currently hurt LR.
4. **Monitor and recalibrate for prevalence shift.** Track daily class mix, Critical precision and recall, and calibration. Refresh the prior correction or recalibrate on recent arrived data.
5. **Set thresholds from real costs.** Replace the assumed cost matrix with actual intervention and delay costs. Re-select the Critical threshold on recent validation data.
6. **Evaluate longer and more varied periods.** Several months, more than 11 days of test clusters, air-freight Critical events, and per-port monitoring with adequate counts.
7. **Drop or regularise the identity one-hots in LR.** They did not help on this data and are a likely source of overfitting to the ports and carriers seen in training.
8. **Keep anomaly detection separate.** Validate it against labelled port-incident data rather than shipment delay labels, if such data exists.
