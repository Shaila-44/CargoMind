# CargoMind — Shipment Delay-Risk Models: Report

**Scope:** pre-departure prediction of `delay_status` (On-Time ≤ 12 h < At-Risk ≤ 48 h < Critical), built on the existing
cleaned data and EDA. Six algorithms: four supervised classifiers and two anomaly detectors.
**Reproduce (from the CargoMind root):** `.venv/bin/python -m pip install -r ML/ML/requirements.txt` then
`.venv/bin/python ML/ML/scripts/run_ml_pipeline.py` (≈ 6 min on 8 cores). The run is deterministic: two runs gave identical metrics.
**Every number below comes from files the pipeline wrote**. The main ones are `ML/ML/results/comparison/model_comparison.md` and
`ML/ML/results/metrics/supervised_metrics.csv`. The data is synthetic (`is_synthetic = True`), so the findings describe this
dataset only. Feature importance is association, not causation.

---

## 1. Layout

| Path | Contents |
|---|---|
| `ML/ML/src/cargomind_ml/config.py` | every modelling decision that is a parameter (cutoff, split, gap, folds, seeds, budgets) |
| `features.py` | modelling table (reuses `cargomind_eda.features` unchanged) + ML leakage fixes + feature catalogue |
| `preprocessing.py` | chronological split, purged expanding-window CV folds, per-algorithm preprocessing pipelines |
| `logistic_regression_model.py`, `random_forest_model.py`, `xgboost_model.py`, `lightgbm_model.py`, `isolation_forest_model.py`, `one_class_svm_model.py` | one module per algorithm: pipeline + search space |
| `train_models.py` · `evaluate_models.py` · `feature_importance.py` · `compare_models.py` | tuning/fitting/saving · metrics & plots · interpretability · comparison tables |
| `ML/ML/models/` | `<model>.joblib` (full fitted pipeline), `<model>_preprocessor.joblib`, `<model>_config.json` (best params, search space, CV scores) |
| `ML/ML/results/` | `data/` (splits, feature catalogue, data summary), `metrics/`, `classification_reports/`, `confusion_matrices/`, `feature_importance/`, `anomaly/`, `comparison/`, `figures/` |

`Dataset/` and the EDA outputs are only read. One side effect: importing the EDA package creates its empty `ML/EDA/reports/tables/` folder.

## 2. Modelling decisions

| # | Decision | Choice and reason |
|---|---|---|
| 1 | Prediction timestamp | **`departure_ts − 1 h`** (EDA convention). Only hourly records with `ts_hour ≤ cutoff`, events already started at the cutoff, and outcomes of shipments that had **arrived** before the cutoff are used. A 24 h cutoff is evaluated as a sensitivity check (§6). |
| 2 | Rows | 9,850 → **9,555**: the 295 shipments whose outcome was never observed in raw are dropped (their labels were filled during cleaning). No other rows are removed. |
| 3 | History features | `hist_*` recomputed on observed shipments only, so filled labels cannot leak in. At least 10 prior arrivals are required, otherwise NaN. `*_n_arrived` counts are excluded because they grow with calendar time and act as a hidden time index. |
| 4 | Reconstructed ETAs | 194 observed shipments had `expected_arrival_ts` rebuilt from the actual arrival. `planned_duration_h` and `planned_vs_route_ratio` are excluded; `planned_transit_hours` and `shipping_mode` carry the same schedule information. The hub-holiday feature is recomputed with a schedule ETA for those rows. It turned out identical, so no row changed. |
| 5 | Features | **51 inputs** in 12 groups, one per EDA redundancy cluster. All 245 columns are listed with a reason in `results/data/feature_catalog.csv`. Exclusions: outcomes, `LEAK_*`, lineage flags, IDs, absolute time, `tr_*` (aggregates of telemetry), visibility (filled at 3 ports), disruption severity/capacity (unknown whether they are known at onset), and destination-holiday-at-ETA (the calendar ends 26 Feb, so most test ETAs are uncovered). |
| 6 | Split | Chronological. **Test = latest 25 % of shipments** (departures ≥ 1 Feb 12:00, n = 2,394). **72 h purge gap** before the test start (709 shipments, used in neither set) so that rolling 24 h port state and multi-day disruptions are not shared across the boundary. **Train** = departures < 29 Jan 12:00 (n = 6,452). |
| 7 | Validation | 3 expanding-window folds inside the training period, each with a 72 h purge. Each validation block is about 970 shipments. The last block (25–29 Jan) is reported as the **validation set**. The test set is used once, for final evaluation only. |
| 8 | Tuning | `RandomizedSearchCV` (20–40 draws) maximising **macro-F1**, never accuracy. Preprocessing is inside the pipeline, so every fold fits its scalers, imputers and encoders on its own training rows. |
| 9 | Imbalance | Balanced class weights in all four classifiers. XGBoost receives balanced sample weights recomputed on each fitted subset. No oversampling (EDA: ≈ 1.1k Critical training examples is enough). |
| 10 | Preprocessing per family | LR: median imputation + missing flags + standard scaling + one-hot. RF: imputation + flags, no scaling. XGBoost/LightGBM: native NaN handling, no scaling. Isolation Forest: per-origin-port z-scores, no global scaling. One-Class SVM: per-port z-scores + global standard scaling. |
| 11 | Anomaly set-up | Fitted on **On-Time training shipments only**. Inputs are 15 origin operating-state features (congestion, dwell, anchor queue, weather, disruptions), standardised **within each origin port**, so the model learns "unusual for this port" rather than "big port". Positive mapping for evaluation: **anomaly-positive = At-Risk or Critical** (Critical-only also reported). Hyperparameters are chosen by validation PR-AUC; labels are never used for fitting. Flag threshold = 95th percentile of training On-Time scores (a 5 % false-alarm budget on normal shipments). One-Class SVM is fitted on a seeded subsample of 3,000 normal rows (cost). |

**Label-availability caveat.** 2,038 of the 6,452 training shipments had not yet arrived at the test start (voyages run 9–800 h). A
production model retrained at that moment would not have their labels yet. The main results train on them, matching the offline-evaluation
convention. §6 shows the cost of the stricter "labels available only" regime.

**Drift.** The Critical share moves from 17 % (train) to about 25 % (validation block) to **9 %** (test). Absolute scores therefore differ
between periods, and the test period is easier for accuracy and harder for Critical precision.

## 3. Supervised classifiers — results

Test period (all shipments). Day-block bootstrap 95 % intervals resample whole departure days.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Critical Recall |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.729** | **0.650** | **0.660** | **0.654** (0.599–0.696) | **0.730** | **0.595** (0.444–0.706) |
| Random Forest | 0.706 | 0.640 | 0.618 | 0.625 (0.570–0.657) | 0.710 | 0.460 (0.278–0.577) |
| XGBoost | 0.709 | 0.645 | 0.648 | 0.644 (0.586–0.679) | 0.717 | 0.540 (0.343–0.662) |
| LightGBM | 0.710 | 0.649 | 0.634 | 0.636 (0.579–0.671) | 0.717 | 0.479 (0.286–0.611) |
| *Majority class (reference)* | *0.619* | *0.206* | *0.333* | *0.255* | *0.473* | *0.000* |

| Model | Critical Precision | At-Risk Recall | ROC-AUC (OvR macro) | Critical → predicted On-Time | Log loss |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.557 | 0.552 | **0.866** | 4.7 % | **0.601** |
| Random Forest | 0.569 | 0.605 | 0.849 | 7.4 % | 0.700 |
| XGBoost | 0.563 | 0.638 | 0.861 | 5.6 % | 0.610 |
| LightGBM | 0.575 | **0.652** | 0.858 | **3.7 %** | 0.616 |

The ranking is consistent across tuning CV (macro-F1: LR 0.633, LightGBM 0.625, XGBoost 0.621, RF 0.583), the validation block
(LightGBM 0.642, LR 0.638, XGBoost 0.628, RF 0.574) and the test period. Sea-only test metrics are about 0.005 lower for every
model (air shipments are always On-Time and trivially predicted), so air does not inflate the comparison.

Observations:
* **The four models are close.** LR, XGBoost and LightGBM differ by ≤ 0.02 macro-F1, and their bootstrap intervals overlap heavily.
  Random Forest is consistently last.
* **Critical recall is where they differ most.** LR catches 60 % of Critical shipments on test (CV 69 %) against 46–54 % for the
  tree models, at the same Critical precision (≈ 0.56–0.58). The tree ensembles mostly move Critical into At-Risk (40–48 % of Critical).
  The severe error (Critical → On-Time) stays rare for all models (4–7 %).
* **At-Risk remains the hardest class**, as the EDA predicted. The boosted models recall it better (0.64–0.65 vs LR 0.55).
  LR pays for its higher Critical recall with more At-Risk ↔ On-Time confusion.
* **Calibration:** LR has the lowest log loss, so its probabilities are the most usable as risk scores.

Confusion matrices: `results/figures/confusion_matrices_{validation,test,test_sea}.png` (counts in
`results/confusion_matrices/`); per-class reports in `results/classification_reports/`.

## 4. Why a linear model wins here

The data generator (`ML/Dataset/Raw/generate_dataset.py`, lines 323–341) defines delay as a **linear sum** of origin and
destination utilisation, origin anchor queue, hinge terms on gusts above 45 km/h, disruption severity, holiday proximity, hub
utilisation, and carrier/priority/cargo offsets, plus Gaussian noise whose SD grows with route distance. Labels are cut from that sum.
A multinomial linear model is therefore close to the true functional form. The trees have no interactions to discover, and their extra
flexibility mainly fits noise. Rows sharing port episodes make the effective sample smaller than 6,452.

This was read *after* model selection, to explain the result. No feature was engineered from the generator. Doing so would tune the model
to a synthetic recipe rather than to logistics. **On real data with interactions (e.g. weather × route exposure), the ranking could
change. Keep the boosted models in the comparison when real data arrives.**

The generator also explains the second finding of §5: destination-port conditions at the departure hour enter the delay formula
directly (`util24[di,h]`, `gust[di,h]`, `ev_sev[di,h]`). This is observed before departure, so it is not leakage. It is, however, a
synthetic-data pattern: physically, a destination disruption that ends long before a 10-day voyage arrives should matter little.

## 5. Feature importance and interpretability

Methods: LR coefficients; RF impurity; XGBoost/LightGBM total gain; per-feature permutation importance (test macro-F1); **group
permutation** (all columns of a group shuffled together, which handles correlated features); SHAP (TreeExplainer) for XGBoost and LightGBM.
Files: `results/feature_importance/*.csv`, `results/figures/group_permutation_importance.png`, `shap_critical_*.png`,
`logistic_regression_coefficients.png`.

**What the models rely on.** Drop in test macro-F1 when a group is permuted:

| Group | LR | RF | XGBoost | LightGBM | Read-out |
|---|---:|---:|---:|---:|---|
| Origin weather (gust, precipitation) | 0.058 | 0.045 | 0.070 | 0.066 | top group for every model; largest effect on Critical recall (−0.13 to −0.17) |
| Destination state at departure | 0.063 | 0.036 | 0.069 | 0.063 | `dis_d_active_any` + `d_util_mean24`: synthetic pattern (§4) |
| Shipment attributes | 0.058 | 0.028 | 0.037 | 0.035 | mostly `shipping_mode` (air ⇒ On-Time) |
| Origin disruptions | **0.079** | 0.011 | 0.019 | 0.011 | LR uses them most; trees reach the same signal through congestion/weather proxies |
| Origin congestion (utilisation, vessel time…) | 0.051 | 0.008 | 0.021 | 0.019 | |
| Hub state | 0.031 | 0.005 | 0.026 | 0.023 | transshipment routes only |
| Anchor queue | 0.028 | 0.012 | 0.022 | 0.020 | |
| Carrier | 0.020 | ≈ 0 | 0.002 | 0.006 | stable carrier effect (EDA δ 0.19) |
| Calendar (holidays, hour, weekday) | 0.012 | ≈ 0 | 0.008 | 0.008 | origin holiday proximity |
| Container dwell | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 | redundant given anchor/congestion |
| Historical delay rates | ≈ 0 | 0.017 | ≈ 0 | ≈ 0 | no added value over current state |
| Route identity (origin/dest/hub/type/distance) | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 | the *conditions* at the ports matter, not which ports they are |

* **Meaningful, not identifier-driven.** Port and carrier IDs carry little. Removing them leaves macro-F1 unchanged or slightly better:
  LR 0.654 → 0.674, RF +0.010, XGBoost −0.018, LightGBM −0.007. Removing the time-varying operational state collapses every model
  to 0.36–0.43. The models are learning from operating conditions, which is what a pre-departure risk model should do.
* **Gain/impurity vs permutation disagree for history and congestion.** Trees spend many splits on `hist_*` and utilisation (10–20 %
  of gain), but permuting them barely changes predictions. That is a known bias of split-based importance towards continuous,
  correlated features. Permutation and SHAP are the more reliable read-outs.
* **SHAP (XGBoost, Critical class):** `dis_d_active_any` 0.58, `wx_o_gust_max24` 0.43, `d_util_mean24` 0.29, `o_anchor_mean24` 0.21,
  origin holiday distance 0.20, carrier 0.17.
* **LR coefficients** (log-odds per SD) read directly. Air → On-Time (+3.4). Destination disruption active → Critical (+0.89).
  Origin max gust → Critical (+0.84). Coral Bay Logistics → Critical (+0.41) vs Oceanic Lines → On-Time (+0.86). Active origin weather
  event → Critical (+0.50). The L1 penalty zeroes the At-Risk column: At-Risk acts as the reference class, which is valid for a softmax model.
* Customs hold, equipment availability, TEU, departure hour/weekday, container dwell and route identity add nothing. This matches the EDA.

## 6. Ablations and sensitivity (tuned hyperparameters, refit only; test macro-F1)

| Variant | LR | RF | XGBoost | LightGBM |
|---|---:|---:|---:|---:|
| Full model (reference) | 0.654 | 0.625 | 0.644 | 0.636 |
| No port/carrier identity | 0.674 | 0.635 | 0.626 | 0.629 |
| No operational state (booking-time info only) | 0.360 | 0.431 | 0.427 | 0.433 |
| No historical delay rates | 0.665 | 0.609 | 0.645 | 0.648 |
| Train only on labels available at test start | 0.633 | 0.603 | 0.609 | 0.618 |
| Exclude shipments with filled inputs (train + test) | 0.655 | 0.633 | 0.645 | 0.643 |
| Sea shipments only (train + test) | 0.650 | 0.619 | 0.637 | 0.642 |
| **Prediction 24 h before departure** | 0.422 | 0.475 | 0.461 | 0.458 |

* **Prediction time dominates everything.** Moving the cutoff from 1 h to 24 h before departure costs about 0.2 macro-F1, consistent with the
  EDA's signal-decay finding. Fixing the product's prediction timestamp remains the most consequential open decision.
  At 24 h, LR keeps Critical recall (0.62), but the tree models drop to about 0.24.
* **Label availability** costs 0.02–0.04 macro-F1 and hurts the trees' Critical recall most (e.g. XGBoost 0.54 → 0.36). LR is the most
  robust (0.595 → 0.572), because the dropped shipments are disproportionately long and delayed.
* **Filled-input shipments** (12.5 % of rows) do not change the conclusions.
* Without port/carrier identity, LR reaches 0.674 macro-F1 and 0.642 Critical recall. This is a candidate simplification. It was found
  on the test period, so confirm it in CV before adopting it.

## 7. Anomaly detection (separate task)

The two detectors are judged on whether unusual origin conditions coincide with delays. They are **not** three-class predictors.

Test period, anomaly-positive = At-Risk or Critical (prevalence 0.381):

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| **Isolation Forest** | **0.882** | 0.180 | 0.298 | **0.677** | **0.628** |
| One-Class SVM (RBF, ν = 0.2, γ = 0.01) | 0.831 | 0.183 | 0.300 | 0.653 | 0.601 |

Positive = Critical only (prevalence 0.090): Isolation Forest precision 0.45, recall 0.39, ROC-AUC 0.79, PR-AUC 0.45; One-Class SVM 0.45 / 0.42 / 0.76 / 0.43.

Score distributions (`results/figures/anomaly_score_distributions.png`): flag rates are **1–2 % of On-Time, 11 % of At-Risk and 39–42 % of Critical**
shipments. Cliff's δ for Critical vs On-Time scores is 0.65 (IF) and 0.58 (OCSVM). So **anomalies are strongly concentrated among
Critical shipments, much less among At-Risk.** Most delays happen under conditions that look normal for the port, so recall for
"delayed" is low by design. A flag is a high-precision warning (≈ 85 % of flagged shipments are delayed), not a delay classifier.

**Best anomaly detector: Isolation Forest.** It is better on validation PR-AUC (0.764 vs 0.755, the selection rule) and on every test
metric except Critical recall. It is also cheaper and needs no subsampling.

## 8. Recommendation

| Question | Answer |
|---|---|
| Best overall supervised classifier | **Logistic Regression**: highest CV and test macro-F1 (0.633 / 0.654; LightGBM edges it on the single validation block, 0.642 vs 0.638), best ROC-AUC and log loss. The margin over XGBoost/LightGBM is within noise for macro-F1. |
| Best Critical-class detector | **Logistic Regression**: Critical recall 0.69 CV / 0.60 test at the same Critical precision as the trees, and the most robust under stricter label availability and an earlier cutoff. LightGBM has the fewest Critical → On-Time errors (3.7 % vs 4.7 %). |
| Most interpretable | **Logistic Regression** (standardised coefficients per class). |
| Best anomaly detector | **Isolation Forest** |

**For the prototype: deploy the multinomial Logistic Regression** (`ML/ML/models/logistic_regression.joblib`, C ≈ 0.126, L1, balanced
weights) as the risk classifier. Show its class probabilities rather than only the argmax, and use the **Isolation Forest** score as a
secondary "unusual port conditions" alert. Keep **XGBoost** as the challenger: it is the strongest tree model, near-equal on macro-F1, and
better on At-Risk recall. It is the likely winner if real data has the non-linear interactions this synthetic generator lacks.

Before production:
1. **Fix the prediction timestamp.** The difference between 1 h and 24 h before departure (−0.2 macro-F1) dwarfs any algorithm difference.
2. Decide the retraining policy for in-transit shipments (label availability, §6).
3. Validate destination-state features on real data. Here they are strong only because the generator uses them.
4. If missing a Critical shipment costs much more than a false alarm, tune a cost-sensitive decision threshold on the CV folds.
   The models already output probabilities.
5. Re-estimate everything on a longer window. Six weeks and a single test period give wide intervals (macro-F1 ± ≈ 0.05).
