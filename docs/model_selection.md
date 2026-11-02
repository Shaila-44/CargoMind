# CargoMind model selection and inference contracts

This document is the current record of which models CargoMind serves and how the Streamlit app calls them. It supersedes the "recommended model" statements in the stage reports (`ML/ML/reports/ML_model_report.md`, `ML/Evaluation/reports/evaluation_report.md`, `DL/Evaluation/reports/DL_final_report.md`). Those reports remain the record of what was measured.

All models were trained and evaluated on **synthetic data**. Nothing here establishes real-world performance.

## 1. Final configuration

| Workflow | Component | Model (identifier) | Role | Artifact | Config / registry |
|---|---|---|---|---|---|
| ML | Delay-class classifier | XGBoost (`xgboost`) | **Provisional primary** (not shown to be better than LR) | `ML/ML/models/xgboost.joblib` | `ML/ML/models/selection.json`, `xgboost_config.json` |
| ML | Anomaly detector | Isolation Forest (`isolation_forest`) | **Primary** (separate output) | `ML/ML/models/isolation_forest.joblib` (pipeline + threshold 0.5609) | `ML/ML/models/selection.json`, `isolation_forest_config.json` |
| ML | — | **Logistic Regression (challenger)**, Random Forest, LightGBM, One-Class SVM, majority-class reference | Research / comparison (loadable by name; artifacts kept) | `ML/ML/models/*.joblib` | — |
| DL | Next-hour dwell forecast | Transformer T_w (`transformer_t_w`, seed 1) | **Primary (default)** | `DL/DL/artifacts/transformer/runs/T_w_transformer_hour_weighted/seed_1.pt` (sha256 `86d1578f8118…`) | `DL/DL/inference/model_registry.json` |
| DL | Next-hour dwell forecast | C_w LSTM (`c_w_lstm`, seed 2) | **Automatic fallback** | `DL/DL/artifacts/runs/C_w_lstm_ewma_residual_hour_weighted/seed_2.pt` (sha256 `f2dab76ee5ef…`) | `DL/DL/inference/model_registry.json` |
| DL | — | GRU, Transformer T, basic LSTM (A), original B_w LSTM | Research / comparison (registered, loadable by name) | `DL/DL/artifacts/…` | `model_registry.json` (`role: research`) |
| DL | — | EWMA (α = 0.2) | Benchmark / reference (runs only when requested: `model="ewma"` or `fallback="ewma"`) | — | `dl.inference.fallback` |
| DL | — | GNN (GCN + LSTM) | **Experimental, excluded** from inference (validation RMSE 10.77 h vs 4.16 h for T_w) | `DL/DL/artifacts/gnn/` | not registered |

The ML classifier and the anomaly detector answer different questions: which delay class, and whether origin conditions are unusual. They are returned as separate sections and are never combined into one score.

The XGBoost and Isolation Forest selections are product decisions. **XGBoost is the provisional primary classifier, and Logistic Regression is kept as the research challenger.** On this synthetic data XGBoost is **not** claimed to be statistically better than Logistic Regression. The independent ML evaluation found:
- They are indistinguishable on test macro-F1 (LR − XGB +0.011, 95 % day-block CI −0.013 to +0.042).
- LR had higher Critical recall at the default decision (0.595 vs 0.540; difference not significant).
- XGBoost loses more Critical recall when trained only on labels available at the test start (§1a).

### 1a. The label-availability experiment, explained

**What it simulates.** The main models were trained on all 6,452 training shipments, including 2,038 whose arrival (and therefore label) happened *after* the test period began (2025-02-01 12:00 UTC). A system retrained at that moment could not have known those labels. The experiment "sensitivity: train only on labels available at test start" retrains each model as that system would have.

**What was done** (`train_models.ablations_and_sensitivity`; rerun with row-level predictions in `ML/Evaluation/src/cargomind_eval/refits.py::ablation_predictions`):
1. **Training rows:** training shipments with `actual_arrival_ts < test_start`, i.e. 4,414 of 6,452.
2. **Same configuration:** each model's tuned hyperparameters and preprocessing are reused unchanged (cloned pipeline, refit only, no retuning). Balanced class weights are recomputed on the 4,414 rows, as in every fit.
3. **Test set:** both the reference and the variant predict the **same 2,394 test shipments** (identical shipment IDs, checked), with argmax decisions.
4. **Metric:** Critical recall = Critical shipments predicted Critical / all 215 Critical test shipments. The change is variant − full model on those same rows. 95 % intervals come from a paired bootstrap that resamples whole departure days (1,000 draws; `robustness.ablation_analysis`).

**The dropped labels are not random.** Long delays arrive last:

| Training rows | n | On-Time | At-Risk | Critical | Median delay |
|---|---:|---:|---:|---:|---:|
| All training shipments | 6,452 | 52.0 % | 30.7 % | 17.3 % (1,118) | 10.8 h |
| Kept (arrived before test start) | 4,414 | 60.1 % | 28.9 % | **11.0 % (487)** | 4.9 h |
| Dropped (not yet arrived) | 2,038 | 34.3 % | 34.7 % | 31.0 % (631) | 26.3 h |

So the restricted models see 56 % fewer Critical examples, and the remaining ones are the less extreme Critical cases.

**Results** (test, 215 Critical shipments; source `ML/Evaluation/validation/robustness_analysis.csv` and `ML/Evaluation/data/ablation_predictions.csv`):

| Model | Critical caught: full → restricted | Critical recall Δ (95 % CI) | Critical alerts raised | Critical precision | Critical ROC-AUC | Median P(Critical) of true Critical |
|---|---|---|---|---|---|---|
| Logistic Regression | 128 → 123 | **−0.023** (−0.085 to +0.027) | 230 → 260 | 0.557 → 0.473 | 0.921 → 0.902 | 0.557 → 0.559 |
| XGBoost | 116 → 77 | **−0.181** (−0.251 to −0.087) | 206 → 133 | 0.563 → 0.579 | 0.920 → 0.889 | 0.496 → 0.352 |

The −0.18 vs −0.02 figures quoted earlier are these values, rounded.

**Interpretation:**
- Both models lose some ability to *rank* Critical shipments (AUC −0.019 LR, −0.031 XGB).
- Most of XGBoost's recall loss comes from its Critical probabilities shrinking with the rarer, milder Critical training examples: median 0.50 → 0.35, so the argmax decision selects Critical far less often (206 → 133 alerts). Its precision is unchanged.
- LR keeps its probability scale and even raises more Critical alerts (230 → 260), but at lower precision (0.557 → 0.473).

So the gap is partly a ranking difference and partly a difference in where the default decision falls. The LR difference is within noise; the XGBoost loss is not. Both measurements come from one synthetic test period of 11 days.

## 2. Choice of the LSTM fallback (validation evidence only)

Candidates are every LSTM configuration with trained checkpoints: `A_lstm`, `A_w_lstm_hour_weighted`, `B_lstm_port_embedding`, `B_w_lstm_port_embedding_hour_weighted` and `C_w_lstm_ewma_residual_hour_weighted`. The `B_w…__no_*` runs are one-feature ablations with fewer input channels, not deployable candidates. The GRU is not an LSTM. **The test split was not used.**

All numbers below are on the same 1,306 validation windows as T_w, means over 5 seeds. Sources:
- `DL/DL/artifacts/experiment_results.csv`
- `DL/Evaluation/evaluation_engineer/experiment_summary.csv` and `seedwise_metrics.csv`
- `DL/Evaluation/results/transformer/validation_comparison.csv`
- `DL/Evaluation/results/evaluation/validation_subgroups_summary.csv`

| Variant | RMSE (h) | Seed SD | Seed range | MAE (h) | Port-balanced RMSE | Seeds beating EWMA | Worst port vs EWMA (RMSE diff) | Registered |
|---|---:|---:|---|---:|---:|---:|---|---|
| **C_w (EWMA-residual LSTM)** | **4.212** | **0.008** | 4.203–4.225 | **3.390** | **4.205** | **5 / 5** | CNSHA +0.06 h; NLRTM +0.00 h | yes (`c_w_lstm`) |
| B_w (port embedding, hour-weighted) | 4.254 | 0.015 | 4.231–4.268 | 3.424 | 4.247 | 5 / 5 | NLRTM +0.16 h | yes (`lstm_b_w_original_selection`) |
| A_w (hour-weighted) | 4.281 | 0.019 | 4.267–4.314 | 3.440 | 4.273 | 4 / 5 | NLRTM +0.20 h | no |
| B (port embedding) | 4.438 | 0.072 | 4.323–4.515 | 3.542 | 4.394 | 0 / 5 | NLRTM +1.31 h; CNSHA +0.44 h | no |
| A (basic) | 4.446 | 0.066 | 4.352–4.519 | 3.575 | 4.407 | 0 / 5 | NLRTM +1.19 h; USLAX +0.62 h | yes (`basic_lstm`) |
| *Reference: T_w (primary)* | *4.162* | *0.007* | *4.155–4.173* | *3.360* | *4.158* | *5 / 5* | | |
| *Reference: EWMA α = 0.2* | *4.296* | — | — | *3.446* | *4.291* | — | | |

**How C_w scores on each criterion:**
1. **Validation RMSE, MAE and port-balanced RMSE:** lowest of all LSTM variants, 0.050 h behind T_w (inside the project's 0.05 h tie threshold).
2. **Seed consistency:** smallest seed SD (0.008 h), and all 5 seeds beat EWMA (B_w also 5 / 5, with twice the SD).
3. **Per-port behaviour:** no port is materially worse than EWMA. The other variants lose 0.16–1.31 h at NLRTM, the port with the strongest regime changes.
4. **Checkpoint availability and integrity:** all 5 checkpoints exist and match their training logs (`checkpoint_matches_log` = True in `seedwise_metrics.csv`). The registry checkpoint (seed 2, the best-validation seed under the registry's seed rule) is SHA-256-verified at load time.
5. **Compatibility:** same 8 input channels, `feature_schema.json` and `scalers.json` as T_w, so the fallback uses exactly the same validated window and preprocessing.
6. **Same windows as T_w:** yes, all 1,306 validation windows (`results/transformer/comparison_info.json`).
7. **Known limitations** (`DL/Evaluation/reports/DL_final_report.md`, `DL_transformer_report.md`):
   - It is worse than EWMA in the first 6 hours of a disruption (+0.50 h), where T_w is about level with EWMA.
   - It was never scored on the test split, which was already spent before it became a candidate.
   - The single seed-2 checkpoint is optimistic relative to the 5-seed mean (4.2025 vs 4.212 h).
   - In return, its EWMA skip connection means it reduces to EWMA when the learned correction is zero. It does not show the out-of-range extrapolation failure seen for the original B_w.

**Uncertainty:**
- C_w's lead over B_w (0.042 h) is about 3 seed SDs, and its lead over A_w (0.069 h) is about 4.
- Both are small in absolute terms, and the comparison rests on one validation period.
- The choice is nonetheless consistent across every criterion (accuracy, stability, worst-port behaviour, integrity and compatibility), so it is the most defensible operational fallback.

## 3. Inference architecture

```text
streamlit_app/services/ml_risk.py ──► cargomind_ml.inference (ML/ML/src/cargomind_ml/inference.py)
    predict_shipments(shipments_df)       shipment attributes ──► build_features_for_shipments ──► predict
    predict_features(feature_df)          51 model features   ──────────────────────────────────► predict
        predict: validate_features ─┬─► XGBoost pipeline (xgboost.joblib) ─► classification section
                                    └─► Isolation Forest bundle (pipeline + threshold) ─► anomaly section

streamlit_app/services/dl_forecasting.py ──► dl.inference (DL/DL/inference/)
    window_from_dataset(port, end_hour)   12 raw hours from DL/Dataset/processed/hourly_panel.csv
    forecast(window_df, port_id)          validate window ─► transformer_t_w ─(load failure / invalid output)─► c_w_lstm
                                          scalers.json transform ─► model ─► inverse transform ─► output check
```

**ML feature construction** (`build_features_for_shipments`):
- It reuses the training code unchanged (`cargomind_eda.features.build_shipment_features` plus the ML-stage steps of `build_modelling_table`). It reads telemetry, weather, disruptions, holidays and historical outcomes from `ML/Dataset/Processed`.
- **Parity was verified:** for shipments in the dataset, the built features equal the stored training table (`ML/ML/results/data/modelling_table.pkl`) exactly (tests).
- `predict` reproduces the stored XGBoost test probabilities (to within 3e-8) and Isolation Forest scores and flags for all 2,394 test shipments.

**DL fallback rules** (`forecast(..., fallback="auto")`, the default):
- The fallback activates when T_w cannot be loaded (missing, altered or incompatible checkpoint or preprocessing artifacts) or returns an invalid value (non-finite, or outside 1–1,000 h).
- Invalid **input** never activates the fallback: both models need the same valid window, so an `InputValidationError` is raised.
- If C_w also fails, an error naming both failures is raised.
- Explicitly requested research models get no automatic fallback.
- Other modes:
  - `fallback="none"` disables the fallback;
  - `fallback="lstm"` applies it to any requested model;
  - `fallback="ewma"` is the legacy EWMA behaviour.

## 4. Output contracts

### ML (`cargomind-ml-inference/1.0`, one dict per shipment)

| Field | Type | Notes |
|---|---|---|
| `contract_version` | str | |
| `shipment_id` | str or None | None if the feature rows carry no `shipment_id` |
| `prediction_timestamp_utc` | ISO str | when the prediction was produced |
| `feature_cutoff_utc` | ISO str or None | `departure_ts − 1 h` (the time the features describe); None if not supplied |
| `classification.model_name` / `model_label` / `model_version` | str | version = artifact path + SHA-256 prefix |
| `classification.predicted_class` | str | `On-Time`, `At-Risk` or `Critical` (argmax) |
| `classification.class_probabilities` | dict[str, float] | the three classes, summing to 1. Raw probabilities from a class-balanced model, **not recalibrated** |
| `classification.decision_rule`, `probabilities_note` | str | |
| `anomaly.model_name` / `model_label` / `model_version` | str | |
| `anomaly.anomaly_score` | float | `−score_samples`; higher = more unusual origin operating state |
| `anomaly.anomaly_flag` | bool | `anomaly_score > threshold` |
| `anomaly.threshold`, `threshold_rule`, `score_definition` | float, str, str | |
| `warnings` | list[str] | missing inputs, unseen categories, short history, holiday-calendar coverage |

`anomaly` is None when called with `anomaly=False`.

### DL (`cargomind-dl-inference/1.0`, `DwellPrediction.to_dict()`)

| Field | Type | Notes |
|---|---|---|
| `port_id` | str | |
| `predicted_dwell_hours` | float | next-hour container dwell, in hours |
| `model_name` | str | the model that actually produced the number |
| `model_role` | str | `primary`, `fallback`, `research` or `baseline` |
| `requested_model` | str or None | |
| `method` | str | `neural`, `neural_fallback`, `ewma_fallback` or `ewma_baseline` |
| `model_version` | str | checkpoint path + SHA-256 prefix |
| `fallback_used` | bool | |
| `fallback_reason` | str or None | optional |
| `window_start_utc`, `window_end_utc` | ISO str | the 12 input hours |
| `window_hours` | int | 12 |
| `target_hour_utc` | ISO str | the forecast hour (window end + 1 h) |
| `forecast_horizon_hours` | int | 1 |
| `generated_at_utc` | ISO str | when the forecast was produced |
| `warnings` | list[str] | fallback notice, reconstructed-dwell input, out-of-range level |
| `note` | str | point forecast only; no calibrated uncertainty |
| `contract_version` | str | |

No uncertainty interval is returned, because none is calibrated.

## 5. Remaining gaps for Streamlit integration

1. **No live data.** ML features and DL windows are built from the processed synthetic tables. They cover departures and hours from 2025-01-01 to 2025-02-11 15:00 UTC; requests outside that range are rejected. Real use needs a feed of port telemetry, weather, disruptions and arrived-shipment outcomes in the same schema.
2. **Probabilities are raw.** Sigmoid and isotonic calibration were tested and not justified (§6), so only raw probabilities are served. The prior correction suggested by the evaluation was not adopted either: it would bake in an assumed class prevalence.
3. **No alert threshold** (§7). Operational alerting needs a cost decision first.
4. **The new feature builder recomputes the full-dataset integration per call** (about 0.5 s). Batch requests or caching are advisable in the app.
5. **Shipments in the same request are treated as having no known outcome.** This is correct for new shipments, but it means scoring historical shipments in a batch can differ slightly from the training table, whose history used their outcomes.

## 6. Probability calibration study (validation only)

`.venv/bin/python ML/Evaluation/scripts/run_calibration_study.py` (`ML/Evaluation/src/cargomind_eval/calibration_study.py`, calibrators in `ML/ML/src/cargomind_ml/calibration.py`) compares one-vs-rest **sigmoid** (Platt) and **isotonic** calibration. No model is retrained: inputs are the stored out-of-fold validation probabilities.

**Protocol (fixed before any result was computed):**
- Calibrators are forward-chained: fit on validation blocks earlier than *k*, evaluate on block *k* (k = 1, 2).
- A method is justified only if it lowers both log loss and top-label ECE on both blocks, and costs at most 0.005 Critical ROC-AUC.
- The test period is scored once, as a diagnostic, and not used for the choice.

**Validation, pooled blocks 1–2** (n = 1,940; Critical share 25.9 %). Source: `ML/Evaluation/calibration_study/validation_forward_chained.csv`.

| Model | Variant | Log loss | Brier | ECE top-label | ECE classwise | Mean P(Critical) | Macro-F1 (argmax) | Critical recall | Critical precision | Critical ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | raw | 0.787 | 0.477 | 0.059 | 0.045 | 0.232 | 0.613 | 0.529 | 0.675 | 0.863 |
| XGBoost | sigmoid | 0.790 | 0.473 | 0.037 | 0.049 | 0.234 | 0.593 | 0.587 | 0.661 | 0.868 |
| XGBoost | isotonic | 0.877 | 0.471 | 0.024 | 0.039 | 0.233 | 0.595 | 0.561 | 0.686 | 0.866 |
| Logistic Regression | raw | 0.787 | 0.474 | 0.034 | 0.074 | 0.307 | 0.626 | 0.656 | 0.624 | 0.877 |
| Logistic Regression | sigmoid | 0.754 | 0.456 | 0.027 | 0.044 | 0.257 | 0.606 | 0.634 | 0.654 | 0.885 |
| Logistic Regression | isotonic | 0.852 | 0.456 | 0.034 | 0.048 | 0.257 | 0.610 | 0.632 | 0.652 | 0.881 |

**Per-block rule check:**
- XGBoost sigmoid: improves log loss on block 1 (0.817 → 0.809) but worsens it on block 2 (0.757 → 0.770).
- XGBoost isotonic: worsens log loss on both blocks, because it outputs exact zeros for some classes.
- LR sigmoid: improves block 1 but worsens top-label ECE on block 2 (0.025 → 0.035).

**Decision: no method is justified for either model** (`ML/Evaluation/calibration_study/decision.json`). Raw probabilities remain the only output, and no calibrator artifact was saved.

**Test diagnostic** (n = 2,394; Critical share 9.0 %; calibrators fitted on all 2,905 validation rows). Source: `test_diagnostic.csv`.

| Model | Variant | Log loss | Brier | ECE top-label | Mean P(Critical) | Macro-F1 | Critical recall |
|---|---|---:|---:|---:|---:|---:|---:|
| XGBoost | raw | 0.610 | 0.371 | 0.033 | 0.127 | 0.644 | 0.540 |
| XGBoost | sigmoid | 0.611 | 0.363 | 0.072 | 0.142 | 0.652 | 0.633 |
| XGBoost | isotonic | 0.606 | 0.364 | 0.074 | 0.139 | 0.653 | 0.540 |

The diagnostic agrees with the validation decision. Calibrators fitted on validation blocks (about 26 % Critical) double XGBoost's top-label ECE on a test period with 9 % Critical. **Calibration on this data mostly learns the period's class prevalence, which moves strongly over time.**

Recalibration should be reconsidered only with a longer, more stable validation history or with prevalence monitoring (`ML/Evaluation/reports/limitations.md`, items 9 and 12). The reliability plot is in `ML/Evaluation/calibration_study/reliability_xgboost_critical.png`.

## 7. Alert thresholds

No P(Critical) alert threshold is applied, and none is recommended yet. The classification output is the argmax class; all three raw probabilities are returned alongside it.

Turning probabilities into operational alerts needs an explicit decision on two costs:
- the cost of a **missed Critical shipment** (false negative);
- the cost of a **false Critical alert** (false positive).

The evaluation's threshold analysis (`ML/Evaluation/metrics/operational_policies.csv`) shows how strongly the trade-off moves. For XGBoost on test:

| Rule (chosen on validation) | Critical recall | False Critical alerts per 1,000 shipments |
|---|---:|---:|
| argmax | 0.54 | 38 |
| recall ≥ 0.80 rule | 0.79 | 117 |
| illustrative 10 : 1 cost matrix (an assumption, not real costs) | 0.95 | 350 |

Any threshold must come from real costs and be re-checked as prevalence moves.
