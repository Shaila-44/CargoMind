# CargoMind Stage 02 — Evaluation Findings

Source of every number: `DL/Evaluation/evaluation_engineer/evaluation_results.json` and the CSVs beside it, produced by `run_independent_evaluation.py`.

**Labels used in this file:**
- **V** = verified by independent recomputation.
- **R** = reported claim from the DL reports.
- **C** = evaluator's conclusion.

**Summary:** 0 critical · 6 major · 6 minor · 6 informational.

---

## Critical

**None found.** No leakage was found, and the test set did not influence model selection (V). The details are in the integrity audit (report section 11).

---

## Major

### M1. The selected model is dominated on validation by two other trained configurations

- **Evidence (V):**

  | Configuration | Mean validation RMSE |
  |---|---|
  | B_w (selected) | 4.254 ± 0.015 h |
  | C_w (EWMA residual) | 4.212 ± 0.008 h |
  | GRU | 4.202 ± 0.008 h |

  The diagnostic test ranking agrees: B_w 4.143, GRU 4.101, C_w 4.067 h. B_w was chosen only because the selection rule's "simplicity order" (A → A_w → B → B_w → C_w) puts C_w last, although C_w has *fewer* parameters (5,409 vs 5,453) and no embedding. The GRU was trained after selection.
- **Component:** `DL/DL/training/config.py` (`CANDIDATES` order), `DL/DL/training/run_experiments.py::select_final`, `DL/DL/artifacts/selection.json`.
- **Why it matters:** the model put forward for integration is not the best one the pipeline produced, on the data used for decisions. The ordering embedded a judgment (an embedding counted as simpler than a residual connection) that the parameter counts contradict.
- **Action:** define "simplicity" objectively (parameter count, or number of added components) before the next selection. Do **not** re-select using this test set (M2, I3); re-select on validation and confirm on new held-out data.

### M2. The improvement over EWMA is small, and weak on the decision data

- **Evidence (V):**
  - **Validation:** the selected configuration beats EWMA by 0.042 h (0.98 %; 4.254 vs 4.296 h). That is inside the project's own 0.05 h tie band.
  - **Validation significance:** a moving-block bootstrap within port (24 h blocks) excludes zero for only 1 of 5 seeds.
  - **Test:** a mean gain of 0.091 h (2.1 %); the designated seed gains 0.126 h (3.0 %).
  - **Test significance:** 4 of 5 seeds are significant (moving-block 24 h; Diebold–Mariano p ≤ 0.018). Seed 4 is not (p = 0.998).
- **Component:** model value proposition; `DL/Evaluation/reports/DL_evaluation_report.md`.
- **Why it matters:**
  - The measurable gain is about 0.1 h on a 4.2 h error, against an hourly noise floor of about 3.9 h.
  - It is statistically detectable on test for most seeds, but not robust on validation and seed-dependent.
  - It does not establish value large enough to justify replacing a one-line EWMA (C).
- **Action:** treat EWMA as the production forecaster. Require a pre-registered minimum effect (e.g. ≥ 0.1 h on new held-out data, consistent across seeds) before adopting any DL model.

### M3. Under-forecasting and failure in extreme and out-of-distribution conditions

- **Evidence (V, test, n and number of ports in brackets):**

  | Condition | Selected model | EWMA | C_w |
  |---|---|---|---|
  | Top-decile dwell per port [137 / 10] | RMSE 7.09, bias −6.21 h | 6.63, bias −5.78 h | 6.54 |
  | Disruption > 24 h old [54 / 2] | 6.09 (5-seed mean) | 4.73 | 4.30 |
  | Strike at a port with no development strikes [102 / 2] | 5.29 | 4.77 | 4.30 |
  | MYPKG | 4.91 | 4.21 | — |

  Seed ranges: 7.03–7.53 (top decile); 5.72–6.93 (disruption > 24 h); 5.04–5.81 (unseen-strike ports).
- **Component:** `DL/DL/models/recurrent.py` (bounded recurrent state + linear head on per-port z-scaled targets).
- **Why it matters:** CargoMind is a *bottleneck* warning system. The model is least reliable exactly when dwell is highest and conditions are novel, and there it is worse than EWMA. The EWMA-residual design does not show this failure (C).
- **Action:** do not integrate B_w. If a DL model is pursued, prefer a design that cannot fall below EWMA when extrapolating (C_w-type). Add an explicit "outside training range" flag to any deployed forecast.

### M4. Systematic negative bias, larger than EWMA's

- **Evidence (V):**
  - Mean error of the selected configuration: −0.59 h on test (5 seeds) and −0.19 h on validation.
  - EWMA: −0.16 h on test and +0.11 h on validation.
  - Normal hours only, on test: −0.45 h vs +0.11 h for EWMA.
- **Component:** model outputs, `DL/DL/training/trainer.py` (no bias diagnostics are logged).
- **Why it matters:** consistent under-prediction of dwell understates congestion. RMSE alone hides it.
- **Action:** log and report bias for every experiment. Investigate whether the hour-weighted loss, which is dominated by NLRTM's variance, shifts the other ports.

### M5. Inference is not integration-ready

- **Evidence (V, probes run against `lstm_best.pt`):**

  | Probe | Result |
  |---|---|
  | NaN input | Returns NaN, no error |
  | 6-hour sequence | Accepted silently (any length is accepted) |
  | Unscaled raw-hour input (dwell = 120 h) | Accepted, returns a meaningless value |
  | Port index 10 | Raw `IndexError` from the embedding layer, not a validation message |

  Other gaps:
  - There is **no public function that turns a raw 12-hour window** (telemetry + weather + disruption log) **into model input**. The scaling logic lives in `cargomind_dl.sequences.transform`, which expects the dataset-build panel. The disruption features need the event table.
  - `load_checkpoint` refuses to load unless `DL/Dataset/processed/*.json` hash-matches, which ties deployment to the dataset build directory.
- **Component:** `DL/DL/models/recurrent.py::forward`, `DL/DL/models/checkpoint.py`, `DL/DL/src/cargomind_dl/sequences.py`.
- **Why it matters:** a Streamlit integration would have to re-implement the preprocessing (risking training/serving skew), and could not detect bad input.
- **Action:** add an inference module, separate from training, that:
  - loads the checkpoint plus the scaler and schema (bundled, or referenced by hash with a clear error);
  - accepts a raw 12-hour frame and port ID;
  - validates the length, channels, NaN and port;
  - applies the saved transforms and returns hours plus a range flag.

### M6. An evaluation-report claim about disruptions is not supported by validation

- **Evidence (V):** `DL_evaluation_report.md` says the LSTM is "far better than EWMA" in the first 24 h of disruptions, and answers "partly yes" to anticipating disruptions. Checking it on both splits:

  | Disruption age | Split | Selected model | EWMA | n |
  |---|---|---|---|---|
  | 1–6 h | Test | 3.96 | 4.37 | 34 |
  | 1–6 h | Validation | 4.47 | 3.90 | 36 |
  | 7–24 h | Test | 4.20 | 4.86 | 57 |
  | 7–24 h | Validation | 4.58 | 4.96 | 44 |

  Only the 7–24 h gain appears in both splits. Each bin comes from a handful of episodes at ≤ 5 ports.
- **Component:** `DL/Evaluation/reports/DL_evaluation_report.md`, sections 3 and 4.
- **Why it matters:** the claim is based on test only and is contradicted at onset on validation. It overstates a capability stakeholders would rely on.
- **Action:** revise the claim to "better than EWMA 7–24 h into disruptions on both splits; inconclusive at onset; worse beyond 24 h".

---

## Minor

| # | Finding | Evidence (V) | Component | Action |
|---|---|---|---|---|
| m1 | Wrong command in the metadata | `test_split_used_for` names `dl.evaluation.evaluate_test`. The real command is `python -m dl.evaluation.evaluate test`. | `DL/DL/artifacts/experiment_metadata.json`, `run_experiments.py::write_metadata` | Correct the text |
| m2 | The test log does not bind to the selection | `selection_sha_basis` holds a reason string, not a hash. `evaluate test` can be re-run; it only appends to the log. | `DL/Evaluation/evaluation/evaluate.py::test` | Store the SHA-256 of `selection.json` and `lstm_best.pt`; require an explicit flag to re-run |
| m3 | Original bootstrap design | 7 day blocks (too coarse) and port-day blocks, which treat a port's consecutive days as independent and gave wider intervals than dependence-aware methods. No moving-block bootstrap or Diebold–Mariano test. 2,000 replicates (adequate). | `DL/Evaluation/evaluation/metrics.py::paired_bootstrap` | Use a moving-block bootstrap within port (12/24/48 h sensitivity) plus Diebold–Mariano with HAC errors, as done here |
| m4 | Headline uses the best-validation seed | The report leads with seed 2 (4.108 h). The configuration's expected performance is the 5-seed mean, 4.143 ± 0.058 h. Both are shown, but the emphasis is optimistic. | `DL_evaluation_report.md` §1 | Lead with the seed mean |
| m5 | Results not versioned | `.gitignore` (`*.csv`) excludes `experiment_results.csv`, all training logs and all evaluation CSVs. | repository `.gitignore` | Allow-list `DL/Evaluation/reports/**/*.csv`, `DL/DL/training/logs/**/*.csv` |
| m6 | No bias or extreme-value reporting in training and evaluation | Only MAE / RMSE / R² / port-balanced RMSE are reported. | `DL/Evaluation/evaluation/metrics.py` | Add bias, top-decile error and an "outside training range" subgroup |

---

## Informational

| # | Note |
|---|---|
| I1 | **Independence is limited.** This audit was run by the same agent session that built the pipeline. Mitigations: separate metric, baseline and bootstrap code; recomputation from primary files; adversarial checks. A second reviewer is still advisable. |
| I2 | **The brief's baseline table mislabels splits.** 5.37 h (persistence) and 4.30 h (12-h mean) are **test** values; on validation they are 5.66 and 4.33 h, and EWMA is 4.30 h. The brief also describes the split as "independent per port"; it actually uses **global hour cut-offs shared by all ports** (V). |
| I3 | **Test-set status.** The DL pipeline used test once (V). This audit additionally scored all 13 experiments on test **as diagnostics**. The test split can therefore no longer serve as an unbiased selector among these experiments. A future model choice needs new held-out data (rolling-origin or a later period). |
| I4 | **Validation is used three times:** early stopping, LR scheduling and model selection. Validation scores of selected configurations are optimistically biased. Test is the unbiased check for the selected model only. |
| I5 | **Small, episode-driven subgroups.** Test strike hours: n = 118 from 3 ports, of which 102 from 2 ports never seen striking. Disruptions > 24 h: n = 54 from 2 ports. Subgroup results are case studies. |
| I6 | **Synthetic data**, 41.7 days. R² (≈ 0.93–0.96) is inflated by between-port level differences and says little about forecasting skill. Nothing here transfers to real ports. |
