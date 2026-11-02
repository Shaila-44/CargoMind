# CargoMind Stage 02 — Independent Evaluation of the DL Component

**Decision: FAIL – NOT READY**, for the selected LSTM as CargoMind's dwell-time forecaster.

- The implementation and evaluation **integrity pass**: no leakage was found, and the test set was not used for selection.
- The decision rests on **insufficient value over EWMA, worse-than-EWMA behaviour in extreme and novel conditions, and missing inference safeguards.** See section 15.

**Reproduce:** `.venv/bin/python -m dl.evaluation_engineer.run_independent_evaluation` (about 2 minutes). It writes only to `DL/Evaluation/evaluation_engineer/`; no file in `DL/Dataset/processed/`, `DL/DL/artifacts/` or `DL/Evaluation/reports/` was changed (checked by modification time after the run).

**Labels:**
- **V** = verified by independent recomputation.
- **R** = reported claim from the DL reports.
- **C** = evaluator's conclusion.

All errors are in **hours**, after inverse transformation. Classified findings with actions are in [critical_findings.md](critical_findings.md). All numbers are in [evaluation_results.json](evaluation_results.json).

---

## 1. Executive summary

1. **Integrity is sound (V).**
   - Windows never cross split boundaries, and every target is hour t+1.
   - Scalers equal training-hour statistics (reconstructed cells excluded).
   - The baselines use only hours t−11…t.
   - Reconstructed targets are excluded.
   - The test split was evaluated once, after selection.
   - All 65 checkpoints reproduce their logged validation RMSE, and two retrains reproduced exactly.
   - The 12 unit tests pass.
2. **The selected model (B_w: LSTM + port embedding + hour-weighted loss) beats EWMA by a small margin:**
   - On test: 0.126 h (3.0 %) for the designated checkpoint; 0.091 h (2.1 %) as the 5-seed mean.
   - On validation, the data used for decisions: 0.042 h (1.0 %), inside the project's own 0.05 h tie band.
   - Statistically, 4 of 5 seeds are significant on test under dependence-aware tests; only 1 of 5 on validation.
3. **It is worse than EWMA where it matters most for a bottleneck system:** at the top decile of dwell, in disruptions older than 24 h, and at ports striking for the first time (+0.5 to +1.4 h). It also under-forecasts on average (bias −0.59 h vs −0.16 h for EWMA on test).
4. **It is not the best model the pipeline trained.** C_w (EWMA residual) and the GRU are better on validation, and also on test as a diagnostic. C_w does not show the extreme-value failure. The selection rule's ordering chose a dominated configuration.
5. **Inference is not ready for Streamlit.** There is no raw-input preprocessing API, and NaN, wrong-length and unscaled inputs are accepted silently.

## 2. Evaluation scope

- **Audited:** `DL/DL/models/`, `DL/DL/training/`, `DL/Evaluation/evaluation/`, `DL/Evaluation/tests/`, all 65 checkpoints in `DL/DL/artifacts/runs/`, `lstm_best.pt`, selection and test logs, `DL/Evaluation/reports/*`, `DL/Dataset/processed/*`.
- **Recomputed independently** (`run_independent_evaluation.py`; its own metric, baseline, bootstrap and Diebold–Mariano code):
  - all validation and test predictions for every checkpoint;
  - baselines rebuilt from `hourly_panel.csv`;
  - per-port and subgroup metrics;
  - four bootstrap schemes plus Diebold–Mariano;
  - two in-memory retrains;
  - inference probes.
- **Not done:** no retraining to improve scores, and no changes to the pipeline, dataset or artifacts.
- **Limits on this audit:**
  - It was performed by the same agent session that built the pipeline (I1). Independence comes from re-derivation, not from a different author.
  - Test predictions of non-selected experiments were computed as **diagnostics** (I3). They are reported to check whether validation rankings generalise, and must not drive future selection.

## 3. Dataset and split verification (V)

| Check | Result |
|---|---|
| `DL/Dataset/processed` outputs and sources vs `manifest.json` SHA-256 | Unchanged |
| Split rule | **Global** hour cut-offs shared by all ports: train targets 0–699, validation 700–849, test 850–999 (not "independent per port" as the brief states) |
| Windows | 6,579 / 1,306 / 1,318 ✓ |
| Inputs t−11…t, target t+1, all within the target's split | All 9,203 windows ✓ (embargo removes cross-boundary windows) |
| Reconstructed targets | 0 among kept windows (447 candidates excluded) |
| Dwell scalers | Equal training-hour, non-reconstructed statistics to 3e-14 h. With all hours included, AEJEA's mean would differ (105.29 vs 105.60), so the training-only fit is real. |
| Test X dwell channel | Equals scaled dwell of hours t−11…t (300 random windows, max error 2e-7 z) |
| Baselines | Recomputed from the hourly panel using hours t−11…t only, they match the stored per-window baselines (EWMA within 5e-5 h). EWMA α = 0.2 was selected on training windows. |

**Baseline values** (V; the brief's table mislabels splits, I2):

| Baseline | Validation RMSE | Test RMSE | Test MAE | Test bias |
|---|---|---|---|---|
| Persistence | 5.659 | 5.372 | 4.293 | −0.049 |
| 12-hour mean | 4.326 | 4.304 | 3.447 | −0.199 |
| **EWMA α = 0.2** | **4.296** | **4.234** | **3.411** | **−0.160** |

## 4. Implementation audit

| Item | Finding | Status |
|---|---|---|
| Input channels | 8 channels in loader order; ablations drop by name; checkpoint channel lists verified against the loader for all 65 runs | V ✓ |
| Sequence | `[B, 12, C]`, oldest hour first; the model uses the last layer's final hidden state | V ✓ |
| Port embedding | `Embedding(10, 4)` concatenated to h_T; different ports produce different outputs; raises if `port_index` is missing (unit-tested) | V ✓ |
| Output and loss shapes | `[B]`; `mse_loss` raises on shape mismatch, so no broadcasting | V ✓ |
| Inverse transform | `cargomind_dl.dataset.to_hours`; this audit's independent inverse matches it | V ✓ |
| Hour-weighted loss | Weights = per-port dwell variance from `scalers.json`, normalised over training samples; equals MSE in hours (unit-tested) | V ✓ |
| Training loop | Seeded shuffling of training only; validation in stored order; gradient clipping 1.0; ReduceLROnPlateau and early stopping on validation RMSE in hours; best state restored | V ✓ |
| Test isolation | The trainer never loads `test`; `evaluate test` refuses to run before `selection.json` exists | V ✓ |
| Determinism | B_w seed 2 and C_w seed 0 retrained in memory: identical best epoch and validation RMSE | V ✓ |
| Checkpoint metadata | Model config, training config, channels, dataset hashes, best epoch | V ✓ |
| Configuration vs report | Parameter counts, channels, best epochs and validation RMSE all match `experiment_results.csv` and the training report | V ✓ |
| Defects | Metadata names a non-existent module (m1); test log is not hash-bound (m2); no bias metric (m6); inference safeguards missing (M5) | see findings |

## 5. Model comparison

5 seeds each. Validation is the decision data; test is a **diagnostic** except for the selected configuration.

| Experiment | Params | Val RMSE (mean ± SD) | Val MAE | Val bias | Test RMSE (mean ± SD) | Test MAE | Test bias | Test port-balanced RMSE |
|---|---|---|---|---|---|---|---|---|
| A: LSTM | 5,409 | 4.446 ± 0.066 | 3.575 | +0.31 | 4.503 ± 0.198 | 3.546 | +0.44 | 4.430 |
| B: + embedding | 5,453 | 4.438 ± 0.072 | 3.542 | +0.05 | 4.245 ± 0.212 | 3.358 | +0.02 | 4.209 |
| A_w: hour-weighted | 5,409 | 4.281 ± 0.019 | 3.440 | −0.23 | 4.175 ± 0.045 | 3.320 | −0.54 | 4.150 |
| **B_w (selected)** | 5,453 | **4.254 ± 0.015** | 3.424 | −0.19 | **4.143 ± 0.058** | 3.300 | −0.59 | 4.121 |
| C_w: EWMA residual | 5,409 | 4.212 ± 0.008 | 3.390 | −0.12 | 4.067 ± 0.010 | 3.266 | −0.25 | 4.057 |
| GRU (B_w settings) | 5,299 | **4.202 ± 0.008** | 3.381 | −0.12 | 4.101 ± 0.018 | 3.270 | −0.49 | 4.084 |
| *EWMA* | 0 | *4.296* | *3.446* | *+0.11* | *4.234* | *3.411* | *−0.16* | *4.227* |

**What each metric shows:**
- **RMSE** is the primary metric. It penalises large misses, which matter for bottlenecks.
- **MAE** shows that typical errors improve less than RMSE, by about 0.11 h on test.
- **Bias** shows that the hour-weighted models systematically under-forecast (M4).
- **Port-balanced RMSE** stays close to pooled RMSE, so no single port dominates after hour-weighting.
- **R²** is 0.93–0.96 for every model, *including the baselines*. Port level differences inflate it, and it is not informative here.
- **Seed SD:**
  - The unweighted A and B are unstable (test SD about 0.2 h).
  - The selected B_w has a test SD of 0.058 h, about half its gain over EWMA.
  - C_w and the GRU are the most stable (≤ 0.018 h).

**Conclusions:**
- On validation, the ranking is GRU ≈ C_w < B_w < A_w ≪ A ≈ B, and the test diagnostic reproduces it.
- The selected B_w is **dominated** (M1).
- The port embedding is consistently slightly helpful (A_w → B_w: −0.027 h validation, −0.032 h test) but below the tie tolerance.

## 6. Baseline comparison (selected configuration)

Improvement (%) = (baseline RMSE − model RMSE) / baseline RMSE × 100. Source: `baseline_comparison.csv` (V).

| Split | Model | vs persistence | vs 12-h mean | **vs EWMA** | Seeds beating EWMA |
|---|---|---|---|---|---|
| Validation | B_w, 5-seed mean (4.254 h) | 1.405 h (24.8 %) | 0.071 h (1.6 %) | **0.042 h (0.98 %)** | 5 / 5 |
| Validation | B_w seed 2 (4.231 h) | 1.429 h (25.2 %) | 0.095 h (2.2 %) | 0.066 h (1.5 %) | — |
| Test | B_w, 5-seed mean (4.143 ± 0.058 h) | 1.229 h (22.9 %) | 0.161 h (3.7 %) | **0.091 h (2.1 %)** | 4 / 5 (seed 4 ties) |
| Test | B_w seed 2, `lstm_best.pt` (4.108 h) | 1.264 h (23.5 %) | 0.196 h (4.5 %) | 0.126 h (3.0 %) | — |

**Does the DL model beat EWMA?**
- On average, yes, by about 0.1 h of RMSE (about 6 minutes).
- That improvement is **not consistent across ports** (section 7) or **across operating conditions** (section 8).
- On validation it falls inside the tie band.
- It is **not a sufficient margin to call the model superior** (C).

**Reported-claim check:** the test headline in `DL_evaluation_report.md` (4.108 h; seeds 4.094–4.237 h; EWMA 4.234 h) is **reproduced exactly** (V; stored predictions match to 1e-14 h). Its emphasis on the best-validation seed is optimistic (m4).

## 7. Per-port evaluation

Selected configuration, 5 seeds; RMSE in hours. Test columns are the selected model's single test evaluation.

| Port | Val EWMA | Val B_w | Val bias | Test EWMA | Test B_w (mean, range) | Test bias | Test C_w (diag.) | EDA group |
|---|---|---|---|---|---|---|---|---|
| INNSA | 4.071 | 4.083 | −0.41 | 4.318 | 3.949 (3.83–4.10) | −1.09 | 3.964 | high-signal |
| USLAX | 4.418 | 4.257 | +0.18 | 3.875 | 3.577 (3.55–3.60) | −0.44 | 3.623 | high-signal |
| CNSHA | 4.010 | 4.055 | −0.65 | 4.328 | 4.062 (4.01–4.22) | +0.04 | 4.147 | high-signal |
| SGSIN | 4.346 | 4.285 | −0.49 | 4.045 | 3.883 (3.84–3.99) | −0.58 | 3.924 | high-signal |
| NLRTM | 4.699 | **4.859** | +1.04 | 4.692 | 4.722 (4.57–4.91) | −0.72 | 4.522 | high-signal |
| LKCMB | 4.129 | 4.008 | +0.40 | 4.203 | 3.967 (3.95–4.00) | −0.18 | 4.039 | intermediate |
| AEJEA | 4.184 | 4.172 | −0.76 | 4.207 | 4.034 (4.00–4.07) | −0.43 | 4.094 | noise-dominated |
| DEHAM | 4.368 | 4.316 | −0.24 | 3.729 | 3.614 (3.57–3.77) | −0.62 | 3.603 | noise-dominated |
| INMAA | 4.539 | 4.378 | −0.42 | 4.663 | 4.489 (4.45–4.51) | −0.27 | 4.573 | noise-dominated |
| MYPKG | 4.150 | 4.054 | −0.55 | 4.208 | **4.914 (4.72–5.26)** | −1.61 | 4.083 | noise-dominated |

**Highest and lowest error:**
- Highest: NLRTM on both splits, and MYPKG on test.
- Lowest: USLAX and DEHAM on test; CNSHA and LKCMB on validation.

**Signal groups (EDA hypothesis, checked):**

| Group | Validation B_w vs EWMA | Test B_w vs EWMA |
|---|---|---|
| High-signal ports | 4.318 vs 4.314 (no gain) | 4.051 vs 4.256 (gain) |
| Noise-dominated ports | 4.235 vs 4.316 (gain) | 4.296 vs 4.215 (loss, entirely MYPKG) |

**The EDA grouping does not predict where the model helps.** The sign of the gain flips between splits for both groups, and for individual ports: NLRTM is worse on validation, while MYPKG is better on validation and worse on test.

**Generalisation:**
- No single port dominates the pooled error (port-balanced ≈ pooled).
- The model does not generalise uniformly: port-level gains depend on which episodes fall in the period (C).

## 8. Disruption and strike evaluation

n = windows; "ports" = distinct ports in the subgroup. **Bold n** marks subgroups too thin for conclusions (fewer than 30 windows, or a single episode or port).

| Subgroup | Split | n (ports) | EWMA RMSE / bias | B_w 5-seed RMSE (range) / bias | C_w (test diag.) |
|---|---|---|---|---|---|
| Normal (no disruption, no strike) | val | 1,204 (10) | 4.289 / +0.11 | 4.248 / −0.16 | |
| Normal | test | 1,168 (10) | 4.168 / +0.11 | 4.027 (4.00–4.07) / −0.45 | 4.044 |
| Disruption active | val | 102 (5) | 4.383 | 4.322 / −0.60 | |
| Disruption active | test | 150 (5) | 4.720 / −2.28 | 4.950 (4.73–5.49) / −1.71 | 4.237 |
| Onset 1–6 h | val | 36 | 3.898 | **4.465** | |
| Onset 1–6 h | test | 34 (5) | 4.367 / −1.81 | 3.957 (3.90–4.01) | 4.002 |
| 7–24 h | val | 44 | 4.961 | 4.582 | |
| 7–24 h | test | 57 (5) | 4.855 / −2.73 | 4.197 (4.04–4.70) | 4.176 |
| > 24 h | val | **16** | 4.025 | 3.575 | |
| > 24 h | test | 54 (**2**) | 4.729 / −2.33 | **6.088** (5.72–6.93) / −3.59 | 4.296 |
| Labour strike | val | **9 (1)** | 4.708 | 4.182 | |
| Labour strike | test | 118 (3) | 4.808 / −2.50 | 5.146 (4.89–5.80) / −2.01 | 4.244 |
| Strike, port with no development strikes | test | 102 (**2**: INNSA, MYPKG) | 4.771 / −2.38 | 5.292 (5.04–5.81) / −2.53 | 4.303 |
| Strike, port with development strikes | test | **16 (1)** | 5.034 | 4.055 (3.45–5.68) | 3.844 |
| Top-decile dwell per port | test | 137 (10) | 6.634 / −5.78 | **7.219** (7.03–7.53) / −6.33 | 6.543 |

**Answers:**
- **Changes during disruptions:** in the 7–24 h window, the model forecasts the rise better than EWMA on **both** splits.
- **Onset (1–6 h):** better on test but worse on validation. **Inconclusive.** The claim in `DL_evaluation_report.md` that it is "far better" in the first 24 h is overstated (M6).
- **Extremes:** the model **underestimates extreme dwell** more than EWMA does (top decile: bias −6.3 vs −5.8 h, RMSE +0.59 h). In long episodes it saturates (MYPKG reaches about 92 h; the model plateaus near 80 h).
- **Strikes at ports without strike history:** **worse than EWMA** (+0.52 h, 2 ports). The evidence comes from one strike episode at MYPKG and one disruption/strike episode at INNSA, so it is a case study, but it is the expected failure mode.
- **C_w** (diagnostic) is better than EWMA in every disruption and strike row. It is the more robust design (C).

## 9. Feature ablations

All relative to B_w. Δ = ablated − full, mean of 5 seeds; negative means better *without* the feature. Validation is the decision evidence; test is a diagnostic.

| Removed | Val RMSE | Δ val | Val seed SD | Test RMSE | Δ test | Verdict (C) |
|---|---|---|---|---|---|---|
| (full B_w) | 4.254 | — | 0.015 | 4.143 | — | |
| All operational channels (dwell history only) | 4.274 | +0.020 | 0.002 | 4.259 | **+0.116** | Operational channels **jointly** add value, more on test |
| Disruption (both channels) | 4.241 | −0.014 | 0.007 | 4.124 | −0.019 | Not needed **given the others** (redundant with utilization and strike) |
| Reconstruction flag | 4.250 | −0.004 | 0.019 | 4.117 | −0.026 | No value; harmless |
| Labour strike | 4.244 | −0.010 | 0.018 | 4.144 | +0.001 | No measurable value. Validation has only 9 strike windows, so it cannot judge this feature. |
| Utilization | 4.244 | −0.010 | 0.013 | 4.137 | −0.006 | No measurable value alone |
| Vessels at anchor | 4.257 | +0.003 | 0.013 | 4.211 | **+0.068** | Neutral on validation, helpful on test. Retain. |
| Wind gust | **4.212** | **−0.042** | 0.007 | **4.113** | −0.030 | **Harmful on both splits.** Consistent with the validation-period gust shift found in the EDA. Candidate for removal. |
| Port embedding (B_w → A_w) | 4.281 | +0.027 | 0.019 | 4.175 | +0.032 | Small, consistent help, below the tie band |

**Conclusions (not causal):**
- Every single-feature effect except wind gust is within seed noise.
- Information about operational conditions matters only as a group.
- No additional ablation is needed for these conclusions. A useful one would be "no wind gust" combined with C_w, evaluated on new data.

## 10. Statistical validation

**The original method** (`DL/Evaluation/evaluation/metrics.py::paired_bootstrap`) was checked:
- Predictions are paired on identical windows (V).
- RMSE difference is computed per resample in hours (V).
- 2,000 replicates and percentile intervals (adequate).
- But the blocks were **7 test days** (too few clusters) or **port × day**. Port × day treats consecutive days of a port as independent, which ignores within-port autocorrelation across midnight. Neither is a time-series block bootstrap (m3).

**This audit added** (5,000 replicates each):
- an iid bootstrap (to show what ignoring dependence does);
- a moving-block bootstrap within port, with block lengths of 12, 24 and 48 h;
- a Diebold–Mariano test on hourly-aggregated squared-error differentials with Newey–West HAC (lag 24).

| B_w checkpoint | Δ RMSE vs EWMA (test) | iid 95 % CI | Port-day CI | MBB L = 24 h CI | MBB L = 48 h CI | DM p (t) |
|---|---|---|---|---|---|---|
| Seed 2 (`lstm_best.pt`) | −0.126 | [−0.211, −0.036] | [−0.272, +0.073] | [−0.241, −0.043] | [−0.234, −0.078] | 0.0007 |
| Seed 0 | −0.073 | [−0.163, +0.023] | [−0.225, +0.111] | [−0.195, +0.024] | [−0.172, −0.008] | 0.018 |
| Seed 1 | −0.116 | [−0.201, −0.026] | [−0.267, +0.089] | [−0.237, −0.029] | [−0.227, −0.065] | 0.003 |
| Seed 3 | −0.141 | [−0.224, −0.053] | [−0.270, +0.026] | [−0.252, −0.068] | [−0.240, −0.101] | 0.0001 |
| Seed 4 | +0.002 | [−0.106, +0.116] | [−0.177, +0.261] | [−0.186, +0.078] | [−0.185, +0.017] | 0.998 |
| Validation, MBB 24 h (seeds 0–4) | −0.039, −0.028, −0.066, −0.048, −0.030 | | | only seed 2 excludes 0 | | |

**Conclusions:**
- On test, for 4 of 5 seeds, the gain over EWMA is statistically detectable under dependence-aware methods.
- The original report's "not statistically secure" (based on port-day blocks) was **too conservative for test**, but **correct for validation**.
- **Statistical significance here does not mean practical significance.** The effect is about 0.1 h, depends on the seed, falls in the tie band on validation, and reverses in extreme conditions (C).

## 11. Leakage and integrity audit

| Question | Answer | Basis |
|---|---|---|
| Future observations in inputs? | No: inputs are hours t−11…t; target t+1 | V (all windows; 300 re-derived from the hourly panel) |
| Sequences across split boundaries? | No (embargo) | V |
| Scalers fitted on training only? | Yes, training hours only, reconstructed cells excluded | V |
| Validation used to fit preprocessing? | No | V (scalers match training-only statistics) |
| Test used for hyperparameter or model selection? | No. Selection used validation (`selection.json`, `selection_round1.json`); test was evaluated once afterwards (`test_evaluation_log.json`, 1 entry) | V. Process note: before modelling, the developer had seen test **baselines** and the EDA's descriptive test-shift notes (I2, I3). No evidence that these influenced choices; round-2 changes are justified by validation diagnostics in the training logs. |
| Repeated test-driven model choice? | No. The GRU and ablations ran after the test evaluation and did not change the selection | V (file times, log) |
| Inverse transformation correct? | Yes | V (independent inverse; stored predictions reproduced to 1e-14 h) |
| Reconstructed targets excluded? | Yes, 0 in kept windows | V |
| Reconstruction using future information? | Reconstructed *inputs* keep the published processed values, whose fill method is undocumented (an upstream data-lineage issue). They are flagged, not re-imputed; the model sees the flag. No evidence of future leakage, but cannot be excluded for those 4.6 % of cells. | V / informational |
| Baselines using the target hour? | No; recomputed from hours t−11…t | V |

**No critical integrity issue.**

## 12. Reproducibility audit

| Item | Status |
|---|---|
| Seeds | Python, NumPy, torch and DataLoader all seeded; deterministic algorithms; fixed thread count. Retrains reproduce exactly (V). |
| Configuration | Full training config in every checkpoint; `lstm_config.json`; `experiment_metadata.json` (one wrong command name, m1) |
| Model loading | `dl.models.load_checkpoint` with no training imports (V). Fails loudly if the dataset JSON files change. |
| Dependencies | `DL/DL/requirements.txt` pins torch 2.5.1, NumPy, pandas and others; Python 3.9.6 recorded |
| Logs | Per-epoch CSVs for all 65 runs, but **git-ignored** (m5) |
| Evaluation scripts | `dl.evaluation.evaluate validation \| test`, `report_figures`; documented commands work |
| Artifact naming | `runs/<experiment>/seed_<s>.pt`, consistent |
| Output consistency | Every checkpoint reproduces its logged validation RMSE (V, 65/65) |
| Tests | 12 / 12 pass (V) |

**Verdict:** reproducible on this platform and library versions. The results CSVs are not versioned.

## 13. Streamlit integration readiness

| Requirement | Status |
|---|---|
| Load the selected checkpoint | ✓ `load_checkpoint("DL/DL/artifacts/lstm_best.pt")` |
| Load the scaler and schema | ⚠ Referenced by path and hash. Loading **requires** `DL/Dataset/processed/*.json` to be present and unchanged. |
| Accept a raw 12-hour sequence and port ID | ✗ **Blocker.** No API turns raw telemetry, weather and the disruption log into the 8 scaled channels. The logic exists only inside the dataset build (`cargomind_dl.sequences.transform` on a build panel; disruption features from the event table). |
| Produce a prediction in hours | ⚠ Possible via `cargomind_dl.dataset.to_hours`, but this is not wrapped in an inference function |
| Handle invalid input safely | ✗ **Blocker.** NaN gives NaN; any sequence length is accepted; unscaled input is accepted; an out-of-range port raises a raw `IndexError` (V, probes) |
| Inference separated from training | ✓ `dl.models` imports no training code (V) |

**Readiness: not ready.** Blockers: a preprocessing API, input validation, and packaging the scaler and schema with the model (M5).

## 14. Limitations

- **Independence:** same-session audit (I1). Recommend a second reviewer.
- **Test-set status:** this audit's diagnostic test scoring of all 13 experiments means the test split can no longer select among them without bias (I3).
- **Validation bias:** validation was used for early stopping, LR scheduling and selection, so validation scores are optimistic (I4).
- **Small subgroups:** subgroups are thin and episode-driven, often 1–2 ports (I5).
- **Synthetic data:** 41.7 days, 10 ports. No real-world claims (I6).
- **Upstream lineage:** the fill method for reconstructed input values is undocumented.

## 15. Final verdict

### FAIL – NOT READY

**What passes:**
- Implementation correctness.
- Leakage controls.
- Test-set discipline.
- Reproducibility.

**The selected model and its integration path do not pass:**

1. **Insufficient evidence of value over EWMA.**
   - The gain is +0.042 h on validation (inside the project's 0.05 h tie band; significant for 1 of 5 seeds), and 0.09–0.13 h on test (2–3 %; significant for 4 of 5 seeds).
   - A small, seed-dependent average gain does not justify replacing EWMA (M2).
2. **Reliability failure where CargoMind needs reliability.**
   - Worse than EWMA, with stronger under-forecasting, at extreme dwell, in long disruptions, and at ports striking for the first time (+0.5 to +1.4 h).
   - Negative bias overall (M3, M4).
3. **The selected configuration is dominated** by C_w and the GRU on validation (M1).
4. **Integration blockers:** no raw-input preprocessing API and no input validation (M5).

**To reach a pass:**
1. Fix the selection rule (objective simplicity measure).
2. Choose among the existing configurations on validation (C_w and the GRU are the evidence-backed candidates; C_w shows no extreme-value failure).
3. Obtain **new held-out data**, such as a rolling-origin evaluation or a later period, for an unbiased final test with a pre-registered minimum gain over EWMA. Report bias, extremes and strike/disruption subgroups.
4. Build a validated inference module.
5. Correct the overstated disruption claim (M6) and the minor items (m1–m6).

**Until then, CargoMind should use EWMA (α = 0.2) for next-hour dwell forecasts.**
