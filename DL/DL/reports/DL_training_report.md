# CargoMind Stage 02 — LSTM Training Report

**Task:** forecast next-hour `container_dwell_hours` per port from the previous 12 hourly observations, evaluated in hours.

**Scope of this report:** architecture, training, validation-based selection, ablations and the GRU challenger.

- All decisions here use **training and validation data only**.
- The one-time test evaluation is in [DL_evaluation_report.md](../../Evaluation/reports/DL_evaluation_report.md).
- Every number comes from `DL/DL/artifacts/experiment_results.csv`, `DL/DL/training/logs/` or `DL/Evaluation/results/evaluation/validation_*.csv`.
- The data is synthetic. No real-world performance is implied.

## 1. Architecture and design decisions

```
x [B, 12, C]  ──►  LSTM (1 layer, hidden 32, batch_first)  ──►  h_T [B, 32]
port_index [B] ──►  Embedding(10, 4)  (configurations B, B_w only) ──►  concat [B, 36]
                                     ──►  Dropout(0.1) ──►  Linear(·, 1) ──►  ŷ [B]  (next-hour dwell, per-port z-score)
optional (C_w):  ŷ = EWMA_α=0.2(dwell channel over the 12 inputs) + Linear(...)   (head initialised at 0)
```

| Decision | Choice | Why |
|---|---|---|
| Sequence summary | Final hidden state h_T of the last layer | The target is one step ahead, so the state after the latest hour is the natural summary. No attention or pooling was added without evidence. |
| Size | 1 layer, 32 units: 5,409 parameters (5,453 with embedding) | The EDA showed most next-hour variance is irreducible noise of about 3.9 h, so a small model limits overfitting. |
| Port embedding | 4-d `nn.Embedding(10, 4)` concatenated to h_T | Ports differ strongly. Tested as A vs B, never assumed. |
| Output | `[B]` (`squeeze(-1)`) | `mse_loss` raises on any shape mismatch, so `[B, 1]` vs `[B]` cannot silently broadcast to `[B, B]` (unit-tested). |
| Target | Saved per-port z-score `y`. Evaluation in hours via `cargomind_dl.dataset.to_hours`. | No scaler was refitted or re-implemented. |
| Device / determinism | CPU, 4 threads, `torch.use_deterministic_algorithms(True)`, fixed seeds for Python, NumPy, torch and the DataLoader | Two runs with the same seed give identical weights (unit-tested). Rerunning round 1 reproduced every number. |

**Code layout:**
- `DL/DL/models/recurrent.py`: model.
- `DL/DL/models/checkpoint.py`: save and load. It imports no training code, so a separate inference module can use it.
- `DL/DL/training/`: config, data adapter, trainer, experiment runner.
- `DL/Evaluation/evaluation/`: metrics, subgroups, evaluation, figures.
- `DL/Evaluation/tests/test_pipeline.py`: 12 unit tests.

## 2. Dataset and loader integration

| Item | Verified value |
|---|---|
| Loader | `cargomind_dl.dataset.load_split(split, mask_channels=("dwell",))`, wrapped by `DL/DL/training/data.py` (feature selection by name only) |
| Input | `X [N, 12, 8]`, float32, already scaled. Channels: `dwell, utilization, labour_strike, wind_gust, disruption_active, disruption_hours_elapsed, vessels_at_anchor, dwell_reconstructed` |
| Windows | train 6,579 · validation 1,306 · test 1,318 (chronological, embargoed) |
| Target | `y [N]` (per-port z-score), `y_hours [N]`, `port_index [N]` |
| Shuffling | Training windows only, with a seeded generator. Validation is evaluated in stored order. The trainer never loads the test split. |
| Inverse scaling | `to_hours(y, port_index)` reproduces `y_hours` to < 1e-3 h (unit-tested) |

## 3. Hyperparameters (all runs unless stated)

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam, lr 1e-3, no weight decay |
| Batch size | 64 |
| Max epochs / early stopping | 150 / patience 15 on **validation RMSE in hours** |
| LR schedule | ReduceLROnPlateau on validation RMSE (factor 0.5, patience 5, min 1e-5). The target is noisy and a fixed LR oscillates near the optimum. |
| Gradient clipping | Global norm 1.0 |
| Dropout | 0.1 on h_T (a single layer has no inter-layer dropout) |
| Checkpoint | Best epoch by validation RMSE (hours). Training loss is never used for selection. |
| Seeds | 0, 1, 2, 3, 4 |
| Tie tolerance | 0.05 h |

**Runtime and environment:**
- A run takes 3–14 s on CPU.
- All 65 runs take a few minutes.
- torch 2.5.1 (CPU wheel, installed into `.venv` for this stage), Python 3.9.6, NumPy 2.0.2.

## 4. Round 1: A vs B, and the loss–metric mismatch it revealed

| Experiment | Mean validation RMSE (h) | SD | Min – max | Mean best epoch |
|---|---|---|---|---|
| A: LSTM | 4.446 | 0.066 | 4.352 – 4.519 | 2.4 |
| B: LSTM + port embedding | 4.438 | 0.072 | 4.323 – 4.515 | 5.2 |
| *EWMA α = 0.2 (reference)* | *4.296* | | | |

- Both configurations were **worse than EWMA**, and most seeds peaked at epoch 1–5.
- The pre-registered rule chose A (B − A = −0.009 h, a tie). This is recorded in `DL/DL/artifacts/selection_round1.json`.

**Diagnosis** (validation only; `DL/Evaluation/figures/training_curves.png`, left column):
- The **unweighted scaled validation loss kept falling** while the **validation RMSE in hours rose**.
- MSE on per-port z-scores weights each port by 1/σ²_port. A 0.1-z error costs about 0.4 h at MYPKG (σ ≈ 4 h) but about 3.1 h at NLRTM (σ ≈ 31 h).
- Seed 3 of A beat EWMA at 8 of 10 ports, and in scaled MSE overall (0.449 vs 0.471). But it was **0.73 h worse at NLRTM and 0.62 h worse at USLAX**, the two ports that dominate hour-scale RMSE.
- The loss was optimising a different objective from the selection metric. This is a pipeline-correctness issue, not a capacity problem.

## 5. Round 2: aligning the loss with the evaluation metric

Two minimal changes. Neither adds parameters, and both were decided on validation evidence only:
1. **Hour-weighted loss (`_w`).** MSE on the scaled target, weighted per sample by the port's dwell variance from the saved `scalers.json`, normalised to mean 1. This equals MSE in hours up to a constant (unit-tested). Nothing is fitted.
2. **EWMA residual (C_w).**
   - The model outputs the window's fixed EWMA (α = 0.2, the dataset baseline's exact definition) plus a learned correction. The head is initialised at zero, so an untrained C_w *is* the EWMA baseline (unit-tested to < 1e-3 h).
   - EWMA is linear, so applying it in z-space equals applying it in hours.
   - This directly tests whether the LSTM adds anything beyond EWMA.

**Final selection rule, fixed before round 2 ran:** find the lowest mean validation RMSE over 5 seeds. Among candidates within 0.05 h of it, choose the simplest, in the order A → A_w → B → B_w → C_w.

| Experiment | Mean val RMSE (h) | SD | Min – max | Val MAE | Mean best epoch | Parameters |
|---|---|---|---|---|---|---|
| A | 4.446 | 0.066 | 4.352 – 4.519 | 3.575 | 2.4 | 5,409 |
| A_w | 4.281 | 0.019 | 4.267 – 4.314 | 3.440 | 6.2 | 5,409 |
| B | 4.438 | 0.072 | 4.323 – 4.515 | 3.542 | 5.2 | 5,453 |
| **B_w (selected)** | **4.254** | **0.015** | **4.231 – 4.268** | **3.424** | 11.8 | 5,453 |
| C_w | 4.212 | 0.008 | 4.203 – 4.225 | 3.390 | 7.8 | 5,409 |
| *EWMA α = 0.2* | *4.296* | | | *3.446* | | 0 |
| *12-hour mean* | *4.326* | | | *3.483* | | 0 |
| *Persistence* | *5.659* | | | *4.514* | | 0 |

**Selection:**
- Best mean: C_w (4.212). Threshold: 4.262. The first candidate in simplicity order under the threshold is **B_w** (4.254).
- **B_w was selected** by the pre-registered rule, and the rule was not changed after seeing the results.
- The checkpoint is B_w seed 2 (validation RMSE 4.231 h), copied to `DL/DL/artifacts/lstm_best.pt`.
- In hindsight, the order ranks C_w as "more complex" even though it has fewer parameters and no embedding. This is discussed under recommendations.

**B_w seed-level results:**

| Seed | Best val RMSE (h) | Val MAE (h) | Val R² | Best epoch | Epochs run | Train RMSE at best (h) |
|---|---|---|---|---|---|---|
| 0 | 4.258 | 3.421 | 0.961 | 11 | 26 | 4.163 |
| 1 | 4.268 | 3.433 | 0.961 | 20 | 35 | 4.127 |
| **2** | **4.231** | **3.407** | 0.962 | 13 | 28 | 4.133 |
| 3 | 4.248 | 3.423 | 0.961 | 13 | 28 | 4.118 |
| 4 | 4.267 | 3.436 | 0.961 | 2 | 17 | 4.239 |
| **Mean ± SD** | **4.254 ± 0.015** | 3.424 | 0.961 | 11.8 | | |

**Consistency:**
- All 5 B_w seeds beat the validation EWMA (4.296), by 0.03–0.07 h.
- The mean gain (0.042 h) is **inside the 0.05 h tie band**.
- On validation, B_w is therefore *practically indistinguishable* from EWMA overall.

## 6. Training curves and overfitting

From `DL/Evaluation/figures/training_curves.png` and `experiment_results.csv`:

| Experiment | Train RMSE at best epoch (h) | Val RMSE (h) | Gap |
|---|---|---|---|
| B_w | 4.156 | 4.254 | 0.098 |
| C_w | 4.137 | 4.212 | 0.075 |
| GRU | 4.110 | 4.202 | 0.093 |
| Dwell history only | 4.286 | 4.274 | −0.012 |

**Observations:**
- **Training error keeps falling while validation flattens** after about 10 epochs, a modest and typical generalisation gap of about 0.1 h. Early stopping (patience 15) and the LR schedule stop training 15 epochs after the best epoch.
- **The operational channels are where the model can overfit.** With dwell history only, the train–validation gap disappears and seed variation collapses (SD 0.002 h).
- **No run fits the training data much below the noise floor.** The best training RMSE is about 4.1 h. The model is noise-limited, not capacity-limited.

## 7. Port embedding

| Comparison (validation) | Δ mean RMSE (h) | Verdict |
|---|---|---|
| B − A (unweighted loss) | −0.009 | Tie |
| B_w − A_w (hour-weighted loss) | −0.027 | Tie (inside 0.05 h) |

The embedding helps slightly and consistently, but not by a practically meaningful margin. It was kept only because the selection rule placed B_w within the tolerance band.

## 8. Ablations of the selected configuration (validation, 5 seeds each)

One factor at a time, relative to B_w (4.254 h). Δ is the change in mean validation RMSE; negative means better *without* the factor.

| Removed | Mean val RMSE (h) | Δ vs B_w | Disruption-active windows (n = 102) | Strike windows (n = 9) |
|---|---|---|---|---|
| — (B_w, all 8 channels) | 4.254 | — | 4.322 | 4.182 |
| `labour_strike` | 4.244 | −0.010 | 4.295 | 4.281 |
| `dwell_reconstructed` | 4.250 | −0.004 | 4.387 | 4.242 |
| `vessels_at_anchor` | 4.257 | +0.003 | 4.025 | 4.445 |
| `utilization` | 4.244 | −0.010 | 4.240 | 4.250 |
| both disruption channels | 4.241 | −0.014 | 4.260 | 4.380 |
| `wind_gust` | 4.212 | −0.042 | 4.147 | 4.246 |
| **all operational channels** (dwell history only) | 4.274 | **+0.020** | **4.406** | **4.789** |
| port embedding (= A_w) | 4.281 | +0.027 | 4.338 | 4.198 |
| *EWMA reference* | *4.296* | | *4.383* | *4.708* |

**Interpretation:**
- **No single feature is individually necessary.** Every one-factor change is inside ±0.05 h. The operational channels overlap: a capacity cut shows up as low utilization, an active disruption or strike, and a growing anchor queue at once.
- **Together, the operational channels help during episodes.** Without them, disruption-hour RMSE rises from 4.32 to 4.41 h, mainly on disruptions 7–24 h old (4.58 → 4.98 h, n = 44), and strike-hour RMSE from 4.18 to 4.79 h (n = 9, too few to rely on).
- **Wind gust looks harmful on validation** (removing it gives −0.042 h, near the tolerance). The EDA found gusts shifted by +1.2 to +1.8 training SDs at INNSA and LKCMB in the validation period. This is consistent with covariate shift rather than useful signal. It is not acted on here: the model was fixed before the ablations, and the effect is inside the tie band.
- **The `dwell_reconstructed` flag makes no measurable difference**, consistent with the EDA. Keeping it costs nothing.
- **`labour_strike` cannot be judged on validation**, which has only 9 strike windows. The test period's strikes at INNSA and MYPKG, ports never seen striking, are a separate generalisation question.

## 9. GRU challenger (validation only)

| Model | Parameters | Mean val RMSE (h) | SD | Mean best epoch |
|---|---|---|---|---|
| B_w (LSTM, 32 units) | 5,453 | 4.254 | 0.015 | 11.8 |
| GRU, same settings, 37 units | 5,299 | **4.202** | 0.008 | 27.8 |

- The GRU is better by 0.052 h, just outside the tolerance, and more stable across seeds.
- It ran **after** the model was fixed and the test was evaluated, so it is a candidate for the next iteration, not a replacement. It was **not** evaluated on test.

## 10. Reproduce

```
.venv/bin/python -m unittest discover -s DL/Evaluation/tests          # 12 tests
.venv/bin/python -m dl.training.run_experiments main            # round 1 (A, B)
.venv/bin/python -m dl.training.run_experiments round2          # round 2 (A_w, B_w, C_w) + final selection -> lstm_best.pt
.venv/bin/python -m dl.training.run_experiments ablations       # one-factor ablations of the selected config
.venv/bin/python -m dl.training.run_experiments gru             # GRU challenger
.venv/bin/python -m dl.evaluation.evaluate validation           # validation diagnostics
.venv/bin/python -m dl.evaluation.report_figures                # figures
```

Training is deterministic. A rerun reproduces every result in `experiment_results.csv` on the same machine, thread count and library versions (`DL/DL/artifacts/experiment_metadata.json`).
