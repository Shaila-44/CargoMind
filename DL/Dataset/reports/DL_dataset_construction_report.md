# CargoMind Stage 02 - DL Dataset Construction

**Task:** forecast next-hour `container_dwell_hours` for each port from the previous 12 hourly observations.

This stage builds the dataset only. No model has been trained.

**Rebuild from the project root:**

```
.venv/bin/python DL/Dataset/data_audit/scripts/run_data_audit.py   # provenance masks (needed once)
.venv/bin/python DL/Dataset/scripts/build_dl_dataset.py            # about 10 s
```

**Sources (read-only):**
- `ML/Dataset/Processed/`: telemetry, weather and disruption events.
- `ML/Dataset/Raw/disruption_events.csv`: only to identify reconstructed event end times.
- `DL/Dataset/data_audit/lineage/`: cell-level provenance from the audit.

Nothing in `Dataset/` was modified. Outputs are in `DL/Dataset/processed/`, and code is in `DL/DL/src/cargomind_dl/`.

---

## 1. Result

| | Train | Validation | Test |
|---|---|---|---|
| Target hours (UTC) | 2025-01-01 12:00 → 01-30 03:00 | 01-30 16:00 → 02-05 09:00 | 02-05 22:00 → 02-11 15:00 |
| Hour range of any input or target | 0 – 699 | 700 – 849 | 850 – 999 |
| **Windows kept** | **6,579** | **1,306** | **1,318** |
| Per port | 653 – 663 | 125 – 135 | 128 – 135 |
| `X` shape | [6579, 12, 7] | [1306, 12, 7] | [1318, 12, 7] |

- These counts reproduce the audit's estimate exactly.
- All 10 integrity and leakage checks pass (`DL/Dataset/processed/integrity_checks.csv`).
- Two independent builds produce identical arrays.
- **Baseline to beat:** EWMA with test RMSE **4.23 h** (persistence 5.37 h, 12-h trailing mean 4.30 h).

## 2. Feature set

All features describe hour t and are known once hour t has closed. Channel order is fixed (`feature_schema.json`).

| # | Channel | Source | Unit | Definition | Scaling (fitted on training hours only) |
|---|---|---|---|---|---|
| 0 | `dwell` | telemetry `container_dwell_hours` | h | Container dwell in hour t | Per-port z-score |
| 1 | `utilization` | telemetry `port_utilization_pct` | % | Throughput / capacity. Censored at 100 when saturated. | Fixed ÷ 100 (no fit) |
| 2 | `labour_strike` | telemetry `labour_status` | 0/1 | 1 if `strike`. "Reduced" counts as 0 (no measurable signal in the audit). | None |
| 3 | `wind_gust` | weather `wind_gusts_10m_kmh`, same (port, hour) | km/h | Maximum gust. 2 negative values clipped to 0 (flagged). | Global z-score |
| 4 | `disruption_active` | `disruption_events` | 0/1 | 1 if an event at this port has started (start ≤ t) and not ended (t < end) | None |
| 5 | `disruption_hours_elapsed` | `disruption_events` | h | Hours since the ongoing disruption episode began (earliest start among overlapping ongoing events); 0 otherwise | log1p, then global z-score |
| 6 | `vessels_at_anchor` | telemetry `vessels_at_anchor` | vessels | Vessels waiting at anchor | log1p, then per-port z-score |

**Target:** dwell at hour t+1, scaled with the port's dwell scaler (`y`). It is also stored in hours (`y_hours`), and `cargomind_dl.dataset.to_hours` inverts the scaling.

**Excluded, as decided:**

| Signal | Reason |
|---|---|
| Throughput | Exact multiple of utilization |
| Customs hold | White noise; most reconstructed signal |
| Temperature | No within-port signal |
| Equipment availability | No next-hour signal |
| Wind speed | Replaced by gusts (r = 0.98) |
| `port_traffic_history` | Daily aggregate of the same telemetry; same-day values leak later hours |

**Scaler choices.** Fitted values are in `scalers.json`.
- **Per-port z-scores** for dwell and anchor counts, because port levels differ far more than hours do. For example, NLRTM dwell has a training mean of 132.6 h and SD 31.4, against AEJEA's 105.6 h and 4.6.
- **Global z-score** for gusts, which share a physical scale across ports.
- **Fixed ÷ 100** for utilization, which is already bounded.
- **Fit data.** Every scaler uses training-period hours (0–699) only, and **reconstructed cells are excluded from the fit**. Each dwell scaler is fitted on 664–673 observed or converted hours per port.

## 3. Disruption features: no retrospective information

- **Only two fields are used:** `event_start` and whether `event_end` has passed. At hour t, "the event is still ongoing" is known. An event that has ended is a past fact.
- **Never used:** severity, affected capacity, planned or remaining duration, the event description, and events that start after t.
- **Assumption:** an event is known from its logged start hour, which is the standard reading of an event log. Real logs may be back-dated, and this cannot be checked on synthetic data.
- **Reconstructed end times.** Three events had no end time in raw; it was reconstructed during cleaning (`disruption_events_with_reconstructed_end.csv`). The hours they affect are flagged in `aux_flags[..., 0]`:

  | Event | Port, type | Kept windows affected |
  |---|---|---|
  | EVT00007 | AEJEA weather | ┐ 52 training windows |
  | EVT00022 | MYPKG weather | ┘ |
  | EVT00052 | MYPKG strike, starts 02-09 | 55 test windows |

  EVT00052's reconstructed end (02-11 16:00) is after the last data hour, so it changes no feature value.

## 4. Policy for reconstructed values

Provenance comes from the audit's value-verified raw ↔ processed join:
- **0 = observed:** equal to the raw value.
- **1 = converted:** a raw value after a unit or format fix.
- **2 = reconstructed:** no raw observation; filled during cleaning by an unknown method.

| | Policy |
|---|---|
| **Target** | A window whose target hour is reconstructed is **excluded from training and from evaluation**. 437 windows were dropped this way: 25–35 / 3–13 / 3–10 per port in train / val / test. |
| **Input values** | **Kept as published** in `ML/Dataset/Processed/`. They are **not re-imputed**: a backward-looking fill would invent new values, and interpolation would use the future. |
| **Input flags** | Stored for every window cell in `provenance` [N, 12, 5] (dwell, utilization, labour, gust, anchor). `dataset.load_split(split)` appends a `dwell_reconstructed` 0/1 channel by default (input [N, 12, 8]) so the model can discount filled hours. `mask_channels=()` gives the 7 features only; other names add more flags. |
| **Sensitivity subset** | `windows.csv` `n_reconstructed_input_cells == 0` selects windows with fully observed inputs: 1,654 / 319 / 351. It is too small for training but usable as an evaluation slice. |

**How common reconstructed inputs are among kept windows:**
- 42 % (train), 48 % (val) and 42 % (test) contain at least one reconstructed *dwell* input.
- About 75 % contain at least one reconstructed cell in any feature.

The values are individually rare (dwell: 313 / 80 / 66 reconstructed hours by split) but scattered (`provenance_summary.csv`).

## 5. Window construction

Sample (port p, hour t):
- inputs = hours t−11, …, t of port p (oldest first);
- target = dwell of port p at hour t+1.

Windows are built per port and never mix ports. All 10 ports share one regular hourly grid (1,000 h, verified at build time), so every candidate window spans 13 consecutive real hours. **No window spans an unobserved gap.** There are none, and the build stops with an error if the grid is ever irregular.

**Example** (SGSIN): inputs 2025-01-08 21:00 … 01-09 08:00 → target 01-09 09:00.

## 6. Chronological split and boundary leakage

1. **Global cut-offs on the target hour, the same for every port.** Train targets are hours 0–699, validation 700–849, test 850–999 (70 / 15 / 15).
2. **Embargo.** A window is kept only if **all of its input hours lie in the same split as its target**. This drops the 12 candidate windows whose targets are the first 12 hours of validation or test, per port and boundary: 240 in total.
3. **Result:**
   - The latest hour any training window touches (input or target) is 699.
   - Validation windows use hours 700–849 only.
   - Test windows use 850–999 only.
   - Test is strictly later than training, and **no hour appears in two splits**, so overlapping windows cannot carry an observation across a boundary.
4. **Fitting happens after the split.** Scalers are fitted after splitting, on training hours only. The EWMA smoothing factor is chosen on training windows only. Nothing is fitted on validation or test.
5. **Shuffling.** Overlapping windows *within* a split are fine, since every window's inputs precede its own target. Training windows may be shuffled after loading. Validation and test should be evaluated in full.

## 7. Sequence counts per port and split

Source: `DL/Dataset/processed/sequence_counts.csv`.

| Port | Train kept | Val kept | Test kept | Dropped: embargo (val + test) | Dropped: reconstructed target (train / val / test) |
|---|---|---|---|---|---|
| AEJEA | 653 | 132 | 133 | 24 | 35 / 6 / 5 |
| CNSHA | 653 | 133 | 129 | 24 | 35 / 5 / 9 |
| DEHAM | 659 | 135 | 131 | 24 | 29 / 3 / 7 |
| INMAA | 658 | 134 | 132 | 24 | 30 / 4 / 6 |
| INNSA | 663 | 132 | 128 | 24 | 25 / 6 / 10 |
| LKCMB | 656 | 131 | 132 | 24 | 32 / 7 / 6 |
| MYPKG | 661 | 126 | 135 | 24 | 27 / 12 / 3 |
| NLRTM | 657 | 131 | 129 | 24 | 31 / 7 / 9 |
| SGSIN | 657 | 127 | 134 | 24 | 31 / 11 / 4 |
| USLAX | 662 | 125 | 135 | 24 | 26 / 13 / 3 |
| **Total** | **6,579** | **1,306** | **1,318** | **240** | **301 / 74 / 62** |

- The candidate windows per port are 688 / 150 / 150 before exclusions.
- Training needs no embargo: its first input hour is 0.

## 8. Baselines

Sources: `DL/Dataset/processed/baseline_metrics.csv`, and per-window predictions in `windows.csv`.

The baselines see exactly what a model sees, namely the 12 input hours, and are scored on the same kept windows in hours:

| Baseline | Train RMSE | Val RMSE | **Test RMSE** | Test MAE |
|---|---|---|---|---|
| Persistence: ŷ = dwell(t) | 5.57 | 5.66 | 5.37 | 4.29 |
| Trailing mean of the 12 inputs | 4.38 | 4.33 | 4.30 | 3.45 |
| **EWMA over the window** (α = 0.2, chosen on train from {0.05 … 0.5}) | 4.32 | 4.30 | **4.23** | **3.41** |

**Test RMSE per port** (EWMA / trailing mean / persistence, in hours):

| Port | EWMA | Trailing mean | Persistence |
|---|---|---|---|
| AEJEA | 4.21 | 4.16 | 5.31 |
| CNSHA | 4.33 | 4.47 | 5.45 |
| DEHAM | 3.73 | 3.67 | 4.79 |
| INMAA | 4.66 | 4.69 | 6.15 |
| INNSA | 4.32 | 4.73 | 4.90 |
| LKCMB | 4.20 | 4.08 | 5.64 |
| MYPKG | 4.21 | 4.34 | 5.18 |
| NLRTM | 4.69 | 4.80 | 5.96 |
| SGSIN | 4.04 | 4.08 | 5.35 |
| USLAX | 3.88 | 3.91 | 4.83 |

- **Persistence is not the benchmark.** It copies the roughly 4-h hourly noise that the audit identified.
- **A DL model should be reported against the EWMA and the trailing mean,** per port, with the remaining headroom expected to lie mainly in disruption and backlog hours. The audit estimated the noise floor at ≤ about 3.9 h RMSE.

## 9. Saved artifacts (`DL/Dataset/processed/`)

| File | Contents |
|---|---|
| `train.npz`, `val.npz`, `test.npz` | `X` [N, 12, 7] float32, scaled. `provenance` [N, 12, 5] int8 (0 / 1 / 2). `aux_flags` [N, 12, 2] (disruption end reconstructed, gust clipped). `y` (scaled target), `y_hours`, `last_dwell_hours`, `port_index`, `target_hour_index`, `window_id` |
| `feature_schema.json` | Channel order, sources, units, definitions, scaling, provenance codes, excluded features, the reconstructed-value policy and the EWMA α |
| `scalers.json` | Fitted per-port and global statistics, with fit counts and fit period |
| `split_metadata.json` | Rule, hour-index and UTC bounds, kept counts overall and per port, exclusions |
| `windows.csv` | Every candidate window (kept or not): port, t, target time, split, exclusion reason, target provenance, reconstructed-input counts, baseline predictions |
| `hourly_panel.csv` | The unscaled hourly feature panel with provenance codes and the split of each hour |
| `sequence_counts.csv`, `provenance_summary.csv`, `baseline_metrics.csv`, `integrity_checks.csv`, `disruption_events_with_reconstructed_end.csv` | Reports |
| `manifest.json` | SHA-256 of every source file (processed CSVs, raw events, audit lineage) and every output, plus seed and library versions |

**Loading in later experiments:**

```python
from cargomind_dl import dataset
tr = dataset.load_split("train")                 # X: [6579, 12, 8] (7 features + dwell_reconstructed)
te = dataset.load_split("test", mask_channels=())  # X: [1318, 12, 7]
hours = dataset.to_hours(predictions, te["port_index"])
```

All three model families can use the arrays directly:
- **LSTM / GRU:** `[N, time, channels]`.
- **1D CNN:** transpose to `[N, channels, time]`.

`port_index` is available for an embedding.

**Version control.** The repository's `.gitignore` excludes `*.csv` outside `Dataset/`. The `.npz` and `.json` files can be committed, but the CSV reports, and the audit lineage CSVs this build reads, are regenerated by the two commands at the top.

## 10. Notes for the modelling stage

1. **Distribution shift between splits.** Measured in training-scale units:
   - Test dwell sits below the training level (mean z −0.25), because training includes the large January backlog episodes.
   - Strikes are much more frequent in test hours (8.5 % vs 3.2 % in training).
   - Disruptions are less frequent in validation and test hours (8.3 % and 11.3 %) than in training (16.9 %).

   Report results per port and for disruption vs normal hours.
2. **Limited episode coverage in validation and test.**
   - AEJEA, CNSHA, INMAA and SGSIN have no disruption hours there.
   - Congestion and cyber events appear only in training.
   - Validation is about 6 days long.

   Use several seeds, and treat small RMSE differences between LSTM, GRU and CNN as ties.
3. **No GNN.** The data has no cross-port dependence to exploit (audit section 11), and none was built.
4. **Synthetic data.** No result built on this dataset implies real-world performance.
