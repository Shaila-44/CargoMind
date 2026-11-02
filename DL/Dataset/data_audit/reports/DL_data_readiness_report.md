# CargoMind Stage 02 - DL Data Readiness Audit

**Objective under audit:** forecast next-hour `container_dwell_hours` per port with an LSTM. Each input is a 12-hour window of hourly signals (8 signals in the PPT design), and the split is chronological.

**Verdict: READY WITH MINOR FIXES** (section 11).

Reproduce with: `.venv/bin/python DL/Dataset/data_audit/scripts/run_data_audit.py` (about 1 minute).

- Data read (read-only): `ML/Dataset/Processed/` (the brief's `Datasets/Processed/` does not exist; the directory is `Dataset/`). `ML/Dataset/Raw/` was read only to trace lineage.
- Neither directory was modified, and no model was trained.
- Every number below comes from a file under `DL/Dataset/data_audit/`; the file is named next to each table.

---

## 1. Summary

| Area | Finding | Status |
|---|---|---|
| Hourly timeline | 10 ports × 1,000 h each (2025-01-01 00:00 → 2025-02-11 15:00 UTC). No missing or duplicate hours, perfectly regular 1-h steps, all UTC. Telemetry and weather keys match 1:1. | ✅ |
| Hidden gap-filling | The processed files are complete, but **120 telemetry hours and 100 weather hours do not exist in the raw data**. **459 target values (4.6 %) have no raw observation.** The fill method is unknown: it is not linear interpolation, forward-fill or the port median, and there are no imputation flags. | ⚠️ fixed by the lineage masks produced here |
| Target | Valid and non-negative (54.6–205.2 h). Each port has its own level. Hour-to-hour change is dominated by white noise (SD ≈ 4 h); forecastable structure comes from multi-day backlog episodes. | ✅ meaningful, but the attainable gain is small |
| PPT 8 signals | 4 of 8 carry measurable next-hour information: dwell history, utilization, wind, labour strike. Throughput duplicates utilization. Customs, temperature and equipment carry ≈ 0. | ⚠️ revise the feature set |
| Better signals in the data | Disruption state (from `disruption_events`), vessels at anchor and wind gusts. | ➕ add |
| Sequences (12 h → 1 h) | 9,640 windows with no lineage filter; **9,203 with an observed target** (6,579 / 1,306 / 1,318 for train / validation / test). | ✅ |
| Leakage | No structural risk in the timeline. Risks to control: fitting on unknown-provenance fills, cross-split windows, scaler fitting, and the daily traffic table. | ✅ with the rules in section 7 |
| Sufficiency | 41.7 days and 10 ports. Enough for a small LSTM / GRU / 1D-CNN comparison, **not** for seasonal claims. Graph-neighbour features have no measurable signal to exploit. | ⚠️ limitations |

## 2. Table-wise schema and quality

Sources: `tables/schema_tables.csv`, `tables/schema_columns.csv`.

| Table | Rows × cols | Key (unique) | Missing cells | Needed for DL? |
|---|---|---|---|---|
| `port_hourly_telemetry` | 10,000 × 16 | `port_id` + `ts_hour` ✓ | 0 | **Yes**: target and port operations |
| `weather_hourly` | 10,000 × 15 | `port_id` + `ts_hour` ✓ | 0 | **Yes**: wind gusts (and temperature, if the PPT set is kept) |
| `disruption_events` | 55 × 11 | `event_id` ✓ | 55 (`source_url` empty) | **Yes**: disruption state known at time t |
| `port_metadata` | 10 × 12 | `port_id` ✓ | 1 | **Yes**: timezone, capacity, coordinates |
| `public_holidays` | 32 × 9 | 3-part key ✓ | 50 (blank subdivision = national) | Optional: holiday flag |
| `port_network` | 90 × 9 | `route_id` ✓ | 52 (blank hub on direct routes) | Only for graph experiments |
| `port_traffic_history` | 420 × 10 | `port_id` + `period_start` ✓ | 0 | **No**: daily aggregate of telemetry (EDA: `vessel_calls` = daily sum of arrivals), so using the same day leaks later hours |
| `shipment_records` | 9,850 × 16 | `shipment_id` ✓ | 0 | **No**: Stage 01 table, different grain |

**Telemetry and weather details:**
- No exact duplicates. All `port_id`s resolve to metadata. Every timestamp is ISO-8601 with `+00:00`.
- `labour_status` takes three values: normal 8,732, reduced 897, strike 371.
- `*_is_synthetic` flags are constant `True` and `data_source` is constant. These, along with `grid_latitude` / `grid_longitude` (constant per port), `weather_code` / `weather_condition` (categorical duplicates) and `wind_direction_10m_deg` (circular; would need sin/cos), are not suitable as sequence inputs.

**Value problems** (flagged here, not altered):

| Problem | Extent |
|---|---|
| `wind_gusts_10m_kmh` < 0 | 2 h |
| Gust < wind speed | 77 h (NLRTM 30) |
| `port_utilization_pct` capped at 100 | 1.1 % (MYPKG) to 40.9 % (NLRTM) of hours, i.e. censored throughput |
| `visibility_m` with no raw source at USLAX, CNSHA, LKCMB | 3,065 values (30.6 %) not observed |

**Units:**

| Signal | Unit |
|---|---|
| Dwell, vessel time in port, customs hold | hours |
| Throughput | TEU / h |
| Utilization | % of nominal capacity |
| Equipment availability | % |
| Temperature | °C |
| Wind speed, wind gusts | km/h |
| Precipitation | mm / h |
| Pressure | hPa |

Utilization equals `throughput / capacity × 100` to within 0.03 pp at every port (`tables/utilization_vs_throughput.csv`).

## 3. Per-port temporal coverage

Source: `tables/temporal_coverage_by_port.csv`.

| Check (both hourly tables, every port) | Result |
|---|---|
| First / last hour | 2025-01-01 00:00 / 2025-02-11 15:00 UTC, identical for all ports |
| Observations / expected hours | 1,000 / 1,000 |
| Missing hours, duplicate timestamps | 0 / 0 |
| Ordering | Monotonic within each port in file order |
| Intervals ≠ 1 h, non-hour-aligned stamps | 0 / 0 |
| Timezone | 100 % UTC `+00:00` |
| Telemetry ↔ weather keys | 10,000 in both; 0 unmatched (`tables/audit_facts.json`) |

**The timeline is complete only because it was filled during cleaning**, which is what the next section addresses.

## 4. Target distribution: `container_dwell_hours`

Sources: `tables/target_profile_by_port.csv`, `figures/dwell_series_by_port.png`, `figures/dwell_autocorrelation.png`.

| Port | Median | p5 – p95 | Max | SD | Lag-1 ACF (level) | Reconstructed |
|---|---|---|---|---|---|---|
| AEJEA | 105.3 | 98.5 – 112.7 | 122.7 | 4.4 | 0.23 | 48 |
| CNSHA | 79.6 | 65.0 – 108.3 | 117.5 | 15.5 | 0.93 | 50 |
| DEHAM | 68.5 | 61.9 – 76.3 | 82.6 | 4.4 | 0.24 | 41 |
| INMAA | 69.4 | 62.8 – 78.1 | 88.6 | 4.8 | 0.25 | 43 |
| INNSA | 82.7 | 74.4 – 106.4 | 120.2 | 9.5 | 0.84 | 46 |
| LKCMB | 106.7 | 99.8 – 120.2 | 129.9 | 6.2 | 0.60 | 45 |
| MYPKG | 67.8 | 61.3 – 77.0 | 93.2 | 5.1 | 0.42 | 43 |
| NLRTM | 115.5 | 103.8 – 190.3 | 205.2 | 27.9 | 0.98 | 50 |
| SGSIN | 77.9 | 70.0 – 102.4 | 120.9 | 9.5 | 0.81 | 48 |
| USLAX | 92.2 | 84.1 – 124.4 | 139.2 | 12.7 | 0.91 | 45 |
| **All** | 84.7 | 64.2 – 123.0 | 205.2 | 22.7 | — | **459 (4.6 %)** |

**Validity and outliers:**
- There are no negatives or zeros.
- There are no single-hour jumps beyond 6 robust SDs. The maxima are the peaks of multi-day backlog episodes, not outliers.

**Structure:**
1. **Ports differ more than hours do.** The pooled SD is 22.7 h against within-port SDs of 4.4–27.9 h. Per-port normalisation or a port embedding is required.
2. **Each hour has about 4 h of white noise on top of the level.**
   - At every port, hourly *changes* have lag-1 autocorrelation of −0.43 to −0.53. That is the signature of independent noise around a slowly moving level.
   - Their SD is 5.4–5.9 h, which implies noise SD ≤ 3.8–4.2 h.
   - At AEJEA, DEHAM, INMAA and MYPKG the series is almost entirely this noise: the *level* has lag-1 autocorrelation of only 0.23–0.42.
3. **Forecastable structure comes from backlog episodes.** These are rises of 10–90 h over 1–3 days that coincide with disruption events, plus CNSHA's plateau during the Spring Festival (28 Jan – 4 Feb). Several of these episodes outlast a 12-h window: NLRTM takes about 2 weeks to decay.
4. **Trends.** Daily-mean trends are within ±1 h/week except at CNSHA (+4.4), NLRTM (+3.7) and SGSIN (−2.4). These come from episodes, not secular drift.
5. **Dwell is not vessel time in port.** Within-port correlation ranges from −0.01 (DEHAM) to 0.82 (NLRTM) (`tables/target_vs_vessel_time.csv`). The two are related only through shared backlog, and `vessel_time_in_port_hours` must not be substituted for the target.

**Is next-hour forecasting meaningful?** Yes, but the attainable improvement is bounded. Fixed-formula reference forecasts on the test hours, with nothing fitted on test (`tables/baseline_floors.csv`, observed targets only, n = 1,424):

| Reference forecast | MAE (h) | RMSE (h) |
|---|---|---|
| Persistence (y<sub>t+1</sub> = y<sub>t</sub>) | 4.31 | 5.42 |
| Trailing mean, 12 h | 3.46 | 4.33 |
| EWMA, α = 0.2 (α chosen on train) | **3.40** | **4.24** |
| Port mean of the training period | 7.39 | 10.04 |
| Implied noise floor | ≈ 3.1 | ≤ ≈ 3.9 |

- Persistence is a **misleading** baseline here: it copies the noise.
- Any DL model must be judged against the EWMA / trailing-mean baseline. The headroom is roughly 0.3 h RMSE overall, concentrated in episode hours at the congested ports.

## 5. Suitability of the eight PPT signals

Sources: `tables/feature_suitability.csv`, `tables/partial_corr_given_trailing_mean.csv`, `lineage/lineage_counts_*.csv`. All signals cover 10 / 10 ports and 100 % of hours in the processed data. "Reconstructed" is the share of cells with no raw observation. "Partial r" is the within-port correlation with next-hour dwell after removing the 12-h trailing dwell mean (training period). It measures what a feature adds beyond the target's own recent history.

| # | PPT signal | Unit / range | Reconstructed | Lag-1 ACF | Partial r | Verdict |
|---|---|---|---|---|---|---|
| 1 | Container dwell (history) | h, 54.6–205.2 | 4.6 % | 0.62 | +0.12 (latest hour) | **Keep.** Core input. |
| 2 | Throughput | TEU/h, 51.6–5,600 | 3.5 % | 0.95 | −0.22 | **Drop.** Same information as utilization (exact ratio to capacity) but on port-specific scales. |
| 3 | Port utilization | %, 9.1–100 (censored) | 1.7 % | 0.95 | **−0.28** | **Keep.** Strongest exogenous signal: a capacity drop now means dwell rises next. Add an "at cap" (=100) indicator. |
| 4 | Customs hold hours | h, 0.1–94.9 | **6.1 %** | 0.04 | +0.01 | **Drop.** Hour-to-hour white noise (ACF 0.04) with no relation to dwell, and the most reconstructed signal. |
| 5 | Temperature | °C, −4.9–34.6 | 2.0 % (+1.6 % °F-converted) | 0.82 | +0.05 | **Drop.** No within-port relation (within-port r with next dwell 0.01). Its pooled correlation (−0.19) only reflects which ports are cold. |
| 6 | Wind speed | km/h, 0.5–66.2 | 1.0 % | 0.97 | +0.12 | **Replace with wind gusts.** Gusts are r = 0.98 with speed and carry the same signal (+0.12); gust thresholds are what stop cranes. |
| 7 | Labour status | normal / reduced / strike | 3.2 % | — | strike +0.17, reduced 0.00 | **Keep, one-hot encoded.** Only `strike` is informative; "reduced" is mostly random. |
| 8 | Equipment availability | %, 57.6–100 | 1.4 % (+2 % `%`-converted) | 0.45 | −0.00 | **Drop or keep for ablation only.** No measurable next-hour signal. |

**Signals present in the data but not in the PPT set:**

| Signal | Source | Partial r | Note |
|---|---|---|---|
| `disruption_active` | `disruption_events` (start ≤ t < end) | **+0.24** | Known at t. Must not use the event's end time or remaining duration. |
| `disruption_hours_elapsed` | `disruption_events` | **+0.21** | Hours since the event started, at t. |
| `vessels_at_anchor` | telemetry | +0.12 (+0.39 given only the current hour) | Count 0–33, 3.2 % reconstructed. A second noisy view of the backlog. |
| `wind_gusts_10m_kmh` | weather | +0.12 | Replaces wind speed. 2.5 % reconstructed; 2 negative values to clip at 0. |
| `vessel_time_in_port_hours` | telemetry | +0.03 | Weak. Optional. |
| `holiday_local` | `public_holidays` + timezone | +0.01 | Weak overall, but marks CNSHA's Spring Festival plateau. Optional. |
| Local hour of day | metadata timezone | −0.01 | No diurnal signal in dwell. Not needed. |

**Availability at prediction time.** Every feature above is a measurement or state for hour t, so all are known once hour t closes.
- `ts_hour` semantics (start vs end of hour) are undocumented. Predict t+1 only after hour t is complete.
- No feature correlates with the next-hour *change* beyond the dwell history itself (all |r| ≤ 0.035, lags 0–24 h). Features help estimate the backlog **level**, not hour-to-hour noise.

## 6. Missing-hour and discontinuity report (lineage)

Sources: `lineage/lineage_summary.csv`, `lineage/lineage_counts_*.csv`, `lineage/cell_lineage_*.csv`, `figures/dwell_reconstructed_hours_heatmap.png`.

**How the lineage was traced.** The audit matched every processed cell to the raw file after resolving the raw issues (with no assumptions left: 0 unmapped rows):
- 4 timestamp formats, including Unix epoch and unmarked IST times (the only `:30` values);
- misspelled and named port IDs;
- unit suffixes, fractions and Fahrenheit values;
- swapped wind columns.

Each cell is labelled by **value comparison**:
- *observed*: equal to the raw value;
- *converted*: equal after a unit or format fix;
- *reconstructed*: no usable raw value.

| | Telemetry | Weather |
|---|---|---|
| Processed hours that do not exist in raw | **120** (+120 duplicate raw rows dropped) | **100** (+100 dropped) |
| Target `container_dwell_hours`: observed / converted / reconstructed | 9,343 / 198 / **459** | — |
| Other reconstructed counts | customs 612, labour 318, anchor 317, throughput 349, arrivals 219, utilization 170, vessel time 150, equipment 145 | visibility 3,065, gusts 249, humidity 230, precipitation 220, pressure 199, temperature 197, wind speed 100 |

**What the reconstructed values are not:**
- They are *not* linear interpolation (1.3 % exact match, MAE 3.8 h), forward-fill or port medians.
- No cleaning code is in the repository, so how they were produced cannot be verified.
- Raw-observed cells match processed values exactly (all 9,343 observed dwell values equal), so the cleaning did not alter real observations.

**Placement.** The reconstructed hours are scattered: 41–50 per port, with no long blocks (red dots in `figures/dwell_series_by_port.png`). They break no window structurally, but they are **not observations** and must not be treated as such.

**Recommended handling (minor fix, no change to `ML/Dataset/Processed/`):**
1. Use `lineage/cell_lineage_port_hourly_telemetry.csv` and `cell_lineage_weather_hourly.csv` as the official mask. They have one row per (port, hour) and a status per column.
2. **Targets:** exclude windows whose target hour is reconstructed from loss and evaluation (policy B below).
3. **Inputs:** keep the processed value, and add a binary "observed" mask channel for the dwell history (and optionally per feature), so the model can discount filled hours. Do not re-impute with future values.
4. **Never span a real gap.** There are none in the processed timeline. If a future data refresh contains true gaps, end the window at the gap: do not interpolate across it, and do not build a window that crosses it.

## 7. Temporal leakage assessment

**Correct alignment** (SGSIN, `tables/alignment_example.csv`):

| Role | ts_hour (UTC) | Dwell | Utilization | Anchor |
|---|---|---|---|---|
| input t−11 | 2025-01-08 21:00 | 106.7 | 100.0 | 6 |
| … | … | … | … | … |
| input t−1 | 2025-01-09 07:00 | 107.7 | 100.0 | 9 |
| input t | 2025-01-09 08:00 | 99.8 | 100.0 | 15 |
| **target t+1** | **2025-01-09 09:00** | **109.0** | (not an input) | (not an input) |

Sample `(port, t)` = features from hours t−11 … t, label = dwell at hour t+1. Nothing from hour t+1 or later enters the inputs.

| Check | Status / rule |
|---|---|
| Inputs strictly before the target hour | Satisfied by the construction above. All inputs are hour-t measurements. |
| Target = next hour's dwell | `y = dwell.shift(-1)` within each port. Never shift across ports. |
| Chronological split before preprocessing | Global hour cut-offs, the same for every port. Train = first 700 h (to 2025-01-30 03:00); validation = 700–849 h (to 2025-02-05 09:00); test = last 150 h. Assign each sample to a split by its **target** hour. |
| Windows across split boundaries | Drop windows whose first input hour lies in an earlier split (an 11-h embargo). This costs 230 windows at W = 12 and keeps validation and test inputs inside their own periods. |
| Scalers / encoders | Fit per-port mean and SD (or robust scaling) on **training hours only**. Apply them unchanged to validation and test. Never fit on full series. |
| Imputation | None needed (no missing cells). Reconstructed values are masked, not re-imputed. Any future imputation must be causal (past-only) and fitted on train. |
| Future fill in history | The provenance of the 459 reconstructed target values is unknown, so they may encode later information. Masking (section 6) removes them from all targets. They remain as inputs at most in ≈ 5 % of hours, with a mask channel. |
| Overlapping windows | Overlap within a split is fine because every window's inputs precede its own target. Leakage would come only from windows that straddle a split, which the embargo removes. Shuffle training windows only *after* splitting. |
| Disruption features | Use event start and an "ongoing" flag at t. **Do not** use `event_end`, remaining duration or `affected_capacity_pct` before it would be known. |
| `port_traffic_history` | Daily aggregate of the same telemetry. **Exclude** (or lag by a full day). |

## 8. Valid sequence-count estimate

Sources: `tables/sequence_counts_*.csv`, `tables/sequence_counts_by_port_w12_*.csv`. Horizon = 1 h, chronological 70 / 15 / 15 split with the embargo.

**Lineage policies:**
- **A**: no lineage filter.
- **B**: target observed; inputs may include masked reconstructed hours. **Recommended.**
- **C**: target and every input cell observed.

| Window | A total (train / val / test) | **B total (train / val / test)** | C, PPT-8 features | C, recommended measured features |
|---|---|---|---|---|
| 6 h | 9,820 (6,940 / 1,440 / 1,440) | 9,374 (6,634 / 1,363 / 1,377) | 3,322 | 4,598 |
| **12 h** | **9,640 (6,880 / 1,380 / 1,380)** | **9,203 (6,579 / 1,306 / 1,318)** | 1,112 | 2,324 |
| 24 h | 9,280 (6,760 / 1,260 / 1,260) | 8,858 (6,463 / 1,190 / 1,205) | 89 | 652 |
| 48 h | 8,560 (6,520 / 1,020 / 1,020) | 8,186 (6,242 / 965 / 979) | 0 | 18 |

- **Per port at W = 12:** 688 / 138 / 138 windows under policy A. Under policy B: 653–663 train, 125–135 validation, 128–135 test.
- **Policy C is not viable.** Requiring every input cell to be observed discards 76–89 % of 12-h windows. 15.6 % of hours have at least one reconstructed PPT-8 signal (11.3 % for the recommended measured set), and a 12-h window needs 12 clean hours in a row. This is another reason to drop customs (the most reconstructed signal).
- **The counts overstate the information.** Consecutive windows share 11 of 12 hours, so the effective sample is far smaller: about 700 training hours per port, and only a handful of backlog episodes.

## 9. Recommended DL feature set (initial)

The input tensor is `[batch, 12, F]`, one series per port. The core set has 8 per-hour channels, the same count as the PPT design but a different composition, plus a mask channel and a port embedding:

| # | Feature | Source | Transform |
|---|---|---|---|
| 1 | `container_dwell_hours` (history) | telemetry | Per-port z-score (training stats) |
| 2 | `dwell_observed` mask | `lineage/cell_lineage_port_hourly_telemetry.csv` | 0/1 |
| 3 | `port_utilization_pct` | telemetry | ÷ 100, plus channel 4 |
| 4 | `util_at_cap` | derived: utilization = 100 | 0/1 |
| 5 | `vessels_at_anchor` | telemetry | log1p, then per-port z-score |
| 6 | `wind_gusts_10m_kmh` | weather (same key) | Clip at ≥ 0; global z-score |
| 7 | `labour_strike` | telemetry `labour_status` | 0/1 (one-hot: strike vs not) |
| 8 | `disruption_active` | `disruption_events` | 0/1 (start ≤ t < end) |
| 9 | `disruption_hours_elapsed` | `disruption_events` | log1p(hours since start), 0 if none |
| static | `port_id` | metadata | Embedding (dim 2–4), or per-port output de-normalisation |

**Optional ablations:** `vessel_time_in_port_hours`, `holiday_local`, `equipment_availability_pct`, and the original PPT-8 set as a comparison arm.

**Dropped, with reasons:** throughput (redundant with utilization); customs, temperature and equipment (no measurable signal; customs is also the most reconstructed); wind speed (replaced by gusts); visibility (fabricated at 3 ports); `port_traffic_history` (leaks within the day).

**Target scaling:** predict the per-port z-scored dwell at t+1, or the change y<sub>t+1</sub> − EWMA<sub>t</sub>. Invert to hours for evaluation.

## 10. Recommended preprocessing strategy

1. **Load** `port_hourly_telemetry` ⨝ `weather_hourly` on (`port_id`, `ts_hour`): a one-to-one inner join, verified at 10,000 / 10,000 rows. Parse `ts_hour` as UTC and sort by (port, ts). Join the lineage masks on the same key.
2. **Derive** disruption features from `disruption_events` using only `event_start` ≤ t < `event_end` and the elapsed time. Derive the optional holiday flag from `public_holidays` on the port's *local* date (`port_metadata.local_timezone`).
3. **Split** on global hour cut-offs (70 / 15 / 15) **before** any fitting.
4. **Fit scalers** on training hours per port (mean / SD, or median / IQR, for dwell, anchor and utilization) and globally for gusts.
5. **Build windows** per port: inputs t−11 … t, target t+1.
   - Drop windows that cross a split boundary.
   - Drop windows whose target is reconstructed (policy B).
   - Never concatenate ports or build windows across a time gap.
6. **Report against references** on the same test windows: persistence, 12-h trailing mean, and EWMA (α chosen on train). Report per port and for episode vs non-episode hours, since that is where any gain will be.
7. **Seeds and repeats.** With about 700 training hours per port, run several seeds and report the spread before comparing LSTM, GRU and CNN.

## 11. Final readiness verdict and sufficiency

**Verdict: READY WITH MINOR FIXES.** The temporal structure is sound: a complete, regular, UTC, aligned grid with no duplicates. Sequence construction can proceed once these fixes are in place:

| # | Minor fix | Effort |
|---|---|---|
| 1 | Use the lineage masks produced here (`DL/Dataset/data_audit/lineage/`) to exclude the 459 reconstructed targets and to mask reconstructed inputs. | Join on an existing file |
| 2 | Replace the PPT-8 signals with the recommended set (section 9): drop throughput, customs, temperature and equipment; switch wind speed to gusts; add disruption state and anchor queue. | Feature list change |
| 3 | Clip 2 negative gusts at 0, and add the utilization-at-cap indicator. | Trivial |
| 4 | Make EWMA / trailing-mean baselines mandatory in evaluation (section 4). | Trivial |

**Sufficiency** (`tables/disruption_hours_by_*_and_split.csv`, `tables/audit_facts.json`):

| Use | Assessment |
|---|---|
| Basic LSTM | **Sufficient** for a small model (1 layer, 16–64 units, few parameters): 6,579 training windows, 10 ports, 700 training hours each. Expect results close to EWMA; the remaining error is mostly irreducible hourly noise (≈ 4 h SD). |
| GRU / 1D-CNN comparison | **Feasible**, but differences will likely be within seed-to-seed and bootstrap noise. The test period is only 150 h (6.25 days) per port. Compare with repeated seeds and per-port intervals, and do not rank on small RMSE gaps. |
| Graph-neighbour features | **Not supported by this data.** The route network is a *complete* directed graph (90 / 90 pairs), so it defines no neighbourhood; only the direct-route subgraph (52 edges, 4 hubs) or distance weighting could. More importantly, cross-port dependence is absent: hourly dwell changes correlate at most \|r\| = 0.10 between ports (0.13 at lags 1–24 h), and deviations from the 24-h mean have a median cross-port r of 0.02. The level correlations (NLRTM–USLAX 0.82, CNSHA–NLRTM 0.74) come from coincident episodes in late January, not propagation. A graph experiment would test an effect the generator does not contain. |
| Seasonality / long-term generalisation | **Not supported.** 41.7 days, one winter, one Spring Festival and 55 disruption events. |

**Temporal diversity is thin in the evaluation periods:**
- Congestion and cyber events occur **only in training**.
- Validation and test contain 21 + 127 strike hours, 88 + 16 weather hours and 16 + 27 equipment hours.
- AEJEA, CNSHA, INMAA and SGSIN have **no disruption hours** in validation or test, so episode performance there cannot be measured.
- CNSHA's Spring Festival backlog straddles the train/validation boundary.

**Further limits:**
- All data is synthetic. The dwell process (a per-port base plus backlog plus white noise) is simple enough that a well-tuned EWMA is a strong competitor.
- Nothing here supports real-world generalisation.
- The 4.6 % reconstructed targets and the unknown fill method remain a provenance gap until the cleaning step is documented.

**Hand-off to the DL Data Engineer:** use the input schema in section 9 and the construction plan in sections 7, 8 and 10 (policy B, W = 12, H = 1, 70 / 15 / 15 global cut-offs with an 11-h embargo). Expected yield: **6,579 / 1,306 / 1,318** windows.
