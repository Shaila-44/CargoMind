# 01 · ML Dataset Engineering

How the CargoMind shipment delay-risk dataset is produced, cleaned, integrated into one row per shipment, labelled, filtered for leakage and split for modelling.

**Related documents:** EDA findings → [02_ML_Exploratory_Data_Analysis.md](02_ML_Exploratory_Data_Analysis.md). Models → [03_Machine_Learning_Development.md](03_Machine_Learning_Development.md). Evaluation → [04_ML_Model_Evaluation.md](04_ML_Model_Evaluation.md).

**Sources inspected:**
- `ML/Dataset/Raw/generate_dataset.py`, `ML/Dataset/Raw/README.md`, `ML/Dataset/Raw/_data_issues_log.csv`
- `ML/Dataset/Processed/*.csv`
- `ML/EDA/src/cargomind_eda/{io,features,lineage}.py`
- `ML/ML/src/cargomind_ml/{config,features,preprocessing,train_models}.py`
- `ML/ML/results/data/{data_summary.json,feature_catalog.csv,splits.csv}`
- `ML/Evaluation/validation/split_verification.json`

All paths are relative to the repository root.

> **The data is synthetic.** Every table except `public_holidays` carries `is_synthetic = True`. Nothing in this document describes real ports or carriers.

---

## A. Dataset overview

The dataset simulates **10 ports** over **1,000 hours**, from 2025-01-01 00:00 UTC to 2025-02-11 15:00 UTC. The ports are INMAA, INNSA, SGSIN, CNSHA, AEJEA, NLRTM, USLAX, DEHAM, LKCMB and MYPKG. The fact table holds the shipments; the other seven tables give each shipment port, weather, network, disruption and calendar context.

### Dataset inventory

| Table | Raw rows (`ML/Dataset/Raw/`) | Processed shape (`ML/Dataset/Processed/`) | Grain / key | Role |
|---|---:|---|---|---|
| `shipment_records` | 10,000 (incl. 150 duplicates) | 9,850 × 16 | shipment / `shipment_id` | fact table: target, outcome, shipment attributes |
| `port_hourly_telemetry` | 10,000 (with gaps and duplicates) | 10,000 × 16 | port × UTC hour / `port_id` + `ts_hour` | congestion and operations |
| `weather_hourly` | 10,000 | 10,000 × 15 | port × UTC hour / `port_id` + `ts_hour` | port weather |
| `port_metadata` | 11 (incl. 1 duplicate) | 10 × 12 | port / `port_id` | capacity, region, country/subdivision, timezone |
| `port_network` | 92 | 90 × 9 | directed route / `route_id` | distance, planned transit hours, direct vs transshipment, hub |
| `port_traffic_history` | 438 (daily + some monthly rows) | 420 × 10 | port × UTC day / `port_id` + `period_start` | daily calls, volumes, deviation |
| `disruption_events` | 56 | 55 × 11 | event / `event_id` | strike, weather, congestion, equipment failure, cyber |
| `public_holidays` | 33 | 32 × 9 | country/subdivision × date | calendar |
| `_data_issues_log.csv` | 136 rows | — | table, column, issue, rows_affected | log of every injected data problem |

Column lists of the processed tables:

| Table | Columns |
|---|---|
| `shipment_records` | shipment_id, route_id, origin_port_id, destination_port_id, carrier, shipping_mode, cargo_type, container_teu, shipment_priority, departure_ts, expected_arrival_ts, actual_arrival_ts, actual_delay_hours, delay_status, data_source, is_synthetic |
| `port_hourly_telemetry` | port_id, ts_hour, throughput_teu, port_utilization_pct, container_dwell_hours, vessel_time_in_port_hours, vessel_arrivals, vessel_departures, vessels_at_anchor, customs_hold_hours, labour_status, equipment_availability_pct, three `*_is_synthetic` flags, data_source |
| `weather_hourly` | port_id, ts_hour, grid_latitude, grid_longitude, temperature_2m_c, relative_humidity_2m_pct, precipitation_mm, wind_speed_10m_kmh, wind_gusts_10m_kmh, wind_direction_10m_deg, surface_pressure_hpa, visibility_m, weather_code, weather_condition, data_source |
| `port_metadata` | port_id, port_name, country_code, subdivision_code, region, latitude, longitude, port_type, capacity_teu_per_hour, local_timezone, data_source, is_synthetic |
| `port_network` | route_id, origin_port_id, destination_port_id, route_type, transshipment_port_id, route_distance_km, planned_transit_hours, data_source, is_synthetic |
| `port_traffic_history` | port_id, period_start, period_granularity, vessel_calls, container_vessel_calls, import_volume_t, export_volume_t, traffic_deviation_pct, avg_vessel_stay_hours, data_source |
| `disruption_events` | event_id, port_id, event_start, event_end, event_type, severity, affected_capacity_pct, event_description, source_url, data_source, is_synthetic |
| `public_holidays` | country_code, subdivision_code, holiday_date, holiday_name, holiday_type, holiday_period_days, port_closed, data_source, is_synthetic |

Coverage facts (EDA notebook 01; [02 · §A](02_ML_Exploratory_Data_Analysis.md#a-purpose-and-scope-of-the-eda)):
- The hourly series are complete in the processed data: 1,000 h × 10 ports, with identical (port, hour) keys in telemetry and weather.
- Departures run from 2 Jan to 11 Feb. 27 % of expected arrivals fall after the last operational hour (up to 12 Mar).
- The network is a complete directed graph: 90 routes, 52 direct and 38 via the hubs SGSIN, LKCMB, MYPKG and AEJEA.

---

## B. Dataset generation

`ML/Dataset/Raw/generate_dataset.py` (seed `SEED = 42`, `rng = np.random.default_rng(SEED)`, `H = 1000` hours from `t0 = 2025-01-01 00:00 UTC`) builds a clean dataset and then deliberately corrupts it. It is run as `python generate_dataset.py <output_dir>`, and it also writes a `_clean_reference` folder holding the uncorrupted data. The script has nine generation sections plus a "MESS IT UP" section (line 435 onwards).

The formulas below are copied from the generator. Nothing was inferred from correlations.

### B.1 Weather (section 4, from line 133)

- Temperature: `p[10] + 4*sin(2π(local_hr − 9)/24) + N(0, 1.2)`. This is a per-port base value plus a diurnal cycle.
- Wind: an AR(1) term `ar[h] = 0.92*ar[h−1] + N(0, 1.6)`, then `wspd = clip(base_w + ar + storm, 0.5, None)`.
- Gusts: `gust = wspd * U(1.35, 1.6) + N(0, 2)`.
- Precipitation, pressure, visibility and weather codes are also simulated. Their exact formulas were not transcribed here (lines 133–175).

### B.2 Disruption events (section 5, from line 176)

- About Poisson(3.5) events per port. Each event's type is drawn from `ev_spec` (probability, duration range in hours, affected-capacity range in %):

  | Type | Probability | Duration (h) | Affected capacity (%) |
  |---|---:|---|---|
  | strike | 0.15 | 12–96 | 30–80 |
  | equipment_failure | 0.40 | 3–30 | 10–40 |
  | congestion | 0.35 | 24–120 | 15–45 |
  | cyber | 0.10 | 6–48 | 20–60 |

- Weather events are derived from the gust series. When the gust exceeds 55 km/h for at least 3 h, an event is logged with `affected = min(85, 20 + (peak − 55)*1.8)`. Weather events do not cut capacity directly; the weather capacity factor below does that.
- Severity (`sev()`, line 195): low if affected < 25 %, medium if < 50 %, otherwise high.

### B.3 Port hourly telemetry (section 6, lines 228–266)

Throughput is computed with a capacity-and-backlog model:

```python
demand  = cap * base_u * diurnal * (1 + ar)            # base_u ~ U(0.62, 0.86); ar: AR(1) 0.95, N(0, 0.02)
demand *= 0.7 if spring festival else 0.88 if holiday else 1.0
wx      = (0.45 if gust > 55 else 0.8 if gust > 45 else 1.0) * (0.85 if prec > 8 else 1.0)
hol_cap = 0.55 if spring festival else 0.75 if holiday else 1.0
cap_eff = cap * (1 - ev_cap_cut) * wx * hol_cap
thr = min(demand + B, cap_eff);  B = max(0, demand + B - cap_eff);  util = thr / cap * 100
```

Every other telemetry variable is driven by the backlog in hours of capacity, `bh = B / cap`:

| Variable | Formula |
|---|---|
| `container_dwell_hours` | `base_dwell (U(60,110)) + 1.6*min(bh,150) + N(0,4)` |
| `vessel_time_in_port_hours` | `18 + 0.35*min(bh,150) + Gamma(2,3)` |
| `vessels_at_anchor` | `Poisson(0.4 + bh/2.5)` |
| `vessel_arrivals` / `vessel_departures` | `Poisson(cap/600 * diurnal)`; departures are also scaled by `clip(cap_eff/demand, 0.2, 1)` |
| `customs_hold_hours` | `Gamma(2,8) + 12*is_holiday` |
| `labour_status` | `strike` during strikes; `reduced` on holidays or in a random 3 % of hours; otherwise `normal` |
| `equipment_availability_pct` | `clip(93 + N(0,2.5) − U(18,35) during equipment failures, 40, 100)` |

`port_traffic_history` is a daily aggregate of this telemetry (section 9). EDA notebook 07 confirmed that `vessel_calls` equals the daily sum of `vessel_arrivals` exactly.

### B.4 Network (section 7, line 267)

- The network is a complete directed graph. `route_distance_km = haversine × U(1.15, 1.35)`.
- About 35 % of routes are transshipment routes via a random hub from `HUBS = ["SGSIN", "AEJEA", "LKCMB", "MYPKG"]`.
- `planned_transit_hours = dist / U(30, 38) + (36 if transshipment) + 12` (line 285).

### B.5 Shipments and the delay formula (section 8, lines 295–366)

`NS = 9850` unique shipments. 150 duplicates are added later in the mess step, giving 10,000 raw rows. The raw delay of shipment *k* is a **linear sum** (lines 332–337):

```python
x = (0.55*(util24[oi,h] - 75) + 0.45*(util24[di,h] - 75) + 1.5*anchor[oi,h]
     + 0.9*max(gmax24[oi,h] - 45, 0) + 0.5*max(gust[di,h] - 45, 0)
     + sev_w[ev_sev[oi,h]] + 0.8*sev_w[ev_sev[di,h]] + 14*hol_near + hub_term
     + carrier_eff[carrier] + {"high": -6, "medium": 0, "low": 3}[prio]
     + (5 if reefer) + (8 if hazardous) + N(0, 7 + route_distance_km/1500))
if air: x *= 0.15
```

The terms are defined as follows:
- `util24` and `gmax24` are the trailing 24 h mean utilisation and maximum gust, over hours `[h−24, h)`.
- `sev_w = [0, 8, 20, 40]` indexes disruption severity (none, low, medium, high).
- `hol_near` is True when a holiday falls within ±24 h of departure at the origin.
- `hub_term = 10 + 0.4*(util24_hub − 75)` on transshipment routes, otherwise 0.
- `carrier_eff` gives Oceanic −4, BlueWave 0, Meridian 3, Pacific Star 6, Coral Bay 10, Northgate −1.

Two consequences matter for modelling:
- The delay depends on **destination conditions at the departure hour** (`util24[di,h]`, `gust[di,h]`, `ev_sev[di,h]`). This is observable before departure, but it is a generator artefact (see [04 · §I](04_ML_Model_Evaluation.md#i-limitations-and-final-verdict)).
- The model is additive with Gaussian noise. No interaction terms exist. [03 · §J](03_Machine_Learning_Development.md#j-limitations-of-the-ml-implementation) explains why this favours a linear model.

Calibration to the target mix (lines 346–349):

```python
q55, q85 = np.quantile(raw_delay, [0.55, 0.85])
delay = (raw_delay - q55) * (36 / (q85 - q55)) + 12      # maps q55 -> 12 h and q85 -> 48 h
delay = np.where(delay < 0, delay * 0.3, delay).round(1)  # early arrivals are modest
```

Air transit time is `route_distance_km/750 + 18`; sea transit uses the route's `planned_transit_hours`. `actual_arrival_ts = expected_arrival_ts + delay`.

### B.6 Injected data problems ("MESS IT UP", line 435 onwards)

Every injected problem is logged in `_data_issues_log.csv` (136 rows; columns `table, column, issue, rows_affected`). The problem types (from `ML/Dataset/Raw/README.md`) are:
- duplicates and hourly gaps;
- mixed timestamp formats (ISO, DD/MM/YYYY, US AM/PM, `Z`, `+05:30`, epoch, unmarked IST);
- messy port codes (lowercase, trailing spaces, typos, port names instead of codes);
- category spelling variants;
- numbers stored as text with units (`"84.2h"`, `"12 TEU"`);
- sentinels (−999, −9999, 9999) and placeholders (`N/A`, `NaN`, `-`);
- unit errors (°F, fractions instead of percentages, days instead of hours);
- impossible values;
- swapped columns, arrival before departure, labels that contradict the delay hours;
- inconsistent booleans.

### B.7 Cleaning (raw → processed): **undocumented**

**The code that produced `ML/Dataset/Processed/` from `ML/Dataset/Raw/` is not in the repository.** Its rules can only be inferred by comparing raw and processed values. `cargomind_eda.lineage` (`raw_vs_processed_summary`, `shipment_fill_flags`) does this comparison. It found (EDA §2):

| What was filled during cleaning | Extent |
|---|---|
| Shipment outcome absent in raw (arrival, delay **and** label all missing) | **295 shipments (3.0 %)** |
| Raw label `Late` re-derived from timestamps | 97 shipments |
| Filled shipment inputs (carrier, cargo, priority, TEU, mode, route, expected arrival) | 1–3 % each; 12.5 % of shipments have at least one |
| Telemetry cells that were empty or sentinels in raw, plus 120 missing hours | ≈ 2–5 % of telemetry |
| Visibility entirely absent in raw at USLAX, CNSHA, LKCMB | ≈ 3,000 port-hours |
| Disruption end times missing in raw | 3 events |

How these values were imputed is unknown. The processed files contain no imputation flags.

---

## C. Data integration

`cargomind_eda.features.build_shipment_features(lag_hours=1, tables=None)` joins all eight tables into **one row per shipment** (9,850 rows). It asserts that the row count is unchanged and that `shipment_id` stays unique. The ML stage reuses it without changes, through `cargomind_ml.features.build_modelling_table`.

| Step | Function | Join logic |
|---|---|---|
| Cutoff | `build_shipment_features` | `cutoff_ts = departure_ts − lag_hours` |
| Route | `build_shipment_features` | `m:1` merge on `route_id`: route_type, hub, distance, planned hours |
| Port constants | `build_shipment_features` | capacity and region mapped for origin `o_`, destination `d_` and hub `h_` |
| Calendar | `_local_hour`, `_local_date` | local departure hour (port timezone), UTC weekday, `dep_day_index` |
| Port state | `_port_hourly_features` + `_attach` | snapshot + trailing-24 h rolling aggregates (window ends at and **includes** the cutoff hour), exact `(port, cutoff hour)` left join for each role |
| Weather | `_port_hourly_features` + `_attach` | same, prefix `wx_o_` / `wx_d_` / `wx_h_` |
| Disruptions | `_disruption_asof` | events with `event_start ≤ cutoff < event_end` (active), plus starts within the prior 72 h |
| Holidays | `_holiday_dates`, `_days_to_nearest_holiday` | national holidays apply to every port in the country, subdivision holidays only to matching ports |
| Traffic | merge | last **complete** UTC day before departure (`tr_` prefix) |
| History | `_historical_outcomes` | outcomes of shipments whose `actual_arrival_ts` is **strictly before** the cutoff, by route, carrier, origin and destination |
| Leakage probes | `_attach`, `_disruption_future` | `LEAK_` columns (destination at expected arrival, origin 24 h after departure, future events, remaining event duration). Built only to measure leakage |

Rolling windows in `_port_hourly_features` include:
- utilisation mean/max/24 h change, anchor mean/max, dwell mean, customs mean, equipment minimum, labour-disrupted share and arrivals sum;
- gust max, wind mean, precipitation sum, visibility minimum, thunderstorm hours (`weather_code ≥ 95`) and 24 h pressure change.

The EDA saved the integrated table to `ML/EDA/data/interim/shipments_integrated.pkl`. The ML stage saves its own version, with the changes below, to `ML/ML/results/data/modelling_table.pkl`.

**ML-specific changes** (`cargomind_ml.features.build_modelling_table`):

1. **Row filter.** The 295 unobserved-outcome shipments are dropped, leaving 9,850 → **9,555** rows. Flags come from `cargomind_eda.lineage.shipment_fill_flags()` (`outcome_not_observed_in_raw`).
2. **History recomputed** on observed shipments only, with `HIST_MIN_N = 10`. Groups with fewer than 10 prior arrivals get NaN.
3. **Schedule ETA** (`_schedule_eta`) for the 194 observed shipments whose `expected_arrival_ts` was reconstructed from the actual arrival during cleaning. For them, ETA = departure + `planned_transit_hours` × the mode's median planned/route ratio. It is used only for `hol_h_days_in_voyage_sched` (`_hub_holidays_in_voyage`).
4. **Encodings.** `hub_port_id` is the hub with an explicit `DIRECT` level. `priority_rank` codes low 0, medium 1, high 2. `hol_o_days_to_nearest_cap14` is capped at 14 days. `dis_o_active_hours_elapsed` is 0 when no event is active.

---

## D. Target construction

| Item | Definition (code) |
|---|---|
| Continuous outcome | `actual_delay_hours = actual_arrival_ts − expected_arrival_ts` (EDA notebook 02 confirmed exact equality) |
| Label | `delay_status`: **On-Time** if delay ≤ 12 h; **At-Risk** if 12 < delay ≤ 48 h; **Critical** if > 48 h (`THRESHOLDS = {"on_time_max": 12.0, "at_risk_max": 48.0}` in `cargomind_ml.config` and `cargomind_eda.config`) |
| Encoding | `y = delay_status.cat.codes`: On-Time 0, At-Risk 1, Critical 2, in ordinal order (`CLASSES` in `cargomind_ml.config`) |

Class counts:

| Class | All processed shipments (9,850) | Observed shipments used by ML (9,555) | Train (6,452) | Purge (709, unused) | Test (2,394) |
|---|---:|---:|---:|---:|---:|
| On-Time | 5,422 (55.0 %) | 5,271 | 3,352 (52.0 %) | 438 (61.8 %) | 1,481 (61.9 %) |
| At-Risk | 2,951 (30.0 %) | 2,858 | 1,982 (30.7 %) | 178 (25.1 %) | 698 (29.2 %) |
| Critical | 1,477 (15.0 %) | 1,426 | 1,118 (17.3 %) | 93 (13.1 %) | 215 (9.0 %) |

Sources:
- The 9,850 column comes from the EDA report §3.
- The train, purge and test columns come from `ML/ML/results/data/data_summary.json`.
- The 9,555 column is the sum of train, purge and test.

Air shipments are always On-Time. The processed data has 503 of them; 485 remain after the row filter (323 train / 34 purge / 128 test).

---

## E. Feature engineering

The model uses **51 input features in 12 groups** (`FEATURE_GROUPS` in `ML/ML/src/cargomind_ml/features.py`). Seven are categorical (`CATEGORICAL`) and 44 are numeric (`NUMERIC`). The groups follow the EDA redundancy clusters ([02 · §I](02_ML_Exploratory_Data_Analysis.md#i-feature-assessment-notebook-09-sections-i1i5)), with one column kept per cluster.

### Feature inventory

| Group | Features | Type | Source / construction |
|---|---|---|---|
| route | `origin_port_id`, `destination_port_id`, `hub_port_id`, `route_type`, `route_distance_km`, `planned_transit_hours` | 4 cat. + 2 num. | `shipment_records` + `port_network`; hub re-encoded with `DIRECT` |
| carrier | `carrier` | cat. | `shipment_records` |
| shipment_attributes | `shipping_mode`, `cargo_type`, `priority_rank`, `container_teu` | 2 cat. + 2 num. | `shipment_records`; priority as ordinal rank |
| calendar | `dep_hour_local`, `dep_dow_utc`, `hol_o_days_to_nearest_cap14`, `hol_o_within_2d`, `hol_h_days_in_voyage_sched` | num. | timezones from `port_metadata`, `public_holidays`; hub holidays bounded by the schedule ETA |
| origin_congestion | `o_port_utilization_pct`, `o_util_max24`, `o_util_change24`, `o_vessel_time_in_port_hours`, `o_arrivals_sum24`, `o_labour_disrupted_share24`, `o_customs_mean24`, `o_equip_min24` | num. | origin telemetry at the cutoff hour + trailing 24 h |
| container_dwell | `o_container_dwell_hours` | num. | origin telemetry snapshot |
| anchor_queue | `o_anchor_mean24` | num. | origin telemetry, 24 h mean |
| weather | `wx_o_gust_max24`, `wx_o_precip_sum24`, `wx_o_thunderstorm_hours24`, `wx_o_pressure_change24` | num. | origin weather, trailing 24 h |
| disruptions | `dis_o_active_any`, `dis_o_active_{weather,strike,congestion,equipment_failure,cyber}`, `dis_o_active_hours_elapsed`, `dis_o_started_72h_n` | num. | `_disruption_asof` at the origin |
| destination_state | `d_util_mean24`, `d_anchor_mean24`, `dis_d_active_any` | num. | destination state **at the cutoff** (not at arrival) |
| hub_state | `h_util_mean24`, `h_anchor_mean24`, `h_container_dwell_hours`, `wx_h_gust_max24`, `dis_h_active_any` | num. | hub state at the cutoff; structurally NaN on direct routes |
| historical_delay_rates | `hist_carrier_pct_delayed`, `hist_carrier_mean_delay_h`, `hist_origin_port_id_pct_delayed`, `hist_destination_port_id_pct_delayed`, `hist_route_id_pct_delayed` | num. | `_historical_outcomes` on observed shipments, min 10 prior arrivals |

Feature subsets defined in `features.py`:
- `IDENTITY_FEATURES` (origin, destination, hub, carrier) is used in the "no identity" ablation.
- `ANOMALY_FEATURES` (15 columns) feeds the anomaly detectors: 10 `ANOMALY_PORT_RELATIVE` columns plus 5 `ANOMALY_RAW` columns. See [03 · §E](03_Machine_Learning_Development.md#e-anomaly-detection).

`feature_catalog(table)` writes one row per column of the modelling table to `ML/ML/results/data/feature_catalog.csv`, with status and reason. The file has 246 rows (51 `used`, 195 `excluded`). `ML_model_report.md` states "245 columns"; the difference of one has not been explained.

---

## F. Leakage prevention and prediction cutoff

**Prediction time.** `LAG_HOURS = 1` gives a cutoff of `departure_ts − 1 h` (`cargomind_ml.config`). The project defines no booking or prediction timestamp, so this is the EDA convention. The config comment says it "is safe whether `ts_hour` marks the start or the end of the hour". A 24 h cutoff is evaluated as a sensitivity check ([04 · §G](04_ML_Model_Evaluation.md#g-robustness-and-statistical-analysis)).

Point-in-time rules applied by the integration code:
- Hourly port and weather records are used only if `ts_hour ≤ cutoff`.
- A disruption counts only if it has **started** by the cutoff and is still active at it. Its remaining duration is a `LEAK_` column, because `event_end` is unknown while the event is ongoing.
- History features use only shipments that had **arrived** before the cutoff.
- Traffic features use only the last complete day before the departure day.

Exclusion rules (`_EXCLUSION_RULES` in `cargomind_ml/features.py`, first match wins):

| Excluded | Reason (from code) |
|---|---|
| `delay_status`, `y` | target |
| `actual_delay_hours`, `actual_arrival_ts` | outcome: reproduces the label exactly |
| `LEAK_*` (17 columns) | future information after the cutoff |
| `filled_*`, `raw_label*`, `outcome_not*` | cleaning-lineage flags: row filtering and sensitivity checks only |
| `shipment_id`, `data_source`, `is_synthetic` | identifier / constant |
| `departure_ts`, `dep_day_index` | absolute time: encodes the period |
| `expected_arrival_ts`, `planned_duration_h`, `planned_vs_route_ratio` | derived from an ETA that was reconstructed for 194 rows |
| `route_id`, `od_pair` | 90-level identifier; route = origin + destination + hub (EDA R² 0.91) |
| `*_n_arrived` | count of prior arrivals grows with time: a hidden time index |
| `tr_*` | exact daily aggregate of telemetry |
| `*visibility*` | filled during cleaning at USLAX, CNSHA, LKCMB |
| `dis_*max_sev*`, `dis_*capacity_pct*` | unknown whether severity and capacity are known at event onset |
| `hol_d_*` | holiday calendar ends 26 Feb, so most test-period ETAs are not covered |
| other redundant cluster members | one column per EDA redundancy cluster |

The EDA demonstrated these leakage mechanisms concretely (notebook 09; see [02 · §I](02_ML_Exploratory_Data_Analysis.md#i-feature-assessment-notebook-09-sections-i1i5)):
- Outcome columns reproduce the label 100 %.
- Naive full-data target encoding looks 4–6× stronger than its point-in-time version.
- Under a random split, 99.8 % of test slots share an origin episode (within ±6 h) with a training shipment.

The ML evaluation stage verified that no outcome column reaches inference and that the history features match a from-scratch recomputation ([04 · §B](04_ML_Model_Evaluation.md#b-split-verification)).

---

## G. Data splitting and preprocessing

### Chronological split (`cargomind_ml.preprocessing.chronological_split`)

- The test start `T` is the departure time of the shipment at position `floor(0.75·n)` in time order (`TEST_FRACTION = 0.25`).
- The purge window is `[T − 72 h, T)` (`PURGE_GAP_HOURS = 72`). Shipments in it are dropped from both sides.
- Train is every departure before `T − 72 h`.

| Set | n | Departures (UTC) | Notes |
|---|---:|---|---|
| Train | 6,452 | 2025-01-02 00:00 → 2025-01-29 11:00 | 323 air; 2,038 had **not yet arrived** at the test start |
| Purge | 709 | 2025-01-29 12:00 → 2025-02-01 11:00 | used for neither training nor testing |
| Test | 2,394 | 2025-02-01 12:00 → 2025-02-11 15:00 | 128 air; 25.05 % of observed shipments |

### Time-series CV (`time_series_folds`)

There are `CV_FOLDS = 3` expanding-window folds. Each validation block holds `CV_VAL_FRACTION = 0.15` of the training rows, and each fold has a 72 h gap. Fold 2's block is the **validation set**.

| Fold | Train n (to) | Validation n (from → to) | Validation class mix (OT / AR / C %) |
|---|---|---|---|
| 0 | 2,840 (→ 01-13 23:00) | 965 (01-17 00:00 → 01-21 01:00) | 43.7 / 33.2 / 23.1 |
| 1 | 3,800 (→ 01-18 01:00) | 970 (01-21 02:00 → 01-25 10:00) | 43.3 / 30.7 / 26.0 |
| 2 | 4,814 (→ 01-22 10:00) | 970 (01-25 11:00 → 01-29 11:00) | 38.6 / 35.6 / 25.9 |

The split assignments are stored in `ML/ML/results/data/splits.csv` (`split`, `cv_val_fold`).

### Preprocessing

All transformers sit inside sklearn `Pipeline`s, so each one is fitted only on the rows of the fold or period being fitted. The per-family pipelines (`make_preprocessor`, `make_anomaly_preprocessor`, `PortRelativeScaler`) are documented in [03 · §C](03_Machine_Learning_Development.md#c-preprocessing). Class imbalance is handled with balanced class or sample weights. No resampling is used.

---

## H. Implementation files

| File | Function / class | Responsibility |
|---|---|---|
| `ML/Dataset/Raw/generate_dataset.py` | script (sections 1–9, "MESS IT UP") | synthetic generation, injected errors, `_clean_reference` |
| `ML/Dataset/Raw/_data_issues_log.csv` | — | log of 136 injected problems |
| *(not in repository)* | — | raw → processed cleaning (**undocumented**) |
| `ML/EDA/src/cargomind_eda/config.py` | `PROCESSED`, `RAW`, `CLASSES`, `THRESHOLDS` | paths and target definition |
| `ML/EDA/src/cargomind_eda/io.py` | `load`, `load_all`, `load_raw_strings` | typed loading of processed tables |
| `ML/EDA/src/cargomind_eda/lineage.py` | `raw_vs_processed_summary`, `shipment_fill_flags` | raw-vs-processed comparison, fill flags |
| `ML/EDA/src/cargomind_eda/features.py` | `build_shipment_features`, `_port_hourly_features`, `_attach`, `_disruption_asof`, `_disruption_future`, `_historical_outcomes`, `_holiday_dates` | leakage-aware integration |
| `ML/ML/src/cargomind_ml/config.py` | `LAG_HOURS`, `TEST_FRACTION`, `PURGE_GAP_HOURS`, `CV_FOLDS`, `CV_VAL_FRACTION`, `SEED` | modelling parameters |
| `ML/ML/src/cargomind_ml/features.py` | `build_modelling_table`, `_schedule_eta`, `_hub_holidays_in_voyage`, `feature_catalog`, `FEATURE_GROUPS` | ML modelling table and feature selection |
| `ML/ML/src/cargomind_ml/preprocessing.py` | `chronological_split`, `time_series_folds` | split and CV folds |
| `ML/ML/src/cargomind_ml/train_models.py` | `prepare_data` | writes `splits.csv`, `modelling_table.pkl`, `feature_catalog.csv`, `data_summary.json` |

---

## I. Findings and limitations

1. **The data is synthetic.** The delay is an additive formula with noise (§B.5), so the models can recover it closely. This is not evidence of real-world performance.
2. **The cleaning code is missing.** The 295 filled outcomes were found and dropped. Other filled inputs (12.5 % of shipments) and visibility at three ports remain, and their provenance is unknown.
3. **The prediction timestamp is a convention** (`departure_ts − 1 h`), not a product decision. Signal decays quickly with earlier cutoffs ([02 · §H](02_ML_Exploratory_Data_Analysis.md#h-integrated-shipment-level-analysis-notebook-08-sections-h1h8)).
4. **The window is short.** It covers about 6 weeks and one season, including Spring Festival. The test period has 11 departure days and 215 Critical shipments.
5. **Labels arrive late.** 2,038 training labels (32 %) were not yet known at the test start, and 42–62 % of each fold's training labels were not known at that fold's validation start (`split_verification.json`).
6. **Prevalence shifts.** Critical makes up 17.3 % of train, 23–26 % of the validation blocks and 9.0 % of test.
7. **History features are right-censored** and drift strongly between train and test (PSI 2.5–4.0; [04 · §G](04_ML_Model_Evaluation.md#g-robustness-and-statistical-analysis)).
8. **The holiday calendar ends 26 Feb,** but arrivals run to 12 Mar. This is why destination-holiday features are excluded.
9. **Semantics are unknown.** It is not known whether `ts_hour` marks the start or end of the hour, whether `departure_ts` is planned or actual, or when disruption severity becomes known.
