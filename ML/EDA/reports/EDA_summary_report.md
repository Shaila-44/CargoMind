# CargoMind — Shipment Delay-Risk EDA: Summary Report

**Scope:** exploratory data analysis and feature assessment of `ML/Dataset/Processed/` for a 3-class delay-risk model
(`delay_status`: On-Time ≤ 12 h < At-Risk ≤ 48 h < Critical). No models were trained.
**Evidence:** everything lives in `ML/EDA/` — notebooks `ML/EDA/notebooks/01–09`, figures in `ML/EDA/reports/figures/`, tables in
`ML/EDA/reports/tables/`, integrated data in `ML/EDA/data/interim/`. Paths below are relative to `ML/EDA/`.
**Reproduce (from the CargoMind root):** `.venv/bin/pip install -r ML/EDA/requirements.txt` then
`.venv/bin/python ML/EDA/scripts/build_notebooks.py` (≈ 1 min; rebuilds every notebook, figure, table and `ML/EDA/data/interim/`
file; the source CSVs in `Dataset/` are only read).

Unless stated otherwise, effect sizes are on **sea shipments** (n = 9,347), because the 503 air shipments are always On-Time.
δ = Cliff's delta (Critical vs On-Time unless stated; |δ| ≥ 0.33 medium, ≥ 0.47 large); V = bias-corrected Cramér's V.
Every table except `public_holidays` is flagged `is_synthetic = True` by the data itself. The findings describe this dataset
and make no claims about real ports or carriers. All associations are observational, not causal.

---

## 1. Dataset overview

All 8 expected tables are present; no extra tables.

| Table | Rows | Grain | Key | Role |
|---|---:|---|---|---|
| shipment_records | 9,850 | shipment | `shipment_id` | fact table: target, outcome, shipment attributes |
| port_hourly_telemetry | 10,000 | port × UTC hour | `port_id`+`ts_hour` | port congestion & operations |
| weather_hourly | 10,000 | port × UTC hour | `port_id`+`ts_hour` | port weather |
| port_metadata | 10 | port | `port_id` | capacity, region, country/subdivision, timezone |
| port_network | 90 | directed route | `route_id` | distance, planned hours, direct/transshipment, hub |
| port_traffic_history | 420 | port × UTC day | `port_id`+`period_start` | daily calls, volumes, deviation |
| disruption_events | 55 | event interval | `event_id` | strikes, weather, congestion, equipment, cyber |
| public_holidays | 32 | country/subdivision × date | 3-part key | calendar |

* All keys are unique and all foreign keys resolve. Each shipment's origin/destination matches its route, and hub is blank ⇔ route is direct.
* Timestamps are 100 % ISO-8601 UTC. Hourly series are complete (1,000 h × 10 ports, 1 Jan 00:00 – 11 Feb 15:00 2025), and
  telemetry and weather share identical (port, hour) keys.
* Departures run 2 Jan – 11 Feb. **27 % of expected arrivals fall after the last operational hour** (up to 12 Mar).
* The network is a complete directed graph (every ordered pair of the 10 ports has exactly one route; 52 direct, 38 via hubs
  SGSIN / LKCMB / MYPKG / AEJEA).

## 2. Data-quality findings

**Valid but unusual (keep as is):**
* `port_utilization_pct` = 100 in 1,628 port-hours (NLRTM 41 % of hours). Utilization is exactly `throughput_teu / capacity × 100`
  (max error 0.03 pp), so 100 % is **saturated, censored throughput**, not an error.
* Heavy tails in anchor queue (max 33), dwell (205 h), gusts (106 km/h) and delay (201 h) cluster at specific ports and dates,
  in line with disruption events. They are **valid extreme operating states**.

**Minor data-quality problems (flagged, not altered):** wind gust < wind speed in 77 h, gust < 0 in 2 h; 7 disruption events
at exactly 85.0 % affected capacity (consistent with a cap).

**Lineage — values that were filled during cleaning (the most important data-quality finding).** The processed files have no
missing measurements and no imputation flags. Comparison with `ML/Dataset/Raw/` shows:

| What | Extent | Implication |
|---|---|---|
| **Shipment outcomes never observed in raw** (arrival, delay and label all empty) | **295 shipments (3.0 %)** | Not observed labels. Provenance unknown. **Exclude from training/evaluation.** Their distribution matches observed rows (KS p = 0.58), which does not make them observations. |
| Raw label `Late` re-derived from timestamps | 97 shipments (58 → On-Time) | Consistent with thresholds; raw label channel was unreliable |
| Filled shipment inputs (carrier, cargo, priority, TEU, mode, route, expected arrival) | 1–3 % each; **12.5 % of shipments have ≥ 1** | Flags saved in `data/interim/shipment_lineage_flags.csv` for sensitivity checks |
| Telemetry cells empty/sentinel in raw (customs 500, dwell 342, anchor 200, labour 200, throughput 200, arrivals 99) + 120 missing hours | ≈ 2–5 % of telemetry | Reconstructed values cannot be located row-by-row without a raw↔processed join |
| **Visibility entirely absent in raw for USLAX, CNSHA, LKCMB** | ≈ 3,000 port-hours | Visibility at those ports is not a measurement |
| Disruption end times missing in raw | 3 events | End times reconstructed |

## 3. Target distribution

| Class | n | % | % (sea only) |
|---|---:|---:|---:|
| On-Time | 5,422 | 55.0 | 52.6 |
| At-Risk | 2,951 | 30.0 | 31.6 |
| Critical | 1,477 | 15.0 | 15.8 |

* The label is an exact function of `actual_delay_hours` (0 disagreements), which equals `actual_arrival_ts − expected_arrival_ts` exactly.
* Delay is right-skewed (median 8.2 h, mean 18.3 h, p99 116 h, max 201 h). The histogram shows a sharp peak just below 0 h and a long tail.
* **Early arrivals:** 39 % of shipments (36 % of sea, 98 % of air), typically 2–7 h early.
* **Air is degenerate:** all 503 air shipments are On-Time (max delay 3.4 h).
* **Threshold proximity:** 5.2 % of shipments lie within ±2 h of the 12 h boundary, 2.1 % within ±2 h of 48 h. Labels there are
  sensitive to small timing errors.
* **Temporal drift:** weekly Critical share ranges from 8 % (week of 3 Feb) to 25 % (week of 20 Jan). Daily % delayed ranges from 26 % to 76 %.

## 4. Important patterns and relationships

### 4.1 What distinguishes the classes

**(a) Time-varying origin conditions before departure (strongest).**

| Pre-departure condition at origin | On-Time | At-Risk | Critical | δ Crit vs OT | δ At-Risk vs OT |
|---|---:|---:|---:|---:|---:|
| Active disruption at departure (share) | 5.6 % | 16.9 % | 44.3 % | 0.39 | 0.11 |
| Anchor queue, 24 h mean (median vessels) | 0.42 | 0.67 | 1.92 | **0.54** | 0.31 |
| Container dwell (median h) | 77.6 | 87.7 | 100.7 | 0.44 | 0.19 |
| Max wind gust, prior 24 h (median km/h) | 34.2 | 36.1 | 48.7 | 0.44 | 0.14 |
| Precipitation, prior 24 h (median mm) | 2.2 | 2.9 | 7.7 | 0.44 | 0.14 |
| Vessel time in port (median h) | 23.6 | 25.2 | 27.0 | 0.33 | 0.17 |

* **Active origin disruption by type** (% delayed / % Critical): weather 98 / **86**, strike 85 / 40, congestion 77 / 33,
  equipment failure 48 / 11, none 41 / 10. By severity: high 96 / 73, medium 71 / 24, low 29 / 11 (V = 0.33).
  Among shipments leaving during an event, the event's affected capacity correlates with delay at ρ = 0.60.
* **All 28 extreme sea delays (> 140 h) departed during an active origin disruption.**
* **87 % of Critical** and **61 % of At-Risk** shipments had ≥ 1 visible adverse origin condition (active event, anchor queue
  > 2, gust > 50 km/h, holiday ±2 d), vs **32 % of On-Time**.

**(b) Route structure.** Transshipment routes: 58.6 % delayed / 22.3 % Critical vs direct 40.7 % / 12.0 % (V = 0.18), at
every distance quintile. Origin + destination + hub explain **R² = 0.91** of route-level delay rates. The residual spread
equals binomial noise, so route ID adds nothing beyond its parts. Distance and planned hours are weak (δ ≈ 0.15–0.18).

**(c) Shipment attributes** (independent of route; Cramér's V with route ≈ 0):
* **carrier** V = 0.12 (37 % delayed for Oceanic Lines vs 63 % for Coral Bay Logistics; the ranking holds within direct and
  transshipment routes);
* **priority** V = 0.08 (high 37 % vs low 54 %);
* **cargo** V = 0.07 (hazardous 59 %, reefer 56 %, general 45 %).

**(d) Calendar.** Departing within 2 days of an origin holiday: 60 % vs 44 % delayed (V = 0.14). Excluding CNSHA origins: 56 % vs 45 % (V = 0.09).
A hub holiday during the voyage is also associated with higher risk (V = 0.13). Spring Festival coincides with a large CNSHA
backlog (+4.3 vessels at anchor, +19 h dwell) and no logged disruption.

**(e) No meaningful association (useful negatives):** container TEU (ε² ≈ 0), departure hour (V = 0.01), weekday (V = 0.05),
customs hold hours (ρ = 0.01), equipment availability (ε² = 0.005), route frequency, traffic deviation (ρ = −0.04),
destination-port state at departure (|δ| ≤ 0.11).

### 4.2 Operational mechanisms (observational)
* Inside disruption intervals, within each port: utilization −24 pp (strikes −38 pp), anchor queue up (δ 0.38), dwell up (δ 0.40).
  Equipment availability barely moves (δ −0.15). Customs hold does not move.
* **Utilization falls when capacity is lost**, so the current-hour value is *lower* for Critical shipments (66 % vs 75 %) while
  its 24 h maximum is higher. Utilization is non-monotonic in risk; read it with the queue length.
* Logged "high winds" events coincide with a median gust of 71 km/h (vs 24). Within port, gusts relate to lower utilization
  (ρ −0.13) and longer queues (+0.10). The weather signal survives within-port centring (δ 0.45), so it is not a climate/port proxy.

### 4.3 Ports, carriers and routes
* **Origins:** NLRTM 68 % delayed, INNSA 58 %, LKCMB 53 %, CNSHA 51 % … MYPKG 31 %. Origin effects are tied to *episodes*
  (e.g. the NLRTM strike of 18–22 Jan, the CNSHA backlog around 28 Jan – 4 Feb), not stable port reputations.
  Point-in-time origin history is weak (δ 0.07).
* **Destinations:** weaker (V = 0.08; INMAA 59 % … MYPKG 39 %). Port risk differs by role. As hubs, SGSIN/LKCMB/MYPKG/AEJEA
  carry 55–61 % delayed.
* **Carriers:** stable over time. Point-in-time carrier history is as informative as the full-data rate (δ 0.19 both).
* **Routes:** 24 routes have CIs entirely above the sea average, and 23 entirely below. The riskiest are LKCMB→INNSA (83 %),
  NLRTM→INNSA (83 %) and NLRTM→SGSIN (78 %).

## 5. Features potentially useful for prediction (pre-departure cutoff)

| Group | Recommended columns (in `data/interim/shipments_integrated.*`) |
|---|---|
| Origin disruption state | `dis_o_active_any` (or `_max_sev`), active event type flags, `dis_o_active_hours_elapsed` |
| Origin congestion | `o_anchor_mean24`, `o_container_dwell_hours`, `o_vessel_time_in_port_hours`; `o_util_max24` with `o_port_utilization_pct` |
| Origin weather (prior 24 h) | `wx_o_gust_max24`, `wx_o_precip_sum24`, `wx_o_thunderstorm_hours24`, pressure anomaly |
| Route structure | `route_type`, hub, origin, destination (encoded per role) |
| Hub state (transshipment only) | `h_util_mean24`, `h_anchor_mean24`, `dis_h_active_any` (structural NA for direct routes) |
| Shipment attributes | `carrier`, `shipment_priority`, `cargo_type`, `shipping_mode` |
| Calendar | `hol_o_days_to_nearest`, `hol_h_days_in_voyage`, `hol_d_within_2d_of_exp_arr` |
| History (point-in-time) | `hist_carrier_*` (others optional) |

Full per-column catalogue with category, rationale and effect sizes: `reports/tables/09_feature_catalog.csv`.

## 6. Features to exclude because of leakage

| Column(s) | Why |
|---|---|
| `actual_arrival_ts`, `actual_delay_hours`, `delay_status` | outcome. Each reproduces the label with 100 % accuracy |
| `LEAK_*` (17 columns) | destination state at expected arrival, origin state in the 24 h after departure, events starting after departure, remaining duration of ongoing events |
| `event_end` / event duration for events ongoing at the cutoff | the end is unknown until the event ends |
| Naive full-data target encodings (route/origin/destination delay rates) | look 4–6× stronger than the deployable point-in-time version (route δ 0.43 vs 0.08) |
| `filled_actual_*`, `outcome_not_observed_in_raw`, `raw_label_was_Late` | lineage of outcome columns. Use only to filter rows |
| `dep_day_index` / raw departure date | encodes the period, will not generalise |
| `shipment_id` | identifier |

**Needs a decision before use:** `dis_*_max_sev` / `dis_*_capacity_pct` (timing is fine, but it is unknown whether severity
and capacity impact are known at onset), visibility features (filled at 3 ports), filled-input lineage flags, and
`expected_arrival_ts` for the 196 rows where it was reconstructed from the actual arrival.

Measured leakage contrast: future information is *weaker* than expected in this dataset. Destination state at arrival
δ ≈ 0, events starting en route δ 0.05–0.10. Risk is largely determined by origin conditions at departure.
These columns still must not be used.

## 7. Redundant or highly correlated features

22 pairs with |ρ| ≥ 0.8 among the 48 informative features (`reports/tables/08_redundant_pairs.csv`). Keep one per cluster:
* `port_utilization_pct` ≡ `throughput_teu` / capacity (deterministic).
* `port_traffic_history` **is an aggregate of telemetry**: `vessel_calls` = daily sum of `vessel_arrivals` (exact, 410/410 days);
  `avg_vessel_stay_hours` = daily mean `vessel_time_in_port_hours` (≤ 0.05 h). Exclude `tr_*`.
* `planned_duration_h` ≡ `planned_transit_hours` for sea (ρ = 1).
* Disruption `active_any` ≈ `active_n` ≈ `active_max_sev` ≈ `started_72h_n` (ρ 0.97–0.999).
* Wind gusts ≈ wind speed (0.97); gust max 24 h ≈ wind mean 24 h (0.93).
* Dwell snapshot ≈ dwell 24 h mean (0.95); anchor 24 h mean ≈ anchor 24 h max (0.88).
* Active-event capacity loss ≈ −current utilization (−0.85).
* `route_id` ≡ `od_pair`; route ≈ origin + destination + hub (R² 0.91).

## 8. Class imbalance assessment

* Moderate imbalance (55 / 30 / 15; majority:minority 3.7:1). ~1,480 Critical examples are enough without synthetic oversampling.
* The classes are **ordinal** and come from one continuous variable. Errors are asymmetric (Critical → On-Time is the costliest).
* **At-Risk is the hardest class.** Almost every feature separates it from On-Time far less than Critical (e.g. gust δ 0.14 vs 0.44).
* Recommended handling: class-weighted loss estimated on the training window; macro-F1, per-class recall (Critical first), and
  an ordinal/cost-aware metric; consider regression/quantile modelling of `actual_delay_hours` + thresholds as a comparison.
  Handle air separately or report sea-only metrics.

## 9. Important limitations

1. **Prediction timestamp is undefined.** The EDA uses `departure_ts − 1 h` as a convention. Signal decays fast with earlier
   cutoffs: anchor δ 0.54 → 0.32 at 24 h; gust δ 0.44 → 0.16 at 24 h and ≈ 0 at 72 h. **This is an open product decision.**
2. **Short window:** ~6 weeks, 10 ports, one season (incl. Spring Festival), ~3 independent fortnights. Associations are stable
   in sign across fortnights, but generalisation to other periods is untested.
3. **Filled values** (Section 2), especially the 295 unobserved outcomes and the filled visibility.
4. **Unknown semantics:** whether `ts_hour` marks the start or end of the hour; whether `departure_ts` is planned or actual;
   when disruption severity/capacity become known.
5. **Holiday calendar ends 26 Feb** while arrivals run to 12 Mar (267 sea shipments); `port_closed` carries no information.
6. **Shared-episode dependence:** 99.8 % of random-split test slots have a training shipment from the same origin within ±6 h.
   Rows are not independent.
7. **Label horizon:** at a time cut, 38–49 % of earlier departures have not arrived yet (and these are disproportionately delayed).
8. All findings are associations. Weather, congestion and disruptions are intertwined; causal effects were not estimated.

## 10. Recommendations for the next stage

1. **Decide the prediction timestamp** (booking vs pre-departure vs en route). Then fix `lag_hours` in `features.build_shipment_features`.
2. **Exclude the 295 unobserved-outcome shipments**. Run a sensitivity check with the filled-input shipments removed.
3. **Time-based validation** with a purge gap ≥ the longest voyage horizon used for labels (or rolling-origin folds). Never use random splits.
4. Start from the recommended feature set (Section 5), one feature per redundancy cluster, with route encoded via
   origin/destination/hub. Add the engineered features to test (notebook 09, I4): backlog trends, recovery-phase timers,
   saturation streaks, multi-window gusts, port-relative z-scores.
5. Treat air shipments with a rule or report sea-only metrics. Treat the task as ordinal and cost-sensitive.
6. Baseline to beat: a simple rule on *origin event active + anchor queue + route type*. In the data this already spans
   26 % → 92 % delayed and 4.5 % → 62 % Critical (notebook 08, H4).
7. Clarify with data owners: `ts_hour` semantics, planned vs actual departure, when disruption severity is known, and how filled values were produced.

**Readiness verdict: ready for predictive modelling, with conditions.** The data is relationally clean and has strong,
legitimately available pre-departure signal. Model development should start only after decisions 1–3 above.
