# Supply Chain Logistics Dataset (raw, uncleaned)

Synthetic data following the "Supply Chain Logistics Dataset Schema" doc. **Every value is simulated.**
Nothing here describes the real performance of any port or carrier. Carrier names are fictional.
Holiday dates were entered from 2025 public calendars and should be verified before real use.

## Files

| File | Rows | Grain |
|---|---|---|
| shipment_records.csv | 10,000 | one row per shipment (incl. 150 duplicates) |
| port_hourly_telemetry.csv | 10,000 | one row per port per hour |
| weather_hourly.csv | 10,000 | one row per port per hour |
| port_traffic_history.csv | 438 | daily rows + a few monthly rows mixed in |
| disruption_events.csv | 56 | one row per event |
| public_holidays.csv | 33 | one row per country/state per holiday date |
| port_network.csv | 92 | one row per route |
| port_metadata.csv | 11 | one row per port (incl. one duplicate) |
| _data_issues_log.csv | 136 | every injected data problem: table, column, issue, rows affected |

Coverage: 10 ports (INMAA, INNSA, SGSIN, CNSHA, AEJEA, NLRTM, USLAX, DEHAM, LKCMB, MYPKG),
1,000 hours from 2025-01-01 00:00 UTC to 2025-02-11 15:00 UTC.

## What drives delays (so the cleaned data has real signal)

`actual_delay_hours` is built from: origin and destination utilization (24 h before departure),
vessels at anchor at origin, wind gusts at origin and destination, active disruption events
(by severity), holidays near departure, transshipment hub congestion, carrier, priority, cargo type
and random noise. Labels: On-Time <= 12 h, At-Risk 12-48 h, Critical > 48 h
(about 55% / 30% / 15% before mess is added).

Inside telemetry, capacity drops during strikes, equipment failures, cyber outages, high gusts and
holidays (Chinese Spring Festival hits hardest), which builds a backlog that raises dwell time,
vessel stay and the anchor queue.

## Kinds of mess to clean (full list in _data_issues_log.csv)

- Exact duplicate rows; gaps in the hourly series
- Mixed timestamp formats: ISO, DD/MM/YYYY, MM-DD-YYYY AM/PM, ISO with Z, +05:30 offset, Unix epoch,
  and some IST local times with no marker (5.5 h off)
- Port codes in lowercase, with trailing spaces, misspelled, or written as port names
- Category spelling variants (carrier, mode, cargo, priority, labour status, severity, labels)
- Numbers stored as text with units ("84.2h", "12 TEU", "93%", "36.5 hrs", "4520 mi", "5,600")
- Sentinels (-999, -9999, 9999) and text placeholders (N/A, NaN, -)
- Unit mistakes: Fahrenheit in a Celsius column, fractions instead of percents, days instead of hours
- Impossible values: utilization > 100%, negative throughput/rainfall/distance, humidity > 100%
- Logic errors: swapped columns, arrival before departure, end before start, labels that contradict delay hours
- Booleans written as TRUE / True / 1 / yes

## Leakage reminder

`actual_arrival_ts`, `actual_delay_hours` and `delay_status` are outcome columns. Use them only to build
the label, never as model features.

## Reproduce

`python generate_dataset.py <output_dir>` (seed 42). It also writes a `_clean_reference` folder with
the same data before the mess was added, useful for checking your cleaning.
