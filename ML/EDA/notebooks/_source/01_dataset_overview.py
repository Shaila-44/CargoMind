# %% [markdown]
# # 01 · Dataset overview and data quality
#
# **Question this notebook answers:** what is in `Dataset/Processed/`, how do the tables relate,
# and can each table be trusted for a shipment delay-risk model?
#
# Rules followed: source files are read-only; no rows are dropped; no values are imputed or
# altered. The only transformations are type conversions (timestamp strings → UTC datetimes,
# ordered categoricals), applied in `eda/src/cargomind_eda/io.py`.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from cargomind_eda import io, lineage, plotting as P
from cargomind_eda.config import PROCESSED, RAW, EXPECTED_TABLES

P.setup()

# %% [markdown]
# ## A1. Directory inventory
# Verify which of the eight expected tables actually exist before analysing anything.

# %%
found, missing = io.available_tables()
inv = pd.DataFrame(
    {
        "file": [p.name for p in found.values()],
        "size_kb": [round(p.stat().st_size / 1024, 1) for p in found.values()],
        "expected": [n in EXPECTED_TABLES for n in found],
    },
    index=list(found),
)
print("Missing expected tables:", missing or "none")
print("Unexpected extra tables:", [n for n in found if n not in EXPECTED_TABLES] or "none")
inv

# %%
T = io.load_all()
S = {name: io.load_raw_strings(name) for name in T}  # string view, for format checks

# %% [markdown]
# ## A2. Shape, dtypes, missingness and duplicates per table

# %%
def profile(name, df):
    return {
        "rows": len(df),
        "cols": df.shape[1],
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "cells_missing": int(df.isna().sum().sum()),
        "cols_with_missing": int((df.isna().sum() > 0).sum()),
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
    }

pd.DataFrame({n: profile(n, d) for n, d in T.items()}).T

# %%
miss = []
for n, d in T.items():
    for c in d.columns:
        k = d[c].isna().sum()
        if k:
            miss.append({"table": n, "column": c, "missing": k, "pct": round(100 * k / len(d), 1)})
pd.DataFrame(miss)

# %% [markdown]
# Only four columns have missing values in the processed data and each one is **structurally**
# missing rather than a gap in measurement:
#
# * `port_network.transshipment_port_id` — blank for direct routes (verified below).
# * `public_holidays.subdivision_code` — blank means a national holiday.
# * `public_holidays.port_closed` — only populated for CN Spring Festival (`False`); unknown elsewhere.
# * `port_metadata.subdivision_code` — Singapore is a city-state with no subdivision.
# * `disruption_events.source_url` — entirely empty (no provenance link for any event).
#
# None of these should be imputed. Section A7 shows, however, that "no missing values" in the
# processed telemetry, weather and shipment tables is the result of filling during cleaning, not of
# complete measurement.

# %%
for n, d in T.items():
    print(f"\n── {n}  {d.shape}")
    print(d.dtypes.astype(str).to_frame("dtype").T.to_string())

# %% [markdown]
# ## A3. Keys, granularity and referential integrity

# %%
keys = {
    "shipment_records": ["shipment_id"],
    "port_hourly_telemetry": ["port_id", "ts_hour"],
    "weather_hourly": ["port_id", "ts_hour"],
    "port_metadata": ["port_id"],
    "port_network": ["route_id"],
    "port_traffic_history": ["port_id", "period_start"],
    "disruption_events": ["event_id"],
    "public_holidays": ["country_code", "subdivision_code", "holiday_date"],
}
grain = {
    "shipment_records": "one row per shipment",
    "port_hourly_telemetry": "one row per port per UTC hour",
    "weather_hourly": "one row per port per UTC hour",
    "port_metadata": "one row per port",
    "port_network": "one row per directed origin→destination route",
    "port_traffic_history": "one row per port per UTC day",
    "disruption_events": "one row per event (interval at one port)",
    "public_holidays": "one row per country/subdivision per holiday date",
}
rows = []
for n, k in keys.items():
    d = T[n]
    rows.append({"table": n, "candidate_key": " + ".join(k), "key_unique": not d.duplicated(k).any(), "grain": grain[n]})
pd.DataFrame(rows).set_index("table")

# %%
ports = set(T["port_metadata"]["port_id"])
ship, net = T["shipment_records"], T["port_network"]
fk = {
    "shipment.origin_port_id ⊂ metadata": set(ship.origin_port_id) <= ports,
    "shipment.destination_port_id ⊂ metadata": set(ship.destination_port_id) <= ports,
    "shipment.route_id ⊂ network": set(ship.route_id) <= set(net.route_id),
    "shipment (origin,dest) matches its route": bool(
        ship.merge(net, on="route_id", suffixes=("", "_n")).eval("origin_port_id == origin_port_id_n and destination_port_id == destination_port_id_n").all()
    ),
    "network.transshipment_port_id ⊂ metadata": set(net.transshipment_port_id.dropna()) <= ports,
    "network: hub blank ⇔ route_type == direct": bool(((net.route_type == "direct") == net.transshipment_port_id.isna()).all()),
    "network: hub ≠ origin and ≠ destination": bool(((net.transshipment_port_id != net.origin_port_id) & (net.transshipment_port_id != net.destination_port_id) | net.transshipment_port_id.isna()).all()),
    "telemetry ports ⊂ metadata": set(T["port_hourly_telemetry"].port_id) <= ports,
    "weather ports ⊂ metadata": set(T["weather_hourly"].port_id) <= ports,
    "traffic ports ⊂ metadata": set(T["port_traffic_history"].port_id) <= ports,
    "disruption ports ⊂ metadata": set(T["disruption_events"].port_id) <= ports,
    "holiday countries ⊂ metadata countries": set(T["public_holidays"].country_code) <= set(T["port_metadata"].country_code),
    "holiday subdivisions ⊂ metadata subdivisions": set(T["public_holidays"].subdivision_code.dropna()) <= set(T["port_metadata"].subdivision_code.dropna()),
}
pd.Series(fk, name="holds").to_frame()

# %%
# Network completeness: is every ordered port pair a route?
print("routes:", len(net), "| ordered port pairs:", len(ports) * (len(ports) - 1))
print("routes used by ≥1 shipment:", ship.route_id.nunique())
print(net.route_type.value_counts().to_dict(), "| hubs:", net.transshipment_port_id.value_counts().to_dict())

# %% [markdown]
# ## A4. Timestamp formats and time coverage
# Every timestamp column is checked as a raw string for one consistent format before parsing.

# %%
fmt_rows = []
iso_ts = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\+00:00$")
iso_d = re.compile(r"^\d{4}-\d{2}-\d{2}$")
for n, cols in {**io.TS_COLUMNS, **io.DATE_COLUMNS}.items():
    for c in cols:
        s = S[n][c]
        pat = iso_d if n in io.DATE_COLUMNS else iso_ts
        fmt_rows.append({"table": n, "column": c, "pct_matching_ISO_UTC": round(100 * s.str.match(pat).mean(), 2),
                         "min": T[n][c].min(), "max": T[n][c].max()})
pd.DataFrame(fmt_rows)

# %%
tel, wx = T["port_hourly_telemetry"], T["weather_hourly"]
for name, d in [("telemetry", tel), ("weather", wx)]:
    g = d.groupby("port_id")["ts_hour"]
    span = g.agg(["min", "max", "nunique"])
    span["expected_hours"] = ((span["max"] - span["min"]).dt.total_seconds() / 3600 + 1).astype(int)
    print(f"{name}: hourly grid complete for all ports →", bool((span["nunique"] == span["expected_hours"]).all()),
          "| hours per port:", span["nunique"].unique())
print("telemetry and weather share identical (port, hour) keys:",
      tel.set_index(["port_id", "ts_hour"]).index.equals(wx.set_index(["port_id", "ts_hour"]).index))
print("shipment departures minute≠0:", (ship.departure_ts.dt.minute != 0).sum(),
      "| actual arrivals minute≠0:", (ship.actual_arrival_ts.dt.minute != 0).sum())

# %%
fig, ax = plt.subplots(figsize=(11, 3.6))
spans = [
    ("telemetry / weather (hourly)", tel.ts_hour.min(), tel.ts_hour.max()),
    ("traffic history (daily)", T["port_traffic_history"].period_start.min().tz_localize("UTC"), T["port_traffic_history"].period_start.max().tz_localize("UTC") + pd.Timedelta("1D")),
    ("disruption events", T["disruption_events"].event_start.min(), T["disruption_events"].event_end.max()),
    ("public holidays (dates)", T["public_holidays"].holiday_date.min().tz_localize("UTC"), T["public_holidays"].holiday_date.max().tz_localize("UTC")),
    ("shipment departures", ship.departure_ts.min(), ship.departure_ts.max()),
    ("shipment expected arrivals", ship.expected_arrival_ts.min(), ship.expected_arrival_ts.max()),
    ("shipment actual arrivals", ship.actual_arrival_ts.min(), ship.actual_arrival_ts.max()),
]
for i, (lab, a, b) in enumerate(spans):
    ax.plot([a, b], [i, i], lw=8, solid_capstyle="round", color="#2a78d6" if "shipment" not in lab else "#52514e")
ax.axvline(tel.ts_hour.max(), color="#e34948", ls="--", lw=1)
ax.text(tel.ts_hour.max(), len(spans) - 0.4, " end of operational data", color="#e34948", fontsize=8)
ax.set_yticks(range(len(spans)), [s[0] for s in spans])
ax.invert_yaxis()
ax.set_xlabel("Date (UTC, 2025)")
ax.set_title("Time coverage: operational tables end on 11 Feb 2025; 27% of planned arrivals fall after that")
P.save(fig, "01_time_coverage")
print("share of shipments whose expected arrival is after the last telemetry hour:",
      round((ship.expected_arrival_ts > tel.ts_hour.max()).mean() * 100, 1), "%")

# %% [markdown]
# ## A5. Numerical and categorical distributions (summary)
# Per-table descriptive statistics. Detailed, target-focused distributions are in notebooks 02–07.

# %%
for n, d in T.items():
    num = d.select_dtypes("number")
    if num.shape[1]:
        print(f"\n── {n}")
        display(num.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T)

# %%
for n, d in T.items():
    cat = [c for c in d.columns if (d[c].dtype == object or str(d[c].dtype) in ("category", "bool")) and d[c].nunique() <= 15]
    for c in cat:
        vc = d[c].value_counts(dropna=False)
        print(f"{n}.{c}: {vc.to_dict()}")

# %% [markdown]
# Constant columns (`data_source`, `is_synthetic`, `*_is_synthetic`) carry no information within this
# dataset. They are metadata, and are excluded from features (notebook 09). Every row in every table
# except `public_holidays` is flagged `is_synthetic = True` / `data_source = synthetic…` by the data itself;
# conclusions in this EDA describe *this dataset* and are not claims about real ports or carriers.

# %% [markdown]
# ## A6. Validity checks — invalid, inconsistent, and extreme-but-valid values

# %%
md = T["port_metadata"].set_index("port_id")
telc = tel.join(md["capacity_teu_per_hour"], on="port_id")
checks = [
    ("telemetry", "port_utilization_pct outside [0,100]", (~tel.port_utilization_pct.between(0, 100)).sum()),
    ("telemetry", "port_utilization_pct == 100 (saturated)", (tel.port_utilization_pct == 100).sum()),
    ("telemetry", "throughput_teu > capacity", (telc.throughput_teu > telc.capacity_teu_per_hour + 1e-9).sum()),
    ("telemetry", "throughput_teu < 0", (tel.throughput_teu < 0).sum()),
    ("telemetry", "equipment_availability_pct outside [0,100]", (~tel.equipment_availability_pct.between(0, 100)).sum()),
    ("telemetry", "negative counts (arrivals/departures/anchor)", (tel[["vessel_arrivals", "vessel_departures", "vessels_at_anchor"]] < 0).sum().sum()),
    ("telemetry", "customs_hold_hours <= 0", (tel.customs_hold_hours <= 0).sum()),
    ("weather", "relative_humidity outside [0,100]", (~wx.relative_humidity_2m_pct.between(0, 100)).sum()),
    ("weather", "relative_humidity == 100", (wx.relative_humidity_2m_pct == 100).sum()),
    ("weather", "precipitation_mm < 0", (wx.precipitation_mm < 0).sum()),
    ("weather", "wind_gusts < 0", (wx.wind_gusts_10m_kmh < 0).sum()),
    ("weather", "wind_gusts < wind_speed", (wx.wind_gusts_10m_kmh < wx.wind_speed_10m_kmh).sum()),
    ("weather", "wind_direction outside [0,360]", (~wx.wind_direction_10m_deg.between(0, 360)).sum()),
    ("weather", "visibility_m outside (0, 100000]", (~wx.visibility_m.between(1, 100000)).sum()),
    ("weather", "surface_pressure outside [950,1050] hPa", (~wx.surface_pressure_hpa.between(950, 1050)).sum()),
    ("shipment", "actual_arrival < departure", (ship.actual_arrival_ts < ship.departure_ts).sum()),
    ("shipment", "expected_arrival <= departure", (ship.expected_arrival_ts <= ship.departure_ts).sum()),
    ("shipment", "container_teu <= 0", (ship.container_teu <= 0).sum()),
    ("shipment", "container_teu non-integer", (ship.container_teu % 1 != 0).sum()),
    ("shipment", "delay ≠ actual − expected (|diff| > 0.05 h)", (((ship.actual_arrival_ts - ship.expected_arrival_ts).dt.total_seconds() / 3600 - ship.actual_delay_hours).abs() > 0.05).sum()),
    ("network", "route_distance_km <= 0", (net.route_distance_km <= 0).sum()),
    ("disruption", "event_end <= event_start", (T["disruption_events"].event_end <= T["disruption_events"].event_start).sum()),
    ("disruption", "affected_capacity_pct outside [0,100]", (~T["disruption_events"].affected_capacity_pct.between(0, 100)).sum()),
    ("traffic", "container_vessel_calls > vessel_calls", (T["port_traffic_history"].container_vessel_calls > T["port_traffic_history"].vessel_calls).sum()),
]
pd.DataFrame(checks, columns=["table", "check", "rows"])

# %% [markdown]
# **Interpretation of the non-zero checks**
#
# * **Utilization = 100 % in 1,628 port-hours.** `port_utilization_pct` equals `throughput_teu / capacity_teu_per_hour × 100`
#   (verified next cell), so 100 % means *throughput hit nominal capacity*. That is a valid saturated state (demand above
#   capacity shows up as a growing anchor queue instead). It is a **censored** measurement: utilization cannot express how far
#   demand exceeds capacity. Not an error; do not treat as outlier.
# * **Wind gust < wind speed (77 h) and gust < 0 (2 h).** Physically a gust (peak) should not be below the mean wind. These are
#   small-magnitude data-quality problems (likely residue of the swapped speed/gust columns noted in the raw issues log). They
#   affect < 1 % of hours; flag, do not silently fix.
# * **Humidity exactly 100 % (6 h)** is physically possible (fog/saturation); consistent with clipping of >100 % raw values. Valid.
# * **Visibility 200 m** minima appear at almost every port — plausible in fog/heavy rain; investigated in notebook 05.

# %%
util_calc = telc.throughput_teu / telc.capacity_teu_per_hour * 100
print("max |utilization − throughput/capacity×100| =", round((util_calc - telc.port_utilization_pct).abs().max(), 3), "pp")
print("→ port_utilization_pct is a deterministic function of throughput_teu and port capacity (redundant).")
sat = tel.assign(sat=tel.port_utilization_pct == 100).groupby("port_id")["sat"].mean().mul(100).round(1)
print("\n% of hours at 100 % utilization by port:\n", sat.sort_values(ascending=False).to_string())

# %%
# Extreme-but-plausible values: how far do tails extend relative to the IQR?
def tail_report(d, cols):
    out = []
    for c in cols:
        x = d[c]
        q1, q3 = x.quantile([0.25, 0.75])
        hi = q3 + 3 * (q3 - q1)
        out.append({"column": c, "p99": x.quantile(0.99), "max": x.max(), "n_beyond_Q3+3IQR": int((x > hi).sum())})
    return pd.DataFrame(out).set_index("column")

display(tail_report(tel, ["container_dwell_hours", "vessel_time_in_port_hours", "vessels_at_anchor", "customs_hold_hours"]))
display(tail_report(wx, ["precipitation_mm", "wind_gusts_10m_kmh", "wind_speed_10m_kmh"]))
display(tail_report(ship, ["actual_delay_hours", "container_teu"]))

# %% [markdown]
# Extreme tails in dwell time, anchor queue and vessel time cluster in time at specific ports (notebook 04 shows they
# coincide with disruption events), so they are treated as **valid extreme operating states**, not errors.
# Heavy precipitation and gust tails are expected for weather data.

# %% [markdown]
# ## A7. Lineage: what did cleaning fill in?
#
# The processed tables contain no missing values in measurement columns and carry no imputation flags.
# To distinguish observed values from values created during cleaning, the processed files are compared
# (read-only) with `Dataset/Raw/`. A value is counted as *filled* when the raw cell was empty, a text
# placeholder (`N/A`, `NaN`, `-`) or a numeric sentinel (`-999`, `-9999`, `9999`).
#
# Note: this compares *cell presence*, not cell values. Values that were transformed (units, formats) are
# not counted here.

# %%
lin = {}
for n in T:
    r = lineage.raw_vs_processed_summary(n)
    r = r[(r.raw_missing_or_placeholder + r.raw_numeric_sentinel) > 0]
    r = r.assign(table=n, raw_rows=r.attrs["raw_rows"], processed_rows=r.attrs["processed_rows"])
    lin[n] = r
lin_df = pd.concat(lin.values()).reset_index().set_index(["table", "column"])
lin_df

# %%
flags = lineage.shipment_fill_flags()
fsum = flags.drop(columns="shipment_id").sum().rename("shipments").to_frame()
fsum["pct"] = (fsum.shipments / len(flags) * 100).round(2)
fsum

# %%
flags.to_csv(ROOT / "data" / "interim" / "shipment_lineage_flags.csv", index=False)
print("saved eda/data/interim/shipment_lineage_flags.csv")
any_feat = flags[[c for c in flags if c.startswith("filled_") and c not in ("filled_actual_arrival_ts", "filled_actual_delay_hours", "filled_delay_status")]].any(axis=1)
print("shipments with ≥1 filled input attribute:", any_feat.sum(), f"({any_feat.mean()*100:.1f}%)")

# %%
# Which ports had visibility entirely missing in raw?
raw_wx = pd.read_csv(RAW / "weather_hourly.csv", dtype=str, keep_default_na=False)
raw_wx["port_norm"] = raw_wx.port_id.str.strip().str.upper()
vis_missing = raw_wx.assign(m=raw_wx.visibility_m.str.strip().isin(lineage.TEXT_PLACEHOLDERS)).groupby("port_norm")["m"].agg(["sum", "size"])
vis_missing = vis_missing[vis_missing["sum"] > 0].sort_values("sum", ascending=False)
vis_missing.head(8)

# %% [markdown]
# ### Lineage findings (important for modelling)
#
# 1. **295 shipments (3.0 %) had no observed outcome in the raw data** — `actual_arrival_ts`, `actual_delay_hours` and
#    `delay_status` were all empty — yet the processed file contains values for all three. The method used to produce
#    these targets cannot be determined from the data. They are not observed labels and should be **excluded from model
#    training and evaluation** (or at minimum flagged and tested in a sensitivity analysis). Notebook 02 compares their
#    distribution to the observed rows.
# 2. **97 shipments carried the ambiguous raw label `Late`**; the processed label was recomputed from timestamps and 58 of
#    them are now `On-Time`. Recomputation from timestamps is consistent with the stated thresholds, but it shows the raw
#    label channel was unreliable.
# 3. **Shipment input attributes were filled** for ~1–3 % of rows each (carrier, cargo type, priority, TEU, mode, route,
#    expected arrival); **1,231 shipments (12.5 %) have at least one filled input attribute**. Route can be derived exactly
#    from origin/destination, but filled carrier/cargo/priority/TEU values are not observations. The flags are saved to
#    `eda/data/interim/shipment_lineage_flags.csv` for sensitivity checks.
# 4. **Telemetry**: ≈ 1,540 cells across throughput, dwell, arrivals, anchor queue, customs hold and labour status were empty or
#    sentinels in raw and are populated in processed (raw also lacked 120 whole port-hours). The largest is
#    `customs_hold_hours` (500 cells). Weather similarly had gusts (150) and humidity (100) gaps filled. These cannot be
#    located in the processed file without a row-level raw↔processed join, so treat telemetry as ~2–5 % reconstructed.
# 5. **Weather visibility was entirely absent in raw for USLAX, CNSHA and LKCMB** (≈ 3,000 port-hours). The processed values for
#    those three ports are therefore not measurements. Visibility features are unreliable for these ports.
# 6. Three disruption events had no end time in raw; their processed end times are reconstructed.

# %% [markdown]
# ## A8. Role of each table in the delay-risk problem
#
# | Table | Grain | Role | Join to shipment |
# |---|---|---|---|
# | `shipment_records` | shipment | **Fact table**: target (`delay_status`), outcome (`actual_*`), static shipment attributes | — |
# | `port_network` | route | Static route attributes: distance, planned hours, direct vs transshipment, hub | `route_id` (m:1) |
# | `port_metadata` | port | Static port attributes: capacity, region, country/subdivision, timezone | origin / destination / hub `port_id` (m:1) |
# | `port_hourly_telemetry` | port × hour | Time-varying port congestion and operations | (`port_id`, hour ≤ prediction cutoff) per port role |
# | `weather_hourly` | port × hour | Time-varying weather at each port | same as telemetry |
# | `disruption_events` | event interval | Strikes, weather, congestion, equipment, cyber events | interval overlap with cutoff, per port role |
# | `port_traffic_history` | port × day | Daily vessel calls, volumes, deviation from normal | last complete day before departure |
# | `public_holidays` | country/subdivision × date | Calendar, known in advance | via port country/subdivision, near departure / expected arrival |
#
# Detailed join logic, temporal matching and duplication checks: notebook 08.
