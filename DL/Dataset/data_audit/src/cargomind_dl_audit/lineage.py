"""Cell-level lineage of the hourly tables: raw (Dataset/Raw) vs processed (Dataset/Processed).

The processed hourly tables are complete and carry no imputation flags, but the raw files have
missing hours, empty cells and sentinels. This module matches every processed (port, hour) cell
to its raw counterpart and labels it, verified by value (both files are read-only):

* observed       - raw value parses directly and equals the processed value;
* converted      - raw value equals the processed value after a unit/format fix ('84.2h', '93%',
                   0.82 -> 82 %, Fahrenheit -> Celsius, swapped wind columns, label spelling);
* reconstructed  - no usable raw value (hour missing from raw, empty, sentinel, impossible value,
                   or a raw value that does not match the processed one). These are not observations.
"""
import difflib
import re

import numpy as np
import pandas as pd

from .config import PROCESSED, RAW


def _port_mapper(meta):
    codes = meta["port_id"].tolist()
    words = {}
    for name, code in zip(meta["port_name"], meta["port_id"]):
        for w in re.findall(r"[a-z]+", name.lower()):
            if w not in ("port", "of"):
                words[w] = code

    def to_code(v):
        s = v.strip().upper()
        if s in codes:
            return s
        for w in re.findall(r"[a-z]+", v.lower()):
            if w in words:
                return words[w]
        same_letters = [c for c in codes if sorted(c) == sorted(s)]
        if same_letters:
            return same_letters[0]
        close = difflib.get_close_matches(s, codes, 1, 0.6)
        return close[0] if close else None

    return to_code


def _to_utc(v):
    """Resolve the raw timestamp formats documented in Dataset/Raw/_data_issues_log.csv."""
    v = v.strip()
    if re.fullmatch(r"\d{7}", v):  # epoch seconds / 1000 (precision lost) -> nearest hour is unique
        return pd.Timestamp(int(v) * 1000, unit="s", tz="UTC").round("h")
    if re.fullmatch(r"\d{10}", v):
        return pd.Timestamp(int(v), unit="s", tz="UTC")
    if "T" in v:
        return pd.Timestamp(v).tz_convert("UTC")
    if "/" in v:
        return pd.to_datetime(v, format="%d/%m/%Y %H:%M").tz_localize("UTC")
    t = pd.Timestamp(v)
    if t.tzinfo is not None:
        return t.tz_convert("UTC")
    if t.minute == 30:  # IST local time without a marker: the only half-hour values in the file
        return (t - pd.Timedelta(hours=5, minutes=30)).tz_localize("UTC")
    return t.tz_localize("UTC")


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(r"[^0-9eE.+-]", "", regex=True).replace("", np.nan), errors="coerce")


NUMERIC_COLS = {
    "port_hourly_telemetry": ["throughput_teu", "port_utilization_pct", "container_dwell_hours", "vessel_time_in_port_hours",
                              "vessel_arrivals", "vessel_departures", "vessels_at_anchor", "customs_hold_hours",
                              "equipment_availability_pct"],
    "weather_hourly": ["temperature_2m_c", "relative_humidity_2m_pct", "precipitation_mm", "wind_speed_10m_kmh",
                       "wind_gusts_10m_kmh", "surface_pressure_hpa", "visibility_m"],
}
CATEGORICAL_COLS = {"port_hourly_telemetry": ["labour_status"], "weather_hourly": ["weather_condition"]}
# spelling variants seen in the raw files (Dataset/Raw/_data_issues_log.csv)
LABEL_VARIANTS = {"norm": "normal", "partial": "reduced", "on strike": "strike"}
SWAP = {"wind_speed_10m_kmh": "wind_gusts_10m_kmh", "wind_gusts_10m_kmh": "wind_speed_10m_kmh"}


def cell_lineage(table):
    """Return (processed frame sorted by port/hour, status frame of the same shape, row-level summary)."""
    meta = pd.read_csv(PROCESSED / "port_metadata.csv")
    proc = pd.read_csv(PROCESSED / f"{table}.csv")
    proc["ts"] = pd.to_datetime(proc["ts_hour"], utc=True)
    proc = proc.sort_values(["port_id", "ts"]).reset_index(drop=True)
    raw = pd.read_csv(RAW / f"{table}.csv", dtype=str, keep_default_na=False)
    raw["port"] = raw["port_id"].map(_port_mapper(meta))
    raw["ts"] = raw["ts_hour"].map(_to_utc)
    n_raw, n_unmapped = len(raw), int(raw["port"].isna().sum())
    raw = raw.dropna(subset=["port"])
    n_dupe_keys = int(raw.duplicated(["port", "ts"]).sum())
    raw = raw.drop_duplicates(["port", "ts"]).set_index(["port", "ts"])
    key = pd.MultiIndex.from_arrays([proc["port_id"], proc["ts"]])
    in_raw = key.isin(raw.index)
    r = raw.reindex(key)

    status = pd.DataFrame(index=proc.index)
    for c in NUMERIC_COLS[table]:
        rv = _num(r[c].fillna(""))
        pv = proc[c].astype(float)
        eq = np.isclose(rv.values, pv.values, atol=0.051)
        conv = np.zeros(len(pv), bool)
        if c == "port_utilization_pct":
            conv = np.isclose(rv.values * 100, pv.values, atol=0.6)
        if c == "temperature_2m_c":
            conv = np.isclose((rv.values - 32) * 5 / 9, pv.values, atol=0.11)
        if c in SWAP:
            conv = np.isclose(_num(r[SWAP[c]].fillna("")).values, pv.values, atol=0.051) & ~eq
        unit_text = r[c].fillna("").str.contains(r"[^0-9eE.+\- ]", regex=True).values
        s = np.where(eq & ~unit_text, "observed", np.where(eq | conv, "converted", "reconstructed"))
        s[np.isnan(pv.values)] = "missing_in_processed"
        status[c] = s
    for c in CATEGORICAL_COLS[table]:
        raw_exact = r[c].fillna("").reset_index(drop=True)
        rv = raw_exact.str.strip().str.lower().map(lambda v: LABEL_VARIANTS.get(v, v))
        pv = proc[c].astype(str).str.lower()
        s = np.where(raw_exact == proc[c].astype(str), "observed", np.where((rv != "") & (rv == pv), "converted", "reconstructed"))
        status[c] = s
    status.loc[~in_raw, :] = "reconstructed"
    summary = {"table": table, "raw_rows": n_raw, "raw_rows_unmapped_port": n_unmapped,
               "raw_duplicate_keys_dropped": n_dupe_keys, "processed_rows": len(proc),
               "processed_hours_absent_from_raw": int((~in_raw).sum())}
    return proc, status, summary


def lineage_tables(save_dir):
    out, summaries = {}, []
    for table in ("port_hourly_telemetry", "weather_hourly"):
        proc, status, summary = cell_lineage(table)
        summaries.append(summary)
        long = status.apply(lambda s: s.value_counts()).fillna(0).astype(int).T
        long.index.name = "column"
        long.insert(0, "table", table)
        long.to_csv(save_dir / f"lineage_counts_{table}.csv")
        flags = proc[["port_id", "ts_hour"]].copy()
        for c in status.columns:
            flags[f"{c}__status"] = status[c].values
        flags["hour_absent_from_raw"] = (status == "reconstructed").all(axis=1).values
        flags.to_csv(save_dir / f"cell_lineage_{table}.csv", index=False)
        out[table] = (proc, status)
    pd.DataFrame(summaries).to_csv(save_dir / "lineage_summary.csv", index=False)
    return out, pd.DataFrame(summaries)
