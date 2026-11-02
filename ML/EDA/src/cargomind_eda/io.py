"""Read-only loaders for the cleaned tables.

Every transformation applied here is a *type* conversion only (timestamp parsing to
UTC-aware datetimes, ordered categoricals). Values, units and row counts are preserved.
"""
import pandas as pd

from .config import PROCESSED, EXPECTED_TABLES, CLASSES

TS_COLUMNS = {
    "shipment_records": ["departure_ts", "expected_arrival_ts", "actual_arrival_ts"],
    "port_hourly_telemetry": ["ts_hour"],
    "weather_hourly": ["ts_hour"],
    "disruption_events": ["event_start", "event_end"],
}
DATE_COLUMNS = {
    "port_traffic_history": ["period_start"],
    "public_holidays": ["holiday_date"],
}


def available_tables():
    """Return {table_name: path} for CSVs that actually exist, and the missing expected names."""
    found = {p.stem: p for p in sorted(PROCESSED.glob("*.csv"))}
    missing = [t for t in EXPECTED_TABLES if t not in found]
    return found, missing


def load_raw_strings(name):
    """Load a processed table with no type inference (for format checks)."""
    return pd.read_csv(PROCESSED / f"{name}.csv", dtype=str, keep_default_na=False)


def load(name, parse=True):
    df = pd.read_csv(PROCESSED / f"{name}.csv")
    if not parse:
        return df
    for c in TS_COLUMNS.get(name, []):
        df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601")
    for c in DATE_COLUMNS.get(name, []):
        df[c] = pd.to_datetime(df[c], format="%Y-%m-%d")
    if name == "shipment_records":
        df["delay_status"] = pd.Categorical(df["delay_status"], categories=CLASSES, ordered=True)
        df["shipment_priority"] = pd.Categorical(
            df["shipment_priority"], categories=["low", "medium", "high"], ordered=True
        )
    if name == "disruption_events":
        df["severity"] = pd.Categorical(df["severity"], categories=["low", "medium", "high"], ordered=True)
    return df


def load_all(parse=True):
    found, _ = available_tables()
    return {name: load(name, parse=parse) for name in found}
