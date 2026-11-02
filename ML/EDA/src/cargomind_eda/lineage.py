"""Raw-vs-processed lineage checks.

The cleaned tables carry no flag for values that were filled in during cleaning. These
helpers compare the processed files with the raw files in Dataset/Raw (read-only) so the
EDA can tell observed values apart from values that were reconstructed or imputed.
Nothing here alters either file.
"""
import numpy as np
import pandas as pd

from .config import RAW, PROCESSED

TEXT_PLACEHOLDERS = {"", "N/A", "NA", "NaN", "nan", "-", "null", "None"}
NUMERIC_SENTINELS = {"-999", "-999.0", "-9999", "-9999.0", "9999", "9999.0"}


def _read_str(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def raw_vs_processed_summary(name):
    """Per column: raw missing/placeholder count, raw sentinel count, processed missing count."""
    raw = _read_str(RAW / f"{name}.csv")
    proc = _read_str(PROCESSED / f"{name}.csv")
    rows = []
    for c in proc.columns:
        r = raw[c].str.strip() if c in raw.columns else pd.Series(dtype=str)
        rows.append(
            {
                "column": c,
                "raw_missing_or_placeholder": int(r.isin(TEXT_PLACEHOLDERS).sum()),
                "raw_numeric_sentinel": int(r.isin(NUMERIC_SENTINELS).sum()),
                "processed_missing": int(proc[c].str.strip().isin({""}).sum()),
            }
        )
    out = pd.DataFrame(rows).set_index("column")
    out.attrs["raw_rows"] = len(raw)
    out.attrs["processed_rows"] = len(proc)
    return out


SHIPMENT_FLAG_COLS = [
    "route_id",
    "carrier",
    "shipping_mode",
    "cargo_type",
    "container_teu",
    "shipment_priority",
    "expected_arrival_ts",
    "actual_arrival_ts",
    "actual_delay_hours",
    "delay_status",
]


def shipment_fill_flags():
    """One row per processed shipment_id: True where the raw value was missing/placeholder.

    Raw duplicates (same shipment_id) are collapsed to the first occurrence, mirroring the
    processed file's one-row-per-shipment grain.
    """
    raw = _read_str(RAW / "shipment_records.csv").drop_duplicates("shipment_id")
    proc = _read_str(PROCESSED / "shipment_records.csv")[["shipment_id"]]
    m = proc.merge(raw, on="shipment_id", how="left", validate="1:1")
    flags = pd.DataFrame({"shipment_id": m["shipment_id"]})
    for c in SHIPMENT_FLAG_COLS:
        v = m[c].str.strip()
        flags[f"filled_{c}"] = v.isin(TEXT_PLACEHOLDERS) | v.isin(NUMERIC_SENTINELS)
    flags["raw_label_was_Late"] = m["delay_status"].str.strip().eq("Late")
    # The outcome was never observed in raw when arrival, delay hours and label are all absent
    flags["outcome_not_observed_in_raw"] = (
        flags["filled_actual_arrival_ts"] & flags["filled_actual_delay_hours"] & flags["filled_delay_status"]
    )
    return flags
