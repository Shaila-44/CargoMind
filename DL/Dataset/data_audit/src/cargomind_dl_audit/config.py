"""Paths and the initial DL design under audit (from the project PPT; treated as a starting point)."""
from pathlib import Path

AUDIT_ROOT = Path(__file__).resolve().parents[2]  # DL/Dataset/data_audit
PROJECT_ROOT = AUDIT_ROOT.parents[2]  # repository root
PROCESSED = PROJECT_ROOT / "ML" / "Dataset" / "Processed"  # shared source data, read-only
RAW = PROJECT_ROOT / "ML" / "Dataset" / "Raw"  # read-only, used only for lineage

OUT = {k: AUDIT_ROOT / k for k in ("reports", "tables", "figures", "lineage")}

TARGET = "container_dwell_hours"
WINDOW = 12  # hours of history per sample
HORIZON = 1  # predict the next hour
PPT_SIGNALS = {
    "container_dwell_hours": "port_hourly_telemetry",
    "throughput_teu": "port_hourly_telemetry",
    "port_utilization_pct": "port_hourly_telemetry",
    "customs_hold_hours": "port_hourly_telemetry",
    "temperature_2m_c": "weather_hourly",
    "wind_speed_10m_kmh": "weather_hourly",
    "labour_status": "port_hourly_telemetry",
    "equipment_availability_pct": "port_hourly_telemetry",
}
# Measured (raw-traceable) columns of the recommended feature set; used for lineage-filtered window counts.
# Derived inputs (disruption state, calendar) are not raw measurements and carry no lineage status.
RECOMMENDED_MEASURED = ["container_dwell_hours", "port_utilization_pct", "vessels_at_anchor",
                        "wind_gusts_10m_kmh", "labour_status"]
SEED = 42


def ensure_dirs():
    for p in OUT.values():
        p.mkdir(parents=True, exist_ok=True)
