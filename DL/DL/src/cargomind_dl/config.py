"""Every DL dataset decision that is a parameter rather than code.

Decisions taken from the Stage 02 data-readiness audit (DL/Dataset/data_audit/reports/DL_data_readiness_report.md)
and the project's DL brief. Changing a value and re-running DL/Dataset/scripts/build_dl_dataset.py rebuilds
the dataset end to end.

This module is also the single source of truth for DL workflow locations (DL/Dataset -> DL/EDA -> DL/DL ->
DL/Evaluation); other DL modules derive their paths from it.
"""
from pathlib import Path

DL_STAGE = Path(__file__).resolve().parents[2]  # DL/DL         (models, training, inference, artifacts)
DL_ROOT = DL_STAGE  # kept for backward compatibility
DL_WORKFLOW = DL_STAGE.parent  # DL/
PROJECT_ROOT = DL_WORKFLOW.parent  # repository root
DATASET_STAGE = DL_WORKFLOW / "Dataset"  # DL/Dataset
EDA_STAGE = DL_WORKFLOW / "EDA"  # DL/EDA
EVALUATION_STAGE = DL_WORKFLOW / "Evaluation"  # DL/Evaluation
SHARED_DATASET = PROJECT_ROOT / "ML" / "Dataset"  # the project's shared source data (owned by the ML workflow; read-only)
PROCESSED = SHARED_DATASET / "Processed"  # source, read-only
RAW = SHARED_DATASET / "Raw"  # read-only, used only for provenance
AUDIT_LINEAGE = DATASET_STAGE / "data_audit" / "lineage"  # cell-level provenance produced by the audit (read-only here)
DATASET = DATASET_STAGE / "processed"  # outputs of the dataset stage (the DL modelling dataset)
REPORTS = DATASET_STAGE / "reports"  # dataset construction report

# Paths recorded in manifest.json before the repository reorganisation -> current locations.
_LEGACY_PREFIXES = {"Dataset/": "ML/Dataset/", "dl/data_audit/": "DL/Dataset/data_audit/", "dl/dataset/": "DL/Dataset/processed/"}


def resolve_recorded_path(recorded: str) -> Path:
    """Absolute path for a project-relative path recorded in an artifact, old layout or new."""
    for old, new in _LEGACY_PREFIXES.items():
        if recorded.startswith(old):
            return PROJECT_ROOT / (new + recorded[len(old):])
    return PROJECT_ROOT / recorded

SEED = 42
TARGET = "container_dwell_hours"
WINDOW = 12  # input hours t-11 .. t
HORIZON = 1  # target hour t+1

# Chronological split on the TARGET hour, using global hour indices shared by all ports.
# 1,000 hours -> train targets in hours [0, 700), validation [700, 850), test [850, 1000).
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
# A window is kept only if all of its input hours lie in the same split as its target
# (embargo of WINDOW-1 hours after each boundary), so no input hour is shared across splits.

# Model input channels, in tensor order. Provenance masks are stored separately (see PROVENANCE_COLUMNS).
FEATURES = [
    "dwell",                     # container_dwell_hours history
    "utilization",               # port_utilization_pct
    "labour_strike",             # labour_status == strike
    "wind_gust",                 # wind_gusts_10m_kmh
    "disruption_active",         # disruption event ongoing at hour t
    "disruption_hours_elapsed",  # hours since the ongoing disruption began
    "vessels_at_anchor",         # vessels_at_anchor
]
# Raw-measured inputs with cell-level provenance from the audit (source column per feature).
PROVENANCE_COLUMNS = {
    "dwell": ("port_hourly_telemetry", "container_dwell_hours"),
    "utilization": ("port_hourly_telemetry", "port_utilization_pct"),
    "labour_strike": ("port_hourly_telemetry", "labour_status"),
    "wind_gust": ("weather_hourly", "wind_gusts_10m_kmh"),
    "vessels_at_anchor": ("port_hourly_telemetry", "vessels_at_anchor"),
}
STATUS_CODES = {"observed": 0, "converted": 1, "reconstructed": 2}

EXCLUDED_FEATURES = {
    "throughput_teu": "exact multiple of utilization (utilization = throughput / capacity x 100); port-specific scale",
    "customs_hold_hours": "hour-to-hour white noise, no relation to next-hour dwell (audit partial r 0.01); most reconstructed signal (6.1 %)",
    "temperature_2m_c": "no within-port relation to next-hour dwell (audit partial r 0.05)",
    "equipment_availability_pct": "no measurable next-hour signal (audit partial r -0.00)",
    "wind_speed_10m_kmh": "replaced by wind gusts (r = 0.98, same signal)",
    "port_traffic_history": "daily aggregate of the same telemetry; same-day values leak later hours",
}

# EWMA baseline: smoothing factor chosen on training windows only, from this grid.
EWMA_ALPHAS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
