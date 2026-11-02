from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # the eda/ folder: all EDA outputs live here
PROJECT_ROOT = ROOT.parent  # ML/: the ML workflow root, which holds the shared source data in ML/Dataset
PROCESSED = PROJECT_ROOT / "Dataset" / "Processed"
RAW = PROJECT_ROOT / "Dataset" / "Raw"
INTERIM = ROOT / "data" / "interim"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"

EXPECTED_TABLES = [
    "shipment_records",
    "port_hourly_telemetry",
    "weather_hourly",
    "port_metadata",
    "port_network",
    "port_traffic_history",
    "disruption_events",
    "public_holidays",
]

# Target definition (unchanged from the project specification)
CLASSES = ["On-Time", "At-Risk", "Critical"]
THRESHOLDS = {"on_time_max": 12.0, "at_risk_max": 48.0}  # hours

# Validated (dataviz validator, light mode, all pairs): blue / yellow / red
CLASS_COLORS = {"On-Time": "#2a78d6", "At-Risk": "#eda100", "Critical": "#e34948"}
SEQ_CMAP = "Blues"
DIV_CMAP = "RdBu_r"
NEUTRAL = "#8a8985"

for _p in (INTERIM, FIGURES, TABLES):
    _p.mkdir(parents=True, exist_ok=True)
