"""Methodology: which models are used, why, and what the outputs do and do not mean."""
import logging

import streamlit as st

from components import ui
from services import dl_forecasting as dl
from services import ml_risk

log = logging.getLogger("cargomind.methodology")

st.set_page_config(page_title="Methodology | CargoMind", layout="wide")
ui.theme()
ui.demo_banner()
ui.header("Methodology",
          "What each model does, how to read its outputs, and the limits of this demo.")

st.header("Models in use")
st.markdown(
    """
| Model | Role | Output |
|---|---|---|
| **XGBoost** | Provisional primary delay classifier | Delay class (On-Time, At-Risk, Critical) and raw probabilities |
| **Isolation Forest** | Primary anomaly detector | Anomaly score and flag for unusual origin operating conditions |
| **Transformer T_w** | Primary next-hour dwell forecaster | Forecast in hours |
| **C_w LSTM** | Automatic fallback for the forecaster | Used only if the primary cannot produce a valid forecast |
| **EWMA** | Benchmark / reference | Shown only when you ask for it. It is not a fallback. |
"""
)

st.header("Why two ML outputs stay separate")
st.markdown(
    """
The delay class and the operating-conditions flag come from different models answering different questions.
They are shown together to help interpretation, but they are **never merged** into one score, ensemble or new class.

An anomaly is not a delay prediction. A shipment can be delayed without an unusual origin, and an unusual origin does
not prove a future delay.
"""
)

st.header("Reading probabilities")
st.markdown(
    """
The XGBoost probabilities are **raw model probabilities**. They have not been calibrated, so they are not real-world
likelihoods. Calibration was tested with sigmoid and isotonic methods, and neither met the acceptance rule, so no
calibrated output exists. No Critical alert threshold has been approved, and the predicted class is the highest raw
probability.
"""
)

st.header("Forecast fallback")
st.markdown(
    """
The Transformer T_w is the primary forecaster. If it cannot load or returns an invalid value, the C_w LSTM is used
automatically and the page says so, with the reason. Invalid input is rejected and is never passed to the fallback.
The C_w LSTM is weaker than EWMA in the first six hours of a disruption and has not been scored on a test set.
"""
)

st.header("Evidence behind the classifier choice")
st.markdown(
    """
XGBoost is the **provisional** primary classifier. Logistic Regression is kept as a challenger, because the evidence
does not show XGBoost is better in general. In a label-availability experiment, both models were tested on the same
2,394 test shipments. Training was limited to the 4,414 of 6,452 training shipments whose outcomes would have been
known at the start of the test period.

| Model | Critical caught: full → restricted | Critical recall change (95% range) | Critical alerts: full → restricted | Precision: full → restricted |
|---|---:|---:|---:|---:|
| Logistic Regression | 128 → 123 | −0.023 (−0.085 to +0.027) | 230 → 260 | 0.557 → 0.473 |
| XGBoost | 116 → 77 | −0.181 (−0.251 to −0.087) | 206 → 133 | 0.563 → 0.579 |

These come from one 11-day synthetic test period and say nothing about real-world performance.
"""
)

st.header("Calibration study")
st.markdown(
    """
Two calibration methods were compared on validation data. Neither met the pre-registered acceptance rule, so neither was
kept. That is why the app shows raw probabilities only.

| Method | Log loss | ECE | Macro-F1 | Critical recall |
|---|---:|---:|---:|---:|
| Raw (in use) | 0.787 | 0.059 | 0.613 | 0.529 |
| Sigmoid | 0.790 | 0.037 | 0.593 | 0.587 |
| Isotonic | 0.877 | 0.024 | 0.595 | 0.561 |
"""
)

st.header("Model provenance")
st.caption("Read from the model selection and forecasting registry used by the services.")
try:
    sel = ml_risk.selection()
    registry = dl.available_models()
except Exception:
    log.exception("provenance unavailable")
    ui.error_box("Model provenance is unavailable right now.")
else:
    p1, p2 = st.columns(2, gap="medium")
    with p1, st.container(border=True):
        ui.panel_title("ML selection")
        ui.rows({"Classifier": sel["classifier"]["name"], "Challenger": sel["classifier"]["challenger"],
                 "Anomaly detector": sel["anomaly_detector"]["name"], "Alert threshold": "None set",
                 "Selected on": sel["selected_on"]})
        st.caption(sel["classifier"]["status"])
    with p2, st.container(border=True):
        ui.panel_title("Forecasting registry")
        ui.rows({name: f"{info['role']} · val RMSE {info['val_rmse_h_5_seed_mean']:.3f} h" for name, info in registry.items()
                 if info["role"] in ("primary", "fallback")})
        st.caption("EWMA (alpha 0.2) is a benchmark, run only on request. Research models stay registered for comparison; "
                   "the GNN is not registered.")

st.header("Limits of this demo")
st.markdown(
    """
- **Synthetic, offline data only.** There is no live logistics, port, weather or telemetry feed.
- Supported shipment departures and port history cover a fixed synthetic period. Requests outside it are rejected.
- Model choices rest on synthetic-data evidence. Real-world generalisation is not established.
- The experimental GNN is not used for inference.
- Outputs are not current real-world port or shipment conditions and must not be read as operational alerts.
"""
)
