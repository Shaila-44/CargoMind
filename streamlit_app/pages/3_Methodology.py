"""Methodology: which models are used, why, and what the outputs do and do not mean."""
import logging

import streamlit as st

from components import ui
from services import dl_forecasting as dl
from services import ml_risk

log = logging.getLogger("cargomind.methodology")

st.set_page_config(page_title="Methodology | CargoMind", layout="wide")
ui.theme()
ui.navbar("Methodology")
ui.header("Methodology", "What each model does, how to read its outputs, and the limits of this demo.")

# --- System diagram ------------------------------------------------------------------------------------------------
with ui.panel("diagram"):
    ui.card_head("How CargoMind produces its outputs", "Two workflows; every output is shown on its own and never merged")
    st.markdown(
        '<div class="cm-lane-label">Shipment workflow (ML) · two independent models on the same features</div>'
        '<div class="cm-flow">'
        '<div class="node"><b>Shipment attributes</b>route, carrier, mode, cargo, departure</div><span class="arrow">→</span>'
        '<div class="node"><b>Feature builder</b>51 features from port, weather, disruption and holiday history</div>'
        '<span class="arrow">→</span>'
        '<div class="node navy"><b>XGBoost</b>delay class + raw probabilities (On-Time, At-Risk, Critical)</div>'
        '<div class="node orange"><b>Isolation Forest</b>anomaly score + flag for unusual origin conditions</div>'
        '</div>'
        '<div class="cm-lane-label" style="margin-top:1rem">Port workflow (DL) · one forecast, with an automatic fallback</div>'
        '<div class="cm-flow">'
        '<div class="node"><b>12-hour port window</b>dwell, utilisation, vessels at anchor, wind gust, disruption and strike state</div>'
        '<span class="arrow">→</span>'
        '<div class="node"><b>Input validation</b>invalid windows are rejected, never passed on</div><span class="arrow">→</span>'
        '<div class="node blue"><b>Transformer T_w</b>primary next-hour dwell forecast (hours)</div>'
        '<span class="arrow">→</span>'
        '<div class="node orange"><b>C_w LSTM</b>used automatically only if T_w cannot load or returns an invalid value</div>'
        '<div class="node dashed"><b>EWMA</b>benchmark for comparison, run only on request; not a fallback</div>'
        '</div>', unsafe_allow_html=True)

# --- Model cards ---------------------------------------------------------------------------------------------------
ui.section("Models in use")
with ui.row("models_a"):
    m1, m2 = st.columns(2, gap="medium")
with m1, ui.panel("xgb_method"):
    ui.card_head("XGBoost", "Delay classification · provisional primary")
    ui.rows({"Output": "Delay class (On-Time, At-Risk, Critical) and raw probabilities",
             "Predicted class": "Highest raw probability",
             "Calibration": "None · raw probabilities", "Alert threshold": "None set",
             "Challenger": "Logistic Regression"})
with m2, ui.panel("iforest_method"):
    ui.card_head("Isolation Forest", "Anomaly detection · primary detector")
    ui.rows({"Output": "Anomaly score and flag", "Question answered": "Did the origin's operating state look unusual?",
             "Compared with": "Normal on-time operating conditions",
             "Relation to delay": "Independent; an anomaly is not a delay prediction"})
with ui.row("models_b"):
    m3, m4 = st.columns(2, gap="medium")
with m3, ui.panel("tw_method"):
    ui.card_head("Transformer T_w", "Next-hour dwell forecasting · primary")
    ui.rows({"Input": "Last 12 observed hours of one port", "Output": "Forecast dwell time for the next hour, in hours",
             "Uncertainty": "Point forecast; no calibrated interval", "Selected on": "Validation data only"})
with m4, ui.panel("cw_method"):
    ui.card_head("C_w LSTM", "Next-hour dwell forecasting · automatic fallback")
    ui.rows({"Used when": "T_w cannot load or returns an invalid value", "Reported as": "Fallback used, with the reason",
             "Known weakness": "Weaker than EWMA in the first six hours of a disruption",
             "Test set": "Not scored on a test set"})

# --- Detail sections -----------------------------------------------------------------------------------------------
ui.section("Details")
with st.expander("Why the two ML outputs stay separate", icon=":material/call_split:", expanded=True):
    st.markdown(
        """
The delay class and the operating-conditions flag come from different models answering different questions.
They are shown together to help interpretation, but they are **never merged** into one score, ensemble or new class.

An anomaly is not a delay prediction. A shipment can be delayed without an unusual origin, and an unusual origin does
not prove a future delay.
"""
    )

with st.expander("Reading the probabilities", icon=":material/percent:"):
    st.markdown(
        """
The XGBoost probabilities are **raw model probabilities**. They have not been calibrated, so they are not real-world
likelihoods. Calibration was tested with sigmoid and isotonic methods, and neither met the acceptance rule, so no
calibrated output exists. No Critical alert threshold has been approved, and the predicted class is the highest raw
probability.
"""
    )

with st.expander("Calibration study", icon=":material/tune:"):
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

with st.expander("Evidence behind the classifier choice", icon=":material/fact_check:"):
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

with st.expander("Forecast fallback", icon=":material/swap_horiz:"):
    st.markdown(
        """
The Transformer T_w is the primary forecaster. If it cannot load or returns an invalid value, the C_w LSTM is used
automatically and the page says so, with the reason. Invalid input is rejected and is never passed to the fallback.
The C_w LSTM is weaker than EWMA in the first six hours of a disruption and has not been scored on a test set.
EWMA (alpha 0.2) is a benchmark only. The experimental GNN is not used for inference.
"""
    )

# --- Provenance ----------------------------------------------------------------------------------------------------
ui.section("Model provenance · read from the selection files the services use")
try:
    sel = ml_risk.selection()
    registry = dl.available_models()
except Exception:
    log.exception("provenance unavailable")
    ui.error_box("Model provenance is unavailable right now.")
else:
    with ui.row("provenance"):
        p1, p2 = st.columns(2, gap="medium")
    with p1, ui.panel("ml_selection"):
        ui.card_head("ML selection", "ML/ML/models/selection.json")
        ui.rows({"Classifier": sel["classifier"]["name"], "Challenger": sel["classifier"]["challenger"],
                 "Anomaly detector": sel["anomaly_detector"]["name"], "Alert threshold": "None set",
                 "Selected on": sel["selected_on"]})
        st.caption(sel["classifier"]["status"])
    with p2, ui.panel("dl_registry"):
        ui.card_head("Forecasting registry", "DL/DL/inference/model_registry.json")
        ui.rows({name: f"{info['role']} · val RMSE {info['val_rmse_h_5_seed_mean']:.3f} h" for name, info in registry.items()
                 if info["role"] in ("primary", "fallback")})
        st.caption("EWMA (alpha 0.2) is a benchmark, run only on request. Research models stay registered for comparison; "
                   "the GNN is not registered.")

# --- Limits --------------------------------------------------------------------------------------------------------
with ui.panel("limits"):
    ui.card_head("Limits of this demo", "Read every output with these in mind")
    st.markdown(
        """
- **Synthetic, offline data only.** There is no live logistics, port, weather or telemetry feed.
- Supported shipment departures and port history cover a fixed synthetic period. Requests outside it are rejected.
- Model choices rest on synthetic-data evidence. Real-world generalisation is not established.
- The experimental GNN is not used for inference.
- Outputs are not current real-world port or shipment conditions and must not be read as operational alerts.
"""
    )
