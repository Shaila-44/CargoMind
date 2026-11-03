"""ML Intelligence: XGBoost delay class and Isolation Forest anomaly, shown together but kept as two separate outputs."""
from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from components import ui
from services import ml_risk, port_ops

log = logging.getLogger("cargomind.ml_page")

st.set_page_config(page_title="ML Intelligence | CargoMind", layout="wide")
ui.theme()
ui.navbar("ML Intelligence")
h_left, h_right = st.columns([1.6, 1], vertical_alignment="bottom")
with h_left:
    ui.header("ML Intelligence", "Delay outlook and unusual conditions at origin for one shipment: two independent model "
                                 "outputs, shown side by side and never combined into one score.")
with h_right:
    st.markdown('<div style="text-align:right">' + ui.badge("Model A · XGBoost classifier", "navy") + " "
                + ui.badge("Model B · Isolation Forest", "orange", ui.FLAG_COLOR) + "</div>", unsafe_allow_html=True)
ui.demo_banner()


@st.cache_data(show_spinner=False)
def _shipments() -> pd.DataFrame:
    s = ml_risk.supported_shipments()
    return s.assign(departure=pd.to_datetime(s["departure_ts"], utc=True, format="ISO8601"))


@st.cache_data(show_spinner=False)
def _names() -> dict[str, str]:
    try:
        return port_ops.port_names()
    except Exception:  # names are a convenience; ids still work without them
        log.exception("port names unavailable")
        return {}


@st.cache_data(show_spinner="Scoring shipment...")
def _score(shipment: pd.DataFrame) -> list[dict]:
    return ml_risk.predict_shipments(shipment[ml_risk.SHIPMENT_INPUT_COLUMNS])


def _origin_context(port: str, departure: pd.Timestamp) -> pd.Series | None:
    """Observed state of the origin port at the last full hour before departure (history, not a model output)."""
    try:
        snap = port_ops.snapshot(departure.floor("h") - pd.Timedelta(hours=1))
    except Exception:
        log.exception("origin context unavailable")
        return None
    hit = snap[snap["port_id"] == port]
    return None if hit.empty else hit.iloc[0]


def _reset():
    for key in ("origin", "shipment", "dest", "carrier", "dates"):
        st.session_state.pop(key, None)


try:
    shipments = _shipments()
except Exception:  # the file read failed; details go to the log, not the page
    log.exception("supported shipment list unavailable")
    ui.error_box("Shipment data is unavailable right now. Try again later.")
    st.stop()
names = _names()
label = lambda p: f"{p} · {names[p]}" if p in names else p  # noqa: E731
origins = sorted(shipments["origin_port_id"].unique())
seed = st.session_state.pop("cm_ml_origin_seed", None)  # set by the Overview's "Open in ML Intelligence"
if seed in origins:
    st.session_state["origin"] = seed
    st.session_state["shipment"] = None

# --- Shipment finder -----------------------------------------------------------------------------------------------
with ui.panel("finder"):
    ui.card_head("Find a shipment", "Synthetic shipments whose departure the models support")
    f1, f2, f3, f4, f5 = st.columns([1.25, 1, 1, 1.25, 0.6], vertical_alignment="bottom")
    with f1:
        origin = st.selectbox("Origin port", origins, index=None, format_func=label, placeholder="Choose an origin port",
                              key="origin", on_change=lambda: st.session_state.update(shipment=None))  # new port: new shipment
    from_origin = shipments[shipments["origin_port_id"] == origin] if origin else shipments.iloc[0:0]
    with f2:
        dests = st.multiselect("Destination", sorted(from_origin["destination_port_id"].unique()), format_func=label,
                               placeholder="Any destination", key="dest", disabled=origin is None)
    with f3:
        carriers = st.multiselect("Carrier", sorted(from_origin["carrier"].unique()), placeholder="Any carrier", key="carrier",
                                  disabled=origin is None)
    with f4:
        lo, hi = ml_risk.SUPPORTED_DEPARTURE_START.date(), ml_risk.SUPPORTED_DEPARTURE_END.date()
        dates = st.date_input("Departure dates (UTC)", value=(lo, hi), min_value=lo, max_value=hi, key="dates",
                              format="YYYY-MM-DD", help="Only the synthetic period the models support can be chosen.",
                              disabled=origin is None)
    with f5:
        st.button("Reset", icon=":material/restart_alt:", on_click=_reset, width="stretch")
    if origin is None:
        st.info("Choose an origin port to list its supported shipments.")
        st.stop()
    candidates = from_origin
    if dests:
        candidates = candidates[candidates["destination_port_id"].isin(dests)]
    if carriers:
        candidates = candidates[candidates["carrier"].isin(carriers)]
    if isinstance(dates, tuple) and len(dates) == 2:
        candidates = candidates[candidates["departure"].dt.date.between(dates[0], dates[1])]
    candidates = candidates.sort_values("departure")
    describe = dict(zip(candidates["shipment_id"],
                        candidates["shipment_id"] + " · to " + candidates["destination_port_id"] + " · "
                        + candidates["departure"].dt.strftime("%d %b %H:%M") + " · " + candidates["carrier"]))
    shipment_id = st.selectbox("Shipment", list(describe), index=None, format_func=describe.get,
                               placeholder=f"{len(candidates)} matching · type to search", key="shipment")

if candidates.empty:
    st.info("No supported shipments match these filters. Widen the filters or press Reset.")
    st.stop()

with st.expander(f"All {len(candidates)} matching shipments from {origin} (sortable table)", icon=":material/table:"):
    if st.toggle("Score every matching shipment", key=f"batch_{origin}"):
        try:
            batch = _score(candidates)
        except Exception:
            log.exception("batch scoring failed for %s", origin)
            ui.error_box("The ML service is unavailable right now. Try again later.")
        else:
            table = pd.DataFrame([{
                "Shipment": r["shipment_id"],
                "Predicted class (XGBoost)": r["classification"]["predicted_class"],
                **{f"Raw P({c})": r["classification"]["class_probabilities"][c] for c in ui.CLASS_ORDER},
                "Unusual conditions (Isolation Forest)": ("Flagged" if r["anomaly"]["anomaly_flag"] else "Not flagged")
                if r["anomaly"] else "No result",
                "Anomaly score": r["anomaly"]["anomaly_score"] if r["anomaly"] else None,
            } for r in batch]).merge(candidates[["shipment_id", "destination_port_id", "departure_ts"]]
                                     .rename(columns={"shipment_id": "Shipment", "destination_port_id": "Destination",
                                                      "departure_ts": "Departure (UTC)"}), on="Shipment")
            st.dataframe(table, hide_index=True, width="stretch",
                         column_config={f"Raw P({c})": st.column_config.NumberColumn(format="%.3f") for c in ui.CLASS_ORDER}
                         | {"Anomaly score": st.column_config.NumberColumn(format="%.3f")})
            st.caption("Click a header to sort. The anomaly score has its own scale and cannot be compared with the probabilities.")

if shipment_id is None:
    st.info("Select a shipment above to see both outputs. The list is searchable: type a shipment id, destination or carrier.")
    st.stop()

row = candidates[candidates["shipment_id"] == shipment_id].reset_index(drop=True)
r0 = row.iloc[0]

try:
    result = _score(row)[0]
except (ml_risk.ShipmentInputError, ml_risk.FeatureValidationError) as err:
    log.warning("shipment rejected by ML service: %s", err)
    ui.error_box("This shipment cannot be scored. Its inputs are outside what the model was built for.")
    st.stop()
except Exception:
    log.exception("ML inference failed for %s", shipment_id)
    ui.error_box("The ML service is unavailable right now. Try again later.")
    st.stop()

classification = result["classification"]
anomaly = result["anomaly"]
predicted = classification["predicted_class"]

# --- Shipment summary ----------------------------------------------------------------------------------------------
with ui.panel("summary"):
    s1, s2 = st.columns([1.1, 2], vertical_alignment="center", gap="large")
    with s1:
        st.markdown(f'<div class="cm-eyebrow">Shipment {shipment_id}</div>'
                    f'<div class="cm-big" style="font-size:1.9rem">{r0["origin_port_id"]} → {r0["destination_port_id"]}</div>'
                    f'<div class="cm-card-sub">{names.get(r0["origin_port_id"], "")} to {names.get(r0["destination_port_id"], "")}</div>',
                    unsafe_allow_html=True)
    with s2:
        ui.context_bar({"Shipment": shipment_id, "Carrier": r0["carrier"], "Mode": r0["shipping_mode"], "Cargo": r0["cargo_type"],
                        "Size": f"{r0['container_teu']:g} TEU", "Priority": r0["shipment_priority"],
                        "Departs": f"{r0['departure']:%a %d %b %Y, %H:%M} UTC",
                        "Expected arrival": f"{pd.Timestamp(r0['expected_arrival_ts']):%d %b %H:%M} UTC"})

ui.section("Two independent outputs · never combined into one score")
with ui.row("models"):
    left, right = st.columns(2, gap="medium")

with left, ui.panel("xgb"):
    ui.card_head("Delay outlook", f"Model A · {classification['model_label']} · primary delay classifier", ui.badge("Classification", "navy"))
    st.markdown(f'<div style="display:flex;align-items:center;gap:0.6rem;margin:0.4rem 0 0.2rem 0">'
                f'<span class="cm-card-sub" style="margin:0">Predicted class</span>{ui.class_chip(predicted)}</div>',
                unsafe_allow_html=True)
    st.altair_chart(ui.probability_chart(classification["class_probabilities"]), use_container_width=True,
                    key=f"ml_probs_{shipment_id}")
    st.caption(ui.UNCALIBRATED_NOTE)
    ui.note("<b>Uncalibrated probabilities.</b> The predicted class is simply the highest raw probability. 0.6 does not mean "
            "a 60 % real-world chance, and no alert threshold has been approved, so the app raises no alerts.")
    ui.help_popover("XGBoost gives a raw probability for each delay class; the predicted class is simply the highest "
                    "one. The probabilities are **not calibrated**: 0.6 does not mean a 60 % real-world chance. No "
                    "alert threshold has been approved, so the app does not raise alerts.")

with right, ui.panel("iforest"):
    ui.card_head("Unusual conditions at origin", "Model B · Isolation Forest · anomaly detector",
                 ui.badge("Anomaly detection", "orange"))
    if anomaly is None:
        st.info("No anomaly result was produced for this shipment.")
    else:
        flagged = bool(anomaly["anomaly_flag"])
        m1, m2 = st.columns([1, 1.2], vertical_alignment="center")
        with m1:
            st.metric("Anomaly score", f"{anomaly['anomaly_score']:.3f}",
                      help=f"{anomaly['score_definition']}. Flag threshold {anomaly['threshold']:.3f}.")
        with m2:
            st.markdown(f'<div class="cm-card-sub">Anomaly flag</div>{ui.flag_chip(flagged)}', unsafe_allow_html=True)
        st.altair_chart(ui.anomaly_scale(anomaly["anomaly_score"], anomaly["threshold"], flagged), use_container_width=True,
                        key=f"ml_anomaly_{shipment_id}")
        ctx = _origin_context(r0["origin_port_id"], r0["departure"])
        if ctx is not None:
            st.markdown(f'<div class="cm-card-sub" style="margin-top:0.3rem">Observed at {r0["origin_port_id"]} '
                        f'the hour before departure (history, for context)</div>', unsafe_allow_html=True)
            ui.rows({"Dwell time": f"{ctx['dwell']:.1f} h", "Utilisation": f"{ctx['utilization']:.0f}%",
                     "Vessels at anchor": f"{ctx['vessels_at_anchor']:.0f}", "Wind gust": f"{ctx['wind_gust']:.0f} km/h",
                     "Disruption · strike": f"{'yes' if ctx['disruption_active'] else 'no'} · {'yes' if ctx['labour_strike'] else 'no'}"})
    ui.help_popover("A separate model compares the origin port's operating state at departure with normal on-time "
                    "conditions. Scores above the threshold are flagged. **An anomaly is not a delay prediction**, and it "
                    "does not explain a delay.")

with ui.panel("reading"):
    ui.card_head("Reading both outputs together", "A plain-language note; it does not change either output")
    st.markdown(ui.interpret(predicted, bool(anomaly["anomaly_flag"]) if anomaly else None))  # None: no anomaly result

if result["warnings"]:
    with st.expander("Data notes for this shipment", icon=":material/info:"):
        for note in result["warnings"]:
            st.write(f"- {note}")

with st.expander("Shipment input context", icon=":material/description:"):
    st.dataframe(row[ml_risk.SHIPMENT_INPUT_COLUMNS], hide_index=True, width="stretch")
