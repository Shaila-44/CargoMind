"""Overview: the executive landing page over the synthetic history, replayed at a chosen hour (never live).

It answers three questions and nothing more: what is happening across the network (KPIs and the map), what needs
attention now (a ranked list), and what the models forecast (a compact next-hour dwell summary). Shipment tables,
trends and model detail live in Risk Prediction and Port Forecast.

Every number is read from the existing services at the replay hour: observed history (port_ops), XGBoost delay classes
and Isolation Forest anomaly flags (ml_risk), and the Transformer T_w / C_w forecast (dl_forecasting). Observed
conditions and model outputs are labelled separately and the two ML outputs are never combined.
"""
from __future__ import annotations

import logging
from datetime import timedelta

import pandas as pd
import pydeck as pdk
import streamlit as st

from components import ui
from services import dl_forecasting as dl
from services import ml_risk, port_ops

log = logging.getLogger("cargomind.overview")

WINDOW_DAYS = 7          # shipments: departures in the 7 days up to the replay hour
MAX_ITEMS = 5
SEVERITY = {"high": (0, "High"), "medium": (2, "Medium"), "low": (5, "Low")}
EVENT_LABELS = {"strike": "Labour strike", "weather": "Weather disruption", "equipment_failure": "Equipment failure",
                "congestion": "Yard congestion", "cyber": "Systems outage"}

ui.theme()
ui.navbar("Overview")

try:
    panel = port_ops.panel()
    first, last = panel["ts_hour"].min(), panel["ts_hour"].max()
    port_names = dict(zip(port_ops.ports()["port_id"], port_ops.ports()["port_name"]))
except Exception:
    log.exception("port history unavailable")
    ui.header("Network Operations", "Logistics intelligence research prototype, trained and evaluated on synthetic data only.")
    ui.error_box("Port history data is unavailable in this copy of the project, so the overview cannot be shown.")
    st.stop()
earliest = first + timedelta(days=WINDOW_DAYS)

# --- A. Heading and replay controls --------------------------------------------------------------------------------
h_left, h_port, h_day, h_hour = st.columns([2.7, 1.2, 0.85, 0.7], gap="small", vertical_alignment="bottom")
with h_port:
    focus = ui.port_picker(list(port_names), port_names, key="ov_port", required=True, label="Focus port")
as_of = ui.hour_picker(earliest, last, default=last, key="replay", label="Replay hour (UTC)", columns=(h_day, h_hour))
with h_left:
    st.markdown(f'<div class="cm-h1 cm-h1-xl">Network Operations</div>'
                f'<div class="cm-lede">Synthetic replay as of {as_of:%a %d %b %Y, %H:%M} UTC · not live data</div>',
                unsafe_allow_html=True)

snap = port_ops.snapshot(as_of)
prev = port_ops.snapshot(as_of - pd.Timedelta(hours=24))
events = port_ops.active_events(as_of)


@st.cache_data(show_spinner="Scoring network shipments...")
def _network_scored(end: pd.Timestamp) -> pd.DataFrame:
    """XGBoost class + raw probabilities and the Isolation Forest flag for every supported shipment departing in the
    WINDOW_DAYS up to `end`. The two outputs stay in separate columns and are never combined."""
    s = ml_risk.supported_shipments()
    dep = pd.to_datetime(s["departure_ts"], utc=True, format="ISO8601")
    s = s.assign(departure=dep)[(dep > end - pd.Timedelta(days=WINDOW_DAYS)) & (dep <= end)]
    if s.empty:
        return pd.DataFrame()
    results = {r["shipment_id"]: r for r in ml_risk.predict_shipments(s[ml_risk.SHIPMENT_INPUT_COLUMNS])}
    rows = []
    for _, row in s.iterrows():
        r = results[row["shipment_id"]]
        an = r["anomaly"]
        rows.append({"shipment_id": row["shipment_id"], "origin": row["origin_port_id"], "destination": row["destination_port_id"],
                     "departure": row["departure"], "predicted_class": r["classification"]["predicted_class"],
                     "p_Critical": r["classification"]["class_probabilities"]["Critical"],
                     "anomaly_flag": bool(an["anomaly_flag"]) if an else None})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Forecasting...")
def _port_forecast(port: str, end: pd.Timestamp) -> dict:
    return dl.forecast(dl.window_from_dataset(port, end), port_id=port).to_dict()


try:
    scored = _network_scored(as_of)
except Exception:
    log.exception("network scoring failed")
    scored = None


def _open_shipment(shipment_id: str, origin: str) -> None:
    st.session_state["cm_ml_origin_seed"] = origin
    st.session_state["cm_ml_shipment_seed"] = shipment_id
    st.switch_page("pages/1_ML_Intelligence.py")


def _open_forecast(port: str) -> None:
    st.session_state[ui.SHARED_PORT] = port
    st.switch_page("pages/2_Port_Forecast.py")


def _open_origin(port: str) -> None:
    st.session_state["cm_ml_origin_seed"] = port
    st.switch_page("pages/1_ML_Intelligence.py")


# --- B. Network KPIs (observed, at the replay hour) ----------------------------------------------------------------
def _change(now: float, before: float | None, unit: str = "", fmt: str = ".0f") -> str:
    """Change against the same hour one day earlier, shown neutrally (not judged good or bad)."""
    if before is None or pd.isna(before):
        return "no data 24 h earlier"
    return f"{now - before:+{fmt}}{unit} vs 24 h earlier"


def _prev(col: str, agg: str) -> float | None:
    return getattr(prev[col], agg)() if len(prev) else None


disrupted = (snap["disruption_active"] == 1) | (snap["labour_strike"] == 1)
disrupted_prev = int(((prev["disruption_active"] == 1) | (prev["labour_strike"] == 1)).sum()) if len(prev) else None
kpis = [(f"{len(snap)}", "", "Ports monitored", "all reporting"),
        (f"{int(disrupted.sum())}", "", "Ports with active disruptions", _change(int(disrupted.sum()), disrupted_prev)),
        (f"{snap['vessels_at_anchor'].sum():.0f}", "", "Vessels at anchor", _change(snap["vessels_at_anchor"].sum(), _prev("vessels_at_anchor", "sum"))),
        (f"{snap['dwell'].mean():.1f}", "h", "Average dwell", _change(snap["dwell"].mean(), _prev("dwell", "mean"), " h", ".1f")),
        (f"{snap['utilization'].mean():.0f}", "%", "Average utilisation", _change(snap["utilization"].mean(), _prev("utilization", "mean"), " pts"))]
st.markdown('<div class="cm-kpistrip">' + "".join(
    f'<div class="cell"><div class="lab">{lab}</div><div class="val">{v}<small>{u}</small></div><div class="chg">{chg}</div></div>'
    for v, u, lab, chg in kpis) + "</div>", unsafe_allow_html=True)

# --- C. Priority attention ------------------------------------------------------------------------------------------
items = []  # (rank, severity, identifier, title, explanation, action label, action)
for ev in events.itertuples(index=False):
    rank, _ = SEVERITY.get(ev.severity, (4, ev.severity.title()))
    outbound = 0 if scored is None or scored.empty else int(((scored["origin"] == ev.port_id) & (scored["departure"] >= ev.event_start)).sum())
    items.append((rank, ev.severity, ev.port_id,
                  f"{EVENT_LABELS.get(ev.event_type, ev.event_type.title())} at {port_names.get(ev.port_id, ev.port_id)}",
                  f"Observed. {ev.event_description}; {ev.affected_capacity_pct:.0f}% of capacity affected, in progress for "
                  f"{ev.hours_elapsed} h. {outbound} shipments have departed since it began.",
                  "Open forecast", (_open_forecast, (ev.port_id,))))
if scored is not None and not scored.empty:
    crit = scored[scored["predicted_class"] == "Critical"].sort_values("p_Critical", ascending=False)
    if len(crit):
        top = crit.iloc[0]
        items.append((1, "high", top["shipment_id"], f"{len(crit)} shipments predicted Critical",
                      f"XGBoost prediction. The highest raw P(Critical) is {top['p_Critical']:.2f}, for "
                      f"{top['origin']} → {top['destination']}. Probabilities are uncalibrated; no alert threshold is set.",
                      "Open shipment", (_open_shipment, (top["shipment_id"], top["origin"]))))
    flags = scored[scored["anomaly_flag"].fillna(False).astype(bool)]
    if len(flags):
        by_port = flags["origin"].value_counts()
        others = ", ".join(f"{p} ({n})" for p, n in by_port.iloc[1:3].items())
        items.append((3, "medium", by_port.index[0], f"{len(flags)} departures under unusual origin conditions",
                      f"Isolation Forest signal, not a delay prediction. Most flags come from {by_port.index[0]} "
                      f"({by_port.iloc[0]})" + (f", then {others}." if others else "."),
                      "Review", (_open_origin, (by_port.index[0],))))
items.sort(key=lambda x: x[0])

att_col, fc_col = st.columns([1.75, 1], gap="large")  # forecast card keeps its own (content) height

with att_col, ui.panel("attention"):
    ui.card_head("Priority attention", "What needs attention now, most severe first")
    if not items:
        st.markdown('<div class="cm-empty">No active incidents, Critical predictions or unusual-condition signals at this hour.</div>',
                    unsafe_allow_html=True)
    for i, (_, sev, ident, title, text, label, (fn, args)) in enumerate(items[:MAX_ITEMS]):
        c_text, c_btn = st.columns([2.7, 1], vertical_alignment="center", gap="medium")
        with c_text:
            st.markdown(f'<div class="cm-item"><div class="top"><span class="sev {sev}">{SEVERITY.get(sev, (0, sev))[1]}</span>'
                        f'<span class="id">{ident}</span></div><div class="t">{title}</div><div class="d">{text}</div></div>',
                        unsafe_allow_html=True)
        with c_btn:
            if st.button(label, key=f"att_{i}", width="stretch"):
                fn(*args)

# --- D. Next-hour dwell forecast (compact) -------------------------------------------------------------------------
with fc_col, ui.panel("forecast"):
    try:
        fc = _port_forecast(focus, as_of)
    except dl.InputValidationError:
        fc, fc_msg = None, "A forecast needs 12 hours of history before the replay hour for this port."
    except Exception:
        log.exception("forecast failed for %s", focus)
        fc, fc_msg = None, "The forecast models are unavailable right now."
    ui.card_head("Next-hour dwell", f"Model forecast for {port_names.get(focus, focus)}")
    if fc is None:
        st.info(fc_msg)
    else:
        primary = fc["model_name"] == dl.DEFAULT_MODEL
        st.markdown(f'<div class="cm-fcval">{float(fc["predicted_dwell_hours"]):.1f}<small>hours</small></div>'
                    f'<div class="cm-card-sub">for {pd.Timestamp(fc["target_hour_utc"]):%a %d %b, %H:%M} UTC</div>',
                    unsafe_allow_html=True)
        ui.rows({"Active model": "Transformer T_w" if primary else "C_w LSTM (fallback)",
                 "LSTM fallback": "Not used" if not fc["fallback_used"] else f"Used · {fc['fallback_reason'] or 'reason not stated'}"})
    st.page_link("pages/2_Port_Forecast.py", label="Open Port Forecast", icon=":material/arrow_outward:")

# --- E. Network overview map ---------------------------------------------------------------------------------------
with ui.panel("map"):
    ui.card_head("Port network", "Observed at the replay hour · circle size shows utilisation · amber marks a disruption or "
                                 f"strike · the outlined port is the focus port, {focus}")
    points = snap.assign(
        radius=70000 + snap["utilization"] * 3200,
        color=snap.apply(lambda r: [228, 160, 26, 230] if r["disruption_active"] or r["labour_strike"] else [67, 139, 232, 215], axis=1),
        line=snap["port_id"].map(lambda p: [22, 25, 31, 255] if p == focus else [255, 255, 255, 255]),
        width=snap["port_id"].map(lambda p: 3 if p == focus else 1.5),
        status=snap.apply(lambda r: "Labour strike" if r["labour_strike"] else "Disruption" if r["disruption_active"] else "Normal", axis=1),
        util=snap["utilization"].round(0), dwell_h=snap["dwell"].round(1), anchor=snap["vessels_at_anchor"].round(0))
    layer = pdk.Layer("ScatterplotLayer", data=points, get_position="[longitude, latitude]", get_radius="radius",
                      get_fill_color="color", pickable=True, stroked=True, get_line_color="line", get_line_width="width",
                      line_width_units="pixels", radius_min_pixels=8, radius_max_pixels=24)
    st.pydeck_chart(pdk.Deck(layers=[layer], map_provider="carto", map_style="light",
                             initial_view_state=pdk.ViewState(latitude=25, longitude=40, zoom=1.25),
                             tooltip={"text": "{port_id} · {port_name}\n{status}\nUtilisation {util}% · dwell {dwell_h} h\n"
                                              "Vessels at anchor {anchor}"}),
                    use_container_width=True, height=480, key=f"map_{focus}_{as_of:%Y%m%d%H}")
