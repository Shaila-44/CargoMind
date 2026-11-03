"""Overview: port operations dashboard over the synthetic history, replayed at a chosen hour (never live).

Row order follows what a user needs first: headline numbers, then the three model outputs (kept as separate panels),
then the observed operations context.
"""
import logging
from datetime import timedelta

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

from components import ui
from services import dl_forecasting as dl
from services import ml_risk, port_ops

log = logging.getLogger("cargomind.overview")

ui.theme()
ui.demo_banner()

try:
    panel = port_ops.panel()
    first, last = panel["ts_hour"].min(), panel["ts_hour"].max()
    port_names = dict(zip(port_ops.ports()["port_id"], port_ops.ports()["port_name"]))
except Exception:
    log.exception("port history unavailable")
    ui.header("CargoMind", "Logistics intelligence research prototype, trained and evaluated on synthetic data only.")
    ui.error_box("Port history data is unavailable in this copy of the project, so the overview cannot be shown.")
    st.stop()

with st.sidebar:
    st.header("Replay settings")
    focus = st.selectbox("Focus port", list(port_names), format_func=lambda p: f"{p} · {port_names[p]}")
    as_of = ui.hour_picker(first + timedelta(days=7), last, default=last, key="replay", label="Replay hour (UTC)")
    st.caption(f"Synthetic history {first:%Y-%m-%d} to {last:%Y-%m-%d} UTC. Replays show the data as it stood at that hour.")

snap = port_ops.snapshot(as_of)
focus_row = snap[snap["port_id"] == focus].iloc[0]

ui.titlebar("Port Operations Dashboard", f"Synthetic replay as of {as_of:%Y-%m-%d %H:%M} UTC · not live data")
ui.context_bar({"Focus port": f"{focus} · {port_names[focus]}", "Replay hour": f"{as_of:%a %d %b %Y, %H:%M} UTC",
                "Shipment window": "departures in the 7 days to the replay hour"})

# --- KPI row -------------------------------------------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5, gap="small")
with k1:
    ui.kpi("Ports monitored", f"{len(snap)}")
with k2:
    ui.kpi("Vessels at anchor", f"{snap['vessels_at_anchor'].sum():.0f}")
with k3:
    ui.kpi("Average dwell time", f"{snap['dwell'].mean():.1f}", "hours")
with k4:
    ui.kpi("Average utilisation", f"{snap['utilization'].mean():.0f}", "%")
with k5:
    ui.kpi("Ports with disruptions", f"{int(((snap['disruption_active'] == 1) | (snap['labour_strike'] == 1)).sum())}")

# --- Model outputs (classifier, detector and forecaster shown separately, never combined) --------------------------


@st.cache_data(show_spinner="Scoring shipments...")
def _scored(port: str, end: pd.Timestamp) -> pd.DataFrame:
    """XGBoost class and Isolation Forest flag for shipments departing `port` in the 7 days up to `end`."""
    s = ml_risk.supported_shipments()
    dep = pd.to_datetime(s["departure_ts"], utc=True, format="ISO8601")
    s = s[(s["origin_port_id"] == port) & (dep > end - pd.Timedelta(days=7)) & (dep <= end)]
    if s.empty:
        return pd.DataFrame(columns=["shipment_id", "predicted_class", "anomaly_flag"])
    results = ml_risk.predict_shipments(s)
    return pd.DataFrame({"shipment_id": [r["shipment_id"] for r in results],
                         "predicted_class": [r["classification"]["predicted_class"] for r in results],
                         "anomaly_flag": [bool(r["anomaly"]["anomaly_flag"]) if r["anomaly"] else None for r in results]})


@st.cache_data(show_spinner="Forecasting...")
def _port_forecast(port: str, end: pd.Timestamp) -> dict:
    return dl.forecast(dl.window_from_dataset(port, end), port_id=port).to_dict()


st.markdown('<div class="cm-eyebrow">Model outputs · three separate signals</div>', unsafe_allow_html=True)
o1, o2, o3 = st.columns(3, gap="medium")
try:
    scored = _scored(focus, as_of)
except Exception:
    log.exception("batch scoring failed for %s", focus)
    scored = None
NO_SHIPMENTS = "No supported shipments departed this port in the 7 days to the replay hour."

with o1, st.container(border=True):
    ui.panel_title("Delay outlook", "XGBoost · primary delay classifier")
    if scored is None:
        ui.error_box("The delay classifier is unavailable right now.")
    elif scored.empty:
        st.info(NO_SHIPMENTS)
    else:
        counts = scored["predicted_class"].value_counts().reindex(ui.CLASS_ORDER, fill_value=0).rename_axis("class")
        chart = alt.Chart(counts.reset_index(name="shipments")).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("shipments:Q", title="Shipments"), y=alt.Y("class:N", sort=ui.CLASS_ORDER, title=None),
            color=alt.Color("class:N", legend=None, scale=alt.Scale(domain=ui.CLASS_ORDER,
                                                                     range=[ui.CLASS_COLORS[c] for c in ui.CLASS_ORDER])),
            tooltip=["class:N", "shipments:Q"])
        labels = chart.mark_text(align="left", dx=4, color=ui.palette()["text"]).encode(text="shipments:Q",
                                                                                         color=alt.value(ui.palette()["text"]))
        st.altair_chart(ui.style((chart + labels).properties(height=140)), use_container_width=True)
        st.caption(f"{len(scored)} shipments. Class = highest raw (uncalibrated) probability.")
    ui.help_popover("Each shipment gets the delay class with the highest **raw** probability from XGBoost. The "
                    "probabilities are not calibrated, and there is no alert threshold. Open **ML Intelligence** to see "
                    "the three probabilities for any shipment.")

with o2, st.container(border=True):
    ui.panel_title("Unusual conditions at origin", "Isolation Forest · anomaly detector")
    if scored is None:
        ui.error_box("The anomaly detector is unavailable right now.")
    elif scored.empty:
        st.info(NO_SHIPMENTS)
    else:
        flagged = int(scored["anomaly_flag"].fillna(False).sum())
        ui.rows({"Shipments flagged": f"{flagged}", "Not flagged": f"{len(scored) - flagged}",
                 "Share flagged": f"{flagged / len(scored):.0%}"})
    ui.help_popover("A separate model asks whether the origin port's operating state looked unusual compared with "
                    "normal on-time conditions. **An anomaly is not a delay prediction**: delayed shipments can have "
                    "normal conditions, and unusual conditions do not prove a delay.")

with o3, st.container(border=True):
    try:
        fc = _port_forecast(focus, as_of)
    except dl.InputValidationError:
        ui.panel_title(f"Next-hour dwell · {focus}", "Forecast")
        st.info("A forecast needs 12 hours of history before the replay hour for this port.")
    except Exception:
        log.exception("forecast failed for %s", focus)
        ui.panel_title(f"Next-hour dwell · {focus}", "Forecast")
        ui.error_box("The forecast models are unavailable right now.")
    else:
        primary = fc["model_name"] == dl.DEFAULT_MODEL
        ui.panel_title(f"Next-hour dwell · {focus}",
                       "Transformer T_w · primary forecaster" if primary else "C_w LSTM · automatic fallback")
        ui.rows({"Forecast for next hour": f"{fc['predicted_dwell_hours']:.1f} h", "Observed at replay hour": f"{focus_row['dwell']:.1f} h"})
        if fc["fallback_used"]:
            st.warning(f"Fallback used: {fc['fallback_reason'] or 'reason not stated'}.", icon=":material/swap_horiz:")
    st.page_link("pages/2_Port_Forecast.py", label="Open full forecast", icon=":material/timeline:")

# --- Operations context (observed synthetic history) ---------------------------------------------------------------
st.markdown('<div class="cm-eyebrow">Port operations · observed synthetic history</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 1, 1.15], gap="medium")
trend = port_ops.daily_trend(as_of, days=7, port_id=focus)
x_day = alt.X("day:T", title=None, scale=alt.Scale(type="utc"), axis=alt.Axis(format="%a %d"))  # UTC days

with c1, st.container(border=True):
    ui.panel_title(f"Dwell time · {focus}", "Daily mean, last 7 days")
    dwell = alt.Chart(trend).mark_line(color=ui.palette()["accent"], strokeWidth=3,
                                       point=alt.OverlayMarkDef(color=ui.palette()["accent"], size=50)).encode(
        x=x_day, y=alt.Y("dwell:Q", title="Hours", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("day:T", timeUnit="utcyearmonthdate", title="Day (UTC)"), alt.Tooltip("dwell:Q", format=".1f", title="Dwell (h)")])
    st.altair_chart(ui.style(dwell.properties(height=190)), use_container_width=True)

with c2, st.container(border=True):
    ui.panel_title(f"Vessels at anchor · {focus}", "Daily mean, last 7 days")
    anchored = alt.Chart(trend).mark_bar(color=ui.FLAG_COLOR, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=x_day, y=alt.Y("anchored:Q", title="Vessels"),
        tooltip=[alt.Tooltip("day:T", timeUnit="utcyearmonthdate", title="Day (UTC)"), alt.Tooltip("anchored:Q", format=".1f", title="At anchor")])
    st.altair_chart(ui.style(anchored.properties(height=190)), use_container_width=True)

with c3, st.container(border=True):
    ui.panel_title("Berth utilisation", "At the replay hour")
    g1, g2 = st.columns(2)
    with g1:
        st.altair_chart(ui.gauge(snap["utilization"].mean(), "All ports (mean)", ui.palette()["accent"]),
                        use_container_width=True)
    with g2:
        st.altair_chart(ui.gauge(focus_row["utilization"], focus, "#16a34a"), use_container_width=True)
    busiest = snap.sort_values("utilization", ascending=False).iloc[0]
    st.caption(f"Busiest: {busiest['port_id']} ({busiest['utilization']:.0f}%).")

d1, d2, d3 = st.columns([1, 1.7, 1.2], gap="medium")

with d1, st.container(border=True):
    ui.panel_title(f"Port performance · {focus}", "At the replay hour")
    ui.rows({"Dwell time": f"{focus_row['dwell']:.1f} h", "Utilisation": f"{focus_row['utilization']:.0f}%",
             "Vessels at anchor": f"{focus_row['vessels_at_anchor']:.0f}", "Wind gust": f"{focus_row['wind_gust']:.0f} km/h",
             "Capacity": f"{focus_row['capacity_teu_per_hour']:.0f} TEU/h"})

with d2, st.container(border=True):
    ui.panel_title("Port map", "Circle size: utilisation · amber: disruption or strike · blue: normal · hover for details")
    points = snap.assign(
        radius=40000 + snap["utilization"] * 2500,
        color=snap.apply(lambda r: [217, 119, 6, 220] if r["disruption_active"] or r["labour_strike"] else [56, 189, 248, 220],
                         axis=1),
        util=snap["utilization"].round(0), dwell_h=snap["dwell"].round(1))
    layer = pdk.Layer("ScatterplotLayer", data=points, get_position="[longitude, latitude]", get_radius="radius",
                      get_fill_color="color", pickable=True, stroked=True, get_line_color=[255, 255, 255],
                      line_width_min_pixels=1, radius_min_pixels=6, radius_max_pixels=18)
    # No text labels: ports cluster (North Europe, India) and labels overlap; hovering a circle shows its name instead.
    st.pydeck_chart(pdk.Deck(layers=[layer], map_provider="carto", map_style=ui.palette()["map"],
                             initial_view_state=pdk.ViewState(latitude=22, longitude=10, zoom=0.6),
                             tooltip={"text": "{port_id} · {port_name}\nUtilisation {util}%\nDwell {dwell_h} h"}), height=300)

with d3, st.container(border=True):
    ui.panel_title("Disruptions", "At the replay hour, all ports")
    affected = snap.loc[(snap["disruption_active"] == 1) | (snap["labour_strike"] == 1), "port_id"].tolist()
    ui.rows({"Active disruptions": f"{int(snap['disruption_active'].sum())}", "Labour strikes": f"{int(snap['labour_strike'].sum())}",
             "Highest wind gust": f"{snap['wind_gust'].max():.0f} km/h", "Affected": ", ".join(affected) if affected else "None"})
    st.write("")
    ui.panel_title(f"Recorded outcomes · from {focus}", "Arrivals in the last 7 days · history, not predictions")
    outcomes = port_ops.shipment_outcomes(as_of, days=7, origin_port_id=focus)
    total = int(outcomes.sum())
    ui.rows({**{c: f"{outcomes[c]}" for c in ui.CLASS_ORDER},
             "On-time share": f"{outcomes['On-Time'] / total:.0%}" if total else "–"})

# --- Modules -------------------------------------------------------------------------------------------------------
st.write("")
m1, m2, m3 = st.columns(3, gap="medium")
with m1, st.container(border=True):
    ui.panel_title("ML Intelligence", "Delay outlook and unusual conditions for one shipment, side by side")
    st.page_link("pages/1_ML_Intelligence.py", label="Open ML Intelligence", icon=":material/insights:")
with m2, st.container(border=True):
    ui.panel_title("Port Forecast", "Next-hour dwell time from the last 12 observed hours")
    st.page_link("pages/2_Port_Forecast.py", label="Open Port Forecast", icon=":material/timeline:")
with m3, st.container(border=True):
    ui.panel_title("Methodology", "How the models work and the limits of this demo")
    st.page_link("pages/3_Methodology.py", label="Read the methodology", icon=":material/menu_book:")
