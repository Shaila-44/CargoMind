"""Overview: port operations dashboard over the synthetic history, replayed at a chosen hour (never live).

Composition follows the reference layout: header with replay controls, then a wide shipment panel (XGBoost) beside the
next-hour forecast (Transformer T_w / C_w fallback), then trends and network numbers beside the independent anomaly
panel (Isolation Forest) and the disruption list, then port performance and the map. The three model outputs stay in
separate panels and are never combined.
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
ui.navbar("Overview")

try:
    panel = port_ops.panel()
    first, last = panel["ts_hour"].min(), panel["ts_hour"].max()
    port_names = dict(zip(port_ops.ports()["port_id"], port_ops.ports()["port_name"]))
except Exception:
    log.exception("port history unavailable")
    ui.header("Port operations overview", "Logistics intelligence research prototype, trained and evaluated on synthetic data only.")
    ui.demo_banner()
    ui.error_box("Port history data is unavailable in this copy of the project, so the overview cannot be shown.")
    st.stop()

# --- Header: title left, replay controls right ---------------------------------------------------------------------
h_left, h_right = st.columns([1.05, 1], gap="large", vertical_alignment="bottom")
with h_right:
    c_port, c_day, c_hour = st.columns([1.35, 1, 0.9], gap="small")
    with c_port:
        focus = ui.port_picker(list(port_names), port_names, key="ov_port", required=True, label="Focus port")
    as_of = ui.hour_picker(first + timedelta(days=7), last, default=last, key="replay", label="Replay hour (UTC)",
                           columns=(c_day, c_hour))
with h_left:
    ui.header("Port operations overview",
              f"Synthetic replay as of {as_of:%a %d %b %Y, %H:%M} UTC · not live data")
ui.demo_banner()

snap = port_ops.snapshot(as_of)
focus_row = snap[snap["port_id"] == focus].iloc[0]


@st.cache_data(show_spinner="Scoring shipments...")
def _scored(port: str, end: pd.Timestamp) -> pd.DataFrame:
    """XGBoost class + raw probabilities and the Isolation Forest flag + score for shipments departing `port` in the
    7 days up to `end`. The two outputs are kept in separate columns and never combined."""
    s = ml_risk.supported_shipments()
    dep = pd.to_datetime(s["departure_ts"], utc=True, format="ISO8601")
    s = s.assign(departure=dep)[(s["origin_port_id"] == port) & (dep > end - pd.Timedelta(days=7)) & (dep <= end)]
    cols = ["shipment_id", "destination_port_id", "departure", "predicted_class", *[f"p_{c}" for c in ui.CLASS_ORDER],
            "anomaly_flag", "anomaly_score", "threshold"]
    if s.empty:
        return pd.DataFrame(columns=cols)
    results = {r["shipment_id"]: r for r in ml_risk.predict_shipments(s[ml_risk.SHIPMENT_INPUT_COLUMNS])}
    out = []
    for _, row in s.iterrows():
        r = results[row["shipment_id"]]
        probs, an = r["classification"]["class_probabilities"], r["anomaly"]
        out.append({"shipment_id": row["shipment_id"], "destination_port_id": row["destination_port_id"],
                    "departure": row["departure"], "predicted_class": r["classification"]["predicted_class"],
                    **{f"p_{c}": probs[c] for c in ui.CLASS_ORDER},
                    "anomaly_flag": bool(an["anomaly_flag"]) if an else None,
                    "anomaly_score": an["anomaly_score"] if an else None, "threshold": an["threshold"] if an else None})
    return pd.DataFrame(out, columns=cols)


@st.cache_data(show_spinner="Forecasting...")
def _port_forecast(port: str, end: pd.Timestamp) -> dict:
    return dl.forecast(dl.window_from_dataset(port, end), port_id=port).to_dict()


try:
    scored = _scored(focus, as_of)
except Exception:
    log.exception("batch scoring failed for %s", focus)
    scored = None
NO_SHIPMENTS = "No supported shipments departed this port in the 7 days to the replay hour."

# --- Row 1: delay outlook (XGBoost) | next-hour forecast (DL) -------------------------------------------------------
with ui.row("models"):
    left, right = st.columns([1.9, 1], gap="medium")

with left, ui.panel("delay"):
    head_l, head_r = st.columns([1.3, 1], vertical_alignment="center")
    with head_l:
        ui.card_head("Shipment delay outlook", f"XGBoost · departures from {focus} in the 7 days to the replay hour")
    with head_r:
        view = st.segmented_control("Filter by predicted class", ["All", *reversed(ui.CLASS_ORDER)], default="All",
                                    key="ov_class_filter", label_visibility="collapsed")
    if scored is None:
        ui.error_box("The delay classifier is unavailable right now.")
    elif scored.empty:
        st.info(NO_SHIPMENTS)
    else:
        shown = scored if view in (None, "All") else scored[scored["predicted_class"] == view]
        shown = shown.sort_values("p_Critical", ascending=False).head(6)
        head = ('<div class="cm-list-head"><span>Shipment</span><span>Route</span><span>Predicted class</span>'
                '<span>Raw P(predicted class) · uncalibrated</span><span>Departs (UTC)</span></div>')
        body = ""
        for _, r in shown.iterrows():
            prob = r[f"p_{r['predicted_class']}"]
            body += (f'<div class="cm-list-row"><span class="id">{r["shipment_id"]}</span>'
                     f'<span>{focus} → {r["destination_port_id"]}<div class="sub">{port_names.get(r["destination_port_id"], "")}</div></span>'
                     f'<span>{ui.class_badge(r["predicted_class"])}</span>'
                     f'<span class="cm-pbar"><span class="num">{prob:.2f}</span>{ui.seg_bar(prob, ui.CLASS_COLORS[r["predicted_class"]])}</span>'
                     f'<span class="sub" style="font-size:0.8rem;color:#4f5b69">{r["departure"]:%d %b %H:%M}</span></div>')
        st.markdown(head + (body or '<div class="cm-list-row">No shipments in this class.</div>'), unsafe_allow_html=True)
        counts = scored["predicted_class"].value_counts().reindex(ui.CLASS_ORDER, fill_value=0)
        f1, f2 = st.columns([2.2, 1], vertical_alignment="center")
        with f1:
            st.markdown(" ".join(ui.class_badge(c) + f'<span style="font-size:0.82rem;color:#4f5b69;margin:0 0.6rem 0 0.3rem">'
                                 f'{counts[c]}</span>' for c in ui.CLASS_ORDER), unsafe_allow_html=True)
            st.caption(f"{len(scored)} shipments; the 6 with the highest raw P(Critical) are listed. Class = highest raw "
                       "(uncalibrated) probability; no alert threshold is set.")
        with f2:
            if st.button("Open in ML Intelligence", icon=":material/arrow_outward:", key="ov_open_ml", width="stretch"):
                st.session_state["cm_ml_origin_seed"] = focus
                st.switch_page("pages/1_ML_Intelligence.py")

with right, ui.panel("forecast"):
    try:
        fc = _port_forecast(focus, as_of)
    except dl.InputValidationError:
        fc, fc_msg = None, "A forecast needs 12 hours of history before the replay hour for this port."
    except Exception:
        log.exception("forecast failed for %s", focus)
        fc, fc_msg = None, "The forecast models are unavailable right now."
    if fc is None:
        ui.card_head(f"Next-hour dwell · {focus}", "Forecast")
        st.info(fc_msg)
    else:
        primary = fc["model_name"] == dl.DEFAULT_MODEL
        model_badge = ui.badge("Transformer T_w · primary", "navy") if primary else ui.badge("C_w LSTM · fallback", "orange")
        ui.card_head(f"Next-hour dwell · {focus}", f"Forecast for {pd.Timestamp(fc['target_hour_utc']):%d %b %H:%M} UTC")
        delta = fc["predicted_dwell_hours"] - focus_row["dwell"]
        ui.big(f"{fc['predicted_dwell_hours']:.1f}", "h", ui.badge(f"{delta:+.1f} h vs now", "blue"))
        week = panel[(panel["port_id"] == focus) & (panel["ts_hour"] > as_of - pd.Timedelta(days=7)) & (panel["ts_hour"] <= as_of)]
        lo, hi = float(week["dwell"].min()), float(week["dwell"].max())
        lo, hi = min(lo, fc["predicted_dwell_hours"]), max(hi, fc["predicted_dwell_hours"])
        ui.meter(lo, hi, fc["predicted_dwell_hours"], mark=float(focus_row["dwell"]),
                 lo_label=f"7-day low {lo:.1f} h", hi_label=f"high {hi:.1f} h")
        st.caption("Knob: forecast · thin mark: observed at the replay hour.")
        t1, t2 = st.columns(2, gap="small")
        with t1, ui.tile("fc_obs"):
            st.markdown(ui.kpi("Observed now", f"{focus_row['dwell']:.1f}", "h"), unsafe_allow_html=True)
        with t2, ui.tile("fc_anchor"):
            st.markdown(ui.kpi("Vessels at anchor", f"{focus_row['vessels_at_anchor']:.0f}"), unsafe_allow_html=True)
        ui.rows({"Active model": model_badge,
                 "Fallback": ui.badge("Not used", "green") if not fc["fallback_used"] else ui.badge("Used", "orange"),
                 "Input window": f"{pd.Timestamp(fc['window_start_utc']):%d %b %H:%M} – {pd.Timestamp(fc['window_end_utc']):%H:%M}"})
        if fc["fallback_used"]:
            ui.note(f"Fallback reason: {fc['fallback_reason'] or 'not stated'}.", warn=True)
    st.page_link("pages/2_Port_Forecast.py", label="Open full forecast", icon=":material/arrow_outward:")

# --- Row 2: trend | network pulse | anomaly + disruptions -----------------------------------------------------------
trend = port_ops.daily_trend(as_of, days=7, port_id=focus)
x_day = alt.X("day:T", title=None, scale=alt.Scale(type="utc"), axis=alt.Axis(format="%a %d", grid=False))  # UTC days

with ui.row("ops"):
    c1, c2, c3 = st.columns([1, 1, 1], gap="medium")
with c1, ui.panel("trend"):
    ui.card_head(f"Dwell trend · {focus}", "Daily mean, last 7 days · observed")
    latest = trend["dwell"].iloc[-1] if len(trend) else float("nan")
    ui.big(f"{latest:.1f}", "h", ui.badge("latest day", "neutral"))
    st.altair_chart(ui.area_line(trend, x_day, "dwell", "Hours",
                                 [alt.Tooltip("day:T", timeUnit="utcyearmonthdate", title="Day (UTC)"),
                                  alt.Tooltip("dwell:Q", format=".1f", title="Dwell (h)")], height=180),
                    use_container_width=True, key=f"ov_trend_{focus}_{as_of:%Y%m%d%H}")
with c2, ui.panel("network"):
    ui.card_head("Network pulse", "All ports at the replay hour · observed")
    disrupted = int(((snap["disruption_active"] == 1) | (snap["labour_strike"] == 1)).sum())
    ui.kpis([ui.kpi("Ports monitored", f"{len(snap)}", accent=True),
             ui.kpi("Vessels at anchor", f"{snap['vessels_at_anchor'].sum():.0f}", accent=True),
             ui.kpi("Average dwell", f"{snap['dwell'].mean():.1f}", "h", accent=True)])
    hourly = panel[(panel["ts_hour"] > as_of - pd.Timedelta(hours=48)) & (panel["ts_hour"] <= as_of)] \
        .groupby("ts_hour", as_index=False)["vessels_at_anchor"].sum()
    st.altair_chart(ui.thin_bars(hourly, alt.X("ts_hour:T", title=None, scale=alt.Scale(type="utc"),
                                                axis=alt.Axis(format="%d %b %Hh", tickCount=4, grid=False)),
                                 "vessels_at_anchor", None,
                                 [alt.Tooltip("ts_hour:T", timeUnit="utcyearmonthdatehoursminutes", title="Hour (UTC)"),
                                  alt.Tooltip("vessels_at_anchor:Q", title="At anchor (all ports)")], height=120),
                    use_container_width=True, key=f"ov_anchor_{as_of:%Y%m%d%H}")
    st.caption("Vessels at anchor across all ports, hourly, last 48 h.")
    st.markdown(ui.badge(f"Average utilisation {snap['utilization'].mean():.0f}%", "blue") + " "
                + ui.badge(f"Ports with disruptions {disrupted}", "orange" if disrupted else "green"),
                unsafe_allow_html=True)
with c3, ui.panel("anomaly"):
    if scored is None:
        ui.card_head("Unusual conditions at origin", "Isolation Forest · anomaly detector")
        ui.error_box("The anomaly detector is unavailable right now.")
    elif scored.empty:
        ui.card_head("Unusual conditions at origin", "Isolation Forest · anomaly detector")
        st.info(NO_SHIPMENTS)
    else:
        flags = scored["anomaly_flag"].fillna(False).astype(bool)
        flagged = int(flags.sum())
        ui.card_head("Unusual conditions at origin", f"Isolation Forest · departures from {focus}",
                     ui.badge("Flagged", "orange", ui.FLAG_COLOR) if flagged else ui.badge("None flagged", "green"))
        ui.big(f"{flagged}", f"of {len(scored)} shipments flagged ({flagged / len(scored):.0%})")
        scores = scored.dropna(subset=["anomaly_score"]).sort_values("departure")
        if len(scores):
            bars = alt.Chart(scores).mark_bar(width=3, cornerRadius=2).encode(
                x=alt.X("departure:T", title=None, axis=None, scale=alt.Scale(type="utc")),
                y=alt.Y("anomaly_score:Q", title=None, axis=None, scale=alt.Scale(zero=False)),
                color=alt.condition("datum.anomaly_flag", alt.value(ui.FLAG_COLOR), alt.value(ui.BLUE_SOFT)),
                tooltip=[alt.Tooltip("shipment_id:N", title="Shipment"), alt.Tooltip("anomaly_score:Q", format=".3f", title="Score")])
            rule = alt.Chart(pd.DataFrame({"t": [float(scores["threshold"].iloc[0])]})).mark_rule(
                strokeDash=[3, 3], color=ui.INK2).encode(y="t:Q")
            st.altair_chart(ui.style((bars + rule).properties(height=110)), use_container_width=True,
                            key=f"ov_anomaly_{focus}_{as_of:%Y%m%d%H}")
        st.caption("Anomaly score per shipment, by departure; dashed line = flag threshold. An anomaly is not a delay "
                   "prediction, and this panel is independent of the delay outlook.")
    ui.help_popover("A separate model asks whether the origin port's operating state looked unusual compared with "
                    "normal on-time conditions. **An anomaly is not a delay prediction**: delayed shipments can have "
                    "normal conditions, and unusual conditions do not prove a delay.")

# --- Row 3: port performance | map ----------------------------------------------------------------------------------
with ui.row("ports"):
    p1, p2 = st.columns([1.15, 1], gap="medium")

with p1, ui.panel("performance"):
    ui.card_head("Port performance", "All ports at the replay hour · observed synthetic history")
    head = ('<div class="cm-port-row head"><span>Port</span><span>Dwell</span><span>Utilisation</span>'
            '<span>At anchor</span><span>Status</span></div>')
    body = ""
    for r in snap.sort_values("dwell", ascending=False).itertuples(index=False):
        status = (ui.badge("Strike", "red") if r.labour_strike else ui.badge("Disruption", "orange") if r.disruption_active
                  else ui.badge("Normal", "green"))
        body += (f'<div class="cm-port-row{" focus" if r.port_id == focus else ""}"><span><b>{r.port_id}</b> '
                 f'<span style="color:#8a94a3">{r.port_name}</span></span><span class="num">{r.dwell:.1f} h</span>'
                 f'<span class="num">{r.utilization:.0f}%</span><span class="num">{r.vessels_at_anchor:.0f}</span><span>{status}</span></div>')
    st.markdown(head + body, unsafe_allow_html=True)
    outcomes = port_ops.shipment_outcomes(as_of, days=7, origin_port_id=focus)
    total = int(outcomes.sum())
    st.markdown(f'<div class="cm-card-sub" style="margin-top:0.7rem">Recorded outcomes · arrivals from {focus}, last 7 days '
                f'(history, not predictions)</div>', unsafe_allow_html=True)
    st.markdown(" ".join(ui.class_badge(c) + f'<span style="font-size:0.82rem;color:#4f5b69;margin:0 0.6rem 0 0.3rem">'
                         f'{outcomes[c]}</span>' for c in ui.CLASS_ORDER)
                + (ui.badge(f"On-time share {outcomes['On-Time'] / total:.0%}", "neutral") if total else ""),
                unsafe_allow_html=True)

with p2, ui.panel("attention"):
    ui.card_head("Disruptions", "At the replay hour, all ports · observed")
    items = []
    for r in snap.itertuples(index=False):
        if r.disruption_active:
            items.append((r.port_id, r.port_name, "Disruption active", ui.badge("Disruption", "orange", ui.FLAG_COLOR)))
        if r.labour_strike:
            items.append((r.port_id, r.port_name, "Labour strike", ui.badge("Strike", "red", ui.CLASS_COLORS["Critical"])))
    gust = snap.sort_values("wind_gust", ascending=False).iloc[0]
    ui.rows({**{f'<b title="{name}">{pid}</b>{what}': b for pid, name, what, b in items[:5]},
             **({} if items else {"Active disruptions or strikes": ui.badge("None", "green")}),
             "Highest wind gust": f"{gust['wind_gust']:.0f} km/h · {gust['port_id']}"})

with p2, ui.panel("map"):
    ui.card_head("Port map", "Circle size: utilisation · orange: disruption or strike · blue: normal · hover for details")
    points = snap.assign(
        radius=40000 + snap["utilization"] * 2500,
        color=snap.apply(lambda r: [224, 132, 26, 220] if r["disruption_active"] or r["labour_strike"] else [47, 125, 225, 210],
                         axis=1),
        util=snap["utilization"].round(0), dwell_h=snap["dwell"].round(1))
    layer = pdk.Layer("ScatterplotLayer", data=points, get_position="[longitude, latitude]", get_radius="radius",
                      get_fill_color="color", pickable=True, stroked=True, get_line_color=[255, 255, 255],
                      line_width_min_pixels=2, radius_min_pixels=6, radius_max_pixels=18)
    # No text labels: ports cluster (North Europe, India) and labels overlap; hovering a circle shows its name instead.
    st.pydeck_chart(pdk.Deck(layers=[layer], map_provider="carto", map_style="light",
                             initial_view_state=pdk.ViewState(latitude=22, longitude=10, zoom=0.6),
                             tooltip={"text": "{port_id} · {port_name}\nUtilisation {util}%\nDwell {dwell_h} h"}), height=300)
