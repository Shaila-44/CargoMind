"""Port Forecast: next-hour dwell-time forecast from a validated 12-hour window of the synthetic port history."""
import logging
from datetime import timedelta

import altair as alt
import pandas as pd
import streamlit as st

from components import ui
from services import dl_forecasting as dl
from services import port_ops

log = logging.getLogger("cargomind.dl_page")

MODEL_LABELS = {
    dl.DEFAULT_MODEL: "Transformer T_w (primary)",
    dl.FALLBACK_MODEL: "C_w LSTM (automatic fallback)",
}
HISTORY, FORECAST = "Observed history", "Forecast (model estimate)"

st.set_page_config(page_title="Port Forecast | CargoMind", layout="wide")
ui.theme()
ui.navbar("Port Forecast")


@st.cache_data(show_spinner=False)
def _ports() -> list[str]:
    return dl.supported_ports()


@st.cache_data(show_spinner=False)
def _coverage() -> tuple[pd.Timestamp, pd.Timestamp]:
    return dl.coverage()


@st.cache_data(show_spinner="Forecasting...")
def _forecast(port: str, window_end: str, benchmark: bool) -> tuple[pd.DataFrame, dict]:
    window = dl.window_from_dataset(port, window_end)
    result = dl.forecast(window, port_id=port, model="ewma" if benchmark else dl.DEFAULT_MODEL)
    return window, result.to_dict()


try:
    ports = _ports()
    first, last = _coverage()
except Exception:
    log.exception("port history unavailable")
    ui.header("Port Forecast", "Next-hour port dwell time in hours, from the last 12 observed hours.")
    ui.error_box("Port history data is unavailable in this copy of the project. Forecasts cannot be produced.")
    st.stop()
try:
    names = port_ops.port_names()
except Exception:
    names = {}

# --- Header: title left, port / hour controls right ----------------------------------------------------------------
h_left, h_right = st.columns([1, 1.15], gap="large", vertical_alignment="bottom")
with h_right:
    c_port, c_day, c_hour = st.columns([1.35, 1, 0.9], gap="small")
    with c_port:
        port = ui.port_picker(ports, names, key="fc_port", required=False, label="Port", placeholder="Choose a port")
    earliest = first + timedelta(hours=11)  # the window needs 12 hours, so the first end hour is 11 h after the start
    window_end = ui.hour_picker(earliest, last, default=last, key="fc", label="Last observed hour (UTC)",
                                columns=(c_day, c_hour))
    benchmark = st.toggle("Show EWMA benchmark instead (reference only)", value=False,
                          help="EWMA is a simple benchmark for comparison. It is not the fallback.")
with h_left:
    ui.header("Port Forecast", "Next-hour port dwell time in hours, from the last 12 observed hours.")

if port is None:
    st.info("Select a port above to see its next-hour forecast. "
            f"History available {first:%Y-%m-%d %H:%M} to {last:%Y-%m-%d %H:%M} UTC; no live feed.")
    st.stop()

end_iso = window_end.isoformat()
try:
    window, result = _forecast(port, end_iso, benchmark)
except dl.InputValidationError as err:
    log.warning("forecast input rejected: %s", err)
    ui.error_box("This port and hour cannot be forecast. Choose an hour inside the available history for this port.")
    st.stop()
except (dl.ModelLoadError, dl.PredictionError):
    log.exception("forecast model failed for %s at %s", port, end_iso)
    ui.error_box("Neither the primary model nor its fallback could produce a forecast. Try again later.")
    st.stop()
except Exception:
    log.exception("unexpected forecast error for %s at %s", port, end_iso)
    ui.error_box("Something went wrong while producing the forecast. Try again later.")
    st.stop()

window = window.assign(ts_hour=pd.to_datetime(window["ts_hour"], utc=True))
target = pd.Timestamp(result["target_hour_utc"])
forecast_hours = float(result["predicted_dwell_hours"])
observed_at_target = dl.actual_dwell(port, target)
model_name = "EWMA benchmark (reference only, not a fallback)" if benchmark else \
    MODEL_LABELS.get(result["model_name"], result["model_name"])
last_obs = float(window["dwell"].iloc[-1])
if benchmark:
    FORECAST = "Benchmark estimate (EWMA)"

# --- Hero: forecast chart | forecast summary -----------------------------------------------------------------------
# Times are plotted on a UTC scale (scale type and tooltips below); otherwise the browser shows the viewer's local time.
hist_times = window["ts_hour"]
history = pd.DataFrame({"time": hist_times, "hours": window["dwell"], "series": HISTORY})
forecast_point = pd.DataFrame({"time": [target], "hours": [forecast_hours], "series": [FORECAST]})
connector = pd.DataFrame({"time": [hist_times.iloc[-1], target], "hours": [last_obs, forecast_hours]})
colors = alt.Scale(domain=[HISTORY, FORECAST], range=[ui.BLUE, ui.FLAG_COLOR])
x = alt.X("time:T", title="Hour (UTC)", scale=alt.Scale(type="utc"), axis=alt.Axis(format="%d %b %H:%M", grid=False))
y = alt.Y("hours:Q", title="Dwell time (hours)", scale=alt.Scale(zero=False), axis=alt.Axis(tickCount=5))
grad = alt.Gradient(gradient="linear", x1=1, x2=1, y1=1, y2=0,
                    stops=[alt.GradientStop(color="rgba(47,125,225,0.0)", offset=0),
                           alt.GradientStop(color="rgba(47,125,225,0.18)", offset=1)])
lo = min(history["hours"].min(), forecast_hours)
area = alt.Chart(history.assign(base=lo - 1)).mark_area(color=grad, interpolate="monotone").encode(x=x, y=y, y2="base:Q")
line = alt.Chart(history).mark_line(strokeWidth=2.6, interpolate="monotone").encode(
    x=x, y=y, color=alt.Color("series:N", title=None, scale=colors))
dots = alt.Chart(history).mark_point(filled=True, size=55, stroke="white", strokeWidth=1.5, color=ui.BLUE).encode(
    x=x, y="hours:Q",
    tooltip=[alt.Tooltip("time:T", timeUnit="utcyearmonthdatehoursminutes", title="Hour (UTC)"), alt.Tooltip("hours:Q", format=".1f", title="Observed (h)")])
band = alt.Chart(pd.DataFrame({"x": [hist_times.iloc[-1]], "x2": [target]})).mark_rect(
    color=ui.FLAG_COLOR, opacity=0.07).encode(x=alt.X("x:T", scale=alt.Scale(type="utc")), x2="x2:T")
dash = alt.Chart(connector).mark_line(strokeDash=[5, 4], color=ui.FLAG_COLOR, strokeWidth=2).encode(x=x, y="hours:Q")
marker = alt.Chart(forecast_point).mark_point(shape="diamond", size=260, filled=True, strokeWidth=3, stroke="white").encode(
    x=x, y="hours:Q", color=alt.Color("series:N", title=None, scale=colors, legend=alt.Legend(orient="top", labelLimit=0)),
    tooltip=[alt.Tooltip("time:T", timeUnit="utcyearmonthdatehoursminutes", title="Hour (UTC)"), alt.Tooltip("hours:Q", format=".1f", title="Forecast (h)")])
label = alt.Chart(forecast_point).mark_text(dy=-20, fontSize=13, fontWeight=600, color=ui.INK, font=ui.FONT).encode(
    x=x, y="hours:Q", text=alt.Text("hours:Q", format=".1f"))

with ui.row("hero"):
    chart_col, side_col = st.columns([2.15, 1], gap="medium")

with chart_col, ui.panel("chart"):
    ui.card_head(f"Dwell time · {port} {names.get(port, '')}",
                 "Last 12 observed hours (line) and the next-hour model estimate (diamond)")
    st.altair_chart(ui.style(alt.layer(band, area, line, dash, dots, marker, label).properties(height=380)), use_container_width=True,
                    key=f"fc_chart_{port}_{end_iso}_{benchmark}")
    st.caption("A single point forecast with no calibrated uncertainty.")

with side_col, ui.panel("summary"):
    ui.card_head("Next-hour forecast", f"For {target:%a %d %b, %H:%M} UTC")
    ui.big(f"{forecast_hours:.1f}", "h", ui.badge(f"{forecast_hours - last_obs:+.1f} h vs last hour", "blue"))
    fb = "Not applicable (benchmark)" if benchmark else ("Used" if result["fallback_used"] else "Not used")
    ui.rows({"Model that produced this forecast": model_name,
             "Automatic fallback": fb,
             "Last observed hour": f"{last_obs:.1f} h",
             "Observed at forecast hour": f"{observed_at_target:.1f} h" if observed_at_target is not None else "Not in the history",
             "Horizon": f"{result['forecast_horizon_hours']} h"})
    if not benchmark and result["fallback_used"]:
        ui.note(f"The primary model could not be used, so the automatic C_w LSTM fallback produced this forecast. "
                f"Reason: {result['fallback_reason'] or 'not stated'}.", warn=True)
    elif benchmark:
        ui.note("EWMA (alpha 0.2) is shown for comparison only. It is a benchmark, not the fallback.")
    st.markdown('<div class="cm-card-sub" style="margin-top:0.4rem">Model chain</div>', unsafe_allow_html=True)
    active = "ewma" if benchmark else result["model_name"]
    chain = [(dl.DEFAULT_MODEL, "Transformer T_w", "primary"), (dl.FALLBACK_MODEL, "C_w LSTM", "automatic fallback"),
             ("ewma", "EWMA", "benchmark only")]
    ui.rows({f"{name} <span style='color:#9aa1ac'>· {role}</span>":
             (ui.badge("active", "blue") if key == active else "standby" if key != "ewma" else "on request")
             for key, name, role in chain})

# --- Context: input window | 7-day history -------------------------------------------------------------------------
with ui.row("context"):
    w_col, h_col = st.columns([1, 1.15], gap="medium")

with w_col, ui.panel("window"):
    ui.card_head("Input window", f"{window['ts_hour'].iloc[0]:%d %b %H:%M} – {window['ts_hour'].iloc[-1]:%d %b %H:%M} UTC · 12 hours")
    cols = [c for c in ("ts_hour", "dwell", "utilization", "vessels_at_anchor", "wind_gust", "disruption_active", "labour_strike")
            if c in window.columns]
    shown = window[cols].copy()
    shown["ts_hour"] = shown["ts_hour"].dt.strftime("%d %b %H:%M")
    st.dataframe(shown.rename(columns={"ts_hour": "Hour (UTC)", "dwell": "Dwell (h)", "utilization": "Util. %",
                                       "vessels_at_anchor": "At anchor", "wind_gust": "Gust km/h",
                                       "disruption_active": "Disruption", "labour_strike": "Strike"}),
                 hide_index=True, width="stretch", height=300,
                 column_config={"Dwell (h)": st.column_config.NumberColumn(format="%.1f"),
                                "Util. %": st.column_config.NumberColumn(format="%.0f")})

with h_col, ui.panel("history"):
    ui.card_head(f"Historical context · {port}", "Daily mean dwell and vessels at anchor, 7 days to the last observed hour")
    trend = port_ops.daily_trend(window_end, days=7, port_id=port)
    if len(trend):
        x_day = alt.X("day:T", title=None, scale=alt.Scale(type="utc"), axis=alt.Axis(format="%a %d", grid=False))
        ui.kpis([ui.kpi("7-day mean dwell", f"{trend['dwell'].mean():.1f}", "h", accent=True),
                 ui.kpi("Lowest day", f"{trend['dwell'].min():.1f}", "h"),
                 ui.kpi("Highest day", f"{trend['dwell'].max():.1f}", "h"),
                 ui.kpi("Mean at anchor", f"{trend['anchored'].mean():.1f}")])
        st.altair_chart(ui.area_line(trend, x_day, "dwell", "Dwell (h)",
                                     [alt.Tooltip("day:T", timeUnit="utcyearmonthdate", title="Day (UTC)"),
                                      alt.Tooltip("dwell:Q", format=".1f", title="Dwell (h)")], height=200),
                        use_container_width=True, key=f"fc_hist_{port}_{end_iso}")
    else:
        st.info("No daily history is available before this hour.")

with st.expander("Forecast details", icon=":material/info:"):
    st.write(f"- Window: {result['window_start_utc']} to {result['window_end_utc']}")
    st.write(f"- Forecast hour: {result['target_hour_utc']} (horizon {result['forecast_horizon_hours']} h)")
    st.write(f"- Method: {result['method']} · model role: {result['model_role']}")
    st.write(f"- Fallback used: {'yes' if result['fallback_used'] else 'no'}")
    st.write(f"- Generated: {result['generated_at_utc']}")
    for note in result["warnings"]:
        st.write(f"- Note: {note}")
