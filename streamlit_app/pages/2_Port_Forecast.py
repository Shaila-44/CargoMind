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
ui.demo_banner()
ui.header("Port Forecast", "Next-hour port dwell time in hours, from the last 12 observed hours.")


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
    ui.error_box("Port history data is unavailable in this copy of the project. Forecasts cannot be produced.")
    st.stop()
try:
    names = port_ops.port_names()
except Exception:
    names = {}

with st.sidebar:
    st.header("Select a port and hour")
    port = st.selectbox("Port", ports, index=None, placeholder="Choose a port",
                        format_func=lambda p: f"{p} · {names[p]}" if p in names else p)
    earliest = first + timedelta(hours=11)  # the window needs 12 hours, so the first end hour is 11 h after the start
    window_end = ui.hour_picker(earliest, last, default=last, key="fc", label="Last observed hour (UTC)")
    benchmark = st.checkbox("Show EWMA benchmark instead (reference only)", value=False,
                            help="EWMA is a simple benchmark for comparison. It is not the fallback.")
    st.caption(f"History available {first:%Y-%m-%d %H:%M} to {last:%Y-%m-%d %H:%M} UTC. No live feed.")

if port is None:
    st.info("Select a port in the sidebar to see its next-hour forecast.")
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

ui.context_bar({"Port": f"{port} · {names.get(port, port)}",
                "Input window": f"{window['ts_hour'].iloc[0]:%d %b %H:%M} – {window['ts_hour'].iloc[-1]:%d %b %H:%M} UTC",
                "Forecast hour": f"{target:%a %d %b %Y, %H:%M} UTC"})

col_a, col_b, col_c = st.columns(3)
col_a.metric("Forecast dwell (next hour)", f"{forecast_hours:.1f} h")
col_b.metric("Model that produced this forecast", model_name)
col_c.metric("Observed at that hour (for reference)" if observed_at_target is not None else "Observed at that hour",
             f"{observed_at_target:.1f} h" if observed_at_target is not None else "Not in the history")

if not benchmark and result["fallback_used"]:
    st.warning(f"The primary model could not be used, so the automatic C_w LSTM fallback produced this forecast. "
               f"Reason: {result['fallback_reason'] or 'not stated'}.", icon=":material/swap_horiz:")

p = ui.palette()
# Times are plotted on a UTC scale (scale type and tooltips below); otherwise the browser shows the viewer's local time.
hist_times, target_utc = window["ts_hour"], target  # timezone-aware UTC
history = pd.DataFrame({"time": hist_times, "hours": window["dwell"], "series": HISTORY})
forecast_point = pd.DataFrame({"time": [target_utc], "hours": [forecast_hours], "series": [FORECAST]})
connector = pd.DataFrame({"time": [hist_times.iloc[-1], target_utc],
                          "hours": [float(window["dwell"].iloc[-1]), forecast_hours]})
colors = alt.Scale(domain=[HISTORY, FORECAST], range=[p["accent"], ui.FLAG_COLOR])
x, y = alt.X("time:T", title="Hour (UTC)", scale=alt.Scale(type="utc"), axis=alt.Axis(format="%d %b %H:%M")), \
    alt.Y("hours:Q", title="Dwell time (hours)", scale=alt.Scale(zero=False))

line = alt.Chart(history).mark_line(point=True, strokeWidth=3).encode(
    x=x, y=y, color=alt.Color("series:N", title=None, scale=colors),
    tooltip=[alt.Tooltip("time:T", timeUnit="utcyearmonthdatehoursminutes", title="Hour (UTC)"), alt.Tooltip("hours:Q", format=".1f", title="Observed (h)")])
dash = alt.Chart(connector).mark_line(strokeDash=[5, 4], color=ui.FLAG_COLOR, strokeWidth=2).encode(x=x, y="hours:Q")
marker = alt.Chart(forecast_point).mark_point(shape="diamond", size=220, filled=False, strokeWidth=3).encode(
    x=x, y="hours:Q", color=alt.Color("series:N", title=None, scale=colors, legend=alt.Legend(orient="top", labelLimit=0)),
    tooltip=[alt.Tooltip("time:T", timeUnit="utcyearmonthdatehoursminutes", title="Hour (UTC)"), alt.Tooltip("hours:Q", format=".1f", title="Forecast (h)")])

st.write("")
with st.container(border=True):
    ui.panel_title(f"Dwell time · {port}", "Last 12 observed hours (solid line) and the next-hour forecast (hollow diamond)")
    st.altair_chart(ui.style(alt.layer(line, dash, marker).properties(height=340)), use_container_width=True)
    ui.help_popover("The solid line is **observed** history. The hollow diamond is a **model estimate** for the next hour, "
                    "joined by a dashed line. It is a single point forecast with no calibrated uncertainty. The Transformer "
                    "T_w is the primary model; if it cannot produce a valid forecast, the C_w LSTM is used automatically "
                    "and this page says so. EWMA is only a benchmark.")

with st.expander("Forecast details", icon=":material/info:"):
    st.write(f"- Window: {result['window_start_utc']} to {result['window_end_utc']}")
    st.write(f"- Forecast hour: {result['target_hour_utc']} (horizon {result['forecast_horizon_hours']} h)")
    st.write(f"- Method: {result['method']} · model role: {result['model_role']}")
    st.write(f"- Fallback used: {'yes' if result['fallback_used'] else 'no'}")
    st.write(f"- Generated: {result['generated_at_utc']}")
    for note in result["warnings"]:
        st.write(f"- Note: {note}")
