"""Shared presentation helpers: theme, status strip, panels, charts and the ML interpretation text.

No model or feature logic lives here. The interpretation only describes the two independent ML outputs side by side;
it never changes either output and never combines them into one score. Probabilities and the anomaly score are always
drawn on separate charts with separate scales.

Light and dark mode follow Streamlit's own theme switch (menu > Settings > Theme); `palette()` reads the active mode so
the custom panels and charts match the built-in widgets.
"""
from __future__ import annotations

from datetime import datetime, time

import altair as alt
import pandas as pd
import streamlit as st

from services import ml_risk

CLASS_ORDER = ["On-Time", "At-Risk", "Critical"]
# One meaning per colour across the whole app. Every use is paired with the class name, so colour is never the only cue.
CLASS_COLORS = {"On-Time": "#16a34a", "At-Risk": "#d97706", "Critical": "#dc2626"}
FLAG_COLOR = "#d97706"
FONT = "Fira Sans, system-ui, -apple-system, Helvetica, Arial, sans-serif"  # chart text; sans-serif fallbacks
UNCALIBRATED_NOTE = "Raw model probabilities, not calibrated: they are not real-world likelihoods, and no alert threshold is set."
DEMO_NOTICE = "Synthetic, offline demo. No live port, shipment or weather feed; nothing here is current real-world status."

_PALETTES = {
    "dark": {"background": "radial-gradient(1200px 600px at 50% -10%, #1d4ed8 0%, rgba(29,78,216,0) 60%), "
                           "linear-gradient(180deg, #0b2a5e 0%, #0a1d3f 55%, #071530 100%)",
             "panel": "rgba(18, 52, 110, 0.55)", "border": "rgba(96, 165, 250, 0.38)", "text": "#eef4ff",
             "muted": "#c3d3ee", "sidebar": "#0a1b3a", "accent": "#38bdf8", "chart_text": "#d5e2fb",
             "grid": "rgba(148, 163, 184, 0.22)", "track": "rgba(148, 163, 184, 0.28)", "strip": "rgba(217, 119, 6, 0.16)",
             "strip_border": "rgba(251, 191, 36, 0.55)", "map": "dark"},
    "light": {"background": "linear-gradient(180deg, #eef4ff 0%, #f6f9ff 45%, #ffffff 100%)",
              "panel": "#ffffff", "border": "#c9daf3", "text": "#0f2a57", "muted": "#3d5378", "sidebar": "#eaf1fc",
              "accent": "#1d4ed8", "chart_text": "#334155", "grid": "rgba(15, 42, 87, 0.12)", "track": "#e2e8f0",
              "strip": "#fff7e6", "strip_border": "#f2b84b", "map": "light"},
}


def mode() -> str:
    """'light' or 'dark', following Streamlit's active theme (dark when unknown, e.g. in tests)."""
    try:
        return "light" if st.context.theme.type == "light" else "dark"
    except Exception:
        return "dark"


def palette() -> dict[str, str]:
    return _PALETTES[mode()]


def theme() -> None:
    """Apply the shared look. Call once near the top of every page."""
    p = palette()
    st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fira+Sans:wght@400;500;600;700&display=swap');
  :root {{ --cm-accent: {p['accent']}; --cm-panel: {p['panel']}; --cm-border: {p['border']}; --cm-text: {p['text']};
          --cm-muted: {p['muted']}; }}
  .stApp {{ background: {p['background']}; background-attachment: fixed; }}
  .stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp td, .stApp th,
  [class^="cm-"] {{ font-family: 'Fira Sans', system-ui, sans-serif; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}
  h1 {{ font-weight: 700; letter-spacing: -0.01em; }}
  div[data-testid="stCaptionContainer"], div[data-testid="stCaptionContainer"] p {{ font-size: 0.875rem; color: var(--cm-muted); }}
  .cm-status {{ display: flex; gap: 0.6rem; align-items: center; padding: 0.5rem 0.9rem; margin-bottom: 0.9rem;
               background: {p['strip']}; border: 1px solid {p['strip_border']}; border-radius: 8px; font-size: 0.9rem;
               color: var(--cm-text); }}
  .cm-status svg {{ flex: none; }}
  .cm-chip {{ display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.12rem 0.6rem; border-radius: 999px;
             font-size: 0.8rem; font-weight: 600; border: 1px solid var(--cm-border); color: var(--cm-text); }}
  .cm-chip .dot {{ width: 0.55rem; height: 0.55rem; border-radius: 50%; display: inline-block; }}
  .cm-eyebrow {{ color: var(--cm-accent); font-size: 0.78rem; font-weight: 600; letter-spacing: 0.12em;
                text-transform: uppercase; margin: 0.4rem 0 0.35rem 0; }}
  .cm-lede {{ color: var(--cm-muted); font-size: 1rem; margin-top: -0.4rem; margin-bottom: 1rem; }}
  .cm-titlebar {{ display: flex; justify-content: space-between; align-items: center; gap: 1rem; padding: 0.75rem 1.1rem;
                 margin-bottom: 0.8rem; border: 1px solid var(--cm-border); border-radius: 10px; background: var(--cm-panel); }}
  .cm-titlebar .t {{ font-size: 1.4rem; font-weight: 700; color: var(--cm-text); }}
  .cm-titlebar .s {{ font-size: 0.85rem; color: var(--cm-muted); }}
  .cm-context {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0 0 0.9rem 0; }}
  .cm-context .item {{ padding: 0.3rem 0.7rem; border-radius: 8px; border: 1px solid var(--cm-border); background: var(--cm-panel);
                      font-size: 0.88rem; color: var(--cm-text); }}
  .cm-context .item b {{ color: var(--cm-muted); font-weight: 500; margin-right: 0.3rem; }}
  .cm-card {{ background: var(--cm-panel); border: 1px solid var(--cm-border); border-radius: 10px; padding: 0.9rem 1.1rem; }}
  .cm-kpi {{ background: var(--cm-panel); border: 1px solid var(--cm-border); border-radius: 10px; padding: 0.8rem 1rem;
            text-align: center; }}
  .cm-kpi .label {{ color: var(--cm-muted); font-size: 0.88rem; }}
  .cm-kpi .value {{ font-size: 1.85rem; font-weight: 700; line-height: 1.2; margin-top: 0.15rem; color: var(--cm-text); }}
  .cm-kpi .unit {{ font-size: 0.95rem; font-weight: 500; color: var(--cm-muted); margin-left: 0.25rem; }}
  .cm-panel-title {{ font-weight: 600; font-size: 1.02rem; color: var(--cm-text); }}
  .cm-panel-sub {{ font-size: 0.82rem; color: var(--cm-muted); margin-bottom: 0.4rem; }}
  .cm-row {{ display: flex; justify-content: space-between; gap: 1rem; padding: 0.42rem 0; border-bottom: 1px solid var(--cm-border); }}
  .cm-row:last-child {{ border-bottom: none; }}
  .cm-row .k {{ color: var(--cm-muted); }}
  .cm-row .v {{ font-weight: 700; font-size: 1.05rem; color: var(--cm-text); text-align: right; }}
  .cm-kpi .value, .cm-row .v, div[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; }}
  div[data-testid="stVerticalBlockBorderWrapper"] {{ background: var(--cm-panel); border-color: var(--cm-border) !important;
                                                     border-radius: 10px; }}
  div[data-testid="stMetric"] {{ background: var(--cm-panel); border: 1px solid var(--cm-border); border-radius: 10px;
                                padding: 0.7rem 1rem; }}
  div[data-testid="stMetricValue"] {{ font-size: 1.4rem; }}
  div[data-testid="stAlert"] {{ border-radius: 10px; }}
  section[data-testid="stSidebar"] {{ background: {p['sidebar']}; border-right: 1px solid var(--cm-border); }}
  div[data-testid="stExpander"] {{ border-radius: 10px; border: 1px solid var(--cm-border); background: var(--cm-panel); }}
  a, button, [role="button"], [data-baseweb="select"] {{ cursor: pointer; }}
  .stApp *:focus-visible {{ outline: 2px solid var(--cm-accent) !important; outline-offset: 2px; }}
  @media (prefers-reduced-motion: reduce) {{ .stApp * {{ transition: none !important; animation: none !important; }} }}
  footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


_INFO_SVG = ('<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#d97706" stroke-width="2" aria-hidden="true">'
             '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>')


def demo_banner() -> None:
    """Slim, always-visible synthetic/offline status with the supported date range (required on every page)."""
    start = ml_risk.SUPPORTED_DEPARTURE_START.strftime("%Y-%m-%d")
    end = ml_risk.SUPPORTED_DEPARTURE_END.strftime("%Y-%m-%d")
    st.markdown(f'<div class="cm-status" role="status">{_INFO_SVG}<span><b>{DEMO_NOTICE}</b> '
                f'Supported departures: {start} to {end} (UTC).</span></div>', unsafe_allow_html=True)


def header(title: str, lede: str, eyebrow: str = "CargoMind") -> None:
    """Page heading: small accent label, title and one-line description."""
    st.markdown(f'<div class="cm-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="cm-lede">{lede}</div>', unsafe_allow_html=True)


def titlebar(title: str, subtitle: str) -> None:
    """Dashboard title strip with a permanent 'synthetic replay' chip."""
    st.markdown(f'<div class="cm-titlebar"><div><div class="t">{title}</div><div class="s">{subtitle}</div></div>'
                f'<span class="cm-chip"><span class="dot" style="background:{FLAG_COLOR}"></span>Synthetic · offline</span></div>',
                unsafe_allow_html=True)


def context_bar(items: dict[str, str]) -> None:
    """What the results below describe: the selected port, shipment, hour, etc."""
    html = "".join(f'<span class="item"><b>{k}</b>{v}</span>' for k, v in items.items())
    st.markdown(f'<div class="cm-context" aria-label="Current selection">{html}</div>', unsafe_allow_html=True)


def class_chip(cls: str) -> str:
    """HTML badge for a delay class: coloured dot plus the class name."""
    return f'<span class="cm-chip"><span class="dot" style="background:{CLASS_COLORS[cls]}"></span>{cls}</span>'


def flag_chip(flagged: bool) -> str:
    color = FLAG_COLOR if flagged else palette()["accent"]
    text = "Unusual conditions detected" if flagged else "Not flagged"
    return f'<span class="cm-chip"><span class="dot" style="background:{color}"></span>{text}</span>'


def card(title: str, body: str) -> None:
    st.markdown(f'<div class="cm-card"><div class="cm-panel-title">{title}</div><p>{body}</p></div>', unsafe_allow_html=True)


def kpi(label: str, value: str, unit: str = "") -> None:
    unit_html = f'<span class="unit">{unit}</span>' if unit else ""
    st.markdown(f'<div class="cm-kpi"><div class="label">{label}</div><div class="value">{value}{unit_html}</div></div>',
                unsafe_allow_html=True)


def rows(items: dict[str, str]) -> None:
    """Label / value list, one per line, for compact metric panels."""
    html = "".join(f'<div class="cm-row"><span class="k">{k}</span><span class="v">{v}</span></div>' for k, v in items.items())
    st.markdown(html, unsafe_allow_html=True)


def panel_title(text: str, subtitle: str | None = None) -> None:
    """Panel heading in plain language, with the producing model (or data source) as a subtitle."""
    sub = f'<div class="cm-panel-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="cm-panel-title">{text}</div>{sub}', unsafe_allow_html=True)


def help_popover(text: str) -> None:
    """Inline 'What does this mean?' explanation next to a panel."""
    with st.popover("What does this mean?", icon=":material/help:"):
        st.markdown(text)


def hour_picker(first_end: pd.Timestamp, last: pd.Timestamp, default: pd.Timestamp, key: str,
                label: str = "Hour (UTC)") -> pd.Timestamp:
    """Date + hour selection limited to [first_end, last] (UTC). Only valid hours are offered."""
    day = st.date_input(f"{label}: date", value=default.date(), min_value=first_end.date(), max_value=last.date(),
                        key=f"{key}_date", format="YYYY-MM-DD")
    hours = [h for h in range(24) if first_end <= pd.Timestamp(datetime.combine(day, time(h)), tz="UTC") <= last]
    index = hours.index(default.hour) if day == default.date() and default.hour in hours else len(hours) - 1
    hour = st.selectbox(f"{label}: hour", hours, index=index, format_func=lambda h: f"{h:02d}:00")
    return pd.Timestamp(datetime.combine(day, time(hour)), tz="UTC")


def style(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    """Theme-matched axes, legend and transparent background for any top-level Altair chart."""
    p = palette()
    return (chart.properties(background="transparent")
            .configure_axis(labelColor=p["chart_text"], titleColor=p["chart_text"], gridColor=p["grid"], domainColor=p["grid"],
                            tickColor=p["grid"], labelFontSize=12, titleFontSize=12, labelFont=FONT, titleFont=FONT)
            .configure_legend(labelColor=p["chart_text"], titleColor=p["chart_text"], labelFont=FONT)
            .configure_view(strokeWidth=0))


def gauge(percent: float, label: str, color: str) -> alt.Chart:
    """Donut gauge for a 0-100 % quantity (used for utilisation only), value in the centre and label below."""
    p = palette()
    pct = max(0.0, min(100.0, float(percent)))
    data = pd.DataFrame({"part": ["value", "rest"], "amount": [pct, 100 - pct]})
    ring = alt.Chart(data).mark_arc(innerRadius=56, outerRadius=76, cornerRadius=4).encode(
        theta=alt.Theta("amount:Q", stack=True),
        color=alt.Color("part:N", scale=alt.Scale(domain=["value", "rest"], range=[color, p["track"]]), legend=None),
        tooltip=alt.value(None))
    text = alt.Chart(pd.DataFrame({"t": [f"{pct:.0f}%"]})).mark_text(fontSize=24, fontWeight="bold", color=p["text"],
                                                                        font=FONT).encode(text="t:N")
    caption = alt.Chart(pd.DataFrame({"t": [label]})).mark_text(dy=96, fontSize=13, color=p["chart_text"],
                                                                 font=FONT).encode(text="t:N")
    return style((ring + text + caption).properties(height=225))


def probability_chart(probabilities: dict[str, float]) -> alt.Chart:
    """Three raw probabilities as labelled bars on a 0-1 scale of their own."""
    p = palette()
    data = pd.DataFrame({"class": CLASS_ORDER, "probability": [probabilities[c] for c in CLASS_ORDER]})
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), title="Raw model probability (not calibrated)"),
        y=alt.Y("class:N", sort=CLASS_ORDER, title=None),
        color=alt.Color("class:N", legend=None, scale=alt.Scale(domain=CLASS_ORDER, range=[CLASS_COLORS[c] for c in CLASS_ORDER])),
        tooltip=[alt.Tooltip("class:N"), alt.Tooltip("probability:Q", format=".3f")])
    labels = bars.mark_text(align="left", dx=4, color=p["text"], font=FONT).encode(
        text=alt.Text("probability:Q", format=".2f"), color=alt.value(p["text"]))
    return style((bars + labels).properties(height=135))


def anomaly_scale(score: float, threshold: float, flagged: bool) -> alt.Chart:
    """The anomaly score on its own scale with the flag threshold marked. Never shares an axis with probabilities."""
    p = palette()
    lo, hi = min(score, threshold), max(score, threshold)
    pad = max((hi - lo) * 0.6, 0.05)
    domain = [lo - pad, hi + pad]
    x = alt.X("x:Q", scale=alt.Scale(domain=domain), title="Anomaly score (own scale; higher = more unusual)")
    track = alt.Chart(pd.DataFrame({"x": [domain[0]], "x2": [domain[1]]})).mark_rule(
        strokeWidth=10, color=p["track"], strokeCap="round").encode(x=x, x2="x2:Q")
    tick = alt.Chart(pd.DataFrame({"x": [threshold], "t": ["Flag threshold"]}))
    marker = alt.Chart(pd.DataFrame({"x": [score], "t": ["This shipment"]}))
    color = FLAG_COLOR if flagged else p["accent"]
    layers = [track,
              tick.mark_tick(thickness=3, size=30, color=p["chart_text"]).encode(x="x:Q"),
              tick.mark_text(dy=-24, fontSize=12, color=p["chart_text"], font=FONT).encode(x="x:Q", text="t:N"),
              marker.mark_point(shape="diamond", filled=True, size=260, color=color).encode(
                  x="x:Q", tooltip=[alt.Tooltip("x:Q", format=".3f", title="Anomaly score")]),
              marker.mark_text(dy=24, fontSize=12, color=p["text"], font=FONT).encode(x="x:Q", text="t:N")]
    return style(alt.layer(*layers).properties(height=110, padding={"top": 26, "bottom": 4, "left": 6, "right": 10}))


def interpret(predicted_class: str, anomaly_flag: bool | None) -> str:
    """Plain-language reading of both outputs together. Keeps them independent; implies no causation."""
    if anomaly_flag is None:
        return f"Predicted delay category: {predicted_class}. No operating-conditions result is available for this shipment."
    if predicted_class == "Critical" and anomaly_flag:
        text = ("This shipment has a high predicted delay category and is also associated with unusual operating conditions. "
                "Consider investigating the underlying conditions.")
    elif predicted_class == "Critical":
        text = "This shipment has a high predicted delay category. Its origin operating conditions look typical."
    else:
        conditions = "unusual operating conditions detected" if anomaly_flag else "no unusual operating conditions flagged"
        text = f"Predicted delay category: {predicted_class}. The origin shows {conditions}."
    return f"{text} An anomaly is not a delay prediction, and neither output proves what will happen."


def error_box(message: str) -> None:
    """A user-facing error: plain words only, no paths, stack traces or internal names."""
    st.error(message, icon=":material/error:")
