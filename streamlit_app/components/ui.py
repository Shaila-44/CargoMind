"""Shared presentation layer: design tokens, navigation, cards, badges, pickers, charts and the ML interpretation text.

No model or feature logic lives here. The interpretation only describes the two independent ML outputs side by side;
it never changes either output and never combines them into one score. Probabilities and the anomaly score are always
drawn on separate charts with separate scales.

Design system (light, flat: grey page, white cards, one dark accent):
  * tokens        -> the constants below and the CSS custom properties in `theme()`
  * layout        -> `navbar()`, `header()`, `panel()` (white card), `tile()` (quiet inner group), `section()`
  * text blocks   -> `card_head()`, `kpi()`, `rows()`, `badge()`, `class_chip()`, `flag_chip()`, `seg_bar()`
  * controls      -> `port_picker()`, `hour_picker()` (selection shared across pages via session state)
  * charts        -> `style()` plus the chart builders; all use the same palette and font
"""
from __future__ import annotations

from datetime import datetime, time

import altair as alt
import pandas as pd
import streamlit as st

from services import ml_risk

# --- Tokens --------------------------------------------------------------------------------------------------------
INK, INK2, MUTED = "#16191f", "#5b6270", "#9aa1ac"
NAVY, BLUE, BLUE_SOFT, BLUE_PALE = "#16191f", "#438be8", "#b9d2f5", "#e8f0fc"
GRID, TRACK = "#eceef1", "#eceef1"
CLASS_ORDER = ["On-Time", "At-Risk", "Critical"]
# One meaning per colour across the whole app. Every use is paired with the class name, so colour is never the only cue.
CLASS_COLORS = {"On-Time": "#159b70", "At-Risk": "#e4a01a", "Critical": "#e65353"}
FLAG_COLOR = "#e4a01a"  # amber: Isolation Forest flag and disruptions in the observed history
FONT = "Inter, system-ui, -apple-system, Helvetica, Arial, sans-serif"  # chart text
UNCALIBRATED_NOTE = "Raw model probabilities, not calibrated: they are not real-world likelihoods, and no alert threshold is set."

NAV = [("Overview", "overview.py", ":material/home:"),
       ("Risk Prediction", "pages/1_ML_Intelligence.py", ":material/insights:"),
       ("Port Forecast", "pages/2_Port_Forecast.py", ":material/timeline:"),
       ("Route Advisor", "pages/4_Route_Advisor.py", ":material/alt_route:")]
# pages/3_Methodology.py stays in the repository as documentation but is not routed or linked from the app.

# Shared filter state (session keys that outlive page switches; widget keys are cleared when a page is left).
SHARED_PORT, SHARED_HOUR = "cm_shared_port", "cm_shared_hour"

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
:root {
  --cm-bg: #f1f2f4; --cm-surface: #ffffff; --cm-ink: #16191f; --cm-ink2: #5b6270; --cm-muted: #9aa1ac;
  --cm-hair: #eceef1; --cm-dark: #16191f; --cm-blue: #438be8; --cm-blue-pale: #e8f0fc;
  --cm-green: #159b70; --cm-amber: #e4a01a; --cm-red: #e65353;
  --cm-r-card: 26px; --cm-r-md: 14px; --cm-r-pill: 999px; --cm-pad: 1.9rem 2.1rem; --cm-gap: 1.6rem;
  /* aliases kept for older selectors */
  --cm-navy: var(--cm-dark); --cm-glass: var(--cm-surface); --cm-solid: var(--cm-surface); --cm-line: var(--cm-hair);
  --cm-orange: var(--cm-amber); --cm-shadow: none;
}
.stApp { background: var(--cm-bg); color: var(--cm-ink); }
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp td, .stApp th, .stApp input,
.stApp button, [class^="cm-"] { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
header[data-testid="stHeader"] { background: transparent; height: 2.2rem; }
section[data-testid="stSidebar"], div[data-testid="stSidebarCollapsedControl"] { display: none; }
footer { visibility: hidden; }
.block-container { max-width: 1280px; padding: 1.4rem 3rem 4rem 3rem !important; }
.block-container > div > [data-testid="stVerticalBlock"] { gap: var(--cm-gap); }
.stApp p { color: var(--cm-ink2); }
div[data-testid="stCaptionContainer"], div[data-testid="stCaptionContainer"] p { font-size: 0.78rem; color: var(--cm-muted); }
.stApp *:focus-visible { outline: 2px solid var(--cm-blue) !important; outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) { .stApp * { transition: none !important; animation: none !important; } }

/* Navigation: brand, one white pill group, status pill */
.st-key-cm_nav { padding: 0 0 1.4rem 0; flex-wrap: wrap; row-gap: 0.6rem; }
.st-key-cm_nav > div, .st-key-cm_nav_pills > div { flex: 0 0 auto !important; width: auto !important; max-width: none !important; }
.cm-brand { display: flex; align-items: center; gap: 0.6rem; font-size: 1.4rem; font-weight: 600; color: var(--cm-ink);
  letter-spacing: -0.02em; line-height: 1.3; white-space: nowrap; }
.cm-brand svg { flex: none; }
.st-key-cm_nav_pills { gap: 0.15rem !important; width: auto !important; flex: 0 1 auto !important; background: var(--cm-surface);
  border-radius: var(--cm-r-pill); padding: 0.3rem; }
[class*="st-key-cm_nav_link"], [class*="st-key-cm_nav_active"] { width: auto !important; flex: 0 0 auto !important; }
.st-key-cm_nav_pills [data-testid="stPageLink"] a, .cm-pill { display: inline-flex; align-items: center; gap: 0.35rem;
  height: 2.5rem; padding: 0 1.15rem !important; border-radius: var(--cm-r-pill); background: transparent; border: none; }
.st-key-cm_nav_pills [data-testid="stPageLink"] a:hover { background: var(--cm-bg); }
.st-key-cm_nav_pills [data-testid="stPageLink"] a p, .st-key-cm_nav_pills [data-testid="stPageLink"] a span,
.cm-pill { color: var(--cm-ink) !important; font-size: 0.92rem !important; font-weight: 500; white-space: nowrap; }
.st-key-cm_nav_pills [data-testid="stPageLink"] a span[data-testid="stIconMaterial"] { display: none; }
[class*="st-key-cm_nav_active"] [data-testid="stPageLink"] a, .cm-pill.active { background: var(--cm-dark) !important; }
[class*="st-key-cm_nav_active"] [data-testid="stPageLink"] a p, [class*="st-key-cm_nav_active"] [data-testid="stPageLink"] a span,
.cm-pill.active { color: #fff !important; }
.cm-status-pill { display: inline-flex; align-items: center; gap: 0.5rem; height: 3.1rem; padding: 0 1.2rem;
  border-radius: var(--cm-r-pill); background: var(--cm-surface); font-size: 0.86rem; color: var(--cm-ink2); white-space: nowrap; }
.cm-status-pill .dot { width: 0.5rem; height: 0.5rem; border-radius: 50%; background: var(--cm-amber); }

/* Page header */
.cm-eyebrow { font-size: 0.95rem; font-weight: 400; color: var(--cm-ink2); margin-bottom: 0.1rem; }
.cm-h1 { font-size: 2.6rem; font-weight: 500; letter-spacing: -0.035em; line-height: 1.1; color: var(--cm-ink); margin: 0; }
.cm-lede { font-size: 0.92rem; color: var(--cm-ink2); margin: 0.45rem 0 0 0; }
.cm-section { font-size: 0.78rem; font-weight: 500; color: var(--cm-muted); margin: 0.6rem 0 -0.2rem 0.2rem; }

/* Cards: keyed containers. card_* = white card, tile_* = quiet inner group. */
[class*="st-key-card_"] { background: var(--cm-surface); border-radius: var(--cm-r-card); padding: var(--cm-pad); gap: 1.1rem; }
[class*="st-key-card_"] [data-testid="stVerticalBlock"] { gap: 0.8rem; }
[class*="st-key-tile_"] { background: var(--cm-bg); border-radius: var(--cm-r-md); padding: 0.85rem 1rem; }
/* Streamlit pulls HTML markdown blocks up by 1rem; inside cards and the header that collapses the spacing. */
[class*="st-key-card_"] [data-testid="stMarkdownContainer"], .block-container [data-testid="stMarkdownContainer"]:has(.cm-h1),
.block-container [data-testid="stMarkdownContainer"]:has(.cm-kpibar),
.block-container [data-testid="stMarkdownContainer"]:has(.cm-kpistrip) { margin-bottom: 0 !important; }
/* Rows of cards stretch to equal height. */
[class*="st-key-row_"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"] { height: 100%; }
[class*="st-key-row_"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > div:last-child:has(> [class*="st-key-card_"]) { flex: 1 1 auto; }
[class*="st-key-row_"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > div:last-child > [class*="st-key-card_"] { height: 100%; }

.cm-card-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 0.8rem; }
.cm-card-title { font-size: 1.3rem; font-weight: 500; color: var(--cm-ink); letter-spacing: -0.02em; }
.cm-card-sub { font-size: 0.86rem; color: var(--cm-muted); margin-top: 0.3rem; line-height: 1.5; }
.cm-big { font-size: 2.2rem; font-weight: 500; letter-spacing: -0.03em; color: var(--cm-ink); line-height: 1.1;
  font-variant-numeric: tabular-nums; }
.cm-big .unit { font-size: 1rem; font-weight: 400; color: var(--cm-muted); margin-left: 0.3rem; letter-spacing: 0; }
.cm-kpi .label { font-size: 0.8rem; color: var(--cm-ink2); white-space: nowrap; }
.cm-kpi .value { font-size: 1.45rem; font-weight: 500; color: var(--cm-ink); letter-spacing: -0.02em; margin-top: 0.15rem;
  font-variant-numeric: tabular-nums; }
.cm-kpi .value .unit { font-size: 0.85rem; font-weight: 400; color: var(--cm-muted); margin-left: 0.2rem; }
.cm-kpi.accent .value { color: var(--cm-ink); }
.cm-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(92px, 1fr)); gap: 1rem; }
.cm-row { display: flex; justify-content: space-between; align-items: center; gap: 1rem; padding: 0.55rem 0;
  border-bottom: 1px solid var(--cm-hair); font-size: 0.86rem; }
.cm-row:last-child { border-bottom: none; }
.cm-row .k { color: var(--cm-ink2); display: inline-flex; align-items: center; gap: 0.45rem; }
.cm-row .k b { color: var(--cm-ink); font-weight: 500; min-width: 3.4rem; }
.cm-row .v { color: var(--cm-ink); font-weight: 500; text-align: right; font-variant-numeric: tabular-nums; }
/* Badges: tinted pills only where colour carries meaning; neutral = plain quiet text. */
.cm-badge { display: inline-flex; align-items: center; gap: 0.35rem; height: 1.6rem; padding: 0 0.7rem; border-radius: var(--cm-r-pill);
  font-size: 0.76rem; font-weight: 500; white-space: nowrap; }
.cm-badge .dot { width: 0.42rem; height: 0.42rem; border-radius: 50%; display: inline-block; }
.cm-badge.neutral { background: none; padding: 0; height: auto; color: var(--cm-muted); }
.cm-badge.blue { background: var(--cm-blue-pale); color: #2a6cc4; }
.cm-badge.navy { background: var(--cm-bg); color: var(--cm-ink); }
.cm-badge.green { background: #e6f5ef; color: #117a58; }
.cm-badge.amber, .cm-badge.orange { background: #fcf3df; color: #8a5e08; }
.cm-badge.red { background: #fde9e9; color: #b53a3c; }
.cm-chip { display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.3rem 0.85rem; border-radius: var(--cm-r-pill);
  font-size: 0.86rem; font-weight: 500; background: var(--cm-bg); color: var(--cm-ink); }
.cm-chip .dot { width: 0.5rem; height: 0.5rem; border-radius: 50%; display: inline-block; }
.cm-context { display: flex; flex-wrap: wrap; gap: 0.4rem 1.4rem; margin: 0; }
.cm-context .item { font-size: 0.86rem; color: var(--cm-ink); }
.cm-context .item b { display: block; color: var(--cm-muted); font-weight: 400; font-size: 0.75rem; }
.cm-note { font-size: 0.8rem; color: var(--cm-ink2); background: var(--cm-bg); border-radius: 12px; padding: 0.6rem 0.8rem; }
.cm-note.warn { background: #fcf3df; color: #6b4e16; }
.cm-table { font-size: 0.85rem; color: var(--cm-ink); }
.cm-table .tr { display: grid; gap: 0.6rem; align-items: center; padding: 0.6rem 0.4rem; border-bottom: 1px solid var(--cm-hair); }
.cm-table .tr:last-child { border-bottom: none; }
.cm-table .th { font-size: 0.78rem; color: var(--cm-muted); }
.cm-table .tr.focus { background: var(--cm-bg); border-radius: 10px; border-bottom-color: transparent; }
.cm-table .num { font-variant-numeric: tabular-nums; }
.cm-table .id { font-weight: 500; }
.cm-table .dim { color: var(--cm-muted); }
.cm-table .cell { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-width: 0; }
.cm-foot { display: flex; align-items: center; flex-wrap: wrap; gap: 0.4rem 1rem; font-size: 0.8rem; color: var(--cm-ink2); }
.cm-foot .n { color: var(--cm-ink); font-weight: 500; margin-left: 0.3rem; }
.cm-pbar { display: flex; align-items: center; gap: 0.55rem; }
.cm-seg { display: flex; gap: 2px; flex: 1; max-width: 120px; }
.cm-seg i { flex: 1; height: 8px; border-radius: 2px; background: var(--cm-hair); }
.cm-meter { position: relative; height: 6px; border-radius: 99px; background: var(--cm-hair); margin: 0.7rem 0 0.3rem 0; }
.cm-meter .fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 99px; background: var(--cm-blue); }
.cm-meter .mark { position: absolute; top: -5px; width: 2px; height: 16px; background: var(--cm-ink2); border-radius: 2px; }
.cm-meter .knob { position: absolute; top: -4px; width: 14px; height: 14px; margin-left: -7px; border-radius: 50%;
  background: #fff; border: 3px solid var(--cm-blue); }
.cm-meter-legend { display: flex; justify-content: space-between; font-size: 0.76rem; color: var(--cm-muted); }

/* Overview: KPI bar (no card), attention list, forecast grid */
.cm-kpibar { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 0.8rem; margin: 0.5rem 0 0.3rem 0; }
.cm-kpibar .cell { display: flex; align-items: center; gap: 0.85rem; min-width: 0; }
.cm-kpibar .ico { flex: none; width: 3.1rem; height: 3.1rem; border-radius: 50%; background: var(--cm-surface);
  display: flex; align-items: center; justify-content: center; color: var(--cm-ink); }
.cm-kpibar .val { font-size: 1.75rem; font-weight: 500; letter-spacing: -0.03em; line-height: 1.1; color: var(--cm-ink);
  font-variant-numeric: tabular-nums; }
.cm-kpibar .val small { font-size: 0.85rem; color: var(--cm-muted); font-weight: 400; margin-left: 0.15rem; letter-spacing: 0; }
.cm-kpibar .lab { font-size: 0.82rem; color: var(--cm-ink2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.cm-kpibar .chg { font-size: 0.74rem; color: var(--cm-muted); }
.cm-alert { display: flex; gap: 0.8rem; align-items: flex-start; }
.cm-alert .sev { flex: none; width: 0.5rem; height: 0.5rem; margin-top: 0.42rem; border-radius: 50%; background: var(--cm-muted); }
.cm-alert .sev.high { background: var(--cm-red); } .cm-alert .sev.medium { background: var(--cm-amber); }
.cm-alert .sev.low { background: var(--cm-blue); }
.cm-alert .t { font-size: 0.92rem; font-weight: 500; color: var(--cm-ink); }
.cm-alert .d { font-size: 0.8rem; color: var(--cm-ink2); margin-top: 0.15rem; line-height: 1.45; }
.st-key-card_attention [data-testid="stHorizontalBlock"] { border-top: 1px solid var(--cm-hair); padding-top: 1.25rem; margin-top: 0.35rem; }
/* Overview landing page */
.st-key-card_attention .stButton button { padding: 0 1rem; } .st-key-card_attention .stButton button p { white-space: nowrap; }
.cm-h1-xl { font-size: 3.4rem; letter-spacing: -0.045em; line-height: 1.05; }
.cm-kpistrip { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 2rem; margin: 1.4rem 0 0.8rem 0;
  padding: 1.8rem 0; border-top: 1px solid #e2e5e9; border-bottom: 1px solid #e2e5e9; }
.cm-kpistrip .cell { min-width: 0; }
.cm-kpistrip .lab { font-size: 0.86rem; color: var(--cm-ink2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.cm-kpistrip .val { font-size: 2.6rem; font-weight: 500; letter-spacing: -0.04em; line-height: 1.15; color: var(--cm-ink);
  margin-top: 0.35rem; font-variant-numeric: tabular-nums; }
.cm-kpistrip .val small { font-size: 1rem; color: var(--cm-muted); font-weight: 400; margin-left: 0.25rem; letter-spacing: 0; }
.cm-kpistrip .chg { font-size: 0.78rem; color: var(--cm-muted); margin-top: 0.3rem; }
.cm-item .top { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.4rem; }
.cm-item .sev { display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.74rem; font-weight: 500; color: var(--cm-ink2);
  text-transform: uppercase; letter-spacing: 0.06em; }
.cm-item .sev::before { content: ""; width: 0.5rem; height: 0.5rem; border-radius: 50%; background: var(--cm-muted); }
.cm-item .sev.high::before { background: var(--cm-red); } .cm-item .sev.medium::before { background: var(--cm-amber); }
.cm-item .sev.low::before { background: var(--cm-blue); }
.cm-item .id { font-size: 0.74rem; font-weight: 500; color: var(--cm-ink2); background: var(--cm-bg); border-radius: 6px; padding: 0.12rem 0.45rem;
  font-variant-numeric: tabular-nums; }
.cm-item .t { font-size: 1.02rem; font-weight: 500; color: var(--cm-ink); letter-spacing: -0.01em; }
.cm-item .d { font-size: 0.86rem; color: var(--cm-ink2); margin-top: 0.3rem; line-height: 1.55; max-width: 60ch; }
.cm-empty { font-size: 0.92rem; color: var(--cm-ink2); padding: 1rem 0; }
.cm-fcval { font-size: 3.6rem; font-weight: 500; letter-spacing: -0.045em; line-height: 1; color: var(--cm-ink); margin-top: 0.6rem;
  font-variant-numeric: tabular-nums; }
.cm-fcval small { font-size: 1.05rem; color: var(--cm-muted); font-weight: 400; margin-left: 0.4rem; letter-spacing: 0; }
@media (max-width: 1100px) { .cm-kpistrip { grid-template-columns: repeat(3, minmax(0, 1fr)); row-gap: 1.6rem; } .cm-h1-xl { font-size: 2.6rem; } }
.cm-fc { display: grid; grid-template-columns: 0.9fr 1.2fr 1.2fr 1.2fr; gap: 2rem; align-items: center; }
@media (max-width: 1100px) { .cm-kpibar { grid-template-columns: repeat(3, minmax(0, 1fr)); } .cm-fc { grid-template-columns: 1fr 1fr; } }

/* Route Advisor */
.cm-verdict { padding: 0.4rem 0 0.2rem 0; }
.cm-verdict .v { font-size: 1.6rem; font-weight: 500; letter-spacing: -0.03em; color: var(--cm-ink); }
.cm-verdict .w { font-size: 0.95rem; color: var(--cm-ink2); margin-top: 0.3rem; }
.cm-verdict .x { font-size: 0.8rem; color: var(--cm-muted); margin-top: 0.5rem; }
.cm-verdict .x b { color: var(--cm-ink2); font-weight: 500; }
.cm-routes .rr { display: grid; grid-template-columns: 9.5rem minmax(0, 1fr) 11rem 3.5rem 6.5rem 4rem; gap: 1rem; align-items: center;
  padding: 1rem 0.2rem; border-bottom: 1px solid var(--cm-hair); font-size: 0.86rem; color: var(--cm-ink); }
.cm-routes .rr:last-child { border-bottom: none; }
.cm-routes .rr.th { font-size: 0.76rem; color: var(--cm-muted); padding-top: 0.2rem; padding-bottom: 0.6rem; }
.cm-routes .num { font-variant-numeric: tabular-nums; }
.cm-routes .score { font-weight: 500; }
.cm-routes .warn { color: #8a5e08; }
.cm-routes .tag { display: inline-flex; width: fit-content; padding: 0.3rem 0.75rem; border-radius: 999px; font-size: 0.78rem;
  font-weight: 500; background: var(--cm-bg); color: var(--cm-ink2); }
.cm-routes .tag.best { background: var(--cm-dark); color: #fff; }
.cm-routes .tag.sched { background: transparent; border: 1px solid #dfe2e7; }
.cm-routes .chain { display: flex; flex-wrap: wrap; align-items: center; gap: 0.35rem; }
.cm-routes .pt { padding: 0.28rem 0.6rem; border-radius: 8px; background: var(--cm-bg); font-weight: 500; font-size: 0.82rem; }
.cm-routes .pt.end { background: var(--cm-blue-pale); color: #2a6cc4; }
.cm-routes .pt.hot { background: #fcf3df; color: #8a5e08; box-shadow: inset 0 0 0 1px #f0d48f; }
.cm-routes .arr { color: var(--cm-muted); }
.cm-routes .why { display: block; font-size: 0.78rem; color: var(--cm-muted); margin-top: 0.45rem; }
@media (max-width: 900px) { .cm-routes .rr { grid-template-columns: 1fr 1fr; } .cm-routes .rr.th { display: none; } }

/* Methodology flow diagram */
.cm-flow { display: flex; align-items: stretch; gap: 0.5rem; flex-wrap: wrap; }
.cm-flow .node { flex: 1 1 140px; background: var(--cm-bg); border-radius: 14px; padding: 0.75rem 0.9rem; font-size: 0.8rem;
  color: var(--cm-ink2); }
.cm-flow .node b { display: block; color: var(--cm-ink); font-size: 0.88rem; font-weight: 500; margin-bottom: 0.15rem; }
.cm-flow .node.navy { box-shadow: inset 3px 0 0 var(--cm-dark); }
.cm-flow .node.orange { box-shadow: inset 3px 0 0 var(--cm-amber); }
.cm-flow .node.blue { box-shadow: inset 3px 0 0 var(--cm-blue); }
.cm-flow .node.dashed { background: transparent; border: 1px dashed #c9ced6; }
.cm-flow .arrow { flex: none; align-self: center; color: var(--cm-muted); font-size: 1.1rem; }
.cm-lane-label { font-size: 0.78rem; font-weight: 500; color: var(--cm-muted); margin: 0.2rem 0 0.4rem 0; }

/* Native widgets */
div[data-baseweb="select"] > div, div[data-testid="stDateInput"] div[data-baseweb="input"] {
  border-radius: var(--cm-r-pill) !important; background: var(--cm-surface) !important; border: none !important;
  min-height: 2.9rem; padding-left: 0.4rem; }
[class*="st-key-card_"] div[data-baseweb="select"] > div, [class*="st-key-card_"] div[data-testid="stDateInput"] div[data-baseweb="input"] {
  background: var(--cm-bg) !important; }
div[data-testid="stDateInput"] input { background: transparent !important; }
/* No "clear" (x) icons on pickers: every control here always holds a value or has its own reset. */
div[data-baseweb="select"] div:has(> svg[title="Clear value"]), div[data-testid="stDateInput"] div:has(> svg[title="Clear value"]) { display: none; }
.stButton button:disabled { background: var(--cm-bg) !important; border-color: transparent !important; color: var(--cm-muted) !important; opacity: 1; }
div[data-baseweb="select"] span, div[data-baseweb="select"] input { font-size: 0.9rem; }
.stApp label p { font-size: 0.8rem !important; color: var(--cm-ink2) !important; }
.stButton button, div[data-testid="stPopover"] button, a[data-testid="stPageLink-NavLink"] { border-radius: var(--cm-r-pill) !important; }
.stButton button { background: var(--cm-surface); border: 1px solid #dfe2e7; color: var(--cm-ink); font-weight: 500; min-height: 2.6rem; }
[class*="st-key-card_"] .stButton button { background: var(--cm-bg); border-color: transparent; }
.stButton button:hover:not(:disabled) { border-color: var(--cm-dark) !important; color: var(--cm-ink); }
.stButton button[kind="primary"] { background: var(--cm-dark) !important; border-color: var(--cm-dark) !important; color: #fff !important; }
.stButton button[kind="primary"] p, .stButton button[kind="primary"] span { color: #fff !important; white-space: nowrap; }
div[data-testid="stPopover"] button { background: transparent; border: none; font-size: 0.8rem; min-height: 2rem; padding: 0.1rem 0.4rem;
  color: var(--cm-muted); }
div[data-testid="stButtonGroup"] { background: var(--cm-bg); border-radius: var(--cm-r-pill); padding: 0.25rem; width: fit-content; }
div[data-testid="stButtonGroup"] button { border-radius: var(--cm-r-pill) !important; border: none !important; background: transparent !important;
  font-size: 0.86rem; min-height: 2.3rem; padding: 0.15rem 1rem; }
div[data-testid="stButtonGroup"] button[kind$="Active"] { background: var(--cm-dark) !important; color: #fff !important; }
div[data-testid="stButtonGroup"] button[kind$="Active"] p { color: #fff !important; }
div[data-testid="stExpander"] details { border-radius: var(--cm-r-card) !important; border: none !important; background: var(--cm-surface); }
div[data-testid="stExpander"] summary { padding: 1rem 1.4rem; }
div[data-testid="stExpander"] summary p { font-weight: 500; color: var(--cm-ink); font-size: 0.95rem; }
div[data-testid="stMetric"] { background: transparent; padding: 0; }
div[data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 500; letter-spacing: -0.03em; color: var(--cm-ink); }
div[data-testid="stAlert"] { border-radius: var(--cm-r-md); background: transparent; border: none; }
div[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) { background: var(--cm-bg) !important; color: var(--cm-ink2); }
div[data-testid="stDataFrame"] { border-radius: var(--cm-r-md); overflow: hidden; }
div[data-testid="stToggle"] label p { font-size: 0.84rem !important; }
div[data-testid="stDeckGlJsonChart"] { border-radius: 18px; overflow: hidden; }

@media (max-width: 900px) {
  .block-container { padding: 1rem 1rem 2rem 1rem !important; }
  .cm-h1 { font-size: 2rem; }
}
@media (max-width: 640px) {
  .st-key-cm_nav > div { max-width: 100% !important; }
  .st-key-cm_nav_pills { flex-wrap: nowrap !important; overflow-x: auto; max-width: 100%; scrollbar-width: none; }
  .cm-brand { padding-bottom: 0.6rem; }
  .cm-kpibar, .cm-fc, .cm-kpistrip { grid-template-columns: 1fr 1fr; }
}
</style>
"""

_LOGO = ('<svg width="30" height="30" viewBox="0 0 32 32" fill="none" aria-hidden="true">'
         '<circle cx="16" cy="16" r="14" stroke="#16191f" stroke-width="2.2"/>'
         '<path d="M6 19c3.2 2.4 6.6 2.4 10 0s6.8-2.4 10 0" stroke="#16191f" stroke-width="2.2" stroke-linecap="round"/>'
         '<path d="M16 7v9M11.5 11.5h9" stroke="#16191f" stroke-width="2.2" stroke-linecap="round"/></svg>')


def mode() -> str:
    """The app uses one light design (set in .streamlit/config.toml)."""
    return "light"


def palette() -> dict[str, str]:
    """Colours for charts and the map, kept for callers that read them by role."""
    return {"text": INK, "muted": INK2, "accent": BLUE, "chart_text": INK2, "grid": GRID, "track": TRACK, "map": "light"}


def theme() -> None:
    """Apply the shared design system. Call once near the top of every page."""
    st.markdown(_CSS, unsafe_allow_html=True)


# --- Layout --------------------------------------------------------------------------------------------------------
def navbar(active: str) -> None:
    """Brand, horizontal pill navigation and the permanent synthetic/offline status pill."""
    with st.container(key="cm_nav", horizontal=True, vertical_alignment="center", horizontal_alignment="distribute"):
        st.markdown(f'<div class="cm-brand">{_LOGO}CargoMind</div>', unsafe_allow_html=True)
        with st.container(key="cm_nav_pills", horizontal=True, vertical_alignment="center", gap=None):
            for i, (label, page, icon) in enumerate(NAV):
                key = f"cm_nav_active_{i}" if label == active else f"cm_nav_link_{i}"
                with st.container(key=key):
                    try:
                        st.page_link(page, label=label, icon=icon)
                    except Exception:  # page registry absent (a page run on its own, e.g. in tests): plain label
                        css = "cm-pill active" if label == active else "cm-pill"
                        st.markdown(f'<span class="{css}">{label}</span>', unsafe_allow_html=True)
        st.markdown('<span class="cm-status-pill" title="All data is synthetic; there is no live feed">'
                    '<span class="dot"></span>Synthetic · offline</span>', unsafe_allow_html=True)


def header(title: str, lede: str, eyebrow: str | None = None) -> None:
    """Page title and one-line description (left side of the header row)."""
    eb = f'<div class="cm-eyebrow">{eyebrow}</div>' if eyebrow else ""
    st.markdown(f'{eb}<div class="cm-h1">{title}</div><div class="cm-lede">{lede}</div>', unsafe_allow_html=True)


def section(text: str) -> None:
    st.markdown(f'<div class="cm-section">{text}</div>', unsafe_allow_html=True)


def panel(name: str):
    """A frosted card. Use as a context manager: `with ui.panel("forecast"): ...`."""
    return st.container(key=f"card_{name}")


def tile(name: str):
    """A solid white inner card, for grouping inside a panel."""
    return st.container(key=f"tile_{name}")


def row(name: str):
    """Wrap a set of columns whose cards should stretch to equal height."""
    return st.container(key=f"row_{name}")


def card_head(title: str, subtitle: str | None = None, right: str = "") -> None:
    """Card heading: title and source/model subtitle on the left, an optional badge on the right."""
    sub = f'<div class="cm-card-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="cm-card-head"><div><div class="cm-card-title">{title}</div>{sub}</div>'
                f'<div>{right}</div></div>', unsafe_allow_html=True)


def panel_title(text: str, subtitle: str | None = None) -> None:
    card_head(text, subtitle)


def context_bar(items: dict[str, str]) -> None:
    """What the results below describe: the selected port, shipment, hour, etc."""
    html = "".join(f'<span class="item"><b>{k}</b>{v}</span>' for k, v in items.items())
    st.markdown(f'<div class="cm-context" aria-label="Current selection">{html}</div>', unsafe_allow_html=True)


# --- Text blocks ---------------------------------------------------------------------------------------------------
def badge(text: str, tone: str = "neutral", dot: str | None = None) -> str:
    d = f'<span class="dot" style="background:{dot}"></span>' if dot else ""
    return f'<span class="cm-badge {tone}">{d}{text}</span>'


def class_chip(cls: str) -> str:
    """HTML badge for a delay class: coloured dot plus the class name."""
    return f'<span class="cm-chip"><span class="dot" style="background:{CLASS_COLORS[cls]}"></span>{cls}</span>'


def class_badge(cls: str) -> str:
    tone = {"On-Time": "green", "At-Risk": "amber", "Critical": "red"}[cls]
    return badge(cls, tone, CLASS_COLORS[cls])


def flag_chip(flagged: bool) -> str:
    color = FLAG_COLOR if flagged else BLUE
    text = "Unusual conditions detected" if flagged else "Not flagged"
    return f'<span class="cm-chip"><span class="dot" style="background:{color}"></span>{text}</span>'


def kpi(label: str, value: str, unit: str = "", accent: bool = False) -> str:
    unit_html = f'<span class="unit">{unit}</span>' if unit else ""
    return f'<div class="cm-kpi{" accent" if accent else ""}"><div class="label">{label}</div><div class="value">{value}{unit_html}</div></div>'


def kpis(items: list[str]) -> None:
    st.markdown(f'<div class="cm-kpis">{"".join(items)}</div>', unsafe_allow_html=True)


def big(value: str, unit: str = "", right: str = "") -> None:
    u = f'<span class="unit">{unit}</span>' if unit else ""
    st.markdown(f'<div style="display:flex;align-items:center;gap:0.6rem;flex-wrap:wrap"><span class="cm-big">{value}{u}</span>'
                f'{right}</div>', unsafe_allow_html=True)


def rows(items: dict[str, str]) -> None:
    """Label / value list, one per line, for compact metric panels."""
    html = "".join(f'<div class="cm-row"><span class="k">{k}</span><span class="v">{v}</span></div>' for k, v in items.items())
    st.markdown(html, unsafe_allow_html=True)


def note(text: str, warn: bool = False) -> None:
    st.markdown(f'<div class="cm-note{" warn" if warn else ""}">{text}</div>', unsafe_allow_html=True)


def seg_bar(value: float, color: str, segments: int = 18) -> str:
    """Segmented bar for a 0-1 quantity (display only; no threshold is implied)."""
    n = round(max(0.0, min(1.0, value)) * segments)
    cells = "".join(f'<i style="background:{color}"></i>' if i < n else "<i></i>" for i in range(segments))
    return f'<span class="cm-seg" aria-hidden="true">{cells}</span>'


def meter(lo: float, hi: float, value: float, mark: float | None = None, lo_label: str = "", hi_label: str = "") -> None:
    """Range meter: track from lo to hi, a knob at `value` and an optional thin mark (e.g. the observed value)."""
    span = (hi - lo) or 1.0
    pos = lambda v: max(0.0, min(100.0, (v - lo) / span * 100))  # noqa: E731
    mk = f'<span class="mark" style="left:{pos(mark):.1f}%"></span>' if mark is not None else ""
    st.markdown(f'<div class="cm-meter"><span class="fill" style="width:{pos(value):.1f}%"></span>{mk}'
                f'<span class="knob" style="left:{pos(value):.1f}%"></span></div>'
                f'<div class="cm-meter-legend"><span>{lo_label}</span><span>{hi_label}</span></div>', unsafe_allow_html=True)


def help_popover(text: str) -> None:
    """Inline 'What does this mean?' explanation next to a panel."""
    with st.popover("What does this mean?", icon=":material/help:"):
        st.markdown(text)


def error_box(message: str) -> None:
    """A user-facing error: plain words only, no paths, stack traces or internal names."""
    st.error(message, icon=":material/error:")


# --- Controls (shared across pages) --------------------------------------------------------------------------------
def port_picker(ports: list[str], names: dict[str, str], key: str, required: bool, label: str = "Port",
                placeholder: str = "Choose a port") -> str | None:
    """Port selectbox seeded from the shared selection, so Overview and Port Forecast follow the same port."""
    shared = st.session_state.get(SHARED_PORT)
    if key not in st.session_state and (shared in ports or required):
        st.session_state[key] = shared if shared in ports else ports[0]
    value = st.selectbox(label, ports, index=None, key=key, placeholder=placeholder, label_visibility="collapsed",
                         format_func=lambda p: f"{p} · {names[p]}" if p in names else p)
    if value is not None:
        st.session_state[SHARED_PORT] = value
    return value


def hour_picker(first_end: pd.Timestamp, last: pd.Timestamp, default: pd.Timestamp, key: str,
                label: str = "Hour (UTC)", columns=None) -> pd.Timestamp:
    """Date + hour selection limited to [first_end, last] (UTC), seeded from the shared replay hour.

    `columns` is an optional pair of containers to place the date and the hour in.
    """
    seed = pd.Timestamp(st.session_state.get(SHARED_HOUR, default))
    seed = min(max(seed, first_end), last)
    dk, hk = f"{key}_date", f"{key}_hour"
    if dk not in st.session_state:
        st.session_state[dk] = seed.date()
    c1, c2 = columns if columns else (st.container(), st.container())
    with c1:
        day = st.date_input(f"{label}: date", value=None, min_value=first_end.date(), max_value=last.date(), key=dk,
                            format="YYYY-MM-DD", label_visibility="collapsed") or seed.date()
    hours = [h for h in range(24) if first_end <= pd.Timestamp(datetime.combine(day, time(h)), tz="UTC") <= last]
    if st.session_state.get(hk) not in hours:
        st.session_state[hk] = seed.hour if seed.hour in hours else hours[-1]
    with c2:
        hour = st.selectbox(f"{label}: hour", hours, key=hk, format_func=lambda h: f"{h:02d}:00 UTC", label_visibility="collapsed")
    ts = pd.Timestamp(datetime.combine(day, time(hour)), tz="UTC")
    st.session_state[SHARED_HOUR] = ts
    return ts


# --- Charts --------------------------------------------------------------------------------------------------------
def style(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    """Shared axes, legend and transparent background for any top-level Altair chart."""
    return (chart.properties(background="transparent")
            .configure_axis(labelColor=MUTED, titleColor=INK2, gridColor=GRID, domainColor=GRID, tickColor=GRID,
                            labelFontSize=11, titleFontSize=11, labelFont=FONT, titleFont=FONT, titleFontWeight=500,
                            labelPadding=6)
            .configure_legend(labelColor=INK2, titleColor=INK2, labelFont=FONT, labelFontSize=11, symbolSize=70)
            .configure_view(strokeWidth=0))


def gauge(percent: float, label: str, color: str = BLUE, size: int = 120) -> alt.Chart:
    """Thin ring gauge for a 0-100 % quantity (used for utilisation only), value in the centre."""
    pct = max(0.0, min(100.0, float(percent)))
    data = pd.DataFrame({"part": ["value", "rest"], "amount": [pct, 100 - pct]})
    r = size / 2 - 6
    ring = alt.Chart(data).mark_arc(innerRadius=r - 7, outerRadius=r, cornerRadius=6).encode(
        theta=alt.Theta("amount:Q", stack=True),
        color=alt.Color("part:N", scale=alt.Scale(domain=["value", "rest"], range=[color, TRACK]), legend=None),
        tooltip=alt.value(None))
    text = alt.Chart(pd.DataFrame({"t": [f"{pct:.0f}%"]})).mark_text(fontSize=17, fontWeight=600, color=INK, font=FONT,
                                                                        dy=-5).encode(text="t:N")
    caption = alt.Chart(pd.DataFrame({"t": [label]})).mark_text(dy=12, fontSize=10, color=MUTED, font=FONT).encode(text="t:N")
    return style((ring + text + caption).properties(height=size, width=size))


def probability_chart(probabilities: dict[str, float]) -> alt.Chart:
    """Three raw probabilities as labelled bars on a 0-1 scale of their own."""
    data = pd.DataFrame({"class": CLASS_ORDER, "probability": [probabilities[c] for c in CLASS_ORDER]})
    base = alt.Chart(data).encode(y=alt.Y("class:N", sort=CLASS_ORDER, title=None, axis=alt.Axis(labelFontSize=12, labelColor=INK2)))
    track = base.mark_bar(cornerRadius=6, color=TRACK, height=14).encode(x=alt.X("one:Q", scale=alt.Scale(domain=[0, 1])))
    bars = base.mark_bar(cornerRadius=6, height=14).encode(
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), title="Raw model probability (not calibrated)",
                axis=alt.Axis(format=".1f", tickCount=5)),
        color=alt.Color("class:N", legend=None, scale=alt.Scale(domain=CLASS_ORDER, range=[CLASS_COLORS[c] for c in CLASS_ORDER])),
        tooltip=[alt.Tooltip("class:N"), alt.Tooltip("probability:Q", format=".3f")])
    labels = base.mark_text(align="left", dx=6, color=INK, font=FONT, fontWeight=600).encode(
        x="probability:Q", text=alt.Text("probability:Q", format=".2f"))
    return style(alt.layer(track, bars, labels).transform_calculate(one="1").properties(height=150))


def anomaly_scale(score: float, threshold: float, flagged: bool) -> alt.Chart:
    """The anomaly score on its own scale with the flag threshold marked. Never shares an axis with probabilities."""
    lo, hi = min(score, threshold), max(score, threshold)
    pad = max((hi - lo) * 0.6, 0.05)
    domain = [lo - pad, hi + pad]
    x = alt.X("x:Q", scale=alt.Scale(domain=domain), title="Anomaly score (own scale; higher = more unusual)",
              axis=alt.Axis(offset=22, tickCount=6))
    track = alt.Chart(pd.DataFrame({"x": [domain[0]], "x2": [domain[1]]})).mark_rule(
        strokeWidth=10, color=TRACK, strokeCap="round").encode(x=x, x2="x2:Q")
    tick = alt.Chart(pd.DataFrame({"x": [threshold], "t": ["Flag threshold"]}))
    marker = alt.Chart(pd.DataFrame({"x": [score], "t": ["This shipment"]}))
    color = FLAG_COLOR if flagged else BLUE
    layers = [track,
              tick.mark_tick(thickness=2, size=28, color=INK2).encode(x="x:Q"),
              tick.mark_text(dy=-24, fontSize=11, color=INK2, font=FONT).encode(x="x:Q", text="t:N"),
              marker.mark_point(shape="circle", filled=True, size=230, color=color, stroke="white", strokeWidth=3).encode(
                  x="x:Q", tooltip=[alt.Tooltip("x:Q", format=".3f", title="Anomaly score")]),
              marker.mark_text(dy=24, fontSize=11, color=INK, font=FONT, fontWeight=600).encode(x="x:Q", text="t:N")]
    return style(alt.layer(*layers).properties(height=96, padding={"top": 26, "bottom": 4, "left": 6, "right": 10}))


def area_line(data: pd.DataFrame, x: alt.X, y_field: str, y_title: str, tooltip: list, height: int = 190) -> alt.Chart:
    """Soft area + line in the accent blue (trend charts)."""
    grad = alt.Gradient(gradient="linear", x1=1, x2=1, y1=1, y2=0,
                        stops=[alt.GradientStop(color="rgba(47,125,225,0.0)", offset=0),
                               alt.GradientStop(color="rgba(47,125,225,0.22)", offset=1)])
    lo, hi = float(data[y_field].min()), float(data[y_field].max())
    pad = max((hi - lo) * 0.35, 0.5)
    y = alt.Y(f"{y_field}:Q", title=y_title, scale=alt.Scale(domain=[lo - pad, hi + pad], nice=False), axis=alt.Axis(tickCount=4))
    base = alt.Chart(data).encode(x=x, y=y)
    return style(alt.layer(base.mark_area(color=grad, line=False, interpolate="monotone", clip=True),
                           base.mark_line(color=BLUE, strokeWidth=2.2, interpolate="monotone"),
                           base.mark_point(color=BLUE, filled=True, size=46, stroke="white", strokeWidth=1.5).encode(tooltip=tooltip))
                 .properties(height=height))


def thin_bars(data: pd.DataFrame, x: alt.X, y_field: str, y_title: str | None, tooltip: list, height: int = 140) -> alt.Chart:
    """Thin rounded bars (the reference's 'wave' bars), shaded by value."""
    return style(alt.Chart(data).mark_bar(width=4, cornerRadius=3).encode(
        x=x, y=alt.Y(f"{y_field}:Q", title=y_title, axis=alt.Axis(tickCount=3)),
        color=alt.Color(f"{y_field}:Q", legend=None, scale=alt.Scale(range=[BLUE_SOFT, BLUE])),
        tooltip=tooltip).properties(height=height))


# --- ML interpretation ---------------------------------------------------------------------------------------------
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
