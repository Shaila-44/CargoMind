"""Alternative Route Advisor: scheduled route vs prototype alternatives that avoid disrupted ports (graph heuristic).

Routing and scoring live in services/routes.py; this page only collects the query and presents the result. Conditions
(disruptions, dwell, utilisation) come from the replay hour shared with the other pages. Results are illustrative:
the network is the synthetic route table, not verified shipping services.
"""
from __future__ import annotations

import logging

import pydeck as pdk
import streamlit as st

from components import ui
from services import port_ops, routes

log = logging.getLogger("cargomind.routes_page")

SCENARIO_NONE = "Current disruptions only"

st.set_page_config(page_title="Route Advisor | CargoMind", layout="wide")
ui.theme()
ui.navbar("Route Advisor")

try:
    panel = port_ops.panel()
    first, last = panel["ts_hour"].min(), panel["ts_hour"].max()
    meta = port_ops.ports().set_index("port_id")
    names = meta["port_name"].to_dict()
    ports = sorted(routes.graph().nodes)
except Exception:
    log.exception("route data unavailable")
    ui.header("Alternative Route Advisor", "Compare the scheduled route with alternatives that avoid disrupted ports.")
    ui.error_box("Port network data is unavailable in this copy of the project, so routes cannot be planned.")
    st.stop()
label = lambda p: f"{p} · {names[p]}" if p in names else p  # noqa: E731

# --- Header: title and replay hour ---------------------------------------------------------------------------------
h_left, h_day, h_hour = st.columns([3.3, 0.9, 0.75], gap="small", vertical_alignment="bottom")
as_of = ui.hour_picker(first, last, default=last, key="ra", label="Conditions at (UTC)", columns=(h_day, h_hour))
with h_left:
    ui.header("Alternative Route Advisor", f"Scheduled route vs alternatives that avoid disrupted ports · conditions as of "
                                           f"{as_of:%a %d %b %Y, %H:%M} UTC (synthetic replay)")

# --- Query ---------------------------------------------------------------------------------------------------------
with ui.panel("ra_query"):
    q1, q2, q3, q4 = st.columns([1.2, 1.2, 1.35, 1], vertical_alignment="bottom", gap="medium")
    with q1:
        origin = st.selectbox("Origin port", ports, index=ports.index(st.session_state.get(ui.SHARED_PORT, ports[0]))
                              if st.session_state.get(ui.SHARED_PORT) in ports else 0, format_func=label, key="ra_origin")
    with q2:
        destination = st.selectbox("Destination port", ports, index=ports.index("NLRTM") if "NLRTM" in ports else 1,
                                   format_func=label, key="ra_destination")
    with q3:
        scenario = st.selectbox("What-if: also treat as disrupted (optional)", [SCENARIO_NONE, *ports], format_func=
                                lambda p: p if p == SCENARIO_NONE else label(p), key="ra_scenario",
                                help="Ports that the data shows as disrupted at this hour are always avoided as stopovers. "
                                     "Pick a port here to explore what happens if it is unavailable too.")
    with q4:
        if st.button("Find Alternative Routes", type="primary", width="stretch", icon=":material/alt_route:"):
            st.session_state["ra_last"] = (origin, destination, None if scenario == SCENARIO_NONE else scenario)

query = st.session_state.get("ra_last")
if not query:
    st.markdown('<div class="cm-empty">Choose an origin and a destination, then select <b>Find Alternative Routes</b>. '
                'The advisor compares the scheduled route with up to three alternatives built from direct services.</div>',
                unsafe_allow_html=True)
    st.stop()

o, d, s = query
if o == d:
    st.info("Origin and destination are the same port, so there is no route to plan. Choose two different ports.")
    st.stop()
try:
    advice = routes.advise(o, d, as_of, scenario_port=s)
except Exception:
    log.exception("route advice failed for %s -> %s", o, d)
    ui.error_box("Routes could not be computed right now.")
    st.stop()

best = advice.alternatives[0] if advice.alternatives else None
sched = advice.scheduled


# --- Recommendation ------------------------------------------------------------------------------------------------
def chain(r: routes.Route) -> str:
    pills = []
    for i, p in enumerate(r.ports):
        cls = "end" if i in (0, len(r.ports) - 1) else "mid"
        if p in advice.disrupted:
            cls += " hot"
        pills.append(f'<span class="pt {cls}" title="{names.get(p, p)}{" · " + advice.disrupted[p] if p in advice.disrupted else ""}">{p}</span>')
    return '<span class="arr">→</span>'.join(pills)


if best and sched and best.score < sched.score:
    verdict = f"Best alternative: {' → '.join(best.ports)}"
    why = " · ".join(best.reasons[:2])
elif best and sched:
    verdict = f"The scheduled route ({' → '.join(sched.ports)}) scores best at this hour"
    why = "Alternatives exist but score higher on the heuristic"
elif best:
    verdict, why = f"Best alternative: {' → '.join(best.ports)}", " · ".join(best.reasons[:2])
else:
    verdict, why = "No alternative route found", ""
disrupted_txt = " · ".join(f"<b>{p}</b> {w}" for p, w in advice.disrupted.items()) or "none"
st.markdown(f'<div class="cm-verdict"><div class="v">{verdict}</div><div class="w">{why}</div>'
            f'<div class="x">Disrupted at this hour: {disrupted_txt}</div></div>', unsafe_allow_html=True)
for note in advice.endpoint_notes:
    st.caption(note)

# --- Route comparison ----------------------------------------------------------------------------------------------
with ui.panel("ra_routes"):
    ui.card_head("Route comparison", "Scheduled route from the route table, alternatives from direct services · lower score is better")
    rows = ([sched] if sched else []) + advice.alternatives
    html = ('<div class="cm-routes"><div class="rr th"><span>Route</span><span>Port sequence</span><span>Disruption exposure</span>'
            '<span>Stops</span><span>Distance</span><span>Score</span></div>')
    for r in rows:
        tag_cls = "best" if r is best and (not sched or r.score < sched.score) else ("sched" if r.kind == "scheduled" else "")
        exposure = ", ".join(f"{p} ({advice.disrupted.get(p, '').split('·')[-1].strip() or 'disrupted'})" for p in r.exposed_ports) or "None"
        reasons = " · ".join(r.reasons) if r.reasons else ("Recorded route for this origin and destination" if r.kind == "scheduled" else "")
        html += (f'<div class="rr"><span class="tag {tag_cls}">{r.label}</span>'
                 f'<span><span class="chain">{chain(r)}</span><span class="why">{reasons}</span></span>'
                 f'<span class="{"warn" if r.exposed_ports else ""}">{exposure}</span><span class="num">{r.stops}</span>'
                 f'<span class="num">{r.distance_km:,.0f} km</span><span class="num score">{r.score:.2f}</span></div>')
    st.markdown(html + "</div>", unsafe_allow_html=True)
    if advice.message:
        st.info(advice.message)
    st.caption(f"Prototype recommendation · illustrative route analysis. Network edges are the direct services in the synthetic "
               f"route table, not verified or bookable services; no transit time or cost is estimated. Score = weighted sum of "
               f"disruption exposure, stopover dwell and utilisation, stops and distance relative to the scheduled route "
               f"(weights in services/routes.py). Up to {routes.MAX_LEGS} legs; disrupted stopovers are avoided.")

# --- Route map -----------------------------------------------------------------------------------------------------
with ui.panel("ra_map"):
    ui.card_head("Route map", "Grey: scheduled route · dark blue: best alternative · light blue: other alternatives · amber ports are disrupted")
    pos = {p: [float(meta.loc[p, "longitude"]), float(meta.loc[p, "latitude"])] for p in meta.index}
    legs = []
    for r, color, width in ([(sched, [140, 148, 160, 200], 3)] if sched else []) + \
            [(a, [22, 70, 160, 235] if a is best else [120, 170, 235, 170], 5 if a is best else 2) for a in advice.alternatives]:
        legs += [{"from": pos[x], "to": pos[y], "color": color, "width": width, "port_id": r.label, "port_name": "",
                  "status": f"leg {x} → {y}"}
                 for x, y in zip(r.ports, r.ports[1:]) if x in pos and y in pos]
    snap = port_ops.snapshot(as_of)
    on_route = {p for r in ([sched] if sched else []) + advice.alternatives for p in r.ports}
    pts = snap.assign(
        color=snap["port_id"].map(lambda p: [228, 160, 26, 235] if p in advice.disrupted else [67, 139, 232, 220]),
        radius=snap["port_id"].map(lambda p: 9 if p in (o, d) else 6 if p in on_route else 4),
        status=snap["port_id"].map(lambda p: advice.disrupted.get(p, "no disruption at this hour")))
    layers = [
        pdk.Layer("ArcLayer", data=legs, get_source_position="from", get_target_position="to", get_source_color="color",
                  get_target_color="color", get_width="width", great_circle=True, pickable=True),
        pdk.Layer("ScatterplotLayer", data=pts, get_position="[longitude, latitude]", get_fill_color="color",
                  get_radius="radius", radius_units="pixels", stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=2,
                  pickable=True),
    ]
    st.pydeck_chart(pdk.Deck(layers=layers, map_provider="carto", map_style="light",
                             initial_view_state=pdk.ViewState(latitude=25, longitude=40, zoom=1.2),
                             tooltip={"text": "{port_id} {port_name}\n{status}"}),
                    use_container_width=True, height=440, key=f"ra_map_{o}_{d}_{s}_{as_of:%Y%m%d%H}")
