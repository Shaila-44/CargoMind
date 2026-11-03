"""Route service: prototype alternative-route search over the synthetic port network (no model, no live data).

The graph has one directed edge per *direct* service in ML/Dataset/Processed/port_network.csv, weighted by that row's
`route_distance_km`. Scheduled transshipment routes are not turned into edges, because their individual legs are not
recorded as services. Every recommended leg is therefore a real row of the route table. The edges are prototype
network connections from synthetic data, not verified or bookable shipping services.

Candidate routes are simple paths (no repeated ports) with at most MAX_LEGS legs. Intermediate ports are excluded only
when the data shows a disruption at the replay hour, or when the user names a port as a what-if scenario. Candidates
are ranked by a transparent heuristic (lower is better) built from fields that exist at the replay hour:
disruption exposure, dwell, utilisation, number of intermediate stops and distance relative to the scheduled route.

    from services import routes
    result = routes.advise("INMAA", "NLRTM", "2025-02-11 15:00+00:00", scenario_port=None)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import networkx as nx
import pandas as pd

from cargomind_eda.config import PROCESSED
from services import port_ops

__all__ = ["WEIGHTS", "MAX_LEGS", "MAX_ALTERNATIVES", "Route", "Advice", "graph", "advise"]

# Heuristic weights (lower score = preferred). Change here to re-balance the ranking.
WEIGHTS = {
    "exposure": 3.0,   # per intermediate port: affected capacity share (0-1) of an active disruption, or 1 for a scenario port
    "dwell": 1.0,      # per intermediate port: dwell at the replay hour / network mean dwell
    "utilisation": 0.5,  # per intermediate port: utilisation at the replay hour / 100
    "stops": 0.75,     # per intermediate stop (transshipment)
    "distance": 1.0,   # route distance / scheduled route distance
}
MAX_LEGS = 3           # at most two intermediate stops
MAX_ALTERNATIVES = 3


@dataclass
class Route:
    ports: list[str]
    kind: str                      # "scheduled" or "alternative"
    distance_km: float
    stops: int
    exposure: float                # sum over intermediate ports
    exposed_ports: list[str]
    score: float
    parts: dict[str, float] = field(default_factory=dict)
    label: str = ""
    reasons: list[str] = field(default_factory=list)


@dataclass
class Advice:
    origin: str
    destination: str
    scheduled: Route | None
    alternatives: list[Route]
    disrupted: dict[str, str]      # port -> why it counts as disrupted (observed event or scenario)
    endpoint_notes: list[str]
    message: str = ""              # set when no alternative exists


@lru_cache(maxsize=1)
def _network() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "port_network.csv",
                       usecols=["origin_port_id", "destination_port_id", "route_type", "transshipment_port_id",
                                "route_distance_km"])


@lru_cache(maxsize=1)
def graph() -> nx.DiGraph:
    """Directed graph of direct services only (one edge per direct row of the route table)."""
    g = nx.DiGraph()
    for r in _network()[_network()["route_type"] == "direct"].itertuples(index=False):
        g.add_edge(r.origin_port_id, r.destination_port_id, distance_km=float(r.route_distance_km))
    return g


def _scheduled(origin: str, destination: str) -> tuple[list[str], float] | None:
    rows = _network()[(_network()["origin_port_id"] == origin) & (_network()["destination_port_id"] == destination)]
    if rows.empty:
        return None
    r = rows.iloc[0]
    via = [r["transshipment_port_id"]] if r["route_type"] == "transshipment" and pd.notna(r["transshipment_port_id"]) else []
    return [origin, *via, destination], float(r["route_distance_km"])


def _conditions(hour_utc) -> tuple[pd.DataFrame, dict[str, float], dict[str, str]]:
    """Observed state at the replay hour, exposure per port (0-1) and the reason a port counts as disrupted."""
    snap = port_ops.snapshot(hour_utc).set_index("port_id")
    exposure, why = {}, {}
    for ev in port_ops.active_events(hour_utc).itertuples(index=False):
        share = float(ev.affected_capacity_pct) / 100
        if share > exposure.get(ev.port_id, 0):
            exposure[ev.port_id] = share
            why[ev.port_id] = f"{ev.event_type.replace('_', ' ')} · {ev.affected_capacity_pct:.0f}% capacity affected"
    for pid, row in snap.iterrows():  # telemetry flags without a matching event record still count
        if (row["disruption_active"] or row["labour_strike"]) and pid not in exposure:
            exposure[pid] = 1.0
            why[pid] = "labour strike" if row["labour_strike"] else "disruption flag"
    return snap, exposure, why


def _score(ports: list[str], distance: float, base_distance: float, snap: pd.DataFrame, exposure: dict[str, float],
           kind: str) -> Route:
    mids = ports[1:-1]
    mean_dwell = float(snap["dwell"].mean()) or 1.0
    parts = {
        "exposure": sum(exposure.get(p, 0.0) for p in mids),
        "dwell": sum(float(snap.loc[p, "dwell"]) / mean_dwell for p in mids if p in snap.index),
        "utilisation": sum(float(snap.loc[p, "utilization"]) / 100 for p in mids if p in snap.index),
        "stops": float(len(mids)),
        "distance": distance / base_distance if base_distance else 1.0,
    }
    score = sum(WEIGHTS[k] * v for k, v in parts.items())
    return Route(ports=ports, kind=kind, distance_km=distance, stops=len(mids), exposure=parts["exposure"],
                 exposed_ports=[p for p in mids if exposure.get(p, 0) > 0], score=score, parts=parts)


def _reasons(alt: Route, base: Route) -> list[str]:
    """Relative, data-backed comparisons with the scheduled route (no invented times or costs)."""
    out = []
    if alt.exposure < base.exposure - 1e-9:
        out.append("Lower disruption exposure")
    elif alt.exposure > base.exposure + 1e-9:
        out.append("Higher disruption exposure")
    else:
        out.append("No disrupted stops" if alt.exposure == 0 else "Same disruption exposure")
    if alt.stops < base.stops:
        out.append("Fewer intermediate stops")
    elif alt.stops > base.stops:
        out.append("Additional transshipment required")
    if base.distance_km:
        diff = (alt.distance_km - base.distance_km) / base.distance_km
        if abs(diff) >= 0.05:
            out.append(f"{abs(diff):.0%} {'longer' if diff > 0 else 'shorter'} by recorded distance")
        else:
            out.append("Similar recorded distance")
    if alt.parts["dwell"] and base.stops and alt.parts["dwell"] / max(alt.stops, 1) < base.parts["dwell"] / base.stops:
        out.append("Less congested stopover")
    return out


def advise(origin: str, destination: str, hour_utc, scenario_port: str | None = None) -> Advice:
    """Scheduled route plus up to MAX_ALTERNATIVES alternatives that avoid disrupted intermediate ports."""
    if origin == destination:
        return Advice(origin, destination, None, [], {}, [], "Origin and destination are the same port, so there is no route to plan.")
    snap, exposure, why = _conditions(hour_utc)
    if scenario_port:
        exposure[scenario_port] = max(exposure.get(scenario_port, 0.0), 1.0)
        why[scenario_port] = (why[scenario_port] + " · also selected as scenario") if scenario_port in why else "selected as a what-if scenario"
    notes = [f"{p} is the {'origin' if p == origin else 'destination'} and is disrupted ({why[p]}); no route can avoid it."
             for p in (origin, destination) if p in why]

    sched = _scheduled(origin, destination)
    base_distance = sched[1] if sched else 0.0
    scheduled = _score(sched[0], sched[1], base_distance, snap, exposure, "scheduled") if sched else None
    if scheduled:
        scheduled.label = "Scheduled route"

    g = graph()
    blocked = {p for p in why if p not in (origin, destination)}  # only ports the data (or the scenario) marks disrupted
    usable = g.subgraph([n for n in g.nodes if n not in blocked])
    candidates = []
    if origin in usable and destination in usable:
        for path in nx.all_simple_paths(usable, origin, destination, cutoff=MAX_LEGS):
            if scheduled and path == scheduled.ports:
                continue
            dist = sum(g[a][b]["distance_km"] for a, b in zip(path, path[1:]))  # every leg is a direct-service edge
            candidates.append(_score(path, dist, base_distance or dist, snap, exposure, "alternative"))
    candidates.sort(key=lambda r: (r.score, r.stops, r.distance_km))
    alternatives = candidates[:MAX_ALTERNATIVES]
    for i, r in enumerate(alternatives):
        r.label = "Best alternative" if i == 0 else f"Alternative {i + 1}"
        r.reasons = _reasons(r, scheduled) if scheduled else []

    message = ""
    if not alternatives:
        avoided = ", ".join(sorted(blocked)) or "none"
        message = (f"No alternative path of up to {MAX_LEGS} direct services connects {origin} to {destination} while avoiding "
                   f"the disrupted ports ({avoided}). {origin} has direct services to "
                   f"{', '.join(sorted(g.successors(origin))) or 'no ports'}.")
    return Advice(origin, destination, scheduled, alternatives, why, notes, message)
