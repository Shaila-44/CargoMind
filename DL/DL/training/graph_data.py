"""Graph view of the existing DL dataset for the GCN + LSTM experiment.

Nothing new is generated. All ten ports share one hourly grid (hours 0..999), so a graph snapshot at hour t is
the ten ports' 12-hour windows ending at t, stacked into X[t] of shape [12, 10, 8]:

* inputs: the hourly panel (DL/Dataset/processed/hourly_panel.csv) passed through the dataset build's own
  train-fitted transform (cargomind_dl.sequences.transform, saved scalers.json), plus the dwell-reconstructed
  flag from provenance - i.e. exactly the 8 channels of dataset.load_split (verified against the stored arrays
  in the tests);
* targets: next-hour dwell for every port; a port's target is used only if that (port, t) window was KEPT by
  the dataset build (windows.csv `keep`). Reconstructed targets and boundary-crossing windows stay excluded,
  so the GNN is trained and evaluated on exactly the same windows as the other models (masked, not filled);
* snapshot hours per split follow the dataset's embargo rule: all 12 input hours and the target hour lie in
  the same split.

Adjacency: an undirected edge joins two ports if a DIRECT route exists in either direction in
port_network.csv. Transshipment routes are not turned into edges (their legs are not routes in the table), and
no edge is added that the network does not contain.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cargomind_dl import config as data_config
from cargomind_dl.sequences import transform

CHANNELS = list(data_config.FEATURES) + ["dwell_reconstructed"]


def port_order() -> list[str]:
    return json.loads((data_config.DATASET / "feature_schema.json").read_text())["arrays"]["port_index"]["ports"]


def direct_route_adjacency(ports: list[str] | None = None) -> np.ndarray:
    """0/1 symmetric adjacency (no self-loops) from direct routes in port_network.csv, in `ports` order."""
    ports = ports or port_order()
    net = pd.read_csv(data_config.PROCESSED / "port_network.csv")
    unknown = set(net["origin_port_id"]) | set(net["destination_port_id"])
    if not unknown <= set(ports):
        raise ValueError(f"network contains ports outside the dataset: {sorted(unknown - set(ports))}")
    a = np.zeros((len(ports), len(ports)), dtype=np.int64)
    for o, d in net.loc[net["route_type"] == "direct", ["origin_port_id", "destination_port_id"]].itertuples(index=False):
        a[ports.index(o), ports.index(d)] = 1
    a = ((a + a.T) > 0).astype(np.int64)  # undirected: a direct sailing in either direction links the two ports
    np.fill_diagonal(a, 0)
    return a


def hourly_tensors(ports: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """F [hours, ports, 8] scaled model channels; Y_z [hours, ports] scaled dwell; Y_h [hours, ports] dwell in hours."""
    ports = ports or port_order()
    panel = pd.read_csv(data_config.DATASET / "hourly_panel.csv")
    scalers = json.loads((data_config.DATASET / "scalers.json").read_text())
    panel = panel.sort_values(["hour_index", "port_id"]).reset_index(drop=True)
    n_hours = panel["hour_index"].nunique()
    if len(panel) != n_hours * len(ports) or sorted(panel["port_id"].unique()) != sorted(ports):
        raise ValueError("hourly panel is not a complete hours x ports grid; graph snapshots cannot be formed")
    scaled = transform(panel, scalers)  # the dataset build's own transform with the saved scalers
    scaled["dwell_reconstructed"] = (panel["prov_dwell"] == 2).astype(float).to_numpy()
    order = [ports.index(p) for p in panel["port_id"]]
    F = np.zeros((n_hours, len(ports), len(CHANNELS)), dtype=np.float32)
    Yz = np.zeros((n_hours, len(ports)), dtype=np.float32)
    Yh = np.zeros((n_hours, len(ports)), dtype=np.float32)
    h = panel["hour_index"].to_numpy()
    F[h, order] = scaled[CHANNELS].to_numpy(np.float32)
    Yz[h, order] = scaled["dwell"].to_numpy(np.float32)
    Yh[h, order] = panel["dwell"].to_numpy(np.float32)
    return F, Yz, Yh, ports


@dataclass
class GraphSplit:
    split: str
    X: np.ndarray          # [S, 12, N, 8] float32
    y: np.ndarray          # [S, N] scaled next-hour dwell (meaningful only where mask)
    y_hours: np.ndarray    # [S, N]
    mask: np.ndarray       # [S, N] bool: the (port, t) window was kept by the dataset build
    t_hour: np.ndarray     # [S] last input hour t
    window_id: np.ndarray  # [S, N] windows.csv id of the kept window, -1 otherwise
    ports: list[str]


def load_graph_split(split: str) -> GraphSplit:
    F, Yz, Yh, ports = hourly_tensors()
    meta = json.loads((data_config.DATASET / "split_metadata.json").read_text())
    lo, hi = meta["bounds_hour_index"][split]
    W = data_config.WINDOW
    t_hours = np.arange(lo + W - 1, hi - 1)  # inputs t-11..t and target t+1 all inside [lo, hi)
    windows = pd.read_csv(data_config.DATASET / "windows.csv")
    kept = windows[windows["keep"] & (windows["split"] == split)]
    wid = np.full((len(t_hours), len(ports)), -1, dtype=np.int64)
    pos = {t: i for i, t in enumerate(t_hours)}
    for t, p, i in kept[["t_hour_index", "port_id", "window_id"]].itertuples(index=False):
        if t not in pos:
            raise ValueError(f"kept {split} window at t={t} lies outside the split's snapshot hours")
        wid[pos[t], ports.index(p)] = i
    X = np.stack([F[t - W + 1:t + 1] for t in t_hours])
    return GraphSplit(split=split, X=X, y=Yz[t_hours + 1], y_hours=Yh[t_hours + 1], mask=wid >= 0, t_hour=t_hours,
                      window_id=wid, ports=ports)
