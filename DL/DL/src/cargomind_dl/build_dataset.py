"""Build the DL-ready dataset: hourly panel -> split -> training-only scalers -> windows -> baselines
-> files under dl/dataset/, plus integrity checks that fail loudly."""
import hashlib
import importlib
import json
import platform

import numpy as np
import pandas as pd

from . import baselines, config, features, sequences

FEATURE_DOC = {
    "dwell": ("port_hourly_telemetry.container_dwell_hours", "h", "Average container dwell time at the port in hour t (also the target, one hour later)."),
    "utilization": ("port_hourly_telemetry.port_utilization_pct", "% of nominal capacity", "Throughput / capacity in hour t; censored at 100 when the port is saturated."),
    "labour_strike": ("port_hourly_telemetry.labour_status", "0/1", "1 if labour_status == 'strike' in hour t ('reduced' carries no signal and is treated as 0)."),
    "wind_gust": ("weather_hourly.wind_gusts_10m_kmh (same port_id, ts_hour)", "km/h", "Maximum 10 m wind gust in hour t; 2 negative values clipped to 0."),
    "disruption_active": ("disruption_events (event_start <= t < event_end, same port)", "0/1", "1 if a logged disruption is ongoing at hour t."),
    "disruption_hours_elapsed": ("disruption_events", "h", "Hours since the ongoing disruption episode began (earliest start among overlapping ongoing events); 0 if none."),
    "vessels_at_anchor": ("port_hourly_telemetry.vessels_at_anchor", "vessels", "Vessels waiting at anchor in hour t."),
}


def sha256(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def versions():
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "pandas"):
        out[pkg] = importlib.import_module(pkg).__version__
    return out


def build():
    panel, events, hours = features.build_panel()
    bounds = sequences.split_bounds(len(hours))
    sc = sequences.fit_scalers(panel, bounds)
    scaled = sequences.transform(panel, sc)
    meta, X, P, F, y_z = sequences.build_windows(panel, scaled, bounds)
    return panel, events, hours, bounds, sc, scaled, meta, X, P, F, y_z


def checks(panel, hours, bounds, sc, meta, X, y_z):
    """Integrity and leakage checks on the kept windows. Returns a list of (check, passed, evidence)."""
    W = config.WINDOW
    k = meta[meta["keep"]]
    res = []
    res.append(("target is the hour immediately after the last input hour",
                bool((k["target_hour_index"] - k["t_hour_index"] == 1).all()), f"{len(k)} windows"))
    first_input = k["t_hour_index"] - (W - 1)
    starts = k["split"].map({s: b[0] for s, b in bounds.items()})
    res.append(("every input hour lies in the target's split (no window straddles a boundary)", bool((first_input >= starts).all()),
                f"embargo removed {int(meta['crosses_split_boundary'].sum())} windows"))
    lo = {s: int((k.loc[k["split"] == s, "t_hour_index"] - (W - 1)).min()) for s in sequences.SPLITS}
    hi = {s: int(k.loc[k["split"] == s, "target_hour_index"].max()) for s in sequences.SPLITS}
    res.append(("all train hours < all val hours < all test hours (inputs and targets)",
                hi["train"] < lo["val"] and hi["val"] < lo["test"], f"train ..{hi['train']}, val {lo['val']}..{hi['val']}, test {lo['test']}.."))
    res.append(("no reconstructed target in any kept window", bool((k["target_status"] != "reconstructed").all()),
                f"{int((~meta['target_observed'] & ~meta['crosses_split_boundary']).sum())} excluded"))
    yh = sequences.inverse_dwell(y_z[k.index], k["port_id"].values, sc)
    res.append(("scaled target inverts to the stored dwell hours", bool(np.allclose(yh, k["target_hours"], atol=1e-3)),
                f"max |Δ| {np.abs(yh - k['target_hours']).max():.2e} h"))
    last = sequences.inverse_dwell(X[k.index, -1, 0], k["port_id"].values, sc)
    res.append(("last input step is hour t's dwell (not t+1)", bool(np.allclose(last, k["last_input_dwell_hours"], atol=1e-3)), ""))
    # independent recomputation of a sample of windows straight from the panel
    rng = np.random.default_rng(config.SEED)
    ok = True
    for i in rng.choice(k.index.values, 200, replace=False):
        r = meta.loc[i]
        g = panel[(panel["port_id"] == r["port_id"])].set_index("hour_index")
        ok &= np.allclose(g.loc[r["t_hour_index"] - W + 1:r["t_hour_index"], "dwell"].values,
                          sequences.inverse_dwell(X[i, :, 0], [r["port_id"]] * W, sc), atol=1e-3)
        ok &= np.isclose(g.loc[r["t_hour_index"] + 1, "dwell"], r["target_hours"])
    res.append(("200 random windows match an independent slice of the hourly panel", bool(ok), "inputs t-11..t, target t+1"))
    res.append(("scalers fitted on training-period hours only",
                sc["_fit_period"]["hour_index"] == list(bounds["train"]), f"hour_index {bounds['train']}"))
    res.append(("all model inputs finite", bool(np.isfinite(X[k.index]).all()), f"shape {X[k.index].shape}"))
    return res


def main():
    out = config.DATASET
    out.mkdir(parents=True, exist_ok=True)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    panel, events, hours, bounds, sc, scaled, meta, X, P, F, y_z = build()

    # determinism: a second build must give identical arrays
    _, _, _, _, sc2, _, meta2, X2, P2, F2, y2 = build()
    deterministic = np.array_equal(X, X2) and np.array_equal(P, P2) and np.array_equal(y_z, y2) and meta.equals(meta2) and sc == sc2

    meta.insert(0, "window_id", np.arange(len(meta)))
    # baselines (hours) on every window; EWMA alpha chosen on kept training windows
    dwell_h = np.stack([sequences.inverse_dwell(X[:, j, 0], meta["port_id"].values, sc) for j in range(config.WINDOW)], 1)
    tr = (meta["keep"] & (meta["split"] == "train")).values
    alpha, alpha_scores = baselines.choose_alpha(dwell_h[tr], meta["target_hours"].values[tr])
    preds = baselines.predictions(dwell_h, alpha)
    for name, p in preds.items():
        meta[f"baseline_{name}_hours"] = p
    kept = meta[meta["keep"]].reset_index(drop=True)
    base_eval = baselines.evaluate(kept, {n: p[meta["keep"].values] for n, p in preds.items()})

    res = checks(panel, hours, bounds, sc, meta, X, y_z)
    res.append(("dataset build is deterministic (two builds identical)", bool(deterministic), "X, provenance, y, metadata, scalers"))
    failed = [c for c, ok, _ in res if not ok]
    if failed:
        raise AssertionError(f"integrity checks failed: {failed}")

    # ----- save -----
    for s in sequences.SPLITS:
        m = (meta["keep"] & (meta["split"] == s)).values
        np.savez_compressed(out / f"{s}.npz", X=X[m], provenance=P[m], aux_flags=F[m], y=y_z[m],
                            y_hours=meta.loc[m, "target_hours"].to_numpy(np.float32),
                            last_dwell_hours=meta.loc[m, "last_input_dwell_hours"].to_numpy(np.float32),
                            port_index=meta.loc[m, "port_index"].to_numpy(np.int16),
                            target_hour_index=meta.loc[m, "target_hour_index"].to_numpy(np.int16),
                            window_id=meta.loc[m, "window_id"].to_numpy(np.int32))
    meta.to_csv(out / "windows.csv", index=False)
    panel_out = panel.drop(columns=["ts"]).copy()
    panel_out["split_of_hour"] = [sequences.split_of(h, bounds) for h in panel["hour_index"]]
    panel_out.to_csv(out / "hourly_panel.csv", index=False)
    (out / "scalers.json").write_text(json.dumps(sc, indent=2))
    base_eval.to_csv(out / "baseline_metrics.csv", index=False)

    ports = sorted(panel["port_id"].unique())
    counts = meta.groupby(["port_id", "split"]).agg(
        candidate=("window_id", "size"), embargo_dropped=("crosses_split_boundary", "sum"),
        target_reconstructed_dropped=("target_observed", lambda s: int((~s & ~meta.loc[s.index, "crosses_split_boundary"]).sum())),
        kept=("keep", "sum"),
        kept_with_reconstructed_inputs=("n_reconstructed_input_cells", lambda s: int(((s > 0) & meta.loc[s.index, "keep"]).sum())),
    ).reset_index()
    counts.to_csv(out / "sequence_counts.csv", index=False)

    schema = {
        "task": "next-hour container_dwell_hours regression, one sample per (port, hour t)",
        "window_hours": config.WINDOW, "horizon_hours": config.HORIZON,
        "arrays": {
            "X": {"shape": "[N, 12, 7]", "dtype": "float32", "axis_1": "input hours t-11 .. t (oldest first)", "axis_2": config.FEATURES},
            "provenance": {"shape": "[N, 12, 5]", "dtype": "int8", "axis_2": list(config.PROVENANCE_COLUMNS), "codes": config.STATUS_CODES},
            "aux_flags": {"shape": "[N, 12, 2]", "dtype": "int8", "axis_2": ["disruption_end_reconstructed", "wind_gust_clipped"]},
            "y": {"shape": "[N]", "dtype": "float32", "meaning": "dwell at hour t+1, per-port z-score (scalers.json['dwell'])"},
            "y_hours": {"shape": "[N]", "meaning": "dwell at hour t+1 in hours (for evaluation)"},
            "last_dwell_hours": {"shape": "[N]", "meaning": "dwell at hour t in hours"},
            "port_index": {"shape": "[N]", "meaning": "index into ports (for an embedding / per-port de-scaling)", "ports": ports},
            "target_hour_index": {"shape": "[N]", "meaning": "global hour index of the target (0 = 2025-01-01 00:00 UTC)"},
            "window_id": {"shape": "[N]", "meaning": "row in windows.csv"},
        },
        "features": [{"name": f, "position": i, "source": FEATURE_DOC[f][0], "unit": FEATURE_DOC[f][1], "definition": FEATURE_DOC[f][2],
                      "scaling": sequences.SCALING[f], "provenance_tracked": f in config.PROVENANCE_COLUMNS}
                     for i, f in enumerate(config.FEATURES)],
        "excluded_features": config.EXCLUDED_FEATURES,
        "reconstructed_input_policy": {
            "values": "kept exactly as in Dataset/Processed; not re-imputed (re-imputation would invent new values or use the future)",
            "flag": "provenance == 2 marks a value with no raw observation; provenance == 1 marks a raw value after a unit/format fix",
            "default_model_input": "X concatenated with the dwell reconstructed flag (provenance[..., 0] == 2) -> [N, 12, 8]; see dataset.load_split",
            "sensitivity": "windows.csv n_reconstructed_input_cells == 0 selects windows with fully observed inputs",
        },
        "target_policy": "windows whose target hour is reconstructed are excluded from training and evaluation",
        "baselines": {"ewma_alpha": alpha, "alpha_selected_on": "training windows (RMSE)", "alpha_grid_rmse": alpha_scores},
    }
    (out / "feature_schema.json").write_text(json.dumps(schema, indent=2))

    split_meta = {
        "rule": "split by the global index of the target hour; same cut-offs for all ports; a window is kept only if all its input hours are in the target's split",
        "bounds_hour_index": {s: list(b) for s, b in bounds.items()},
        "bounds_utc": {s: [str(pd.Timestamp(hours[b[0]]).tz_localize("UTC")), str(pd.Timestamp(hours[b[1] - 1]).tz_localize("UTC"))]
                       for s, b in bounds.items()},
        "kept_windows": {s: int((meta["keep"] & (meta["split"] == s)).sum()) for s in sequences.SPLITS},
        "kept_windows_per_port": {s: meta[meta["keep"] & (meta["split"] == s)].groupby("port_id").size().to_dict() for s in sequences.SPLITS},
        "excluded": {"embargo": int(meta["crosses_split_boundary"].sum()),
                     "reconstructed_target": int((~meta["target_observed"] & ~meta["crosses_split_boundary"]).sum())},
        "first_and_last_kept_hour": {s: [int(meta.loc[meta["keep"] & (meta["split"] == s), "t_hour_index"].min() - (config.WINDOW - 1)),
                                         int(meta.loc[meta["keep"] & (meta["split"] == s), "target_hour_index"].max())] for s in sequences.SPLITS},
    }
    (out / "split_metadata.json").write_text(json.dumps(split_meta, indent=2))

    prov_summary = []
    for f, (table, col) in config.PROVENANCE_COLUMNS.items():
        for s in sequences.SPLITS:
            m = panel["hour_index"].between(*[bounds[s][0], bounds[s][1] - 1])
            vc = panel.loc[m, f"prov_{f}"].value_counts()
            prov_summary.append({"feature": f, "source": f"{table}.{col}", "split_hours": s,
                                 **{name: int(vc.get(code, 0)) for name, code in config.STATUS_CODES.items()}})
    pd.DataFrame(prov_summary).to_csv(out / "provenance_summary.csv", index=False)
    ev = events[events["end_reconstructed"]][["event_id", "port_id", "event_type", "event_start", "event_end"]]
    ev.to_csv(out / "disruption_events_with_reconstructed_end.csv", index=False)

    pd.DataFrame(res, columns=["check", "passed", "evidence"]).to_csv(out / "integrity_checks.csv", index=False)
    sources = sorted(config.PROCESSED.glob("*.csv")) + [config.RAW / "disruption_events.csv"] + \
        sorted(config.AUDIT_LINEAGE.glob("cell_lineage_*.csv"))
    outputs = sorted(p for p in out.iterdir() if p.name != "manifest.json")
    manifest = {"seed": config.SEED, "versions": versions(),
                "sources_sha256": {str(p.relative_to(config.PROJECT_ROOT)): sha256(p) for p in sources},
                "outputs_sha256": {p.name: sha256(p) for p in outputs}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return meta, counts, base_eval, res, schema, split_meta
