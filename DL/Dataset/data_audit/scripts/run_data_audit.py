"""DL data-readiness audit for CargoMind (next-hour container dwell forecasting).

Run from the CargoMind project root:

    .venv/bin/python DL/Dataset/data_audit/scripts/run_data_audit.py

Requires the project packages to be installed once (`pip install -e .` at the repository root).
Reads ML/Dataset/Processed (and ML/Dataset/Raw for lineage only); never writes to either.
Outputs go to DL/Dataset/data_audit/{tables,figures,lineage}. No model is trained.
"""
from cargomind_dl_audit import audit, config, lineage


def main():
    config.ensure_dirs()
    tables = audit.load()
    print("schema ...", flush=True)
    audit.schema(tables)
    print("temporal integrity ...", flush=True)
    _, join = audit.temporal(tables)
    print("lineage (raw vs processed) ...", flush=True)
    lin, lin_summary = lineage.lineage_tables(config.OUT["lineage"])
    tel_proc, tel_status = lin["port_hourly_telemetry"]
    wx_proc, wx_status = lin["weather_hourly"]
    print("hourly panel, target, features ...", flush=True)
    df = audit.panel(tables)
    assert (df["port_id"].values == tel_proc["port_id"].values).all() and (df["ts"].values == tel_proc["ts"].values).all()
    assert (df["port_id"].values == wx_proc["port_id"].values).all() and (df["ts"].values == wx_proc["ts"].values).all()
    audit.target(df, tel_status)
    statuses = {**{c: tel_status[c] for c in tel_status}, **{c: wx_status[c] for c in wx_status}}
    audit.features(df, statuses)
    print("joins and graph ...", flush=True)
    _, graph = audit.joins_and_graph(tables, df)
    print("sequence counts ...", flush=True)
    counts, cuts = audit.sequence_counts(df, tel_status, wx_status, list(config.PPT_SIGNALS), "ppt8")
    rec, _ = audit.sequence_counts(df, tel_status, wx_status, config.RECOMMENDED_MEASURED, "recommended")
    print("baseline floors ...", flush=True)
    audit.baselines(df, tel_status, cuts)
    audit.alignment_example(df)
    audit.event_coverage(tables, cuts)
    audit.figures(df, tel_status, tables)
    audit.dump_json({"telemetry_weather_join": join, "graph": graph, "split_cuts": cuts,
                     "lineage": lin_summary.to_dict("records")}, "audit_facts")
    print(counts.to_string())
    print(rec.to_string())
    print("done")


if __name__ == "__main__":
    main()
