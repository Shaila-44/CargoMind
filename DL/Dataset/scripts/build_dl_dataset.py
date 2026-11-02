"""Build the CargoMind Stage 02 DL dataset (no model training).

Run from the CargoMind project root, after the data-readiness audit:

    .venv/bin/python DL/Dataset/data_audit/scripts/run_data_audit.py   # provenance masks (once)
    .venv/bin/python DL/Dataset/scripts/build_dl_dataset.py

Requires the project packages to be installed once (`pip install -e .` at the repository root).
Reads ML/Dataset/Processed (and the raw disruption table + audit lineage for provenance); never writes
to them. Writes DL/Dataset/processed/.
"""
from cargomind_dl import build_dataset


def main():
    meta, counts, base_eval, res, schema, split_meta = build_dataset.main()
    print("kept windows:", split_meta["kept_windows"], "| excluded:", split_meta["excluded"])
    print(f"EWMA alpha (train): {schema['baselines']['ewma_alpha']}")
    b = base_eval[base_eval["port_id"] == "ALL"].pivot(index="baseline", columns="split", values="rmse_h").round(3)
    print("baseline RMSE (h):\n", b)
    for check, ok, ev in res:
        print(("PASS " if ok else "FAIL ") + check, f"({ev})" if ev else "")


if __name__ == "__main__":
    main()
