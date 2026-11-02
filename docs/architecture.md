# CargoMind architecture

CargoMind is one project containing **two independent modelling workflows**. Each has its own data preparation, EDA, modelling and evaluation stages, and its own Python packages.

```
ML/  Dataset ──► EDA ──► ML ──► Evaluation        shipment delay-risk classification (+ anomaly detection)
DL/  Dataset ──► EDA ──► DL ──► Evaluation        next-hour port dwell-time forecasting
                ▲
                └── DL/Dataset reads ML/Dataset (shared source data, read-only)

streamlit_app/  ──► services ──► ML inference (cargomind_ml.inference) / DL inference (dl.inference)
docs/           project-wide documentation
```

## Stage responsibilities

| Workflow | Stage | Folder | Python package (import name → source folder) | Writes to |
|---|---|---|---|---|
| ML | Dataset | `ML/Dataset/{Raw,Processed}` | — (generator script in `Raw/`) | only `Raw/` via the generator |
| ML | EDA | `ML/EDA` | `cargomind_eda` → `ML/EDA/src/cargomind_eda` | `ML/EDA/{reports,data}` |
| ML | ML | `ML/ML` | `cargomind_ml` → `ML/ML/src/cargomind_ml` | `ML/ML/{models,results,reports}` |
| ML | Evaluation | `ML/Evaluation` | `cargomind_eval` → `ML/Evaluation/src/cargomind_eval` | `ML/Evaluation/*` |
| DL | Dataset | `DL/Dataset` | `cargomind_dl_audit` → `DL/Dataset/data_audit/src/…`; dataset code in `cargomind_dl` | `DL/Dataset/{data_audit,processed,reports}` |
| DL | EDA | `DL/EDA` | scripts in `DL/EDA/scripts` | `DL/EDA/{plots,reports}` |
| DL | DL | `DL/DL` | `dl` → `DL/DL` (`dl.models`, `dl.training`, `dl.inference`); `cargomind_dl` → `DL/DL/src/cargomind_dl` | `DL/DL/{artifacts,training/logs,reports}` |
| DL | Evaluation | `DL/Evaluation` | `dl.evaluation`, `dl.evaluation_engineer`, `dl.tests` → `DL/Evaluation/{evaluation,evaluation_engineer,tests}` | `DL/Evaluation/{results,figures,reports,evaluation_engineer}` |

**Packaging.** The import names map to their stage folders through the root `pyproject.toml`, installed once in editable mode (`pip install -e .`). The DL package keeps its original name `dl`, so existing commands such as `python -m dl.training.run_experiments` and the inference import `from dl.inference import forecast` are unchanged, even though its modelling and evaluation sub-packages live in different stage folders.

**Path resolution.** No code depends on the current working directory:
- **ML:** each package computes its locations from its own file (`cargomind_ml.config`, `cargomind_eda.config`, `cargomind_eval.common`).
- **DL:** every module takes its locations from one module, `cargomind_dl.config`: `PROJECT_ROOT`, `SHARED_DATASET`, `DATASET`, `DL_STAGE`, `EDA_STAGE`, `EVALUATION_STAGE`.

## Shared dataset

`ML/Dataset` is the only copy of the source data, and it belongs to the ML workflow, which uses it directly. The DL workflow:
- reads it read-only via `cargomind_dl.config.SHARED_DATASET` (`PROCESSED`, `RAW`);
- builds its own modelling dataset in `DL/Dataset/processed`, with a manifest of source SHA-256 hashes;
- never copies or modifies the source data.

`DL/Dataset/processed/manifest.json` records source paths in the layout used when it was built. `cargomind_dl.config.resolve_recorded_path()` maps those recorded paths to the current layout, so integrity checks keep working without rewriting the artifact.

## Inference interfaces (consumed by `streamlit_app/`)

| Workflow | Interface | Notes |
|---|---|---|
| DL | `dl.inference.forecast(window_df, port_id)`; `dl.inference.windows.window_from_dataset(port_id, end_hour)` | Transformer T_w primary, C_w LSTM automatic fallback on load failure or invalid output; raw 12-hour windows validated; checkpoint and scaler hashes verified. Adapter: `streamlit_app/services/dl_forecasting.py`. |
| ML | `cargomind_ml.inference.predict_shipments(shipments_df)` / `predict(feature_df)` | XGBoost classification and Isolation Forest anomaly score/flag as separate sections; shipment attributes are turned into the 51 features with the training code. Adapter: `streamlit_app/services/ml_risk.py`. |

Model selection, the LSTM fallback evidence and both output contracts: [model_selection.md](model_selection.md).

## Known gaps

- **No live data feed.** `cargomind_ml.inference.build_features_for_shipments` turns shipment attributes into model features, but it reads port telemetry, weather, disruptions, holidays and arrived outcomes from `ML/Dataset/Processed`. It therefore works only for departures whose cutoff hour lies in 2025-01-01 .. 2025-02-11 15:00 UTC. Likewise, `window_from_dataset` serves DL windows only from `DL/Dataset/processed/hourly_panel.csv`. Real use needs a feed of these inputs in the same schema.
- **DL inference expects pre-computed disruption state** and a reconstruction flag when windows come from outside the panel (see the DL final report, limitations).
- **Probabilities are served raw.** Recalibration and decision thresholds evaluated in `ML/Evaluation` are not applied.
- **Both workflows are synthetic-data prototypes.**
