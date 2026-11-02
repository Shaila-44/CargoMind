# CargoMind

AI-powered supply-chain logistics bottleneck prediction. CargoMind is a **research prototype built on synthetic data**: its models have not been validated on real port or shipment data.

The repository holds **two independent modelling workflows** under one project, plus documentation and a UI shell:

| Directory | Purpose |
|---|---|
| [`ML/`](ML/) | **Shipment delay-risk classification** (On-Time / At-Risk / Critical), plus anomaly detection of unusual port conditions. Workflow: **Dataset → EDA → ML → Evaluation** |
| [`DL/`](DL/) | **Next-hour container dwell-time forecasting** per port (LSTM, GRU, Transformer; EWMA baseline). Workflow: **Dataset → EDA → DL → Evaluation** |
| [`docs/`](docs/) | Project-wide documentation: architecture, setup and reproduction, repository history |
| [`streamlit_app/`](streamlit_app/) | Streamlit application (scaffold). It consumes the ML and DL inference interfaces and contains no model code. |

## The two workflows

### ML workflow (`ML/`)

| Stage | Folder | Contents |
|---|---|---|
| Dataset | `ML/Dataset/` | **The project's shared source data.** `Raw/` holds the generator and messy synthetic data; `Processed/` holds the cleaned tables (shipments, port telemetry, weather, disruptions, holidays, network). |
| EDA | `ML/EDA/` | Notebooks (and their `.py` sources), `cargomind_eda` package, reports, figures, scripts |
| ML | `ML/ML/` | `cargomind_ml` package: feature table, chronological split, 4 classifiers + 2 anomaly detectors, saved models, results, model report |
| Evaluation | `ML/Evaluation/` | `cargomind_eval` package: independent evaluation (calibration, robustness, uncertainty, reproducibility), metrics, figures, reports |

### DL workflow (`DL/`)

| Stage | Folder | Contents |
|---|---|---|
| Dataset | `DL/Dataset/` | `data_audit/` (readiness audit and cell-level provenance), `processed/` (the 12-hour window dataset, scalers, schema, manifest), `scripts/` (builder), `reports/` |
| EDA | `DL/EDA/` | Plots, report and summary, `scripts/` |
| DL | `DL/DL/` | The `dl` package: `models/`, `training/` (logs), `inference/` (+ model registry), `src/cargomind_dl` (dataset code and loader), `artifacts/` (checkpoints, selection), `reports/` |
| Evaluation | `DL/Evaluation/` | `evaluation/` (code), `evaluation_engineer/` (independent audit), `results/`, `figures/`, `reports/` (incl. **`DL_final_report.md`**), `tests/` |

### The shared dataset

`ML/Dataset/` is the single copy of the source data. The ML workflow owns it. The DL workflow reads it, read-only, through one documented path (`cargomind_dl.config.SHARED_DATASET`) and builds its own modelling dataset in `DL/Dataset/processed/`. Neither workflow modifies the source data.

### Streamlit

`streamlit_app/` is the user interface layer:
- `services/ml_risk.py` wraps the ML workflow's saved classifier;
- `services/dl_forecasting.py` wraps `dl.inference.forecast`.

All model logic, feature engineering and preprocessing stay inside `ML/` and `DL/`. The dashboards are not implemented yet.

## Setup

Python 3.9; all commands run from the repository root.

```bash
python3.9 -m venv .venv
.venv/bin/python -m pip install --upgrade pip "setuptools>=68"
.venv/bin/python -m pip install -r requirements.txt        # both workflows + editable install of the project packages
.venv/bin/python -m pip install -r streamlit_app/requirements.txt   # only for the Streamlit app
```

The editable install (`pip install -e .`, included in `requirements.txt`) registers each workflow's packages from its stage folder:
- `cargomind_eda`, `cargomind_ml`, `cargomind_eval` (ML);
- `cargomind_dl_audit`, `cargomind_dl`, `dl` (DL).

Imports therefore work from any directory, without `sys.path` changes.

## Common commands

| Task | Command |
|---|---|
| DL tests (52) | `.venv/bin/python -m unittest discover -s DL/Evaluation/tests` |
| DL sample forecast | `.venv/bin/python -m dl.inference.demo --model transformer_t_w --port SGSIN --hour 811` |
| ML pipeline (retrains; overwrites `ML/ML/models` and `ML/ML/results`) | `.venv/bin/python ML/ML/scripts/run_ml_pipeline.py` |
| ML independent evaluation | `.venv/bin/python ML/Evaluation/scripts/run_evaluation.py` |
| Rebuild the DL dataset (deterministic) | `.venv/bin/python DL/Dataset/data_audit/scripts/run_data_audit.py` then `.venv/bin/python DL/Dataset/scripts/build_dl_dataset.py` |
| Run the app (scaffold) | `.venv/bin/streamlit run streamlit_app/app.py` |

The full list, including the DL training phases and evaluation reports, is in [`docs/setup_and_reproduction.md`](docs/setup_and_reproduction.md).

```python
from dl.inference import forecast                       # DL: next-hour dwell forecast in hours
result = forecast(window_df, port_id="SGSIN")           # 12 raw hourly rows, oldest first
```

## Status

| Workflow | Status |
|---|---|
| ML | Complete. **Selected for inference: XGBoost** (delay class; provisional primary, not shown to be better than the Logistic Regression challenger) **and Isolation Forest** (anomaly score and flag, a separate output). Raw probabilities are served (calibration was not justified on validation), and no alert threshold is set. The other models are kept for research and comparison. See `docs/model_selection.md`; measured results in `ML/ML/reports/ML_model_report.md` and `ML/Evaluation/reports/evaluation_report.md`. |
| DL | **Complete with limitations.** **Transformer T_w is the primary model and the C_w LSTM the automatic fallback**; EWMA (α = 0.2) is kept as a benchmark, and the GNN is experimental and excluded from inference. See `docs/model_selection.md` and `DL/Evaluation/reports/DL_final_report.md`. |

Neither workflow's results establish real-world performance.
