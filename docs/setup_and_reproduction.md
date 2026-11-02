# Setup and reproduction

All commands run from the repository root with the project virtualenv (Python 3.9.6 was used).

## Environment

```bash
python3.9 -m venv .venv
.venv/bin/python -m pip install --upgrade pip "setuptools>=68"
.venv/bin/python -m pip install -r requirements.txt     # ML + DL dependencies, then `pip install -e .` (project packages)
```

Pinned dependencies:
- `ML/EDA/requirements.txt` and `ML/ML/requirements.txt` (scikit-learn, xgboost, lightgbm, shap, …);
- `DL/DL/requirements.txt` (torch 2.5.1 CPU, …);
- `streamlit_app/requirements.txt` (streamlit 1.50.0).

The editable install maps the import names to the stage folders (see `pyproject.toml`). If a package moves, rerun `.venv/bin/python -m pip install -e . --no-deps`.

## Verification (no retraining, nothing overwritten)

| Check | Command |
|---|---|
| DL unit tests (84) | `.venv/bin/python -m unittest discover -s DL/Evaluation/tests` |
| ML inference tests (20) | `.venv/bin/python -m unittest discover -s ML/ML/tests` |
| DL sample prediction | `.venv/bin/python -m dl.inference.demo --port SGSIN --hour 811` (production call: Transformer T_w, automatic C_w LSTM fallback) |
| DL model registry (regenerate from result tables) | `.venv/bin/python -m dl.inference.build_registry` |

## ML workflow

| Step | Command | Writes |
|---|---|---|
| EDA notebooks (built from `ML/EDA/notebooks/_source/*.py`) | `.venv/bin/python ML/EDA/scripts/build_notebooks.py` | `ML/EDA/notebooks/*.ipynb` |
| Run a notebook | open it from `ML/EDA/notebooks/` (notebooks resolve paths from that folder) | `ML/EDA/reports/` |
| Train / evaluate / compare (full ML stage) | `.venv/bin/python ML/ML/scripts/run_ml_pipeline.py` | `ML/ML/{models,results}` (**overwrites**) |
| Independent ML evaluation | `.venv/bin/python ML/Evaluation/scripts/run_evaluation.py` | `ML/Evaluation/*` |
| Calibration study (sigmoid vs isotonic, validation only; no retraining) | `.venv/bin/python ML/Evaluation/scripts/run_calibration_study.py` | `ML/Evaluation/calibration_study/`; saves `ML/ML/models/xgboost_calibrator.joblib` only if calibration is justified (currently not) |

## DL workflow

| Step | Command | Writes |
|---|---|---|
| Data readiness audit (provenance masks) | `.venv/bin/python DL/Dataset/data_audit/scripts/run_data_audit.py` | `DL/Dataset/data_audit/` |
| Build the DL dataset (deterministic; must reproduce `manifest.json`) | `.venv/bin/python DL/Dataset/scripts/build_dl_dataset.py` | `DL/Dataset/processed/` (**overwrites**) |
| DL EDA | `.venv/bin/python DL/EDA/scripts/run_eda.py` | `DL/EDA/{plots,reports}` |
| Training (only when intentionally required) | `.venv/bin/python -m dl.training.run_experiments main \| round2 \| ablations \| gru \| transformer` | `DL/DL/{artifacts,training/logs}` (**overwrites that phase**) |
| Validation diagnostics | `.venv/bin/python -m dl.evaluation.evaluate validation` | `DL/Evaluation/results/evaluation/` |
| Transformer comparison (validation) | `.venv/bin/python -m dl.evaluation.compare_transformer` | `DL/Evaluation/results/transformer/`, `DL/Evaluation/figures/transformer/` |
| Report figures | `.venv/bin/python -m dl.evaluation.report_figures` | `DL/Evaluation/figures/` |
| GNN experiment (supplementary; GCN + LSTM, 5 seeds; `--control` = no-edge diagnostic) | `.venv/bin/python -m dl.training.gnn_trainer` | `DL/DL/{artifacts/gnn,training/logs/gnn}` |
| GNN validation comparison | `.venv/bin/python -m dl.evaluation.compare_gnn` | `DL/Evaluation/results/gnn/`, `DL/Evaluation/figures/gnn/` |
| Independent DL audit | `.venv/bin/python -m dl.evaluation_engineer.run_independent_evaluation` | `DL/Evaluation/evaluation_engineer/` |

The one-time DL test-set evaluation (`python -m dl.evaluation.evaluate test`) has already been run. It must not be re-run for model selection.

## Streamlit

```bash
.venv/bin/python -m pip install -r streamlit_app/requirements.txt
.venv/bin/streamlit run streamlit_app/app.py
```
