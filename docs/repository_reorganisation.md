# Repository reorganisation (four top-level folders)

The repository was reorganised from `Dataset/`, `eda/`, `ml/`, `dl/` into `ML/`, `DL/`, `docs/`, `streamlit_app/`.

This was a structural refactor only:
- no data was regenerated and no model retrained;
- every one of the 566 pre-existing files was moved with its contents unchanged (SHA-256 verified), apart from the code and documentation path updates listed below;
- checkpoints, datasets and results are byte-identical.

## Old → new locations

| Old | New | Stage |
|---|---|---|
| `Dataset/Raw/`, `Dataset/Processed/` | `ML/Dataset/Raw/`, `ML/Dataset/Processed/` | ML Dataset (shared source data) |
| `eda/` (notebooks, `_source`, `src/cargomind_eda`, reports, scripts, `data/interim`, requirements) | `ML/EDA/` (same internal layout) | ML EDA |
| `ml/` (models, results, reports, scripts, `src/cargomind_ml`, requirements) | `ML/ML/` (same internal layout) | ML modelling |
| `ml/evaluation/` (data, figures, metrics, reports, scripts, `src/cargomind_eval`, validation) | `ML/Evaluation/` | ML evaluation |
| `dl/data_audit/` | `DL/Dataset/data_audit/` | DL Dataset |
| `dl/dataset/` | `DL/Dataset/processed/` | DL Dataset |
| `dl/scripts/build_dl_dataset.py` | `DL/Dataset/scripts/build_dl_dataset.py` | DL Dataset |
| `dl/reports/DL_dataset_construction_report.md` | `DL/Dataset/reports/` | DL Dataset |
| `dl/eda/plots/` | `DL/EDA/plots/` | DL EDA |
| `dl/eda/DL_EDA_Report.md`, `eda_summary.json` | `DL/EDA/reports/` | DL EDA |
| `dl/eda/run_eda.py`, `report.py` | `DL/EDA/scripts/` | DL EDA |
| `dl/__init__.py`, `dl/models/`, `dl/training/` (incl. logs), `dl/inference/`, `dl/src/`, `dl/requirements.txt` | `DL/DL/` (same names) | DL modelling |
| `dl/artifacts/` (checkpoints, selection, metadata, test log, `transformer/runs`) | `DL/DL/artifacts/` | DL modelling |
| `dl/reports/experiment_results.csv` | `DL/DL/artifacts/experiment_results.csv` | DL modelling (training records) |
| `dl/reports/DL_training_report.md` | `DL/DL/reports/` | DL modelling |
| `dl/evaluation/`, `dl/evaluation_engineer/`, `dl/tests/` | `DL/Evaluation/evaluation/`, `evaluation_engineer/`, `tests/` | DL evaluation |
| `dl/reports/DL_evaluation_report.md`, `DL_transformer_report.md`, `DL_final_report.md` | `DL/Evaluation/reports/` | DL evaluation |
| `dl/reports/evaluation/` (test/validation prediction and metric tables) | `DL/Evaluation/results/evaluation/` | DL evaluation |
| `dl/reports/figures/` (incl. `transformer/`) | `DL/Evaluation/figures/` | DL evaluation |
| `dl/artifacts/transformer/validation_*.csv`, `comparison_info.json` | `DL/Evaluation/results/transformer/` | DL evaluation (comparison tables; the Transformer checkpoints stay in `DL/DL/artifacts/transformer/`) |

**Python import names are unchanged** (`dl.*`, `cargomind_dl`, `cargomind_dl_audit`, `cargomind_ml`, `cargomind_eda`, `cargomind_eval`). They now resolve through the root `pyproject.toml` (editable install). Module commands such as `python -m dl.training.run_experiments` and `python -m dl.inference.demo` are unchanged.

## Changed path and import references

| File | Change |
|---|---|
| `pyproject.toml` (new) | Maps each import name to its stage folder; replaces all `sys.path` insertions |
| `DL/DL/src/cargomind_dl/config.py` | Single source of DL locations (`PROJECT_ROOT`, `SHARED_DATASET`, `PROCESSED`, `RAW`, `DATASET`, `AUDIT_LINEAGE`, `DL_STAGE`, `EDA_STAGE`, `EVALUATION_STAGE`); `resolve_recorded_path()` for paths recorded in old artifacts |
| `DL/DL/src/cargomind_dl/features.py`, `build_dataset.py` | Raw disruption table via `config.RAW` |
| `DL/Dataset/data_audit/src/cargomind_dl_audit/config.py` | Project root and shared dataset paths |
| `DL/DL/training/config.py` | Adds `DATASET`, `EVALUATION`, `EVAL_RESULTS`, `EVAL_FIGURES` |
| `DL/DL/training/run_experiments.py` | Results CSV now in `DL/DL/artifacts/`; schema path; metadata text (also fixes audit finding m1) |
| `DL/DL/training/data.py`, `DL/DL/inference/predictor.py`, `fallback.py` | `sys.path` insertion removed (installed package) |
| `DL/DL/models/checkpoint.py` | Dataset directory from `cargomind_dl.config.DATASET` |
| `DL/DL/inference/registry.py`, `demo.py`; `model_registry.json` (regenerated, identical hashes) | Checkpoint and results paths `DL/DL/artifacts/…` |
| `DL/Evaluation/evaluation/frames.py`, `evaluate.py`, `report_figures.py`, `compare_transformer.py` | Outputs to `DL/Evaluation/{results,figures}`; dataset from config |
| `DL/Evaluation/evaluation_engineer/run_independent_evaluation.py` | Locations from config; manifest sources via `resolve_recorded_path`; new test command |
| `DL/Evaluation/tests/*.py` | Fixture paths from config; documented test command |
| `DL/EDA/scripts/run_eda.py`, `report.py` | Locations from config; report written to `DL/EDA/reports/` with `../plots/` links |
| `DL/Dataset/scripts/build_dl_dataset.py`, `DL/Dataset/data_audit/scripts/run_data_audit.py` | `sys.path` insertion removed; documented commands |
| `ML/ML/src/cargomind_ml/config.py` | `ML_WORKFLOW`, `PROJECT_ROOT`, `EDA_SRC` = `ML/EDA/src` |
| `ML/ML/src/cargomind_ml/features.py`, `ML/Evaluation/src/cargomind_eval/common.py` | `sys.path` insertion removed; `ML_ROOT` = `ML/ML` |
| `ML/ML/scripts/run_ml_pipeline.py`, `ML/Evaluation/scripts/run_evaluation.py` | `sys.path` insertion removed; documented commands |
| `ML/ML/requirements.txt` | `-r ../EDA/requirements.txt` |
| `ML/EDA/src/cargomind_eda/config.py` | Comment only (path logic already resolves `ML/Dataset`) |
| `DL/DL/requirements.txt` | Install command text |
| `.gitignore` | Dataset CSV allow-list → `ML/Dataset/**`; ignore `*.egg-info/`, `build/` |
| 15 Markdown reports under `ML/` and `DL/` | Path mentions updated; 25 relative image and report links fixed (all links resolve) |

## Left unchanged on purpose

| Item | Reason |
|---|---|
| EDA notebooks (`.ipynb`) and their `_source/*.py` | They locate data via `Path.cwd().parent` and `cargomind_eda.config`, which still resolve correctly inside `ML/EDA/`. Executed notebooks were not rewritten. |
| JSON / CSV artifacts containing old path strings (e.g. `manifest.json`, `lstm_config.json`, `selection.json`, `evaluation_results.json`) | Historical records, and in the case of `manifest.json` a hashed dataset artifact. Code resolves recorded paths with `resolve_recorded_path()`. |
| `.gitattributes` | Its LFS patterns refer to `datasets/…` files that never existed in this repository |
| `DL/Dataset/artifacts/` (in the target sketch) | Not created: every DL dataset artifact belongs to the single `processed/` directory the loader and manifest expect. Splitting it would change the dataset's layout. |
