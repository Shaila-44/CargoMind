# CargoMind documentation

| Document | Contents |
|---|---|
| [architecture.md](architecture.md) | The two workflows, stage responsibilities, packages, shared dataset, inference interfaces, known gaps |
| [model_selection.md](model_selection.md) | Production models (XGBoost, Isolation Forest, Transformer T_w + C_w LSTM fallback), LSTM selection evidence, inference output contracts, integration gaps |
| [setup_and_reproduction.md](setup_and_reproduction.md) | Environment setup and every command to verify or reproduce the ML and DL workflows |
| [repository_reorganisation.md](repository_reorganisation.md) | Old-to-new directory mapping and every changed path reference |

Stage-specific reports stay with their stages:

- `ML/EDA/reports/EDA_summary_report.md`
- `ML/ML/reports/ML_model_report.md`
- `ML/Evaluation/reports/`
- `DL/Dataset/data_audit/reports/`, `DL/Dataset/reports/`
- `DL/EDA/reports/`
- `DL/DL/reports/`
- `DL/Evaluation/reports/` (incl. `DL_final_report.md`)
