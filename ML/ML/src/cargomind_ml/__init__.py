"""CargoMind shipment delay-risk models (pre-departure prediction).

Modules
-------
config         : paths, seeds, prediction cutoff, split and tuning settings
features       : modelling table built on the EDA integration + ML leakage fixes, feature catalogue
preprocessing  : chronological split, time-series CV folds, per-algorithm preprocessing pipelines
*_model        : one module per algorithm (pipeline + search space / grid)
train_models   : data preparation, tuning, fitting and saving of all six models
evaluate_models: metrics, classification reports, confusion matrices, anomaly evaluation
feature_importance : coefficients, native importances, (group) permutation importance, SHAP
compare_models : consolidated comparison tables and the written recommendation
"""
