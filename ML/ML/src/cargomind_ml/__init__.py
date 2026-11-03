"""CargoMind shipment delay-risk models (pre-departure prediction).

=============================================================================
CARGOMIND MACHINE LEARNING PACKAGE (`cargomind_ml`)
=============================================================================
This package contains all the machine learning code for CargoMind, designed
to predict cargo shipment delay risks before vessels depart.

MODULE ROADMAP:
---------------
• config.py
    Global settings: file paths, random seeds, cutoff times, and tuning budgets.

• features.py
    Builds the 51 predictive features from raw tables, rigorously preventing
    data leakage (peeking into future data).

• preprocessing.py
    Data transformations: chronological train/test splits, time-series folds,
    scaling, missing value handling, and categorical encoding pipelines.

• Model Architecture Modules:
    - logistic_regression_model.py : Fast, transparent baseline linear model.
    - random_forest_model.py       : Robust bagged-tree ensemble model.
    - xgboost_model.py             : High-performance gradient boosted decision trees.
    - lightgbm_model.py            : Fast, memory-efficient leaf-wise boosted trees.
    - isolation_forest_model.py    : Unsupervised anomaly detector for weird port conditions.
    - one_class_svm_model.py       : Kernel-based boundary detector for normal operating conditions.

• train_models.py
    Coordinates hyperparameter tuning, model training, and saves model artifacts (.joblib).

• evaluate_models.py
    Computes performance metrics (F1 score, recall, precision, ROC-AUC) on test data.

• feature_importance.py
    Explainable AI (SHAP values and permutation importance) to reveal *why* models predict delays.

• calibration.py
    Tunes predicted probabilities so risk percentages reflect true real-world likelihoods.

• compare_models.py
    Compiles comparison scorecards and selects champion models.

• inference.py
    Production prediction engine: takes new shipment attributes and returns risk classifications
    and anomaly detection scores.
=============================================================================
"""
