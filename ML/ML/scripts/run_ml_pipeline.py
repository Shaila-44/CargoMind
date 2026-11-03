"""Run the full ML stage: data -> splits -> tuning/fitting -> evaluation -> importance -> comparison.

Run from the CargoMind project root:

    .venv/bin/python ML/ML/scripts/run_ml_pipeline.py

Requires the project packages to be installed once (`pip install -e .` at the repository root).
Reads ML/Dataset/Processed (via the EDA integration code) and never writes to it. Writes models to
ML/ML/models/ and every table and figure to ML/ML/results/.
"""
# =============================================================================
# THE MASTER ORCHESTRATOR FOR CARGOMIND MACHINE LEARNING
# =============================================================================
# This script is the "Big Red Button" that runs the entire machine learning
# pipeline from start to finish!
#
# If you run this file in your terminal, it will execute the complete scientific
# workflow in 4 distinct phases:
#
#   PHASE 1: DATA PREPARATION & TRAINING (train_models.py)
#     - Loads raw port, weather, calendar, and shipment tables.
#     - Engineers 51 leakage-free features.
#     - Chronologically splits data into Train (earlier) and Test (later).
#     - Trains 4 supervised models (Logistic Regression, Random Forest, XGBoost, LightGBM).
#     - Trains 2 anomaly detectors (Isolation Forest, One-Class SVM).
#     - Saves the trained models into `ML/ML/models/`.
#
#   PHASE 2: EVALUATION (evaluate_models.py)
#     - Grades each model on unseen test shipments.
#     - Computes precision, recall, F1 scores, ROC-AUC, and confusion matrices.
#     - Saves detailed metric CSVs and diagnostic charts.
#
#   PHASE 3: FEATURE IMPORTANCE (feature_importance.py)
#     - Figures out *why* models make their decisions (explainable AI / SHAP).
#     - Identifies which factors (e.g. weather, port dwell time, strikes) drive delays.
#
#   PHASE 4: COMPARISON & MODEL SELECTION (compare_models.py)
#     - Compares all models side-by-side in clear summary tables.
#     - Selects the official champion models and writes `selection.json`.
# =============================================================================

# --- Standard library imports ---
import time  # For stopwatch timing so we know how many seconds the entire pipeline took

# --- Internal project imports ---
# Import the four main sub-modules that handle each phase of the workflow
from cargomind_ml import (
    compare_models,      # Phase 4: Compares all models and selects the winner
    config,              # Global settings, paths, and random seeds
    evaluate_models,     # Phase 2: Computes report cards and error metrics
    feature_importance,  # Phase 3: Explains feature contributions and SHAP values
    train_models,        # Phase 1: Prepares data and trains all 6 models
)


def main():
    """Execute the end-to-end Machine Learning pipeline."""
    # Start the master stopwatch
    t0 = time.time()
    print("=" * 70)
    print("CARGOMIND: STARTING COMPLETE MACHINE LEARNING PIPELINE")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # PHASE 1: Data Preparation, Hyperparameter Tuning, and Model Fitting
    # -------------------------------------------------------------------------
    # ctx contains the prepared datasets (train, test, cross-validation folds, feature lists)
    # The models are saved to disk as .joblib files inside ML/ML/models/
    ctx, _, _ = train_models.main()

    # -------------------------------------------------------------------------
    # PHASE 2: Comprehensive Evaluation
    # -------------------------------------------------------------------------
    # Evaluates every supervised classifier and anomaly detector on the held-out test set
    # Returns metric DataFrames for supervised ('sup') and anomaly ('ano') models
    print("\n--- PHASE 2: EVALUATING MODELS ON TEST DATA ---", flush=True)
    sup, ano = evaluate_models.main(ctx, train_models.SUPERVISED, train_models.ANOMALY)

    # -------------------------------------------------------------------------
    # PHASE 3: Feature Importance and Explainability (XAI)
    # -------------------------------------------------------------------------
    # Calculates tree feature importances, permutation importances, and SHAP explanations
    print("\n--- PHASE 3: COMPUTING FEATURE IMPORTANCES (SHAP & PERMUTATION) ---", flush=True)
    feature_importance.main(ctx, train_models.SUPERVISED)

    # -------------------------------------------------------------------------
    # PHASE 4: Model Comparison and Official Selection
    # -------------------------------------------------------------------------
    # Generates final comparison tables and declares tournament winners (saved to selection.json)
    print("\n--- PHASE 4: FINAL MODEL COMPARISON & SELECTION ---", flush=True)
    picks = compare_models.main(sup, ano, train_models.SUPERVISED)

    # Print summary results to console
    print("\nOfficial Model Selections:")
    for role, winner in picks.items():
        print(f"  • {role}: {winner}")

    # Calculate total elapsed time
    total_time = time.time() - t0
    print("=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.1f} SECONDS")
    print(f"Full report available at: {config.RESULT_DIRS['comparison'] / 'model_comparison.md'}")
    print("=" * 70)


# Standard Python entry point check
if __name__ == "__main__":
    main()
