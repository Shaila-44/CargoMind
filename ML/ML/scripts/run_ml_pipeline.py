"""Run the full ML stage: data -> splits -> tuning/fitting -> evaluation -> importance -> comparison.

Run from the CargoMind project root:

    .venv/bin/python ML/ML/scripts/run_ml_pipeline.py

Requires the project packages to be installed once (`pip install -e .` at the repository root).
Reads ML/Dataset/Processed (via the EDA integration code) and never writes to it. Writes models to
ML/ML/models/ and every table and figure to ML/ML/results/.
"""
import time

from cargomind_ml import compare_models, config, evaluate_models, feature_importance, train_models


def main():
    t0 = time.time()
    ctx, _, _ = train_models.main()
    print("evaluating ...", flush=True)
    sup, ano = evaluate_models.main(ctx, train_models.SUPERVISED, train_models.ANOMALY)
    print("feature importance ...", flush=True)
    feature_importance.main(ctx, train_models.SUPERVISED)
    print("comparison ...", flush=True)
    picks = compare_models.main(sup, ano, train_models.SUPERVISED)
    print(picks)
    print(f"done in {time.time() - t0:.0f}s; see {config.RESULT_DIRS['comparison'] / 'model_comparison.md'}")


if __name__ == "__main__":
    main()
