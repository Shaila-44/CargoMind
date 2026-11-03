"""Feature importance and interpretability.

Methods (all on the saved models; permutation and SHAP on the test period):
* Logistic Regression coefficients (standardised numeric inputs, log-odds per SD).
* Native tree importances: Random Forest impurity decrease, XGBoost and LightGBM total gain.
* Permutation importance per feature (drop in macro-F1).
* Group permutation importance: all columns of a feature group are permuted together (same row
  shuffle), which handles correlated features that would otherwise share/hide importance. Reported
  as the drop in macro-F1 and in Critical recall.
* SHAP (TreeExplainer) for XGBoost and LightGBM.

One-hot levels and missing-value indicators are summed back to their source column so every
method reports on the same 51 input features. None of this is causal.
"""

# ── What is "Feature Importance"? ──────────────────────────────────────────────
# After we train a model, we want to know: *which inputs (features) mattered
# most* for the model's predictions?  For example, does "days until departure"
# matter more than "carrier name"?  This module answers that question using
# several different techniques so we can cross-check the answer.
# ───────────────────────────────────────────────────────────────────────────────

# Suppress non-critical warnings so the output stays clean.
import warnings

# joblib lets us load previously-saved model files from disk.
import joblib

# matplotlib is the standard Python library for creating charts and plots.
import matplotlib

# "Agg" is a non-interactive backend — it creates image files without trying
# to open a window (important when running on a server with no screen).
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# numpy provides fast math operations on arrays of numbers.
import numpy as np  # noqa: E402

# pandas provides DataFrames — spreadsheet-like tables — for data manipulation.
import pandas as pd  # noqa: E402

# permutation_importance is a scikit-learn utility that measures how much the
# model's accuracy drops when you randomly shuffle one feature's values.
# "Permutation importance" works like this: take one input column, randomly
# scramble its values so they no longer match the correct rows, then see how
# much worse the model performs.  If the model gets a lot worse, that feature
# was important.  If performance barely changes, the feature didn't matter much.
from sklearn.inspection import permutation_importance  # noqa: E402

# f1_score and recall_score are metrics that measure how well the model's
# predictions match reality.  f1_score balances precision and recall;
# recall_score measures how many of the actual positive cases were caught.
from sklearn.metrics import f1_score, recall_score  # noqa: E402

# Import project-wide settings (file paths, random seeds, etc.).
from . import config  # noqa: E402

# Import the list of features (input columns), which ones are categorical
# (text-based categories) vs. numeric (numbers), and feature group definitions.
from .features import CATEGORICAL, FEATURE_GROUPS, FEATURE_TO_GROUP, FEATURES, NUMERIC  # noqa: E402

# Directory where feature importance results (CSV files) will be saved.
OUT = config.RESULT_DIRS["feature_importance"]

# How many times to repeat each permutation experiment.  Repeating gives us a
# more stable average (less affected by random luck in any single shuffle).
N_REPEATS = 5


def source_feature(encoded):
    """Map an encoded (transformed) feature name back to the original input name.

    During preprocessing, features get renamed.  For example:
    - A missing-value indicator for "weight" becomes "missingindicator_weight".
    - A categorical column like "carrier" with value "DHL" becomes "carrier_DHL".
    This function reverses those transformations so we can report importance
    using the original, human-readable feature names.
    """
    # If the name starts with "missingindicator_", strip that prefix to get
    # the original column name (e.g., "missingindicator_weight" → "weight").
    if encoded.startswith("missingindicator_"):
        return encoded[len("missingindicator_"):]
    # If the name is already a known numeric column, return it as-is.
    if encoded in NUMERIC:
        return encoded
    # For categorical (one-hot encoded) columns, find which original category
    # name this encoded column belongs to.  One-hot encoding creates columns
    # like "carrier_DHL", "carrier_FedEx", etc. from a single "carrier" column.
    # We look for the longest matching prefix among our known categorical names.
    matches = [c for c in CATEGORICAL if encoded.startswith(c + "_")]
    return max(matches, key=len) if matches else encoded


def _to_source(values, names):
    """Aggregate encoded-level importance values back to their source features.

    For example, if "carrier_DHL" has importance 0.3 and "carrier_FedEx" has 0.2,
    we sum them to get "carrier" = 0.5.  This way every method reports importance
    for the same 51 original input features.
    """
    # Create a pandas Series with the encoded feature names mapped back to
    # their original source feature names using source_feature().
    s = pd.Series(values, index=[source_feature(n) for n in names])
    # Group by the source feature name (so all one-hot columns of the same
    # category are combined) and sum their values.  Reindex to ensure we
    # include all 51 features, filling missing ones with zero.
    return s.groupby(level=0).sum().reindex(FEATURES).fillna(0)


def native_importances(models):
    """Extract built-in ("native") feature importance scores from tree-based models.

    What are "native importances"?
    Tree-based models (Random Forest, XGBoost, LightGBM) internally keep track
    of how useful each feature was during training.  For example:
    - Random Forest tracks "impurity decrease": how much each feature helped
      separate the data into purer groups at each tree split.
    - XGBoost and LightGBM track "gain": how much each feature improved the
      model's predictions across all the trees.
    These are cheap to compute (no extra calculations needed) because the model
    already recorded them during training.
    """
    out = {}
    # Loop through each tree-based model and extract its native importance scores.
    # "key" is the internal name (e.g., "random_forest"), "label" is what we
    # call the resulting column (e.g., "rf_impurity").
    for key, label in [("random_forest", "rf_impurity"), ("xgboost", "xgb_gain"), ("lightgbm", "lgbm_gain")]:
        # Get the saved pipeline (preprocessing + model) for this algorithm.
        pipe = models[key]
        # Get the feature names that the preprocessor created (these are the
        # encoded names like "carrier_DHL", not the original names).
        names = pipe.named_steps["prep"].get_feature_names_out()
        # Get the actual model (estimator) from the pipeline.
        est = pipe.named_steps["model"]
        # Each model type stores its importances slightly differently:
        if key == "xgboost":
            # XGBoost: get the total gain for each feature from its internal
            # booster object.  Features are referred to as "f0", "f1", etc.
            gain = est.get_booster().get_score(importance_type="total_gain")
            vals = np.array([gain.get(f"f{i}", 0.0) for i in range(len(names))])
        elif key == "lightgbm":
            # LightGBM: get gain-based importance directly from the booster.
            vals = est.booster_.feature_importance(importance_type="gain")
        else:
            # Random Forest: scikit-learn stores impurity-based importances in
            # the .feature_importances_ attribute.
            vals = est.feature_importances_
        # Map encoded feature importances back to the 51 source features.
        s = _to_source(vals, names)
        # Normalise so all values add up to 1.0 (i.e., convert to proportions).
        out[label] = s / s.sum()
    # Return a DataFrame with one column per model, one row per source feature.
    return pd.DataFrame(out)


def lr_coefficients(model):
    """Extract and save logistic regression coefficients.

    Logistic regression assigns a numeric "weight" (coefficient) to each feature.
    A large positive coefficient means that feature pushes the prediction toward
    a particular class; a large negative one pushes it away.  These coefficients
    are measured in log-odds per standard deviation, which is a way of making
    the numbers comparable across features with different scales.
    """
    # Import the helper function that extracts coefficients from the LR model.
    from .logistic_regression_model import coefficients

    # Get the coefficients table (rows = features, columns = classes + max_abs).
    coef = coefficients(model)
    # Save the full coefficient table to a CSV file for reference.
    coef.to_csv(OUT / "logistic_regression_coefficients.csv")
    # For a single summary number per feature, take the maximum absolute
    # coefficient across all classes (ignoring the "max_abs" column itself).
    # This tells us: across all classes, how strongly does this feature
    # influence any single class prediction?
    agg = coef.drop(columns="max_abs").abs().max(axis=1)
    # Return both the full table and the aggregated source-level importance.
    return coef, _to_source(agg.values, agg.index).rename("lr_max_abs_coef")


def permutation_by_feature(models, test):
    """Compute permutation importance for every model, one feature at a time.

    How permutation importance works:
    1. Start with the model's normal score on the test data (baseline).
    2. Pick one feature column and randomly shuffle its values across rows,
       breaking the real relationship between that feature and the outcome.
    3. Re-score the model.  If the score drops a lot, that feature was important.
    4. Repeat the shuffle N_REPEATS times to get a stable average.
    5. Do this for every feature, for every model.

    The metric used here is "macro-F1", which averages the F1 score across all
    classes equally (so rare classes matter just as much as common ones).
    """
    res = {}
    # Loop through each model (logistic regression, random forest, etc.).
    for key, pipe in models.items():
        # Suppress warnings that can arise from edge-case predictions.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Use scikit-learn's built-in permutation_importance function.
            # It shuffles each feature one at a time and measures the drop in
            # macro-F1 score on the test data.
            r = permutation_importance(pipe, test[FEATURES], test["y"], scoring="f1_macro", n_repeats=N_REPEATS,
                                       random_state=config.SEED, n_jobs=config.N_JOBS)
        # Store the average importance drop and its standard deviation.
        res[f"{key}_perm_mean"] = pd.Series(r.importances_mean, index=FEATURES)
        res[f"{key}_perm_std"] = pd.Series(r.importances_std, index=FEATURES)
    return pd.DataFrame(res)


def permutation_by_group(models, test):
    """Compute group permutation importance — shuffle entire feature groups together.

    Why groups?  Some features are correlated (e.g., "departure_city" and
    "departure_country" carry overlapping information).  When you shuffle just
    one of them, the model can still rely on the other, making both look
    unimportant individually.  By shuffling all features in a group together,
    we see the true combined importance of that group.

    This function measures two things when a group is shuffled:
    - Drop in macro-F1 (overall accuracy across all classes).
    - Drop in Critical recall (how many truly "Critical" shipments we catch).
    """
    # Create a random number generator with a fixed seed for reproducibility.
    rng = np.random.default_rng(config.SEED)
    # Separate the input features (X) from the true labels (y).
    X, y = test[FEATURES].reset_index(drop=True), test["y"].values
    rows = []
    # Loop through each model.
    for key, pipe in models.items():
        # Get the model's predictions on the unmodified test data (baseline).
        base = pipe.predict(X)
        # Calculate baseline macro-F1 and Critical-class recall.
        base_f1 = f1_score(y, base, average="macro")
        base_rc = recall_score(y, base, labels=[2], average="macro", zero_division=0)
        # Loop through each feature group (e.g., "shipment_timing", "carrier").
        for group, cols in FEATURE_GROUPS.items():
            d_f1, d_rc = [], []
            # Repeat the shuffle experiment N_REPEATS times for stability.
            for _ in range(N_REPEATS):
                # Make a copy of the data so we don't modify the original.
                Xp = X.copy()
                # Create a random permutation (shuffled row order).
                perm = rng.permutation(len(X))
                # Shuffle ALL columns in this group together using the same
                # permutation, so within-group relationships are preserved
                # but the group's relationship to the label is broken.
                Xp[cols] = X[cols].values[perm]
                # Get the model's predictions on the shuffled data.
                p = pipe.predict(Xp)
                # Record how much worse the model did compared to baseline.
                d_f1.append(base_f1 - f1_score(y, p, average="macro"))
                d_rc.append(base_rc - recall_score(y, p, labels=[2], average="macro", zero_division=0))
            # Store the average drop and standard deviation for this group.
            rows.append({"model": key, "group": group, "n_features": len(cols),
                         "macro_f1_drop": np.mean(d_f1), "macro_f1_drop_std": np.std(d_f1),
                         "critical_recall_drop": np.mean(d_rc), "critical_recall_drop_std": np.std(d_rc)})
    return pd.DataFrame(rows)


def shap_values(models, test, labels):
    """Compute SHAP values for tree-based models (XGBoost and LightGBM).

    What are SHAP values?
    SHAP (SHapley Additive exPlanations) is a method from game theory that
    explains *individual predictions*.  For each prediction, SHAP tells you
    how much each feature pushed the prediction higher or lower compared to
    the average.  For example: "the shipment being 3 days early pushed the
    Critical risk score DOWN by 0.15, while the carrier being XYZ pushed it
    UP by 0.08."

    Here we use TreeExplainer, which is a fast, exact SHAP algorithm designed
    specifically for tree-based models.  We compute SHAP values for every
    sample in the test set, then average the absolute values to get an overall
    importance score per feature.
    """
    # Import the shap library (imported here so the rest of the code doesn't
    # fail if shap isn't installed but this function isn't called).
    import shap

    out = {}
    # SHAP TreeExplainer only works with tree-based models, so we loop over
    # XGBoost and LightGBM (skipping logistic regression and others).
    for key in ("xgboost", "lightgbm"):
        pipe = models[key]
        # Get the encoded feature names from the preprocessing step.
        names = pipe.named_steps["prep"].get_feature_names_out()
        # Transform the raw test data through the preprocessing pipeline
        # (scaling, encoding, etc.) to get the matrix the model actually sees.
        Xt = pipe.named_steps["prep"].transform(test[FEATURES])
        # Compute SHAP values using TreeExplainer.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sv = shap.TreeExplainer(pipe.named_steps["model"]).shap_values(Xt)
        # Ensure the result has shape (num_samples, num_features, num_classes).
        # Some SHAP versions return a list of arrays (one per class) instead.
        sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)  # -> (n, p, classes)
        # For each class (On-Time, At-Risk, Critical), compute the mean absolute
        # SHAP value per feature — this tells us the average "push" each feature
        # exerts on that class's prediction.
        for i, c in enumerate(config.CLASSES):
            out[f"{key}_shap_{c}"] = _to_source(np.abs(sv[:, :, i]).mean(0), names)
        # ── Create a "beeswarm" plot for the Critical class ──────────────────
        # A beeswarm plot shows every test sample as a dot, positioned by its
        # SHAP value, colored by the feature's actual value.  This reveals not
        # just which features matter, but HOW they affect the prediction.
        # beeswarm for the Critical class on encoded features
        fig = plt.figure()
        shap.summary_plot(sv[:, :, 2], pd.DataFrame(Xt, columns=names), max_display=20, show=False, plot_size=(8, 7))
        plt.title(f"{labels[key]} - SHAP values for the Critical class (test period)", fontsize=11)
        plt.tight_layout()
        plt.savefig(config.RESULT_DIRS["figures"] / f"shap_critical_{key}.png", bbox_inches="tight", dpi=150)
        plt.close("all")
    return pd.DataFrame(out)


def _plot_groups(groups, labels):
    """Create a horizontal bar chart showing group permutation importance.

    This produces a side-by-side figure with two panels:
    - Left panel: drop in macro-F1 when each group is shuffled.
    - Right panel: drop in Critical recall when each group is shuffled.
    Higher bars mean the model depends more on that feature group.
    """
    # Sort feature groups by their average macro-F1 drop (least to most important).
    order = groups.groupby("group")["macro_f1_drop"].mean().sort_values().index
    # Get the unique model names, preserving their original order.
    keys = list(dict.fromkeys(groups["model"]))
    # Define colors for each model's bars.
    colors = ["#8a8985", "#2a78d6", "#1b9e77", "#7b3fb5"]
    # Create a figure with two side-by-side subplots that share the same y-axis.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    # Calculate the height of each bar (bars for different models are stacked
    # vertically next to each other for the same group).
    h = 0.8 / len(keys)
    # Draw bars for both metrics (macro-F1 drop and Critical recall drop).
    for ax, metric, title in [(axes[0], "macro_f1_drop", "Drop in macro-F1"), (axes[1], "critical_recall_drop", "Drop in Critical recall")]:
        for j, key in enumerate(keys):
            # Filter the data for this model and reindex to the sorted order.
            g = groups[groups["model"] == key].set_index("group").reindex(order)
            # Draw horizontal bars, offset vertically so bars don't overlap.
            ax.barh(np.arange(len(order)) + j * h - 0.4 + h / 2, g[metric], height=h, color=colors[j % len(colors)], label=labels[key])
        # Draw a vertical line at x=0 as a reference baseline.
        ax.axvline(0, color="#0b0b0b", lw=0.8)
        # Label the y-axis with the group names (underscores replaced by spaces).
        ax.set_yticks(range(len(order)), [o.replace("_", " ") for o in order])
        ax.set_title(f"{title} when the group is permuted (test)")
        ax.set_xlabel("drop (higher = model relies more on the group)")
    # Add a legend to the first panel.
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    # Save the figure to disk.
    fig.savefig(config.RESULT_DIRS["figures"] / "group_permutation_importance.png", bbox_inches="tight")
    plt.close(fig)


def _plot_lr(coef):
    """Create a horizontal bar chart of the 20 largest logistic regression coefficients.

    Shows three bars per feature (one per class: On-Time, At-Risk, Critical),
    so you can see which features push toward which class.
    """
    # Take the top 20 features by coefficient size and reverse the order
    # so the most important feature appears at the top of the chart.
    top = coef.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    # Create y-positions for the bars.
    y = np.arange(len(top))
    # Draw three sets of horizontal bars: one for each class, with different colors.
    for j, (c, col) in enumerate([("On-Time", "#2a78d6"), ("At-Risk", "#eda100"), ("Critical", "#e34948")]):
        ax.barh(y + (j - 1) * 0.27, top[c], height=0.27, color=col, label=c)
    # Label the y-axis with the feature names.
    ax.set_yticks(y, top.index, fontsize=8)
    # Draw a vertical reference line at zero.
    ax.axvline(0, color="#0b0b0b", lw=0.8)
    ax.set_xlabel("coefficient (log-odds per SD for numeric inputs)")
    ax.set_title("Logistic Regression: 20 largest coefficients by class")
    ax.legend(fontsize=8)
    fig.tight_layout()
    # Save the chart as an image file.
    fig.savefig(config.RESULT_DIRS["figures"] / "logistic_regression_coefficients.png", bbox_inches="tight")
    plt.close(fig)


def main(ctx, supervised_modules):
    """Run all feature importance analyses and save results.

    This is the main entry point for the feature importance module.  It:
    1. Extracts logistic regression coefficients.
    2. Computes native (built-in) importances from tree models.
    3. Runs single-feature permutation importance for every model.
    4. Computes SHAP values for tree models.
    5. Runs group permutation importance (shuffling related features together).
    6. Saves everything to CSV files and generates charts.
    """
    # Get the test dataset from the shared context dictionary.
    test = ctx["test"]
    # Build a dictionary mapping model names to their human-readable labels
    # (e.g., "xgboost" → "XGBoost").
    labels = {m.NAME: m.LABEL for m in supervised_modules}
    # Load the saved (trained) model files from disk for each algorithm.
    models = {m.NAME: joblib.load(config.MODELS / f"{m.NAME}.joblib") for m in supervised_modules}

    # ── Step 1: Logistic Regression coefficients ─────────────────────────────
    coef, lr_src = lr_coefficients(models["logistic_regression"])
    # Plot the top 20 LR coefficients as a bar chart.
    _plot_lr(coef)

    # ── Step 2: Native (built-in) importances from tree models ───────────────
    native = native_importances(models)

    # ── Step 3: Single-feature permutation importance ────────────────────────
    perm = permutation_by_feature(models, test)

    # ── Step 4: SHAP values for XGBoost and LightGBM ────────────────────────
    shp = shap_values(models, test, labels)

    # ── Combine all per-feature importance measures into one table ───────────
    by_feature = pd.concat([lr_src, native, perm, shp], axis=1)
    # Add a "group" column so we know which feature group each feature belongs to.
    by_feature.insert(0, "group", [FEATURE_TO_GROUP[c] for c in by_feature.index])
    # Save the combined importance table to a CSV file.
    by_feature.to_csv(OUT / "importance_by_feature.csv")

    # ── Step 5: Group permutation importance ─────────────────────────────────
    groups = permutation_by_group(models, test)
    # Save the group-level results to CSV.
    groups.to_csv(OUT / "group_permutation_importance.csv", index=False)
    # Create the group importance bar chart.
    _plot_groups(groups, labels)

    # ── Step 6: Compute each group's share of total importance ───────────────
    # For each importance method, calculate what fraction of the total
    # importance falls in each feature group.  This shows which groups
    # dominate across different measurement techniques.
    # share of each method's importance that falls in each group (permutation: positive drops only)
    share = {}
    for col in ["lr_max_abs_coef", "rf_impurity", "xgb_gain", "lgbm_gain", "xgboost_shap_Critical", "lightgbm_shap_Critical"] + \
            [f"{k}_perm_mean" for k in models]:
        # Clip negative values to zero (a negative drop means the feature
        # actually *helped* when shuffled — we ignore those here).
        v = by_feature[col].clip(lower=0)
        # Calculate each group's proportion of the total importance.
        share[col] = v.groupby(by_feature["group"]).sum() / v.sum()
    # Save the share-by-group table, rounded to 4 decimal places.
    pd.DataFrame(share).round(4).to_csv(OUT / "importance_share_by_group.csv")
    return by_feature, groups
