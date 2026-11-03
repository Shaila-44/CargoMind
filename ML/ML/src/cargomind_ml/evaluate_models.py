"""Evaluation of the saved models.

Supervised: accuracy, per-class / macro / weighted precision-recall-F1, one-vs-rest ROC-AUC,
log loss, ordinal error, Critical->On-Time miss rate, confusion matrices and classification
reports, on the validation block and on the test period (all shipments and sea only). Test-period
uncertainty is estimated with a bootstrap that resamples whole departure days, because shipments
leaving the same port on the same day share conditions and are not independent.

Anomaly: scores are compared with the known classes; they are never turned into class predictions.
Positive mapping: anomaly-positive = delayed (At-Risk or Critical); Critical-only is reported too.
"""
import json

import joblib
import matplotlib

# Use a non-interactive backend so matplotlib can save images without
# needing a display window (important when running on a server).
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as ss  # noqa: E402
from sklearn.base import clone  # noqa: E402
# Import a collection of standard ML evaluation metrics from scikit-learn.
# Each metric measures a different aspect of how well a model is performing:
#   - accuracy_score:  % of predictions that are exactly correct
#   - average_precision_score:  summarises precision at every recall level (area under the PR curve)
#   - classification_report:  a handy text/table summary of precision, recall, F1 per class
#   - confusion_matrix:  a grid showing how many items of each true class were predicted as each class
#   - f1_score:  the harmonic mean of precision and recall — one number that balances both
#   - log_loss:  measures how confident the model is — penalises confident wrong predictions heavily
#   - precision_recall_fscore_support:  returns precision, recall, F1, and support (count) per class
#   - precision_score:  of all items the model flagged as a class, what % actually belong to it
#   - recall_score:  of all items that truly belong to a class, what % did the model find
#   - roc_auc_score:  measures how well the model separates classes across all confidence thresholds
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config  # noqa: E402
from .features import FEATURES  # noqa: E402
from .preprocessing import ANOMALY_INPUT_COLUMNS  # noqa: E402

# The three shipment classes encoded as integers:
#   0 = On-Time, 1 = At-Risk, 2 = Critical
LABELS = [0, 1, 2]

# Consistent colours for plotting each class (blue, amber, red).
CLASS_COLORS = {"On-Time": "#2a78d6", "At-Risk": "#eda100", "Critical": "#e34948"}  # same as EDA config


def setup_plots():
    """Set up a clean, consistent visual style for all plots created during evaluation."""
    import seaborn as sns

    # "whitegrid" gives a light grey background grid that makes charts easier to read.
    sns.set_theme(style="whitegrid", context="notebook")
    # Fine-tune plot appearance: hide top/right borders, set DPI (resolution), etc.
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.titlesize": 11, "axes.titleweight": "semibold",
                         "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#e6e5e1", "legend.frameon": False})


# ---------------------------------------------------------------------------------------------
# Supervised metrics
# ---------------------------------------------------------------------------------------------
# "Supervised" means the model was trained with labeled examples (we told it the correct
# answer for each training shipment).  Below we measure how well each supervised model
# does on data it has NOT seen during training.
# ---------------------------------------------------------------------------------------------

def classification_metrics(y, proba):
    """Compute a comprehensive dictionary of classification quality metrics.

    Parameters
    ----------
    y : array-like
        The true class labels (0 = On-Time, 1 = At-Risk, 2 = Critical).
    proba : np.ndarray, shape [n_samples, 3]
        The model's predicted probability for each of the three classes.

    Returns
    -------
    dict
        A flat dictionary containing every metric described below.

    Metrics explained in plain English
    -----------------------------------
    * **Accuracy**: Out of ALL shipments, what percentage did the model label correctly?
    * **Precision** (per class): When the model says "this is At-Risk", how often is it right?
    * **Recall** (per class): Out of all truly At-Risk shipments, how many did the model catch?
    * **F1 score**: A single number that balances Precision and Recall (their harmonic mean).
      If either is low, F1 will be low too.
    * **Macro average**: Compute the metric for each class separately, then take the simple
      average.  Treats every class as equally important, even if some classes are rare.
    * **Weighted average**: Like macro, but each class's metric is weighted by how many
      samples it has.  Gives more influence to larger classes.
    * **Log loss**: Measures the quality of the predicted *probabilities*.  A model that is
      very confident and wrong gets a high (bad) log-loss penalty.
    * **ROC-AUC (OvR)**: "Area Under the ROC Curve, One-vs-Rest".  For each class, we ask
      "how well does the model separate this class from the other two?" and measure the area
      under the trade-off curve between true positives and false positives.  A perfect model
      scores 1.0; random guessing scores 0.5.
    * **Ordinal MAE**: Since our classes have a natural order (On-Time < At-Risk < Critical),
      we also measure the average "distance" between the predicted and true class on this
      0-1-2 scale.  Predicting Critical when the truth is On-Time (distance = 2) is worse
      than predicting At-Risk (distance = 1).
    * **Critical→On-Time miss rate**: The single costliest mistake — how often a truly
      Critical shipment is mislabeled as On-Time.  This is the error with the biggest
      real-world consequence (we'd take no corrective action on a critical delay).
    """
    # Make sure y is a NumPy array so we can do array math on it.
    y = np.asarray(y)

    # Convert probabilities to hard predictions by picking the class with the highest probability.
    pred = proba.argmax(1)

    # Get per-class precision (p), recall (r), F1 (f), and support/count (s) in one call.
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)

    # Build the output dictionary, starting with aggregate (overall) metrics.
    out = {
        # Total number of samples evaluated.
        "n": int(len(y)),

        # Accuracy: fraction of predictions that match the true label.
        "accuracy": accuracy_score(y, pred),

        # --- Macro averages (treat every class equally) ---
        "macro_precision": precision_score(y, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),

        # --- Weighted averages (weight each class by its frequency) ---
        "weighted_precision": precision_score(y, pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y, pred, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),

        # Log loss: penalises confidently-wrong probability estimates.
        "log_loss": log_loss(y, proba, labels=LABELS),

        # mean |predicted class - true class| on the ordinal scale 0/1/2
        "ordinal_mae": float(np.abs(pred - y).mean()),

        # costliest error: a Critical shipment predicted On-Time
        # Numerator: how many Critical shipments were called On-Time.
        # Denominator: total number of Critical shipments (at least 1 to avoid division by zero).
        "critical_predicted_on_time_rate": float(((pred == 0) & (y == 2)).sum() / max(1, (y == 2).sum())),
    }

    # Add per-class metrics (precision, recall, F1, support) for each of the three classes.
    for i, c in enumerate(config.CLASSES):
        key = c.lower().replace("-", "_")
        out.update({f"{key}_precision": p[i], f"{key}_recall": r[i], f"{key}_f1": f[i], f"{key}_support": int(s[i])})

    # ROC-AUC requires all three classes to be present in the data.
    # "OvR" = One-vs-Rest: for each class, treat it as "positive" and the other two as "negative",
    # then compute the AUC.  This gives a per-class and overall measure of separability.
    if len(np.unique(y)) == 3:
        out["roc_auc_ovr_macro"] = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=LABELS)
        out["roc_auc_ovr_weighted"] = roc_auc_score(y, proba, multi_class="ovr", average="weighted", labels=LABELS)
        # Also store each class's individual ROC-AUC.
        for i, c in enumerate(config.CLASSES):
            out[f"{c.lower().replace('-', '_')}_roc_auc_ovr"] = roc_auc_score(y == i, proba[:, i])
    return out


def day_block_bootstrap(y, proba, days, reps=config.BOOTSTRAP_REPS, seed=config.SEED):
    """95 % percentile intervals, resampling departure days with replacement.

    Why do we need this?
    --------------------
    We want to know how *confident* we can be in the test-set numbers.  If we just reported
    one accuracy value, we wouldn't know whether that number could easily swing by ±5 % or is
    rock-solid.  A **confidence interval** gives a range: "we're 95 % sure the true metric
    lies between X and Y."

    Why resample whole *days* instead of individual shipments?
    ----------------------------------------------------------
    Shipments departing from the same port on the same day share weather, port congestion,
    and carrier schedules.  They are NOT independent.  If we resampled individual shipments,
    we'd underestimate the uncertainty.  By resampling entire departure days (called "block
    bootstrap"), we keep correlated shipments together and get a more honest confidence
    interval.

    How it works step-by-step
    -------------------------
    1. Identify the unique departure days in the test set.
    2. Repeat ``reps`` times (e.g. 2 000):
       a. Randomly draw days *with replacement* (some days appear twice, some not at all).
       b. Collect all shipments from those drawn days.
       c. Compute the metric (e.g. macro-F1) on that resampled collection.
    3. The 2.5th and 97.5th percentiles of the collected metric values form the 95 % CI.
    """
    # Create a reproducible random number generator.
    rng = np.random.default_rng(seed)

    y, pred = np.asarray(y), proba.argmax(1)

    # Find every unique departure day.
    uniq = np.unique(days)

    # Pre-build a lookup: for each day, which row indices belong to it.
    idx_by_day = {d: np.flatnonzero(days == d) for d in uniq}

    # We'll collect metric values from each bootstrap iteration.
    vals = {"macro_f1": [], "critical_recall": [], "accuracy": [], "weighted_f1": []}

    for _ in range(reps):
        # Draw a random set of days (same count as original, but with replacement).
        idx = np.concatenate([idx_by_day[d] for d in rng.choice(uniq, len(uniq), replace=True)])
        # Gather true labels and predictions for the resampled rows.
        yy, pp = y[idx], pred[idx]
        # Compute each metric on this resampled dataset.
        vals["macro_f1"].append(f1_score(yy, pp, average="macro", labels=LABELS, zero_division=0))
        vals["weighted_f1"].append(f1_score(yy, pp, average="weighted", labels=LABELS, zero_division=0))
        vals["accuracy"].append((yy == pp).mean())
        vals["critical_recall"].append(recall_score(yy, pp, labels=[2], average="macro", zero_division=0))

    # Return the 2.5 % and 97.5 % percentiles as the 95 % confidence interval for each metric.
    return {f"{k}_ci95": (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in vals.items()}


def save_classification_outputs(name, label, y, proba, tag):
    """Save a per-model classification report (text + CSV) and confusion matrix CSV.

    A **classification report** is a table showing precision, recall, and F1 for every class
    plus macro/weighted averages — a quick summary of model quality.

    A **confusion matrix** is a grid where:
      - Each *row* represents the TRUE class.
      - Each *column* represents the PREDICTED class.
      - The cell (row i, col j) counts how many items of true class i were predicted as class j.
    Diagonal cells are correct predictions; off-diagonal cells are mistakes.
    """
    # Hard predictions: pick the class with the highest probability for each sample.
    pred = proba.argmax(1)

    # Generate a scikit-learn classification report as a dictionary, then save as CSV.
    rep = classification_report(y, pred, labels=LABELS, target_names=config.CLASSES, digits=3, zero_division=0, output_dict=True)
    pd.DataFrame(rep).T.to_csv(config.RESULT_DIRS["classification_reports"] / f"{name}_{tag}.csv")

    # Also save a human-readable text version of the same report.
    (config.RESULT_DIRS["classification_reports"] / f"{name}_{tag}.txt").write_text(
        f"{label} - {tag}\n\n" + classification_report(y, pred, labels=LABELS, target_names=config.CLASSES, digits=3, zero_division=0))

    # Compute the confusion matrix and save it as a labelled CSV.
    cm = confusion_matrix(y, pred, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=[f"true {c}" for c in config.CLASSES], columns=[f"pred {c}" for c in config.CLASSES])
    cm_df.to_csv(config.RESULT_DIRS["confusion_matrices"] / f"{name}_{tag}.csv")
    return cm


def plot_confusion_grid(cms, tag, title):
    """One panel per model: cell = share of the true class (row-normalised recall), with counts."""
    # Create a row of subplots — one per model.
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6))
    # Ensure axes is always an array (even if there's only one model).
    axes = np.atleast_1d(axes)

    for ax, (label, cm) in zip(axes, cms.items()):
        # Row-normalise: divide each cell by the total for that true class.
        # This shows the "recall" for each cell — what fraction of the true class
        # ended up in each predicted bucket.
        share = cm / cm.sum(1, keepdims=True).clip(min=1)
        # Draw the matrix as a colour-coded image (darker blue = higher share).
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        # Annotate each cell with the percentage and raw count.
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{share[i, j]:.0%}\n({cm[i, j]:,})", ha="center", va="center", fontsize=8.5,
                        color="white" if share[i, j] > 0.55 else "#0b0b0b")
        # Label axes with the class names.
        ax.set_xticks(range(3), config.CLASSES, fontsize=8)
        ax.set_yticks(range(3), config.CLASSES, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true" if ax is axes[0] else "")
        ax.set_title(label)
        ax.grid(False)

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(config.RESULT_DIRS["figures"] / f"confusion_matrices_{tag}.png", bbox_inches="tight")
    plt.close(fig)


def evaluate_supervised(ctx, modules):
    """Run full evaluation of every supervised (classification) model.

    Steps
    -----
    1. Load each saved model from disk.
    2. For the **validation set** (last CV fold), re-fit a clone of the model on only the
       fold's training rows, then predict the held-out validation rows.  This ensures the
       validation predictions are truly "unseen" by that copy of the model.
    3. For the **test set** (latest 25 % of shipments chronologically), use the full model
       that was trained on all training data.
    4. Compute classification metrics (accuracy, F1, etc.) on each split.
    5. On the test set, also run a day-block bootstrap to get 95 % confidence intervals.
    6. Save confusion matrices, classification reports, and a CSV of all predictions.
    """
    # Unpack the context dictionary that was prepared earlier in the pipeline.
    train, test, folds, k = ctx["train"], ctx["test"], ctx["folds"], ctx["val_fold"]

    # Get the row indices for the training and validation portions of the last CV fold.
    tr_idx, val_idx = folds[k]
    val = train.iloc[val_idx]

    # Create a subset of the test data that only contains sea shipments (for separate evaluation).
    test_sea = test[test["shipping_mode"] == "sea"]

    # Extract the departure day for every test shipment (used for day-block bootstrap).
    days = test["departure_ts"].dt.floor("D").values

    rows, probas = [], {}
    cms = {"validation": {}, "test": {}, "test_sea": {}}

    # Build the list of models to evaluate, including a "dummy" baseline that always
    # predicts the majority class (to see how much better real models are).
    entries = [(m.NAME, m.LABEL) for m in modules] + [("dummy_prior", "Majority-class reference")]

    for name, label in entries:
        # Load the pre-trained model from a .joblib file.
        model = joblib.load(config.MODELS / f"{name}.joblib")

        # validation: same hyperparameters, refit on the fold's training rows only
        # "clone" creates a fresh copy with the same settings but no learned parameters.
        fold_model = clone(model).fit(train.iloc[tr_idx][FEATURES], train.iloc[tr_idx]["y"])
        # Predict class probabilities on each data split.
        p_val = fold_model.predict_proba(val[FEATURES])
        p_test = model.predict_proba(test[FEATURES])
        p_sea = model.predict_proba(test_sea[FEATURES])
        probas[name] = p_test

        # Loop over each evaluation split and compute metrics.
        for tag, df, p in [("validation", val, p_val), ("test", test, p_test), ("test_sea", test_sea, p_sea)]:
            m = classification_metrics(df["y"], p)
            # Only for the test set: add bootstrap confidence intervals.
            if tag == "test":
                m.update(day_block_bootstrap(df["y"].values, p, days))
            rows.append({"model": label, "key": name, "set": tag, **m})
            # Save per-model classification report and confusion matrix files.
            cm = save_classification_outputs(name, label, df["y"].values, p, tag)
            # Skip the dummy baseline when collecting confusion matrices for plots.
            if name != "dummy_prior":
                cms[tag][label] = cm

    # Plot side-by-side confusion matrix grids for each evaluation split.
    for tag, title in [("validation", "Validation block (last CV fold)"), ("test", "Test period - all shipments"),
                       ("test_sea", "Test period - sea shipments")]:
        plot_confusion_grid(cms[tag], tag, f"Confusion matrices, row-normalised (recall per true class) - {title}")

    # Collect all per-model, per-split metrics into one big DataFrame and save.
    metrics = pd.DataFrame(rows)
    metrics.to_csv(config.RESULT_DIRS["metrics"] / "supervised_metrics.csv", index=False)

    # Build a test-predictions CSV: for every test shipment, store the model's predicted
    # probabilities and final predicted class alongside the true label.
    pred_df = test[["shipment_id", "departure_ts", "origin_port_id", "shipping_mode", "delay_status", config.OUTCOME]].copy()
    for name, p in probas.items():
        for i, c in enumerate(config.CLASSES):
            pred_df[f"{name}_p_{c}"] = p[:, i]
        pred_df[f"{name}_pred"] = np.array(config.CLASSES)[p.argmax(1)]
    pred_df.to_csv(config.RESULT_DIRS["metrics"] / "test_predictions.csv", index=False)
    return metrics


# ---------------------------------------------------------------------------------------------
# Anomaly evaluation
# ---------------------------------------------------------------------------------------------
# Anomaly detection works DIFFERENTLY from classification:
#   - A classifier assigns each shipment to one of three classes.
#   - An anomaly detector gives each shipment a continuous "anomaly score" indicating how
#     unusual/abnormal its operational features look.  Higher scores = more suspicious.
#   - A threshold is used to flag shipments above a certain score as "anomalous".
#   - We then check whether the flagged shipments tend to be the delayed ones.
# Anomaly models are NOT trying to pick among three classes; they just say "this looks weird."
# That's why we evaluate them with different metrics than the supervised classifiers.
# ---------------------------------------------------------------------------------------------

def _cliffs_delta(a, b):
    """P(a > b) - P(a < b) via the Mann-Whitney U statistic.

    Cliff's delta is an "effect size" measure.  It answers: "If I randomly pick one score
    from group A and one from group B, how much more likely is A's score to be higher?"
    Values range from -1 to +1:
      -  0 means the two groups are indistinguishable.
      - +1 means every score in A exceeds every score in B.
      - -1 means every score in B exceeds every score in A.
    We use this to check whether anomaly scores for delayed shipments are meaningfully
    higher than scores for on-time shipments.
    """
    if len(a) == 0 or len(b) == 0:
        return np.nan
    # The Mann-Whitney U test compares two groups without assuming a normal distribution.
    u = ss.mannwhitneyu(a, b, alternative="two-sided").statistic
    # Convert the U statistic to Cliff's delta.
    return 2 * u / (len(a) * len(b)) - 1


def anomaly_metrics(y, score, flag):
    """Compute metrics for an anomaly detector on a labelled dataset.

    Parameters
    ----------
    y : array
        True class labels (0 = On-Time, 1 = At-Risk, 2 = Critical).
    score : array
        Continuous anomaly score produced by the model (higher = more unusual).
    flag : array of bool
        Binary flag: True if the model considers this shipment anomalous (score > threshold).

    How anomaly evaluation works
    ----------------------------
    Since anomaly detectors don't predict one of three classes, we can't use accuracy or
    macro-F1 directly.  Instead, we define two binary "positive" groups:
      1. **Delayed** (At-Risk OR Critical) — any shipment that wasn't on time.
      2. **Critical** — only the most severely delayed shipments.
    For each group we compute binary precision, recall, F1, ROC-AUC, and PR-AUC to see how
    well the anomaly score separates the positive group from the rest.

    We also report the flag rate within each true class — ideally, most On-Time shipments
    should NOT be flagged, while most Critical shipments SHOULD be.
    """
    y = np.asarray(y)
    out = {"n": int(len(y)), "anomaly_detection_rate": float(flag.mean())}

    # Evaluate against two different definitions of "positive":
    #   - "delayed" = At-Risk + Critical  (label > 0)
    #   - "critical" = Critical only       (label == 2)
    for pos_name, pos in [("delayed", y > 0), ("critical", y == 2)]:
        out.update({
            # What fraction of all shipments are actually in this positive group.
            f"{pos_name}_prevalence": float(pos.mean()),
            # Of all flagged shipments, how many truly belong to this positive group.
            f"{pos_name}_precision": precision_score(pos, flag, zero_division=0),
            # Of all truly positive shipments, how many did the flag catch.
            f"{pos_name}_recall": recall_score(pos, flag, zero_division=0),
            # Harmonic mean of precision and recall.
            f"{pos_name}_f1": f1_score(pos, flag, zero_division=0),
            # ROC-AUC: how well the continuous score separates positive from negative.
            f"{pos_name}_roc_auc": roc_auc_score(pos, score),
            # PR-AUC: area under the precision-recall curve (better for imbalanced data).
            f"{pos_name}_pr_auc": average_precision_score(pos, score),
        })

    # Per-class flag rate: what fraction of each true class gets flagged as anomalous.
    for i, c in enumerate(config.CLASSES):
        out[f"flag_rate_{c}"] = float(flag[y == i].mean())
        out[f"median_score_{c}"] = float(np.median(score[y == i]))

    # Cliff's delta: how much higher are scores for delayed/critical vs on-time?
    out["cliffs_delta_AtRisk_vs_OnTime"] = _cliffs_delta(score[y == 1], score[y == 0])
    out["cliffs_delta_Critical_vs_OnTime"] = _cliffs_delta(score[y == 2], score[y == 0])

    # Kruskal-Wallis test: are the score distributions across all three classes significantly
    # different?  A very small p-value means "yes, the classes have different score distributions."
    out["kruskal_p"] = float(ss.kruskal(*[score[y == i] for i in LABELS]).pvalue)
    return out


def evaluate_anomaly(ctx, modules):
    """Run full evaluation of every anomaly-detection model.

    Steps
    -----
    1. Load each saved anomaly model (pipeline + learned threshold) from disk.
    2. Score every shipment in the validation set, the full test set, and the sea-only test set.
    3. Compute anomaly metrics (ROC-AUC, PR-AUC, Cliff's delta, etc.) on each split.
    4. Save score distributions and flagged-shipment outputs.
    5. Plot violin charts showing how scores differ across the three true classes.
    """
    train, test, folds, k = ctx["train"], ctx["test"], ctx["folds"], ctx["val_fold"]
    val = train.iloc[folds[k][1]]
    rows, scores = [], {}

    for m in modules:
        # Each anomaly model is stored as a bundle containing the fitted pipeline and
        # the threshold above which a shipment is flagged as anomalous.
        bundle = joblib.load(config.MODELS / f"{m.NAME}.joblib")
        pipe, thr = bundle["pipeline"], bundle["threshold"]

        for tag, df in [("validation", val), ("test", test), ("test_sea", test[test["shipping_mode"] == "sea"])]:
            # Compute the anomaly score for each shipment using the model-specific scoring function.
            s = m.anomaly_score(pipe, df[ANOMALY_INPUT_COLUMNS])
            # Compute metrics, using the score and the binary flag (score > threshold).
            rows.append({"model": m.LABEL, "key": m.NAME, "set": tag, "threshold": thr, **anomaly_metrics(df["y"].values, s, s > thr)})
            # Keep test-set scores for later plotting and output.
            if tag == "test":
                scores[m.LABEL] = (s, thr)

    # Collect all anomaly metrics into a DataFrame and save.
    metrics = pd.DataFrame(rows)
    metrics.to_csv(config.RESULT_DIRS["anomaly"] / "anomaly_metrics.csv", index=False)

    # --- Build detailed output CSVs for the test set ---
    y = test["y"].values
    out = test[["shipment_id", "departure_ts", "origin_port_id", "delay_status"]].copy()
    dist = []
    for label, (s, thr) in scores.items():
        key = label.lower().replace(" ", "_").replace("-", "_")
        # Add each model's score and flag column to the output DataFrame.
        out[f"{key}_score"], out[f"{key}_flag"] = s, s > thr
        # Compute score distribution statistics (percentiles) for each true class.
        for i, c in enumerate(config.CLASSES):
            q = np.percentile(s[y == i], [5, 25, 50, 75, 95])
            dist.append({"model": label, "class": c, "n": int((y == i).sum()), "mean": s[y == i].mean(),
                         **{f"p{p}": v for p, v in zip([5, 25, 50, 75, 95], q)}, "flag_rate": float((s[y == i] > thr).mean())})
    out.to_csv(config.RESULT_DIRS["anomaly"] / "test_anomaly_scores.csv", index=False)
    pd.DataFrame(dist).to_csv(config.RESULT_DIRS["anomaly"] / "score_distribution_by_class.csv", index=False)

    # Visualise how scores are distributed for each true class (violin plots).
    _plot_anomaly_distributions(scores, y)
    return metrics


def _plot_anomaly_distributions(scores, y):
    """Create violin plots showing the anomaly-score distribution for each true class.

    A violin plot is like a sideways histogram — wider sections mean more shipments have
    that score.  If the Critical violin sits noticeably higher than the On-Time violin,
    the anomaly detector is doing a good job.
    """
    fig, axes = plt.subplots(1, len(scores), figsize=(6 * len(scores), 3.8))
    for ax, (label, (s, thr)) in zip(np.atleast_1d(axes), scores.items()):
        # Split scores by true class.
        data = [s[y == i] for i in LABELS]
        # Draw the violin plot (one "violin" per class).
        parts = ax.violinplot(data, showextrema=False, showmedians=True)
        # Colour each violin with its class colour.
        for body, c in zip(parts["bodies"], config.CLASSES):
            body.set_facecolor(CLASS_COLORS[c])
            body.set_alpha(0.75)
        parts["cmedians"].set_color("#0b0b0b")
        # Draw a horizontal dashed line at the flagging threshold.
        ax.axhline(thr, color="#52514e", ls="--", lw=1)
        ax.text(3.45, thr, " flag\n threshold", va="center", fontsize=8, color="#52514e")
        # Show what percentage of each class gets flagged.
        rates = [(d > thr).mean() for d in data]
        ax.set_xticks([1, 2, 3], [f"{c}\n{r:.0%} flagged" for c, r in zip(config.CLASSES, rates)])
        ax.set_ylabel("anomaly score (higher = more unusual)")
        ax.set_title(f"{label}: score by true class (test period)")
    fig.tight_layout()
    fig.savefig(config.RESULT_DIRS["figures"] / "anomaly_score_distributions.png", bbox_inches="tight")
    plt.close(fig)


def main(ctx, supervised_modules, anomaly_modules):
    """Entry point: evaluate all supervised and anomaly models, then return their metrics."""
    setup_plots()
    # Evaluate all supervised (classification) models on validation and test data.
    sup = evaluate_supervised(ctx, supervised_modules)
    # Evaluate all anomaly-detection models on validation and test data.
    ano = evaluate_anomaly(ctx, anomaly_modules)
    return sup, ano
