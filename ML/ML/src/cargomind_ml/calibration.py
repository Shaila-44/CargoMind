"""Post-hoc probability calibration for the 3-class classifiers (one-vs-rest, then renormalised).

Same scheme as sklearn's CalibratedClassifierCV for multiclass problems: one calibrator per class, fitted on that
class's predicted probability against the one-vs-rest outcome, then the three calibrated values are normalised to sum
to 1 (uniform if all three are 0).

* "sigmoid"  - Platt scaling: logistic regression on the logit of the class probability;
* "isotonic" - isotonic regression on the class probability (monotone, non-parametric).

A calibrator only re-maps probabilities. It is fitted on held-out (validation) predictions, never on the rows the
model was trained on, and it does not change the saved model.
"""
# =============================================================================
# WHAT IS PROBABILITY CALIBRATION?
# =============================================================================
# Think of a weather forecaster:
# If the forecaster says: "There is an 80% chance of rain today," and you look at
# 100 different days where they made that 80% claim... it should actually rain on
# about 80 of those days!
# If it only rained on 30 of those days, the forecaster is wildly OVERCONFIDENT.
#
# Many machine learning models (especially tree-based algorithms like XGBoost and
# LightGBM) are great at ranking (deciding shipment A is riskier than shipment B),
# but their raw probability numbers can be pushed towards 0% or 100%.
#
# Calibration is like a "truth tuner":
# It adjusts the model's confidence scores so that an "80% risk" prediction
# genuinely means an 80% chance in real life.
#
# TWO METHODS USED HERE:
# 1. "sigmoid" (Platt Scaling): Fits a smooth S-curve (logistic regression) to
#    re-map probabilities. Works great with smaller validation sets.
# 2. "isotonic" (Isotonic Regression): Fits a non-parametric staircase step function
#    that is guaranteed never to go downward. Very flexible when plenty of data is available.
# =============================================================================

from __future__ import annotations

# --- Third-party library imports ---
import numpy as np  # For array math, logarithms, clipping, and matrix operations
from sklearn.isotonic import IsotonicRegression   # For isotonic (stepwise monotone) calibration
from sklearn.linear_model import LogisticRegression # For Platt scaling (smooth sigmoid calibration)

# Small epsilon value to avoid math errors (division by zero or log of zero)
EPS = 1e-6
# Supported calibration methods
METHODS = ("sigmoid", "isotonic")


def _logit(p):
    """Convert a probability (between 0 and 1) into log-odds ('logit').

    Formula: logit(p) = ln(p / (1 - p))
    For example:
      p = 0.5 (even odds 50/50) -> logit = 0.0
      p = 0.9 (high confidence) -> logit = +2.2
      p = 0.1 (low confidence)  -> logit = -2.2

    We clip p slightly away from 0 and 1 to prevent taking log(0) or dividing by zero.
    """
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class OvRCalibrator:
    """One-vs-Rest (OvR) multiclass probability calibrator.

    HOW MULTICLASS CALIBRATION WORKS:
    Our problem has 3 classes: [0: On-Time, 1: At-Risk, 2: Critical].
    We train 3 independent mini-calibrators (one for each class):
      - Calibrator 0: Tunes probabilities for "On-Time vs Not On-Time"
      - Calibrator 1: Tunes probabilities for "At-Risk vs Not At-Risk"
      - Calibrator 2: Tunes probabilities for "Critical vs Not Critical"

    After each calibrator outputs its refined probability, the numbers might not
    sum to exactly 1.0 (100%). We normalise them by dividing each by their total sum.
    """

    def __init__(self, method: str):
        # Validate that the requested calibration method is supported
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.method = method

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "OvRCalibrator":
        """Learn the calibration adjustment curves from held-out validation data.

        Parameters:
          proba : Array of shape [N, 3] with raw predicted probabilities from the model
          y     : True labels (0, 1, or 2)
        """
        proba, y = np.asarray(proba, float), np.asarray(y)
        # Sanity check: must have 3 columns corresponding to our 3 delay classes
        if proba.ndim != 2 or proba.shape[1] != 3 or len(proba) != len(y):
            raise ValueError("proba must be [n, 3] and aligned with y")

        self.calibrators_ = []

        # Train one calibrator per class (k = 0, 1, 2)
        for k in range(3):
            # Binary target: 1 if this shipment belonged to class k, 0 otherwise
            target = (y == k).astype(int)

            if self.method == "sigmoid":
                # Platt scaling: Logistic regression on the logit-transformed probability
                # C=1e6 means virtually no regularisation penalty (unconstrained fit)
                c = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(proba[:, k])[:, None], target)
            else:
                # Isotonic regression: Fits a free-form monotonically increasing curve
                # bounded strictly between 0.0 and 1.0
                c = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(proba[:, k], target)

            self.calibrators_.append(c)

        # Store metadata for auditability
        self.n_fit_ = int(len(y))
        self.class_counts_ = [int((y == k).sum()) for k in range(3)]
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        """Apply calibration to fresh probability predictions.

        Takes uncalibrated probabilities [N, 3] and outputs calibrated,
        trustworthy probabilities [N, 3] that sum to 1.0 for every shipment.
        """
        proba = np.asarray(proba, float)
        cols = []

        # Calibrate each class's probability column independently
        for k, c in enumerate(self.calibrators_):
            if self.method == "sigmoid":
                # For sigmoid, extract probability of the positive class (column 1)
                cols.append(c.predict_proba(_logit(proba[:, k])[:, None])[:, 1])
            else:
                # For isotonic, predict directly maps values along the learned step function
                cols.append(c.predict(proba[:, k]))

        # Stack the 3 calibrated columns side-by-side into an [N, 3] array
        q = np.column_stack(cols)

        # Sum across the 3 classes for each shipment row
        s = q.sum(1, keepdims=True)

        # Normalise so the row sums to exactly 1.0 (100%).
        # If all 3 were zero (an extreme edge case), assign equal 1/3 (33.3%) to each class.
        return np.where(s > 0, q / np.where(s > 0, s, 1), 1.0 / 3)
