"""Tests for the calibration study and the one-vs-rest calibrator (no model is retrained).

Run from the project root:  .venv/bin/python -m unittest discover -s ML/ML/tests
"""
import json
import unittest

import numpy as np

from cargomind_ml import config, inference as mli
from cargomind_ml.calibration import OvRCalibrator

STUDY = config.ML_WORKFLOW / "Evaluation" / "calibration_study"


class Calibrator(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.p = rng.dirichlet([2, 1, 1], 600)
        self.y = np.array([rng.choice(3, p=row) for row in self.p])

    def test_outputs_are_probabilities_and_inputs_unchanged(self):
        for method in ("sigmoid", "isotonic"):
            before = self.p.copy()
            q = OvRCalibrator(method).fit(self.p, self.y).transform(self.p)
            self.assertEqual(q.shape, self.p.shape)
            np.testing.assert_allclose(q.sum(1), 1.0, atol=1e-9)
            self.assertTrue(((q >= 0) & (q <= 1)).all())
            np.testing.assert_array_equal(self.p, before)

    def test_sigmoid_is_monotone_per_class_before_normalisation(self):
        cal = OvRCalibrator("sigmoid").fit(self.p, self.y)
        grid = np.linspace(0.01, 0.99, 50)
        out = cal.calibrators_[2].predict_proba(np.log(grid / (1 - grid))[:, None])[:, 1]
        self.assertTrue((np.diff(out) >= 0).all())

    def test_rejects_bad_method_and_shape(self):
        with self.assertRaises(ValueError):
            OvRCalibrator("beta")
        with self.assertRaises(ValueError):
            OvRCalibrator("sigmoid").fit(self.p[:, :2], self.y)


class StudyOutcome(unittest.TestCase):
    def test_decision_is_recorded_and_raw_outputs_are_kept(self):
        info = json.loads((STUDY / "decision.json").read_text())
        self.assertEqual({k: v["chosen_method"] for k, v in info["decision"].items()}, {"xgboost": None, "logistic_regression": None})
        # not justified -> no production calibrator, and the inference output stays raw
        self.assertFalse((config.MODELS / "xgboost_calibrator.joblib").exists())
        import pandas as pd
        rows = pd.read_pickle(config.RESULT_DIRS["data"] / "modelling_table.pkl").head(2)
        r = mli.predict(rows, anomaly=False)[0]["classification"]
        self.assertIn("Not recalibrated", r["probabilities_note"])
        self.assertEqual(set(r), {"model_name", "model_label", "model_version", "predicted_class", "class_probabilities",
                                  "decision_rule", "probabilities_note"})  # no calibrated field added
        sel = mli.load_selection()["classifier"]
        self.assertIsNone(sel["alert_threshold"])
        self.assertEqual(sel["challenger"], "logistic_regression")


if __name__ == "__main__":
    unittest.main()
