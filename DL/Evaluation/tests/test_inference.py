"""Tests for the inference interface (dl.inference, in DL/DL/inference). Synthetic windows are built in memory; the project
dataset is only read (one equivalence test), never modified.

Run from the project root:  .venv/bin/python -m unittest discover -s DL/Evaluation/tests
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dl.inference import (DEFAULT_MODEL, REQUIRED_COLUMNS, DwellForecaster, InputValidationError, ModelLoadError, ewma_forecast,
                          forecast, load_model, load_registry)
from cargomind_dl.config import DATASET, DL_STAGE
from dl.inference.registry import PROJECT_ROOT, REGISTRY_PATH
from dl.models import RecurrentForecaster, TransformerForecaster

ARTIFACTS = DL_STAGE / "artifacts"


def window(port="SGSIN", dwell=None, start="2025-03-01 00:00:00+00:00"):
    """A valid synthetic 12-hour raw window (not taken from the dataset)."""
    rng = np.random.default_rng(7)
    dwell = np.round(80 + 3 * np.sin(np.arange(12)) + rng.normal(0, 1, 12), 1) if dwell is None else dwell
    return pd.DataFrame({
        "ts_hour": pd.date_range(start, periods=12, freq="h"),
        "dwell": dwell,
        "utilization": np.linspace(65, 80, 12),
        "labour_strike": [0] * 12,
        "wind_gust": np.linspace(18, 30, 12),
        "disruption_active": [0] * 6 + [1] * 6,
        "disruption_hours_elapsed": [0] * 6 + list(range(0, 6)),
        "vessels_at_anchor": [1, 2, 2, 3, 1, 0, 2, 4, 5, 5, 6, 7],
        "dwell_reconstructed": [0] * 11 + [1],
    })


class Registry(unittest.TestCase):
    def test_registry_lists_candidates_and_default(self):
        reg = load_registry()
        self.assertEqual(reg["default_model"], DEFAULT_MODEL)
        self.assertEqual(DEFAULT_MODEL, "transformer_t_w")
        self.assertNotIn("ewma", reg["models"])  # EWMA is a baseline, not a registered neural model
        for name in ("transformer_t_w", "gru", "c_w_lstm", "transformer_t", "basic_lstm"):
            self.assertIn(name, reg["models"])


class LoadingEachArchitecture(unittest.TestCase):
    def test_every_registered_checkpoint_loads_with_the_right_class(self):
        expected = {"transformer_t_w": TransformerForecaster, "transformer_t": TransformerForecaster, "gru": RecurrentForecaster,
                    "c_w_lstm": RecurrentForecaster, "basic_lstm": RecurrentForecaster, "lstm_b_w_original_selection": RecurrentForecaster}
        for name, cls in expected.items():
            f = load_model(name)
            self.assertIsInstance(f.model, cls, name)
            self.assertEqual(f.channels[-1], "dwell_reconstructed")
        self.assertEqual(load_model("gru").model.cfg.cell, "gru")
        self.assertEqual(load_model("c_w_lstm").model.cfg.residual_ewma_alpha, 0.2)
        self.assertGreater(load_model("gru").model.cfg.port_embedding_dim, 0)

    def test_old_checkpoints_without_model_type_still_load(self):
        f = DwellForecaster(ARTIFACTS / "lstm_best.pt", "lstm_best")
        self.assertNotIn("model_type", f.meta)
        self.assertIsInstance(f.model, RecurrentForecaster)

    def test_ablation_checkpoint_with_fewer_channels_selects_by_name(self):
        ck = ARTIFACTS / "runs" / "B_w_lstm_port_embedding_hour_weighted__no_labour_strike" / "seed_0.pt"
        f = DwellForecaster(ck, "ablation")
        x, _, _ = f.preprocess(window(), "SGSIN")
        self.assertEqual(x.shape, (1, 12, 7))
        self.assertNotIn("labour_strike", f.channels)


class OutputAndDeterminism(unittest.TestCase):
    def test_prediction_output_format(self):
        r = forecast(window(), "SGSIN")
        d = r.to_dict()
        for k in ("predicted_dwell_hours", "method", "model_name", "model_version", "port_id", "window_end_utc", "target_hour_utc", "warnings"):
            self.assertIn(k, d)
        self.assertEqual(r.method, "neural")
        self.assertEqual(r.model_name, "transformer_t_w")
        self.assertIn("@", r.model_version)
        self.assertEqual(r.target_hour_utc, "2025-03-01T12:00:00+00:00")
        self.assertTrue(np.isfinite(r.predicted_dwell_hours))
        self.assertTrue(20 < r.predicted_dwell_hours < 300)  # hours, not a z-score
        self.assertIn("input contains hours flagged as reconstructed dwell", r.warnings)
        self.assertNotIn("confidence", json.dumps(d).lower().replace("no calibrated uncertainty", ""))

    def test_deterministic_and_weights_unchanged(self):
        f = load_model("transformer_t_w")
        before = {k: v.clone() for k, v in f.model.state_dict().items()}
        a = f.predict(window(), "SGSIN").predicted_dwell_hours
        b = f.predict(window(), "SGSIN").predicted_dwell_hours
        self.assertEqual(a, b)
        for k, v in f.model.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]))
        self.assertFalse(f.model.training)

    def test_port_identifier_changes_prediction_for_port_specific_models(self):
        a = forecast(window(), "SGSIN", model="gru").predicted_dwell_hours
        b = forecast(window(), "NLRTM", model="gru").predicted_dwell_hours
        self.assertNotAlmostEqual(a, b, places=3)  # per-port scaling and embedding


class ScalingEquivalence(unittest.TestCase):
    """Read-only check: inference preprocessing reproduces the stored dataset input exactly."""

    def test_inference_input_equals_stored_test_window_and_inverse_is_correct(self):
        panel = pd.read_csv(DATASET / "hourly_panel.csv")
        d = np.load(DATASET / "test.npz")
        win = pd.read_csv(DATASET / "windows.csv").set_index("window_id")
        f = load_model("gru")
        sc = json.loads((DATASET / "scalers.json").read_text())
        for i in (0, 400, 900):
            w = win.loc[int(d["window_id"][i])]
            rows = panel[(panel["port_id"] == w["port_id"]) & panel["hour_index"].between(w["t_hour_index"] - 11, w["t_hour_index"])]
            raw = rows.assign(dwell_reconstructed=(rows["prov_dwell"] == 2).astype(int))
            x, _, _ = f.preprocess(raw, w["port_id"])
            # stored X holds the 7 feature channels; the loader appends the reconstruction flag from provenance
            np.testing.assert_allclose(x[0, :, :7], d["X"][i], atol=1e-5)
            np.testing.assert_array_equal(x[0, :, 7], (d["provenance"][i, :, 0] == 2).astype(np.float32))
            r = f.predict(raw, w["port_id"])
            with torch.no_grad():
                z = f.model(torch.from_numpy(x), torch.from_numpy(d["port_index"][i:i + 1].astype(np.int64))).item()
            st = sc["dwell"]["per_port"][w["port_id"]]
            self.assertAlmostEqual(r.predicted_dwell_hours, z * st["std"] + st["mean"], places=3)


class InputRejection(unittest.TestCase):
    def assertRejects(self, w, port="SGSIN", contains=""):
        with self.assertRaises(InputValidationError) as cm:
            forecast(w, port)
        self.assertIn(contains, str(cm.exception))

    def test_wrong_sequence_length(self):
        self.assertRejects(window().iloc[:11], contains="exactly 12")
        self.assertRejects(pd.concat([window(), window(start="2025-03-01 12:00:00+00:00").iloc[:1]]), contains="exactly 12")

    def test_missing_feature(self):
        self.assertRejects(window().drop(columns=["wind_gust"]), contains="missing required columns")

    def test_positional_array_rejected(self):
        self.assertRejects(window()[REQUIRED_COLUMNS[1:]].to_numpy(), contains="DataFrame")

    def test_nan_and_infinity(self):
        w = window()
        w.loc[3, "utilization"] = np.nan
        self.assertRejects(w, contains="NaN")
        w = window()
        w.loc[5, "wind_gust"] = np.inf
        self.assertRejects(w, contains="infinite")

    def test_invalid_port(self):
        self.assertRejects(window(), port="XXXXX", contains="unknown port_id")
        self.assertRejects(window(), port="sgsin", contains="unknown port_id")

    def test_ordering(self):
        self.assertRejects(window().iloc[::-1], contains="reverse time order")
        w = window()
        w.loc[6, "ts_hour"] = w.loc[6, "ts_hour"] + pd.Timedelta(hours=3)
        self.assertRejects(w, contains="consecutive")

    def test_already_scaled_input(self):
        w = window()
        w["dwell"] = np.linspace(-1.2, 1.5, 12)  # per-port z-scores
        self.assertRejects(w, contains="standardised")
        w = window()
        w["utilization"] = w["utilization"] / 100  # fractions
        self.assertRejects(w, contains="fraction")

    def test_out_of_range_and_inconsistent_values(self):
        w = window()
        w.loc[0, "labour_strike"] = 2
        self.assertRejects(w, contains="only 0 or 1")
        w = window()
        w.loc[0, "disruption_hours_elapsed"] = 4  # but disruption_active == 0
        self.assertRejects(w, contains="disruption_active == 0")
        w = window()
        w.loc[0, "vessels_at_anchor"] = 2.5
        self.assertRejects(w, contains="whole numbers")

    def test_no_silent_repair(self):
        w = window()
        w.loc[2, "dwell"] = np.nan
        with self.assertRaises(InputValidationError):
            forecast(w, "SGSIN", fallback="none")


class Fallback(unittest.TestCase):
    def test_ewma_function_matches_the_dataset_baseline(self):
        win = pd.read_csv(DATASET / "windows.csv")
        panel = pd.read_csv(DATASET / "hourly_panel.csv").set_index(["port_id", "hour_index"])
        w = win[win["keep"]].iloc[123]
        dwell = [panel.loc[(w["port_id"], h), "dwell"] for h in range(w["t_hour_index"] - 11, w["t_hour_index"] + 1)]
        self.assertAlmostEqual(ewma_forecast(dwell), w["baseline_ewma_alpha_0.2_hours"], places=3)

    def test_ewma_as_explicit_model_is_labelled_as_baseline(self):
        r = forecast(window(), "SGSIN", model="ewma")
        self.assertEqual((r.method, r.model_name), ("ewma_baseline", "ewma_alpha_0.2"))
        self.assertAlmostEqual(r.predicted_dwell_hours, ewma_forecast(window()["dwell"]))

    def test_fallback_only_when_enabled(self):
        w = window()
        w.loc[4, "wind_gust"] = np.nan  # neural input invalid, dwell history fine
        with self.assertRaises(InputValidationError):
            forecast(w, "SGSIN")
        r = forecast(w, "SGSIN", fallback="ewma")
        self.assertEqual(r.method, "ewma_fallback")
        self.assertTrue(any("not used" in m for m in r.warnings))
        self.assertAlmostEqual(r.predicted_dwell_hours, ewma_forecast(w["dwell"]))

    def test_fallback_fails_without_valid_dwell_history(self):
        w = window()
        w.loc[4, "dwell"] = np.nan
        with self.assertRaises(InputValidationError) as cm:
            forecast(w, "SGSIN", fallback="ewma")
        self.assertIn("EWMA fallback also unavailable", str(cm.exception))
        with self.assertRaises(InputValidationError):
            ewma_forecast([80.0] * 11)

    def test_unknown_model_never_substituted(self):
        with self.assertRaises(ModelLoadError):
            forecast(window(), "SGSIN", model="does_not_exist")
        r = forecast(window(), "SGSIN", model="does_not_exist", fallback="ewma")
        self.assertEqual(r.method, "ewma_fallback")  # only because fallback was requested explicitly


class MissingOrIncompatibleArtifacts(unittest.TestCase):
    def test_missing_checkpoint(self):
        with self.assertRaises(ModelLoadError):
            DwellForecaster(ARTIFACTS / "does_not_exist.pt", "missing")

    def test_missing_scalers(self):
        with tempfile.TemporaryDirectory() as d:
            shutil.copy(DATASET / "feature_schema.json", Path(d) / "feature_schema.json")
            with self.assertRaises(ModelLoadError) as cm:
                DwellForecaster(PROJECT_ROOT / load_registry()["models"]["gru"]["checkpoint"], "gru", dataset_dir=Path(d))
            self.assertIn("scalers.json", str(cm.exception))

    def test_incompatible_scalers(self):
        with tempfile.TemporaryDirectory() as d:
            shutil.copy(DATASET / "feature_schema.json", Path(d) / "feature_schema.json")
            sc = json.loads((DATASET / "scalers.json").read_text())
            sc["dwell"]["per_port"]["SGSIN"]["mean"] += 1.0
            (Path(d) / "scalers.json").write_text(json.dumps(sc))
            with self.assertRaises(ModelLoadError) as cm:
                DwellForecaster(PROJECT_ROOT / load_registry()["models"]["gru"]["checkpoint"], "gru", dataset_dir=Path(d))
            self.assertIn("incompatible", str(cm.exception))

    def test_checkpoint_not_matching_registry_hash(self):
        entry = load_registry()["models"]["gru"]
        with self.assertRaises(ModelLoadError):
            DwellForecaster(PROJECT_ROOT / entry["checkpoint"], "gru", expected_sha256="0" * 64)

    def test_missing_registry(self):
        with self.assertRaises(ModelLoadError):
            load_model("gru", registry_path=REGISTRY_PATH.parent / "nope.json")


if __name__ == "__main__":
    unittest.main()
