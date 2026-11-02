"""Tests for the production DL configuration: Transformer T_w primary, C_w LSTM automatic fallback, output contract.

Failures are simulated with temporary registries and patched model outputs; no checkpoint or dataset file is modified.

Run from the project root:  .venv/bin/python -m unittest discover -s DL/Evaluation/tests
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

from cargomind_dl.config import DATASET
from dl.inference import (CONTRACT_VERSION, DEFAULT_MODEL, FALLBACK_MODEL, InputValidationError, ModelLoadError, PredictionError,
                          forecast, load_model, load_registry)
from dl.inference.registry import PROJECT_ROOT, REGISTRY_PATH
from dl.inference.windows import actual_dwell, coverage, window_from_dataset
from dl.models import RecurrentForecaster, TransformerForecaster

from test_inference import window  # discover puts this folder on sys.path

CONTRACT_FIELDS = {"predicted_dwell_hours", "method", "model_name", "model_version", "port_id", "window_end_utc", "target_hour_utc",
                   "warnings", "note", "model_role", "requested_model", "fallback_used", "fallback_reason", "forecast_horizon_hours",
                   "window_start_utc", "window_hours", "generated_at_utc", "contract_version"}


def registry_with(**changes):
    """A temporary registry file: changes = {model_name: {field: value}}."""
    reg = load_registry()
    for name, fields in changes.items():
        reg["models"][name].update(fields)
    d = tempfile.mkdtemp()
    path = Path(d) / "model_registry.json"
    path.write_text(json.dumps(reg))
    return path


def dataset_window(port="SGSIN", hour=811):
    """A real validation-period window from the hourly panel (read-only)."""
    panel = pd.read_csv(DATASET / "hourly_panel.csv")
    p = panel[panel["port_id"] == port]
    rows = p[p["hour_index"].between(hour - 11, hour)]
    cols = ["ts_hour", "dwell", "utilization", "labour_strike", "wind_gust", "disruption_active", "disruption_hours_elapsed",
            "vessels_at_anchor"]
    return rows[cols].assign(dwell_reconstructed=(rows["prov_dwell"] == 2).astype(int).values).reset_index(drop=True)


class ProductionRegistry(unittest.TestCase):
    def test_primary_fallback_and_roles(self):
        reg = load_registry()
        self.assertEqual((reg["default_model"], reg["fallback_model"]), ("transformer_t_w", "c_w_lstm"))
        self.assertEqual((DEFAULT_MODEL, FALLBACK_MODEL), ("transformer_t_w", "c_w_lstm"))
        roles = {k: v["role"] for k, v in reg["models"].items()}
        self.assertEqual(roles.pop("transformer_t_w"), "primary")
        self.assertEqual(roles.pop("c_w_lstm"), "fallback")
        self.assertTrue(all(r == "research" for r in roles.values()))

    def test_gnn_and_ewma_are_not_registered(self):
        reg = load_registry()
        self.assertFalse(any("gnn" in k.lower() or "gcn" in v["experiment"].lower() for k, v in reg["models"].items()))
        self.assertNotIn("ewma", reg["models"])

    def test_selected_checkpoints_are_the_documented_ones(self):
        reg = load_registry()["models"]
        self.assertEqual(reg["transformer_t_w"]["checkpoint"], "DL/DL/artifacts/transformer/runs/T_w_transformer_hour_weighted/seed_1.pt")
        self.assertEqual(reg["c_w_lstm"]["checkpoint"], "DL/DL/artifacts/runs/C_w_lstm_ewma_residual_hour_weighted/seed_2.pt")
        self.assertEqual(reg["c_w_lstm"]["seed"], 2)


class PrimaryAndFallbackModelsLoad(unittest.TestCase):
    def test_transformer_loads_and_forecasts_a_real_window(self):
        f = load_model("transformer_t_w")
        self.assertIsInstance(f.model, TransformerForecaster)
        r = forecast(dataset_window(), "SGSIN")
        self.assertEqual((r.model_name, r.method, r.model_role, r.fallback_used), ("transformer_t_w", "neural", "primary", False))
        self.assertIsNone(r.fallback_reason)
        self.assertAlmostEqual(r.predicted_dwell_hours, f.predict(dataset_window(), "SGSIN").predicted_dwell_hours, places=9)
        self.assertTrue(40 < r.predicted_dwell_hours < 250)

    def test_selected_lstm_loads_with_verified_hash(self):
        f = load_model("c_w_lstm")
        self.assertIsInstance(f.model, RecurrentForecaster)
        self.assertEqual(f.model.cfg.cell, "lstm")
        self.assertEqual(f.model.cfg.residual_ewma_alpha, 0.2)
        self.assertEqual(f.role, "fallback")
        self.assertIn(load_registry()["models"]["c_w_lstm"]["checkpoint_sha256"][:12], f.version)
        self.assertEqual(f.channels, load_model("transformer_t_w").channels)  # same feature schema as the primary


class OutputContract(unittest.TestCase):
    def test_all_contract_fields_present_and_typed(self):
        d = forecast(window(), "SGSIN").to_dict()
        self.assertEqual(set(d), CONTRACT_FIELDS)
        self.assertEqual(d["contract_version"], CONTRACT_VERSION)
        self.assertEqual((d["forecast_horizon_hours"], d["window_hours"]), (1, 12))
        self.assertEqual(d["window_start_utc"], "2025-03-01T00:00:00+00:00")
        self.assertEqual(d["window_end_utc"], "2025-03-01T11:00:00+00:00")
        self.assertEqual(d["target_hour_utc"], "2025-03-01T12:00:00+00:00")
        self.assertEqual(d["requested_model"], "transformer_t_w")
        self.assertTrue(pd.Timestamp(d["generated_at_utc"]).tzinfo is not None)
        json.dumps(d)  # serialisable for the frontend

    def test_ewma_benchmark_is_labelled_as_baseline(self):
        r = forecast(window(), "SGSIN", model="ewma")
        self.assertEqual((r.method, r.model_role, r.fallback_used), ("ewma_baseline", "baseline", False))


class AutomaticFallback(unittest.TestCase):
    def assertFellBack(self, r, reason_contains):
        self.assertEqual((r.model_name, r.method, r.model_role), ("c_w_lstm", "neural_fallback", "fallback"))
        self.assertEqual(r.requested_model, "transformer_t_w")
        self.assertTrue(r.fallback_used)
        self.assertIn(reason_contains, r.fallback_reason)
        self.assertTrue(r.warnings[0].startswith("fallback model 'c_w_lstm' used"))
        self.assertIn("C_w_lstm_ewma_residual_hour_weighted/seed_2.pt", r.model_version)
        direct = load_model("c_w_lstm").predict(window(), "SGSIN").predicted_dwell_hours
        self.assertAlmostEqual(r.predicted_dwell_hours, direct, places=9)

    def test_altered_transformer_checkpoint_triggers_fallback(self):
        path = registry_with(transformer_t_w={"checkpoint_sha256": "0" * 64})
        self.assertFellBack(forecast(window(), "SGSIN", registry_path=path), "does not match the registry")

    def test_missing_transformer_checkpoint_triggers_fallback(self):
        path = registry_with(transformer_t_w={"checkpoint": "DL/DL/artifacts/transformer/runs/missing/seed_9.pt"})
        self.assertFellBack(forecast(window(), "SGSIN", registry_path=path), "not found")

    def test_non_finite_transformer_output_triggers_fallback(self):
        t = load_model("transformer_t_w")
        with mock.patch.object(t.model, "forward", lambda x, port: torch.full((x.shape[0],), float("nan"))):
            self.assertFellBack(forecast(window(), "SGSIN"), "non-finite")

    def test_implausible_transformer_output_triggers_fallback(self):
        t = load_model("transformer_t_w")
        with mock.patch.object(t.model, "forward", lambda x, port: torch.full((x.shape[0],), 1e6)):
            self.assertFellBack(forecast(window(), "SGSIN"), "implausible")

    def test_runtime_error_in_transformer_triggers_fallback(self):
        t = load_model("transformer_t_w")

        def boom(x, port):
            raise RuntimeError("simulated tensor failure")
        with mock.patch.object(t.model, "forward", boom):
            self.assertFellBack(forecast(window(), "SGSIN"), "simulated tensor failure")

    def test_fallback_disabled_raises(self):
        path = registry_with(transformer_t_w={"checkpoint_sha256": "0" * 64})
        with self.assertRaises(ModelLoadError):
            forecast(window(), "SGSIN", fallback="none", registry_path=path)
        t = load_model("transformer_t_w")
        with mock.patch.object(t.model, "forward", lambda x, port: torch.full((x.shape[0],), float("nan"))):
            with self.assertRaises(PredictionError):
                forecast(window(), "SGSIN", fallback="none")

    def test_both_models_failing_raises_naming_both(self):
        path = registry_with(transformer_t_w={"checkpoint_sha256": "0" * 64}, c_w_lstm={"checkpoint_sha256": "1" * 64})
        with self.assertRaises(ModelLoadError) as cm:
            forecast(window(), "SGSIN", registry_path=path)
        self.assertIn("'transformer_t_w' failed", str(cm.exception))
        self.assertIn("'c_w_lstm' also failed", str(cm.exception))

    def test_invalid_input_is_rejected_not_redirected(self):
        for bad in (window().iloc[:11], window().drop(columns=["utilization"]), window().iloc[::-1]):
            with self.assertRaises(InputValidationError):
                forecast(bad, "SGSIN")
        w = window()
        w.loc[3, "wind_gust"] = np.nan
        with self.assertRaises(InputValidationError):
            forecast(w, "SGSIN")

    def test_research_models_get_no_automatic_fallback(self):
        path = registry_with(gru={"checkpoint_sha256": "0" * 64})
        with self.assertRaises(ModelLoadError):
            forecast(window(), "SGSIN", model="gru", registry_path=path)
        r = forecast(window(), "SGSIN", model="gru", fallback="lstm", registry_path=path)
        self.assertEqual((r.model_name, r.requested_model, r.fallback_used), ("c_w_lstm", "gru", True))

    def test_legacy_ewma_fallback_still_available(self):
        path = registry_with(transformer_t_w={"checkpoint_sha256": "0" * 64})
        r = forecast(window(), "SGSIN", fallback="ewma", registry_path=path)
        self.assertEqual((r.method, r.model_name, r.fallback_used), ("ewma_fallback", "ewma_alpha_0.2", True))

    def test_registry_file_is_not_modified_by_inference(self):
        before = REGISTRY_PATH.read_bytes()
        forecast(window(), "SGSIN")
        self.assertEqual(REGISTRY_PATH.read_bytes(), before)
        self.assertTrue((PROJECT_ROOT / load_registry()["models"]["c_w_lstm"]["checkpoint"]).exists())



class DatasetWindows(unittest.TestCase):
    def test_window_from_dataset_matches_the_panel_window(self):
        w = window_from_dataset("SGSIN", "2025-02-03 19:00:00+00:00")
        ref = dataset_window("SGSIN", 811)
        self.assertEqual(list(w.columns), list(ref.columns))
        pd.testing.assert_frame_equal(w.drop(columns="ts_hour"), ref.drop(columns="ts_hour"), check_dtype=False)
        self.assertEqual(pd.Timestamp(w["ts_hour"].iloc[-1]), pd.Timestamp("2025-02-03 19:00:00+00:00"))
        r = forecast(w, "SGSIN")
        self.assertEqual(r.window_end_utc, "2025-02-03T19:00:00+00:00")
        self.assertIsInstance(actual_dwell("SGSIN", r.target_hour_utc), float)

    def test_window_outside_coverage_or_unknown_port_is_rejected(self):
        first, last = coverage()
        with self.assertRaises(InputValidationError):
            window_from_dataset("SGSIN", first + pd.Timedelta(hours=5))  # needs 11 earlier hours
        with self.assertRaises(InputValidationError):
            window_from_dataset("SGSIN", last + pd.Timedelta(hours=1))
        with self.assertRaises(InputValidationError):
            window_from_dataset("XXXXX", last)
        self.assertIsNone(actual_dwell("SGSIN", last + pd.Timedelta(hours=1)))


if __name__ == "__main__":
    unittest.main()
