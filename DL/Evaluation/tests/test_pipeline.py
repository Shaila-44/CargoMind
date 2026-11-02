"""Unit tests for the Stage 02 model, data adapter, loss, checkpoints and reproducibility.

Run from the project root:  .venv/bin/python -m unittest discover -s DL/Evaluation/tests
"""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from dl.models import ModelConfig, RecurrentForecaster, load_checkpoint, save_checkpoint
from dl.training import data
from dl.training.config import A, B
from dl.training.trainer import mse_loss, predict, restore, train


class ModelShapes(unittest.TestCase):
    def test_forward_shape_lstm_and_gru(self):
        for cell in ("lstm", "gru"):
            m = RecurrentForecaster(ModelConfig(input_size=8, cell=cell))
            out = m(torch.randn(5, 12, 8))
            self.assertEqual(tuple(out.shape), (5,))

    def test_wrong_channel_count_raises(self):
        m = RecurrentForecaster(ModelConfig(input_size=8))
        with self.assertRaises(ValueError):
            m(torch.randn(2, 12, 7))

    def test_port_embedding_behaviour(self):
        torch.manual_seed(0)
        m = RecurrentForecaster(ModelConfig(input_size=8, port_embedding_dim=4, dropout=0.0)).eval()
        x = torch.randn(1, 12, 8).repeat(2, 1, 1)
        out = m(x, torch.tensor([0, 7]))
        self.assertNotAlmostEqual(out[0].item(), out[1].item(), places=6)  # same input, different port -> different output
        with self.assertRaises(ValueError):
            m(x)  # embedding model requires port_index
        plain = RecurrentForecaster(ModelConfig(input_size=8, dropout=0.0)).eval()
        o = plain(x, torch.tensor([0, 7]))
        self.assertAlmostEqual(o[0].item(), o[1].item(), places=6)  # no embedding -> port ignored


class ResidualAndWeighting(unittest.TestCase):
    def test_untrained_residual_model_reproduces_ewma_baseline_in_hours(self):
        from dl.evaluation import frames

        va = data.load("val")
        m = RecurrentForecaster(ModelConfig(input_size=8, residual_ewma_alpha=0.2)).eval()
        hours = data.to_hours(predict(m, va.X, va.port_index), va.port_index)
        stored = frames.context(va.window_id, va.y_hours)["baseline_ewma_alpha_0.2_hours"].values
        self.assertLess(np.abs(hours - stored).max(), 1e-3)

    def test_hour_weighted_loss_equals_mse_in_hours(self):
        from dl.training.trainer import sample_weights

        va = data.load("val")
        rng = np.random.default_rng(0)
        pred = va.y + rng.normal(0, 0.3, len(va.y)).astype(np.float32)
        w = sample_weights(replace(A, loss_weighting="hours"), va)
        weighted = mse_loss(torch.from_numpy(pred), torch.from_numpy(va.y), torch.from_numpy(w)).item()
        hours = np.mean((data.to_hours(pred, va.port_index) - va.y_hours) ** 2)
        sd = data.dwell_std_by_port_index()[va.port_index]
        self.assertAlmostEqual(weighted * np.mean(sd ** 2), hours, delta=1e-3 * hours)


class Loss(unittest.TestCase):
    def test_mse_matches_manual_and_rejects_broadcasting(self):
        p, t = torch.tensor([1.0, 2.0, 4.0]), torch.tensor([1.0, 0.0, 1.0])
        self.assertAlmostEqual(mse_loss(p, t).item(), (0 + 4 + 9) / 3, places=6)
        with self.assertRaises(ValueError):
            mse_loss(p.unsqueeze(-1), t)  # [3,1] vs [3] would silently broadcast to [3,3]


class DataAdapter(unittest.TestCase):
    def test_default_input_has_8_channels_with_reconstruction_flag(self):
        va = data.load("val")
        self.assertEqual(va.X.shape[1:], (12, 8))
        self.assertEqual(va.channels[-1], "dwell_reconstructed")
        self.assertEqual(va.y.shape, (len(va.X),))

    def test_inverse_scaling_matches_stored_hours(self):
        va = data.load("val")
        self.assertLess(np.abs(data.to_hours(va.y, va.port_index) - va.y_hours).max(), 1e-3)

    def test_feature_drop_by_name(self):
        va = data.load("val", mask_channels=(), drop_features=("labour_strike",))
        self.assertNotIn("labour_strike", va.channels)
        self.assertNotIn("dwell_reconstructed", va.channels)
        self.assertEqual(va.X.shape[-1], 6)
        with self.assertRaises(ValueError):
            data.load("val", drop_features=("not_a_channel",))


class CheckpointsAndReproducibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tr, cls.va = data.load("train"), data.load("val")

    def test_checkpoint_round_trip(self):
        res = train(B, self.tr, self.va, max_epochs=2)
        model = restore(res)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.pt"
            save_checkpoint(path, model, res.channels, {"note": "test"})
            loaded, meta = load_checkpoint(path)
        self.assertEqual(meta["input_channels"], res.channels)
        np.testing.assert_array_equal(predict(model, self.va.X, self.va.port_index), predict(loaded, self.va.X, self.va.port_index))

    def test_same_seed_same_result_different_seed_differs(self):
        r1 = train(A, self.tr, self.va, max_epochs=2)
        r2 = train(A, self.tr, self.va, max_epochs=2)
        r3 = train(replace(A, seed=1), self.tr, self.va, max_epochs=2)
        self.assertEqual([h["val_rmse_h"] for h in r1.history], [h["val_rmse_h"] for h in r2.history])
        for k in r1.best_state:
            self.assertTrue(torch.equal(r1.best_state[k], r2.best_state[k]))
        self.assertNotEqual(r1.history[-1]["val_rmse_h"], r3.history[-1]["val_rmse_h"])

    def test_best_checkpoint_reproduces_logged_validation_rmse(self):
        res = train(A, self.tr, self.va, max_epochs=3)
        yh = data.to_hours(predict(restore(res), self.va.X, self.va.port_index), self.va.port_index)
        self.assertAlmostEqual(float(np.sqrt(np.mean((yh - self.va.y_hours) ** 2))), res.best_val_rmse_h, places=4)


if __name__ == "__main__":
    unittest.main()
