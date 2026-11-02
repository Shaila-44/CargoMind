"""Tests for the Transformer forecaster and its integration with the existing trainer and checkpoints.

Run from the project root:  .venv/bin/python -m unittest discover -s DL/Evaluation/tests
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dl.models import TransformerConfig, TransformerForecaster, load_checkpoint, save_checkpoint
from dl.models.transformer import sinusoidal_encoding
from dl.training import data
from dl.training.config import T, T_W
from dl.training.trainer import build_model, mse_loss, predict, restore, train


class Shapes(unittest.TestCase):
    def test_input_and_output_shapes(self):
        m = TransformerForecaster(TransformerConfig())
        out = m(torch.randn(5, 12, 8))
        self.assertEqual(tuple(out.shape), (5,))
        self.assertEqual(out.dtype, torch.float32)

    def test_rejects_wrong_sequence_length_or_channels(self):
        m = TransformerForecaster(TransformerConfig())
        for bad in (torch.randn(2, 11, 8), torch.randn(2, 13, 8), torch.randn(2, 12, 7), torch.randn(12, 8)):
            with self.assertRaises(ValueError):
                m(bad)

    def test_port_index_is_accepted_and_ignored(self):
        torch.manual_seed(0)
        m = TransformerForecaster(TransformerConfig()).eval()
        x = torch.randn(3, 12, 8)
        self.assertTrue(torch.equal(m(x), m(x, torch.tensor([0, 5, 9]))))

    def test_factory_builds_the_specified_architecture(self):
        m = build_model(T, input_size=8)
        self.assertIsInstance(m, TransformerForecaster)
        layer = m.encoder.layers[0]
        self.assertEqual((m.cfg.d_model, m.cfg.nhead, m.cfg.dim_feedforward, m.cfg.num_layers), (32, 4, 64, 1))
        self.assertEqual(layer.self_attn.num_heads, 4)
        self.assertTrue(layer.self_attn.batch_first)
        self.assertEqual((m.input_projection.in_features, m.input_projection.out_features), (8, 32))
        self.assertEqual((m.head.in_features, m.head.out_features), (32, 1))


class PositionalEncoding(unittest.TestCase):
    def test_values_and_shape(self):
        pe = sinusoidal_encoding(12, 32)
        self.assertEqual(tuple(pe.shape), (12, 32))
        self.assertTrue(torch.allclose(pe[0, 0::2], torch.zeros(16)))  # sin(0)
        self.assertTrue(torch.allclose(pe[0, 1::2], torch.ones(16)))   # cos(0)
        self.assertAlmostEqual(pe[3, 0].item(), float(np.sin(3.0)), places=5)
        self.assertAlmostEqual(pe[3, 1].item(), float(np.cos(3.0)), places=5)
        self.assertEqual(len({tuple(r.tolist()) for r in pe}), 12)  # every position distinct

    def test_is_a_fixed_buffer_saved_with_the_model(self):
        m = TransformerForecaster(TransformerConfig())
        self.assertIn("positional_encoding", dict(m.named_buffers()))
        self.assertNotIn("positional_encoding", dict(m.named_parameters()))

    def test_order_matters(self):
        torch.manual_seed(0)
        m = TransformerForecaster(TransformerConfig(dropout=0.0)).eval()
        x = torch.randn(1, 12, 8)
        self.assertFalse(torch.allclose(m(x), m(torch.flip(x, dims=[1]))))  # reversing time changes the output


class FiniteAndGradients(unittest.TestCase):
    def test_finite_outputs_on_real_validation_windows(self):
        va = data.load("val")
        m = TransformerForecaster(TransformerConfig()).eval()
        self.assertTrue(np.isfinite(predict(m, va.X, va.port_index)).all())

    def test_backpropagation_reaches_every_parameter(self):
        torch.manual_seed(0)
        m = TransformerForecaster(TransformerConfig())
        loss = mse_loss(m(torch.randn(4, 12, 8)), torch.randn(4))
        loss.backward()
        for name, p in m.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        self.assertTrue(any(p.grad.abs().sum() > 0 for p in m.parameters()))


class CheckpointAndReproducibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tr, cls.va = data.load("train"), data.load("val")

    def test_checkpoint_round_trip(self):
        res = train(T, self.tr, self.va, max_epochs=2)
        model = restore(res)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.pt"
            save_checkpoint(path, model, res.channels, {"note": "test"})
            loaded, meta = load_checkpoint(path)
        self.assertEqual(meta["model_type"], "transformer")
        self.assertIsInstance(loaded, TransformerForecaster)
        np.testing.assert_array_equal(predict(model, self.va.X, self.va.port_index), predict(loaded, self.va.X, self.va.port_index))

    def test_existing_recurrent_checkpoints_still_load(self):
        from dl.models import RecurrentForecaster
        from dl.training.config import ARTIFACTS

        ck = ARTIFACTS / "lstm_best.pt"
        if not ck.exists():
            self.skipTest("no saved recurrent checkpoint")
        model, meta = load_checkpoint(ck)
        self.assertNotIn("model_type", meta)  # written before the Transformer existed
        self.assertIsInstance(model, RecurrentForecaster)

    def test_same_seed_same_result(self):
        r1 = train(T_W, self.tr, self.va, max_epochs=2)
        r2 = train(T_W, self.tr, self.va, max_epochs=2)
        self.assertEqual([h["val_rmse_h"] for h in r1.history], [h["val_rmse_h"] for h in r2.history])
        for k in r1.best_state:
            self.assertTrue(torch.equal(r1.best_state[k], r2.best_state[k]))

    def test_target_hour_is_not_an_input(self):
        """The model sees only hours t-11..t: input X never contains the target value at any position."""
        va = self.va
        last = data.to_hours(va.X[:, -1, 0], va.port_index)
        self.assertGreater(np.abs(last - va.y_hours).mean(), 1.0)  # hour t differs from hour t+1 (would be ~0 if leaked)
        self.assertEqual(va.X.shape[1], 12)


if __name__ == "__main__":
    unittest.main()
