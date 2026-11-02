"""Tests for the GCN + LSTM graph experiment (graph construction, message passing, training, checkpoints).

Run from the project root:  .venv/bin/python -m unittest discover -s DL/Evaluation/tests
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cargomind_dl import config as data_config
from cargomind_dl import dataset as dl_dataset
from dl.models import GCNLSTMForecaster, GNNConfig, load_checkpoint, save_checkpoint
from dl.models.gnn import GraphConv, normalized_adjacency
from dl.training.gnn_trainer import GNN, build_model, masked_weighted_mse, predict_hours, rmse_hours, train
from dl.training.graph_data import CHANNELS, direct_route_adjacency, load_graph_split, port_order


def model_with(adjacency, seed=0):
    torch.manual_seed(seed)
    n = len(adjacency)
    return GCNLSTMForecaster(GNNConfig(adjacency=[list(map(int, r)) for r in adjacency], ports=[f"P{i}" for i in range(n)],
                                       dropout=0.0)).eval()


class GraphConstruction(unittest.TestCase):
    def test_node_ordering_matches_the_dataset(self):
        self.assertEqual(port_order(), sorted(pd.read_csv(data_config.PROCESSED / "port_metadata.csv")["port_id"]))
        self.assertEqual(CHANNELS[-1], "dwell_reconstructed")

    def test_adjacency_comes_from_direct_routes_only(self):
        ports = port_order()
        a = direct_route_adjacency()
        self.assertEqual(a.shape, (10, 10))
        self.assertTrue((a == a.T).all())
        self.assertEqual(int(np.trace(a)), 0)  # raw adjacency has no self-loops; they are added in normalisation
        net = pd.read_csv(data_config.PROCESSED / "port_network.csv")
        direct = {frozenset(p) for p in net.loc[net["route_type"] == "direct", ["origin_port_id", "destination_port_id"]].itertuples(index=False)}
        for i in range(10):
            for j in range(i + 1, 10):
                self.assertEqual(bool(a[i, j]), frozenset((ports[i], ports[j])) in direct, (ports[i], ports[j]))
        self.assertEqual(int(a.sum() // 2), 37)  # 37 of 45 possible undirected edges: a dense graph

    def test_normalisation_formula_and_self_loops(self):
        a = torch.tensor(direct_route_adjacency(), dtype=torch.float32)
        an = normalized_adjacency(a)
        a_hat = a + torch.eye(10)
        d = torch.diag(a_hat.sum(1).pow(-0.5))
        self.assertTrue(torch.allclose(an, d @ a_hat @ d, atol=1e-6))
        self.assertTrue((torch.diagonal(an) > 0).all())
        self.assertTrue(torch.allclose(an, an.T))
        with self.assertRaises(ValueError):
            normalized_adjacency(torch.tensor([[0.0, 1.0], [0.0, 0.0]]))  # directed input rejected


class MessagePassing(unittest.TestCase):
    def test_graph_conv_aggregates_neighbours(self):
        a = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.float32)
        conv = GraphConv(1, 1)
        with torch.no_grad():
            conv.linear.weight.fill_(1.0)
            conv.linear.bias.zero_()
        x = torch.tensor([[1.0], [3.0], [5.0]])
        out = conv(x, normalized_adjacency(a)).squeeze(-1)
        # nodes 0,1 have degree 2 (incl. self-loop): (1+3)/2 = 2 each; isolated node 2 keeps its own value
        self.assertTrue(torch.allclose(out, torch.tensor([2.0, 2.0, 5.0])))

    def test_connected_node_input_changes_neighbour_output_but_not_a_non_neighbour(self):
        ports = port_order()
        adj = direct_route_adjacency()
        src, nbr, far = ports.index("AEJEA"), ports.index("CNSHA"), ports.index("NLRTM")
        self.assertEqual((adj[src, nbr], adj[src, far]), (1, 0))  # CNSHA is a direct-route neighbour; NLRTM is not
        m = model_with(adj)
        x = torch.randn(2, 12, 10, 8, generator=torch.Generator().manual_seed(1))
        x2 = x.clone()
        x2[:, :, src, :] += 3.0  # change only AEJEA's inputs
        with torch.no_grad():
            d = (m(x2) - m(x)).abs()
        self.assertGreater(d[:, nbr].min().item(), 1e-6)  # neighbour's prediction moves: information flowed along the edge
        self.assertEqual(d[:, far].max().item(), 0.0)  # non-neighbour unchanged (one GCN layer = one-hop messages)

    def test_without_edges_ports_are_independent(self):
        m = model_with(np.zeros((10, 10), dtype=int))
        x = torch.randn(1, 12, 10, 8)
        x2 = x.clone()
        x2[:, :, 0, :] += 3.0
        with torch.no_grad():
            d = (m(x2) - m(x)).abs()[0]
        self.assertGreater(d[0].item(), 0.0)
        self.assertEqual(d[1:].max().item(), 0.0)


class ShapesAndGradients(unittest.TestCase):
    def setUp(self):
        self.m = GCNLSTMForecaster(GNNConfig(adjacency=direct_route_adjacency().tolist(), ports=port_order()))

    def test_input_output_dimensions(self):
        out = self.m(torch.randn(4, 12, 10, 8))
        self.assertEqual(tuple(out.shape), (4, 10))
        self.assertTrue(torch.isfinite(out).all())
        for bad in (torch.randn(4, 11, 10, 8), torch.randn(4, 12, 9, 8), torch.randn(4, 12, 10, 7), torch.randn(12, 10, 8)):
            with self.assertRaises(ValueError):
                self.m(bad)

    def test_gradients_reach_gcn_lstm_and_head(self):
        y, mask = torch.randn(4, 10), torch.ones(4, 10, dtype=torch.bool)
        masked_weighted_mse(self.m(torch.randn(4, 12, 10, 8)), y, mask, torch.ones(10)).backward()
        for name, p in self.m.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertGreater(p.grad.abs().sum().item(), 0.0, name)
        self.assertEqual({k for k, _ in self.m.named_buffers()}, {"adjacency", "a_norm"})  # graph is fixed, not learned

    def test_masked_loss_ignores_unkept_targets(self):
        pred, y = torch.zeros(1, 3), torch.tensor([[1.0, 100.0, 1.0]])
        mask = torch.tensor([[True, False, True]])
        self.assertAlmostEqual(masked_weighted_mse(pred, y, mask, torch.ones(3)).item(), 1.0)


class DataAdapter(unittest.TestCase):
    def test_graph_snapshots_reproduce_the_stored_windows(self):
        g = load_graph_split("val")
        d = dl_dataset.load_split("val")
        self.assertEqual(g.X.shape[1:], (12, 10, 8))
        self.assertEqual(int(g.mask.sum()), len(d["y"]))  # exactly the kept validation windows
        idx = {w: i for i, w in enumerate(d["window_id"])}
        s, n = np.nonzero(g.mask)
        rows = np.array([idx[w] for w in g.window_id[s, n]])
        np.testing.assert_array_equal(g.X[s, :, n, :], d["X"][rows])
        np.testing.assert_allclose(g.y_hours[s, n], d["y_hours"][rows], atol=1e-4)


class TrainingCheckpointInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tr, cls.va = load_graph_split("train"), load_graph_split("val")

    def test_training_is_reproducible_and_checkpoint_round_trips(self):
        m1, r1 = train(GNN, self.tr, self.va, max_epochs=2)
        m2, r2 = train(GNN, self.tr, self.va, max_epochs=2)
        self.assertEqual([h["val_rmse_h"] for h in r1["history"]], [h["val_rmse_h"] for h in r2["history"]])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "g.pt"
            save_checkpoint(path, m1, CHANNELS, {"note": "test"})
            loaded, meta = load_checkpoint(path)
        self.assertEqual(meta["model_type"], "gnn")
        self.assertIsInstance(loaded, GCNLSTMForecaster)
        self.assertTrue(torch.equal(loaded.adjacency, m1.adjacency))
        np.testing.assert_array_equal(predict_hours(m1, self.va), predict_hours(loaded, self.va))
        self.assertAlmostEqual(rmse_hours(loaded, self.va), r1["best_val_rmse_h"], places=4)

    def test_inference_on_a_sample_graph_snapshot(self):
        m = build_model(self.va)
        m.eval()
        with torch.no_grad():
            out = m(torch.from_numpy(self.va.X[:1]))
        self.assertEqual(tuple(out.shape), (1, 10))
        hours = predict_hours(m, self.va)[:1]
        self.assertTrue(np.isfinite(hours).all())


if __name__ == "__main__":
    unittest.main()
