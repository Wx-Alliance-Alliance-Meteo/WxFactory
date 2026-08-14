"""Conservation tests for the metric-weighted exponential filter."""

import os
import types
import unittest

import torch
from wx_test import WxTestCase

from wx_factory.common import Configuration, load_default_schema, readfile
from wx_factory.simulation import Simulation

FILTER = {"strength": 8.0, "order": 4, "cutoff": 0.25}


class ExponentialFilterTestCase(WxTestCase):
    def setUp(self) -> None:
        super().setUp()
        here = os.path.dirname(os.path.realpath(__file__))
        config = Configuration(readfile(os.path.join(here, "filter_config.ini")), load_default_schema())
        self.sim = Simulation(config)
        self.geom = self.sim.geometry
        self.ops = self.sim.rhs.full.ops
        self.num_solpts = self.geom.num_solpts
        self.filter_matrix = self.ops.make_filter_3d(geom=self.geom, **FILTER)

    def quadrature_weights(self):
        """Return tensor-product element quadrature weights."""
        w = self.geom.glweights
        return torch.einsum("i,j,k->ijk", w, w, w).reshape(-1)

    def rough_state(self):
        """Return a positive state that excites all filter modes."""
        gen = torch.Generator().manual_seed(11)
        return 1.0 + torch.rand(self.sim.Q.shape, generator=gen, dtype=torch.float64)

    def varying_metric(self, shape):
        """Return a strongly varying synthetic metric."""
        gen = torch.Generator().manual_seed(7)
        sqrtG = 1.0 + 3.0 * torch.rand(shape, generator=gen, dtype=torch.float64)
        return types.SimpleNamespace(sqrtG_new=sqrtG, inv_sqrtG_new=1.0 / sqrtG)

    def test_quadrature_weights_are_double_precision(self):
        w = self.geom.glweights
        self.assertEqual(w.dtype, torch.float64)
        self.assertAlmostEqual(w.sum().item(), 2.0, delta=1.0e-15)

    def test_lowest_mode_is_unfiltered(self):
        constant = torch.ones((4, self.num_solpts**3), dtype=torch.float64)
        filtered = constant @ self.filter_matrix
        torch.testing.assert_close(filtered, constant, rtol=0.0, atol=1.0e-14)

    def test_high_modes_are_attenuated(self):
        identity = torch.eye(self.num_solpts**3, dtype=torch.float64)
        self.assertGreater((self.filter_matrix - identity).abs().max().item(), 1.0e-3)

    def test_preserves_metric_weighted_integral(self):
        state = self.rough_state()
        metric = self.varying_metric(state.shape[1:])
        weights = self.quadrature_weights()

        before = ((metric.sqrtG_new * state) * weights).sum(dim=-1)
        filtered = self.ops.apply_filter_3d(state, metric, self.filter_matrix)
        after = ((metric.sqrtG_new * filtered) * weights).sum(dim=-1)

        error = (after - before).abs().max().item() / before.abs().max().item()
        self.assertLess(error, 1.0e-13, f"filter changed the metric-weighted integral by {error:.3e}")

    def test_unweighted_filtering_would_not_conserve(self):
        state = self.rough_state()
        metric = self.varying_metric(state.shape[1:])
        weights = self.quadrature_weights()

        before = ((metric.sqrtG_new * state) * weights).sum(dim=-1)
        naive = state @ self.filter_matrix  # Filter the unweighted state.
        after = ((metric.sqrtG_new * naive) * weights).sum(dim=-1)

        error = (after - before).abs().max().item() / before.abs().max().item()
        self.assertGreater(error, 1.0e-6, "sqrt(g) does not vary enough for this test to mean anything")


if __name__ == "__main__":
    unittest.main()
