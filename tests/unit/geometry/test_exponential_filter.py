import os
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
        w = self.geom.glweights
        return torch.einsum("i,j,k->ijk", w, w, w).reshape(-1)

    def rough_state(self):
        """Return a positive state that excites all filter modes."""
        gen = torch.Generator().manual_seed(11)
        return 1.0 + torch.rand(self.sim.Q.shape, generator=gen, dtype=torch.float64)

    def varying_metric(self, shape):
        gen = torch.Generator().manual_seed(7)
        sqrtG = 1.0 + 3.0 * torch.rand(shape, generator=gen, dtype=torch.float64)
        from copy import copy

        metric = copy(self.sim.initial_state.metric)
        metric.volume = copy(metric.volume)
        metric.volume.volume_jacobian = sqrtG
        return metric

    def test_constant_and_highest_legendre_mode(self):
        constant = torch.ones((4, self.num_solpts**3), dtype=torch.float64)
        filtered = constant @ self.filter_matrix
        torch.testing.assert_close(filtered, constant, rtol=0.0, atol=1.0e-14)

        from numpy.polynomial.legendre import legval

        coefficients = [0] * (self.num_solpts - 1) + [1]
        mode = torch.from_numpy(legval(self.geom.solutionPoints.cpu().numpy(), coefficients))
        for axis in range(3):
            shape = [1, 1, 1]
            shape[axis] = self.num_solpts
            field = mode.reshape(shape).expand([self.num_solpts] * 3).flatten()
            attenuation = torch.exp(torch.tensor(-FILTER["strength"], dtype=torch.float64))
            torch.testing.assert_close(field @ self.filter_matrix, attenuation * field, rtol=1e-10, atol=1e-14)

    def test_preserves_metric_weighted_integral(self):
        state = self.rough_state()
        metric = self.varying_metric(state.shape[1:])
        weights = self.quadrature_weights()

        before = ((metric.volume.volume_jacobian * state) * weights).sum(dim=-1)
        naive = state @ self.filter_matrix
        naive_integral = ((metric.volume.volume_jacobian * naive) * weights).sum(dim=-1)
        self.assertGreater((naive_integral[[0, 4]] - before[[0, 4]]).abs().max().item(), 1e-6)
        filtered = self.ops.apply_filter_3d(state, metric, self.filter_matrix)
        after = ((metric.volume.volume_jacobian * filtered) * weights).sum(dim=-1)

        error = (after[[0, 4]] - before[[0, 4]]).abs().max().item() / before[[0, 4]].abs().max().item()
        self.assertLess(error, 1.0e-13, f"filter changed the metric-weighted integral by {error:.3e}")


if __name__ == "__main__":
    unittest.main()
