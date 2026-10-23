"""Tests for two-dimensional cubed-sphere metric identities."""

import os
import unittest

import torch

from tests.unit.mpi_test import MpiTestCase
from wx_factory.common import Configuration, load_default_schema, readfile
from wx_factory.simulation import Simulation

CONFIG = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "restart", "shallow_water.ini")


class Metric2DIdentityTestCase(MpiTestCase):
    def setUp(self):
        super().setUp()
        config = Configuration(readfile(CONFIG), load_default_schema())
        self.sim = Simulation(config)
        self.geom = self.sim.geometry
        self.metric = self.sim.rhs.full.metric

    def test_volume_factor_matches_the_covariant_determinant(self):
        """Check sqrt(g) against the determinant of the covariant metric."""
        for fields in (self.metric.volume, *self.metric.faces):
            g = fields.covariant
            determinant = g[0, 0] * g[1, 1] - g[0, 1] * g[1, 0]
            expected = torch.sqrt(determinant)
            torch.testing.assert_close(fields.volume_jacobian, expected, rtol=1e-13, atol=0)

    def test_metrics_are_inverses(self):
        """The covariant and contravariant forms must invert each other after rescaling."""
        for fields in (self.metric.volume, *self.metric.faces):
            product = torch.einsum("ij...,jk...->ik...", fields.covariant, fields.contravariant)
            identity = torch.eye(2, dtype=product.dtype).reshape((2, 2) + (1,) * (product.ndim - 2))
            torch.testing.assert_close(product, identity.expand_as(product), rtol=1e-12, atol=1e-12)

    def test_global_integral_recovers_the_area_of_the_sphere(self):
        """Integrating one over the cubed sphere recovers its area."""
        from wx_factory.output.diagnostic import global_integral_2d

        ones = torch.ones_like(self.metric.sqrtG)
        area = global_integral_2d(ones, self.sim.operators_real, self.metric, self.geom.num_solpts, comm=self.comm)

        expected = 4.0 * torch.pi * self.geom.earth_radius**2
        # The gnomonic area element is not a polynomial, so quadrature is not exact.
        self.assertLess(abs(area - expected) / expected, 1.0e-6)


if __name__ == "__main__":
    unittest.main()
