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
        determinant = self.metric.H_cov_11 * self.metric.H_cov_22 - self.metric.H_cov_12 * self.metric.H_cov_21
        expected = torch.sqrt(determinant)

        relative = float((self.metric.sqrtG - expected).abs().max() / expected.abs().max())
        self.assertLess(relative, 1.0e-13)

    def test_metrics_are_inverses(self):
        """The covariant and contravariant forms must invert each other after rescaling."""
        a = self.metric.H_cov_11 * self.metric.H_contra_11 + self.metric.H_cov_12 * self.metric.H_contra_21
        b = self.metric.H_cov_11 * self.metric.H_contra_12 + self.metric.H_cov_12 * self.metric.H_contra_22
        c = self.metric.H_cov_21 * self.metric.H_contra_11 + self.metric.H_cov_22 * self.metric.H_contra_21
        d = self.metric.H_cov_21 * self.metric.H_contra_12 + self.metric.H_cov_22 * self.metric.H_contra_22

        self.assertLess(float((a - 1.0).abs().max()), 1.0e-12)
        self.assertLess(float(b.abs().max()), 1.0e-12)
        self.assertLess(float(c.abs().max()), 1.0e-12)
        self.assertLess(float((d - 1.0).abs().max()), 1.0e-12)

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
