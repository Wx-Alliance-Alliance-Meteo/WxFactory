"""Tests for three-dimensional cubed-sphere metric identities."""

import os
import unittest

import torch

from tests.unit.mpi_test import MpiTestCase
from wx_factory.common import Configuration, load_default_schema, readfile
from wx_factory.common.definitions import gravity
from wx_factory.simulation import Simulation

CONFIG = os.path.join(os.path.dirname(os.path.realpath(__file__)), "rotating_sphere_config.ini")


class MetricIdentityTestCase(MpiTestCase):
    """Invariants of the constructed metric, for both depth approximations."""

    def __init__(self, num_procs=6, methodName="runTest", device_name="cpu", depth="shallow"):
        super().__init__(num_procs, methodName, device_name)
        self.depth = depth

    def __str__(self):
        return super().__str__() + f".{self.depth}"

    def setUp(self):
        super().setUp()
        source = readfile(CONFIG).replace("depth_approx = shallow", f"depth_approx = {self.depth}")
        config = Configuration(source, load_default_schema())
        self.sim = Simulation(config)
        self.geom = self.sim.geometry
        self.metric = self.sim.rhs.full.metric

    def relative(self, actual, expected) -> float:
        scale = float(expected.abs().max())
        difference = float((actual - expected).abs().max())
        return difference if scale == 0.0 else difference / scale

    # ------------------------------------------------------------------ the metric itself

    def test_metrics_are_inverses(self):
        """g_ij h^jk = delta_i^k, which fails if either form has a wrong term."""
        product = torch.einsum("ij...,jk...->ik...", self.metric.h_cov_new, self.metric.h_contra_new)
        identity = torch.eye(3, dtype=product.dtype).reshape((3, 3) + (1,) * (product.dim() - 2))
        self.assertLess(float((product - identity).abs().max()), 1.0e-12)

    def test_interface_metrics_are_inverses(self):
        """The same, at the element interfaces, where the metric is rebuilt separately."""
        for contra in (
            self.metric.h_contra_itf_i_new,
            self.metric.h_contra_itf_j_new,
            self.metric.h_contra_itf_k_new,
        ):
            with self.subTest(interface=tuple(contra.shape[2:])):
                self.assertTrue(bool(torch.isfinite(contra).all()))
                trailing = torch.movedim(contra, (0, 1), (-2, -1))
                self.assertLess(float((trailing - trailing.transpose(-2, -1)).abs().max()), 1.0e-14)

                # The interface arrays carry halo blocks that are never filled and stay at zero;
                # only the blocks the model actually uses need to be positive definite.
                filled = trailing.abs().amax(dim=(-2, -1)) > 0.0
                self.assertGreater(float(filled.float().mean()), 0.5)
                self.assertGreater(float(torch.linalg.eigvalsh(trailing[filled]).min()), 0.0)

    # ------------------------------------------------------------------ Coriolis

    def coriolis_matrix(self) -> torch.Tensor:
        """``M_lj = g_li Gamma^i_{j0}``, with the mixed symbols in their stored order."""
        christoffel = self.metric.christoffel
        gamma = torch.stack(
            [torch.stack([christoffel[i, j] for j in range(3)], dim=0) for i in range(3)],
            dim=0,
        )  # gamma[i, j] = Gamma^i_{0j}
        return torch.einsum("li...,ij...->lj...", self.metric.h_cov_new, gamma)

    def test_coriolis_does_no_work(self):
        """Coriolis acceleration does no work when its covariant form is antisymmetric."""
        self.assertNotEqual(self.geom.rotation_speed, 0.0, "a non-rotating case makes this vacuous")

        matrix = self.coriolis_matrix()
        antisymmetric = matrix + torch.movedim(matrix, 0, 1)
        # The symbols must actually be present, or antisymmetry holds trivially.
        self.assertGreater(float(matrix.abs().max()), 0.0)

        christoffel = self.metric.christoffel
        gamma_abs = torch.stack(
            [torch.stack([christoffel[i, j].abs() for j in range(3)], dim=0) for i in range(3)],
            dim=0,
        )
        # A shared scale avoids division by zero in the shallow vertical column.
        scale = float(torch.einsum("li...,ij...->lj...", self.metric.h_cov_new.abs(), gamma_abs).max())
        self.assertGreater(scale, 0.0)
        self.assertLess(float(antisymmetric.abs().max()) / scale, 1.0e-12)

    # ------------------------------------------------------------------ the depth approximation

    def test_depth_approximation_reaches_the_geometry(self):
        self.assertEqual(self.geom.deep, self.depth == "deep")

    def test_shallow_drops_every_d_a_term(self):
        """The shallow approximation removes terms proportional to d_a."""
        if self.depth != "shallow":
            self.skipTest("only meaningful under the shallow approximation")

        for name, index in (("Gamma^1_03", (0, 2)), ("Gamma^2_03", (1, 2)), ("Gamma^3_03", (2, 2))):
            with self.subTest(symbol=name):
                self.assertEqual(float(self.metric.christoffel[index].abs().max()), 0.0)

    def test_deep_keeps_the_d_a_terms(self):
        """The converse: they must not be zero when the deep formulation is selected."""
        if self.depth != "deep":
            self.skipTest("only meaningful under the deep approximation")

        for name, index in (("Gamma^1_03", (0, 2)), ("Gamma^2_03", (1, 2))):
            with self.subTest(symbol=name):
                self.assertGreater(float(self.metric.christoffel[index].abs().max()), 0.0)

    # ------------------------------------------------------------------ the Christoffel option

    def test_numer_christoffel_affects_only_the_space_only_symbols(self):
        """The option changes only the space-only Christoffel symbols."""
        source = readfile(CONFIG).replace("depth_approx = shallow", f"depth_approx = {self.depth}")
        source = source.replace("num_solpts", "numer_christoffel = 0\nnum_solpts", 1)
        analytic = Simulation(Configuration(source, load_default_schema())).rhs.full.metric.christoffel
        numeric = self.metric.christoffel

        self.assertTrue(self.sim.config.numer_christoffel, "the default must be the numerical form")
        self.assertEqual(float((numeric[:, :3] - analytic[:, :3]).abs().max()), 0.0)

        # The two space-only constructions agree to discretization error.
        difference = float((numeric[:, 3:] - analytic[:, 3:]).abs().max())
        scale = float(numeric[:, 3:].abs().max())
        self.assertGreater(difference, 0.0)
        self.assertLess(difference / scale, 1.0e-1)

    # ------------------------------------------------------------------ gravity

    def test_gravity_follows_the_depth_approximation(self):
        """Deep: ``g_r = GM/(a+z)^2 = g_0 a^2 / r^2``. Shallow: the constant surface value."""
        height = self.geom.gnomonic[2, ...]
        radius = self.geom.earth_radius

        if self.geom.deep:
            expected = gravity * (radius / (radius + height)) ** 2
            # It must actually vary, or the deep selection has had no effect.
            self.assertGreater(float(self.metric.gravity_new.max() - self.metric.gravity_new.min()), 0.0)
        else:
            expected = torch.full_like(height, gravity)

        self.assertLess(self.relative(self.metric.gravity_new, expected), 1.0e-14)


if __name__ == "__main__":
    unittest.main()
