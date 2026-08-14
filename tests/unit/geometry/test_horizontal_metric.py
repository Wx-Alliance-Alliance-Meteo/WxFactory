"""Tests for the horizontal metric used to exchange terrain gradients."""

import unittest

import torch

from tests.unit.wx_test import WxTestCase
from wx_factory.geometry.metric3d import horizontal_metric_2d


class HorizontalMetricTestCase(WxTestCase):
    def setUp(self) -> None:
        self.dtype = torch.float64
        # The gnomonic coordinates span [-1, 1] on each panel.
        axis = torch.arange(-10, 11, dtype=self.dtype) / 10.0
        self.X, self.Y = torch.meshgrid(axis, axis, indexing="ij")

    def test_forms_are_exact_inverses(self):
        """g_ij h^jk = delta_i^k, the property the raise-and-lower round trip depends on."""
        contra, cov = horizontal_metric_2d(self.X, self.Y)
        product = torch.einsum("ij...,jk...->ik...", cov, contra)

        identity = torch.eye(2, dtype=self.dtype).reshape(2, 2, 1, 1)
        self.assertLess(float((product - identity).abs().max()), 1.0e-14)

    def test_both_forms_are_symmetric(self):
        for name, metric in zip(("contravariant", "covariant"), horizontal_metric_2d(self.X, self.Y)):
            with self.subTest(form=name):
                self.assertLess(float((metric[0, 1] - metric[1, 0]).abs().max()), 0.0 + 1.0e-16)

    def test_both_forms_are_positive_definite(self):
        for name, metric in zip(("contravariant", "covariant"), horizontal_metric_2d(self.X, self.Y)):
            with self.subTest(form=name):
                trailing = torch.movedim(metric, (0, 1), (-2, -1))
                self.assertGreater(float(torch.linalg.eigvalsh(trailing).min()), 0.0)

    def test_reduces_to_the_identity_at_a_panel_centre(self):
        """At X = Y = 0 the gnomonic chart is locally isometric to the unit sphere."""
        zero = torch.zeros(1, dtype=self.dtype)
        contra, cov = horizontal_metric_2d(zero, zero)

        identity = torch.eye(2, dtype=self.dtype).reshape(2, 2, 1)
        self.assertLess(float((contra - identity).abs().max()), 1.0e-15)
        self.assertLess(float((cov - identity).abs().max()), 1.0e-15)

    def test_off_diagonal_vanishes_only_on_the_axes(self):
        """The off-diagonal term vanishes only on the coordinate axes."""
        contra, _ = horizontal_metric_2d(self.X, self.Y)
        on_axis = (self.X == 0.0) | (self.Y == 0.0)

        self.assertLess(float(contra[0, 1][on_axis].abs().max()), 1.0e-16)
        self.assertGreater(float(contra[0, 1][~on_axis].abs().min()), 0.0)


if __name__ == "__main__":
    unittest.main()
