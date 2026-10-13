"""DFR must preserve constants in both working precisions."""

import os
import unittest

import torch
from wx_test import WxTestCase

from wx_factory.common import Configuration, load_default_schema, readfile
from wx_factory.geometry.operators import DFROperators
from wx_factory.simulation import Simulation


class OperatorIdentityTestCase(WxTestCase):
    def setUp(self) -> None:
        super().setUp()
        here = os.path.dirname(os.path.realpath(__file__))
        config = Configuration(readfile(os.path.join(here, "filter_config.ini")), load_default_schema())
        self.sim = Simulation(config)

    def operators(self, dtype):
        return DFROperators(self.sim.geometry, self.sim.context, dtype=dtype)

    def constants(self, ops, dtype):
        """Return a constant element field and a constant set of interface traces."""
        num_solpts = self.sim.geometry.num_solpts
        volume = torch.ones(1, num_solpts**3, dtype=dtype)
        interface = torch.ones(1, 2 * num_solpts**2, dtype=dtype)
        return volume, interface

    def test_derivative_annihilates_a_constant(self):
        """D.1 + C.1 == 0, in both working precisions."""
        for dtype, tol in ((torch.float64, 1e-14), (torch.float32, 0.0)):
            with self.subTest(dtype=dtype):
                ops = self.operators(dtype)
                volume, interface = self.constants(ops, dtype)
                for derivative, correction in (
                    (ops.derivative_x, ops.correction_WE),
                    (ops.derivative_y, ops.correction_SN),
                    (ops.derivative_z, ops.correction_DU),
                ):
                    residual = (volume @ derivative + interface @ correction).abs().max().item()
                    self.assertLessEqual(residual, tol)

    def test_extrapolation_reproduces_a_constant(self):
        """E.1 == 1, in both working precisions."""
        for dtype, tol in ((torch.float64, 1e-14), (torch.float32, 0.0)):
            with self.subTest(dtype=dtype):
                ops = self.operators(dtype)
                volume, _ = self.constants(ops, dtype)
                for extrapolation in (ops.extrap_x, ops.extrap_y, ops.extrap_z):
                    residual = (volume @ extrapolation - 1.0).abs().max().item()
                    self.assertLessEqual(residual, tol)

    def test_single_precision_matmul_is_not_reduced(self):
        """TF32 would silently defeat the conditioning the operands rely on."""
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")


if __name__ == "__main__":
    unittest.main()
