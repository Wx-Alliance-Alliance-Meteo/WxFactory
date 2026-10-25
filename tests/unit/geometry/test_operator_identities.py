"""DFR must preserve constants in both working precisions."""

import os
import unittest
from types import SimpleNamespace

import sympy
import torch
from wx_test import WxTestCase

from wx_factory.common import Configuration, load_default_schema, readfile
from wx_factory.common.matmul import DFRDerivative, sparse_dfr_derivative
from wx_factory.geometry.operators import DFROperators
from wx_factory.geometry.quadrature import gauss_legendre
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

    def test_sparse_dfr_differentiates_polynomials_with_strided_inputs(self):
        def sparse_action(operator, volume, faces):
            return sparse_dfr_derivative(
                volume, faces, operator.indices, operator.weights, operator.face_indices, operator.face_weights
            )

        for nodes in (2, 3, 5):
            symbolic, points, weights = gauss_legendre(nodes)
            grid = SimpleNamespace(
                num_solpts=nodes,
                solutionPoints=points,
                glweights=weights,
                extension_sym=[sympy.Integer(-1), *symbolic, sympy.Integer(1)],
                is_3d_euler_grid=True,
            )
            for dtype in (torch.float32, torch.float64):
                ops = DFROperators(grid, self.sim.context, dtype=dtype)
                working_points = points.to(dtype)
                z, y, x = torch.meshgrid(working_points, working_points, working_points, indexing="ij")
                degree = nodes - 1
                polynomial = (x**degree + 2 * y**degree + 3 * z**degree).flatten()
                volume = torch.stack((polynomial, -polynomial, 2 * polynomial))[::2]
                for direction, (derivative, correction, coordinate) in enumerate(
                    zip(
                        (ops.derivative_x, ops.derivative_y, ops.derivative_z),
                        (ops.correction_WE, ops.correction_SN, ops.correction_DU),
                        (x, y, z),
                    )
                ):
                    with self.subTest(nodes=nodes, dtype=dtype, direction=direction):
                        traces = []
                        for side in (-1.0, 1.0):
                            axes = [working_points, working_points, working_points]
                            axes[2 - direction] = torch.tensor([side], dtype=dtype)
                            face_z, face_y, face_x = torch.meshgrid(*axes, indexing="ij")
                            traces.append((face_x**degree + 2 * face_y**degree + 3 * face_z**degree).flatten())
                        trace = torch.cat(traces)
                        padded = torch.zeros((3, 2, trace.numel()), dtype=dtype)
                        padded[:, 1] = torch.stack((trace, -trace, 2 * trace))
                        faces = padded[::2, 1]
                        expected = ((direction + 1) * degree * coordinate ** (degree - 1)).flatten()
                        expected = torch.stack((expected, 2 * expected))
                        operator = DFRDerivative(derivative, correction, action=sparse_action)
                        actual = operator(volume, faces)
                        tolerance = 128 * torch.finfo(dtype).eps
                        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

    def test_single_precision_matmul_is_not_reduced(self):
        """TF32 would silently defeat the conditioning the operands rely on."""
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")


if __name__ == "__main__":
    unittest.main()
