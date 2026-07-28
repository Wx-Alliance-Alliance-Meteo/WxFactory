import types
import unittest

import sympy
import torch
from wx_test import WxTestCase

from wx_factory.geometry.operators import DFROperators
from wx_factory.geometry.quadrature import gauss_legendre


class PrecisionConstructionTestCases(WxTestCase):
    @staticmethod
    def _geometry(num_solpts: int):
        symbolic, nodes, weights = gauss_legendre(num_solpts)
        extension = [sympy.sympify("-1"), *symbolic, sympy.sympify("1")]
        return types.SimpleNamespace(
            solutionPoints=nodes,
            solutionPoints_sym=symbolic,
            extension_sym=extension,
            glweights=weights,
            num_solpts=num_solpts,
            is_3d_euler_grid=True,
        )

    def test_quadrature_is_constructed_in_double_precision(self):
        _, nodes, weights = gauss_legendre(4)
        self.assertEqual(nodes.dtype, torch.float64)
        self.assertEqual(weights.dtype, torch.float64)

    def test_single_precision_operators_are_rounded_double_operators(self):
        geom = self._geometry(4)
        double = DFROperators(geom, types.SimpleNamespace(real_dtype=torch.float64))
        single = DFROperators(geom, types.SimpleNamespace(real_dtype=torch.float32))

        for name in ("quad_weights", "highfilter", "highfilter_k"):
            self.assertTrue(
                torch.equal(getattr(single, name), getattr(double, name).to(torch.float32))
            )

        filter_double = double.make_filter_3d(36.0, 8, 0.25, geom)
        filter_single = single.make_filter_3d(36.0, 8, 0.25, geom)
        self.assertTrue(torch.equal(filter_single, filter_double.to(torch.float32)))


if __name__ == "__main__":
    unittest.main()
