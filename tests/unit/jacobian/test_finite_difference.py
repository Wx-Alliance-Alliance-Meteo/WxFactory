import numpy
import torch
from wx_test import WxTestCase

from wx_factory.jacobian import FiniteDifferenceJacobian, fd_jacobian_matvec, fd_norm


class FiniteDifferenceJacobianTestCases(WxTestCase):
    def test_mixed_precision_probe_preserves_small_linear_response(self):
        class AccurateRhs:
            def __call__(self, value):
                return 3 * value

            def jacobian_rhs(self, value):
                self.last_dtype = value.dtype
                return 3 * value

        state = torch.tensor([1e5, 300.0], dtype=torch.float32)
        direction = torch.tensor([0.01, 1.0], dtype=torch.float32)
        rhs = AccurateRhs()
        operator = FiniteDifferenceJacobian(2.0, state, rhs)
        result = operator(direction)
        self.assertEqual(rhs.last_dtype, torch.float64)
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, 6 * direction, rtol=2e-4, atol=1e-5)
        direct = fd_jacobian_matvec(direction, 2.0, state, rhs(state), rhs)
        torch.testing.assert_close(direct, result, rtol=0, atol=0)

    def test_fd_norm_uses_working_precision(self):
        vector = torch.tensor([3.0, 4.0], dtype=torch.float32)

        norm = fd_norm(vector, self.comm)

        self.assertIsInstance(norm, numpy.float32)
        self.assertEqual(norm, numpy.float32(5.0))

    def test_fd_matches_equations_10_and_14(self):
        state = torch.zeros(2, dtype=torch.float32)
        direction = torch.tensor([5.0, 12.0], dtype=torch.float32)
        rhs_inputs = []

        def rhs(value):
            rhs_inputs.append(value.clone())
            return 3.0 * value

        result = fd_jacobian_matvec(direction, 2.0, state, rhs(state), rhs)

        epsilon = numpy.sqrt(numpy.float64(numpy.finfo(numpy.float32).eps)) / 13.0
        expected_state = state + (direction.to(torch.float64) * epsilon).to(state.dtype)
        torch.testing.assert_close(rhs_inputs[1], expected_state, rtol=0.0, atol=0.0)
        torch.testing.assert_close(result, 6.0 * direction, rtol=5.0e-4, atol=5.0e-4)

    def test_fd_preserves_working_precision(self):
        state = torch.zeros(2, dtype=torch.float32)
        direction = torch.tensor([5.0, 12.0], dtype=torch.float32)
        rhs_dtypes = []

        def rhs(value):
            rhs_dtypes.append(value.dtype)
            return value

        base = rhs(state)
        result = fd_jacobian_matvec(direction, 1.0, state, base, rhs)

        self.assertEqual(rhs_dtypes, [torch.float32, torch.float32])
        self.assertEqual(result.dtype, torch.float32)
