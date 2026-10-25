import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from mpi4py import MPI

from wx_factory.context import Context
from wx_factory.integrators.srerk import Srerk


def dense_projection(request):
    vectors = request.vectors
    size = vectors.shape[1]
    order = vectors.shape[0] - 1
    identity = torch.eye(size, dtype=vectors.dtype, device=vectors.device)
    matrix = torch.stack([request.operator(column) for column in identity], dim=1)
    augmented = vectors.new_zeros((size + order, size + order))
    augmented[:size, :size] = matrix
    augmented[:size, size:] = vectors[1:].T.flip(1)
    if order > 1:
        augmented[size:-1, size + 1 :] = torch.eye(order - 1, dtype=vectors.dtype, device=vectors.device)
    initial = vectors.new_zeros(size + order)
    initial[:size], initial[-1] = vectors[0], 1
    result = torch.stack([(torch.matrix_exp(float(time) * augmented) @ initial)[:size] for time in request.tau_out])
    return SimpleNamespace(value=result, final_krylov_size=None)


class LinearRhs:
    def __init__(self, matrix):
        self.matrix = matrix
        self.linearizations = 0

    def __call__(self, state):
        return (self.matrix @ state.flatten()).reshape_as(state)

    def linearize(self, state):
        self.linearizations += 1
        return lambda vector: self.matrix @ vector.flatten()


class QuadraticRhs:
    def __call__(self, state):
        return state**2

    def linearize(self, state):
        return lambda vector: (2 * state * vector.reshape_as(state)).flatten()


class SrerkTestCase(unittest.TestCase):
    def setUp(self):
        self.default_device = torch.get_default_device()
        with patch.dict(os.environ, {"WX_FACTORY_DIFFERENTIABLE": "1"}):
            self.context = Context(MPI.COMM_SELF, "cpu")
        self.config = SimpleNamespace(
            tolerance=1e-12,
            verbose_solver=False,
            krylov_mmax=32,
            exponential_solver="pmex",
            jacobian="analytic",
            exode_method="BS3(2)",
            exode_controller="deadbeat",
        )
        self.state = torch.tensor([[0.3, 0.5], [0.7, 0.9]], dtype=torch.float64)
        self.matrix = torch.tensor(
            [[-0.1, 0.2, 0, 0], [0, -1, 0.3, 0], [0, 0, -3, 0.1], [0, 0, 0, -10]], dtype=torch.float64
        )

    def tearDown(self):
        torch.set_default_device(self.default_device)

    def integrator(self, order, rhs, **kwargs):
        method = Srerk(self.config, order, rhs, context=self.context, **kwargs)
        method.solve_exponential = dense_projection
        return method

    def test_all_orders_linear_exact_solution_and_one_linearization_per_step(self):
        for order in range(3, 10):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(order=order, dtype=dtype):
                    state = self.state.to(dtype)
                    initial = state.clone()
                    rhs = LinearRhs(self.matrix.to(dtype))
                    method = self.integrator(order, rhs)
                    with patch(
                        "wx_factory.integrators.srerk.fd_jacobian_matvec", side_effect=AssertionError("FD used")
                    ):
                        actual = method.step(state, 0.2)
                    expected = (torch.matrix_exp(0.2 * rhs.matrix) @ initial.flatten()).reshape_as(state)
                    tolerance = 2e-6 if dtype == torch.float32 else 2e-13
                    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
                    torch.testing.assert_close(state, initial, atol=0, rtol=0)
                    self.assertEqual(rhs.linearizations, 1)
                    self.assertEqual(actual.dtype, state.dtype)
                    self.assertEqual(actual.device, state.device)
                    self.assertEqual(actual.shape, state.shape)

    def test_nonlinear_accuracy_improves_for_every_order(self):
        time = 0.8
        exact = self.state / (1 - time * self.state)
        for order in range(3, 10):
            errors = []
            for steps in (8, 16):
                method = self.integrator(order, QuadraticRhs())
                state = self.state.clone()
                for step in range(steps):
                    state = method.step(state, time / steps)
                errors.append((state - exact).abs().max().item())
            with self.subTest(order=order):
                self.assertGreater(errors[0] / errors[1], 2 ** (order - 1))
                self.assertLess(errors[1], 2e-3)

    def test_pmex_linear_solution_for_all_orders(self):
        expected = (torch.matrix_exp(0.2 * self.matrix) @ self.state.flatten()).reshape_as(self.state)
        for order in range(3, 10):
            with self.subTest(order=order):
                rhs = LinearRhs(self.matrix)
                method = Srerk(self.config, order, rhs, context=self.context)
                actual = method.step(self.state, 0.2)
                torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)

    def test_custom_jacobian_and_finite_difference_fallback(self):
        expected = (torch.matrix_exp(0.1 * self.matrix) @ self.state.flatten()).reshape_as(self.state)
        calls = []

        def scaled_jacobian(vector, state, dt):
            calls.append(dt)
            return dt * self.matrix @ vector.flatten()

        rhs = lambda state: (self.matrix @ state.flatten()).reshape_as(state)
        for custom in (None, scaled_jacobian):
            with self.subTest(custom=custom is not None):
                method = self.integrator(4, rhs, jac=custom)
                actual = method.step(self.state, 0.1)
                torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-8)
        self.assertTrue(calls)
        self.assertTrue(all(dt == 0.1 for dt in calls))
        self.config.jacobian = "finite_difference"
        rhs = LinearRhs(self.matrix)
        actual = self.integrator(4, rhs).step(self.state, 0.1)
        torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-8)
        self.assertEqual(rhs.linearizations, 0)


if __name__ == "__main__":
    unittest.main()
