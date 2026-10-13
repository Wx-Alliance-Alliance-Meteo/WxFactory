import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from mpi4py import MPI

from tests.unit.integrators.test_srerk import LinearRhs, QuadraticRhs, dense_projection
from wx_factory.context import Context
from wx_factory.integrators.epi import Epi
from wx_factory.integrators.epi_stiff import EpiStiff


class EpiTestCase(unittest.TestCase):
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
            exode_controller="basic",
        )

    def tearDown(self):
        torch.set_default_device(self.default_device)

    @staticmethod
    def projection(request):
        from dataclasses import replace

        dtype = request.vectors.dtype
        precise = replace(
            request,
            vectors=request.vectors.to(torch.float64),
            operator=lambda vector: request.operator(vector.to(dtype)).to(torch.float64),
        )
        result = dense_projection(precise)
        result.value = result.value.to(dtype)
        return result

    def method(self, kind, order, rhs):
        method = kind(self.config, order, rhs, context=self.context, init_substeps=10)
        method.solve_exponential = self.projection
        if method.init_method is not None:
            method.init_method.solve_exponential = self.projection
        return method

    def test_all_orders_past_startup_and_history_reset(self):
        for kind, orders in ((Epi, range(2, 7)), (EpiStiff, range(2, 10))):
            for order in orders:
                for dtype in (torch.float32, torch.float64):
                    with self.subTest(method=kind.__name__, order=order, dtype=dtype):
                        matrix = torch.tensor([[-1.0, 0.2], [0.0, -2.0]], dtype=dtype)
                        state = torch.tensor([0.3, 0.7], dtype=dtype)
                        initial = state.clone()
                        rhs = LinearRhs(matrix)
                        method = self.method(kind, order, rhs)
                        with patch(
                            "wx_factory.integrators.epi_stiff.FiniteDifferenceJacobian",
                            side_effect=AssertionError("finite difference used"),
                        ):
                            steps = method.n_prev + 2
                            for _ in range(steps):
                                previous = state.clone()
                                state = method.step(state, 0.05)
                                torch.testing.assert_close(
                                    previous, method.previous_Q[0] if method.n_prev else previous, atol=0, rtol=0
                                )
                            state = method.step(state, 0.025)
                        expected = (torch.matrix_exp((steps * 0.05 + 0.025) * matrix.double()) @ initial.double()).to(
                            dtype
                        )
                        tolerance = 3e-6 if dtype == torch.float32 else 1e-12
                        torch.testing.assert_close(state, expected, atol=tolerance, rtol=tolerance)
                        self.assertEqual(state.dtype, dtype)
                        self.assertGreater(rhs.linearizations, steps)

    def test_nonlinear_refinement(self):
        initial = torch.tensor([0.3], dtype=torch.float64)
        exact = initial / (1 - 0.8 * initial)
        for kind, orders in ((Epi, range(3, 7)), (EpiStiff, range(3, 10))):
            for order in orders:
                errors = []
                for steps in (16, 32):
                    method = self.method(kind, order, QuadraticRhs())
                    state = initial.clone()
                    for _ in range(steps):
                        state = method.step(state, 0.8 / steps)
                    errors.append((state - exact).abs().max().item())
                with self.subTest(method=kind.__name__, order=order):
                    self.assertLess(errors[1], errors[0])
                    self.assertGreater(errors[0] / errors[1], 4)

    def test_prescribed_transport_internal_clock(self):
        from wx_factory.integrators.prescribed_transport import PrescribedTransportIntegrator

        class TimeDependentTransport:
            period = 1.0
            time = 0.0

            def __call__(self, state):
                return torch.stack((state[0] * state[1], torch.ones_like(state[1])))

            def linearize(self, state):
                def action(vector):
                    delta = vector.reshape_as(state)
                    return torch.stack(
                        (state[1] * delta[0] + state[0] * delta[1], torch.zeros_like(delta[1]))
                    ).flatten()

                return action

            def restore(self, state, time):
                return state.clone()

        initial = torch.ones((6, 1), dtype=torch.float64)
        exact = torch.exp(torch.tensor(0.8**2 / 2, dtype=torch.float64))
        for kind in (Epi, EpiStiff):
            errors = []
            for steps in (8, 16):
                rhs = TimeDependentTransport()
                method = self.method(kind, 3, rhs)
                wrapper = PrescribedTransportIntegrator(self.config, method, rhs, context=self.context)
                wrapper.sim_time = 0.0
                state = initial.clone()
                for _ in range(steps):
                    state = wrapper.step(state, 0.8 / steps)
                torch.testing.assert_close(state[:5], initial[:5], atol=0, rtol=0)
                self.assertEqual(state.shape, initial.shape)
                self.assertAlmostEqual(rhs.time, 0.8)
                errors.append(abs(state[5].item() - exact.item()))
            self.assertGreater(errors[0] / errors[1], 6)

    def test_stiff_finite_difference_selection(self):
        matrix = torch.tensor([[-1.0, 0.2], [0.0, -2.0]], dtype=torch.float64)
        rhs = LinearRhs(matrix)
        method = self.method(EpiStiff, 3, rhs)
        state = torch.tensor([0.3, 0.7], dtype=torch.float64)
        self.config.jacobian = "finite_difference"
        for _ in range(4):
            state = method.step(state, 0.05)
        torch.testing.assert_close(
            state, torch.matrix_exp(0.2 * matrix) @ torch.tensor([0.3, 0.7], dtype=state.dtype), atol=2e-9, rtol=2e-9
        )
        self.assertEqual(rhs.linearizations, 0)
