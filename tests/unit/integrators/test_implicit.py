import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from mpi4py import MPI

from wx_factory.context import Context
from wx_factory.integrators.backward_euler import BackwardEuler
from wx_factory.integrators.bdf2 import Bdf2
from wx_factory.integrators.crank_nicolson import CrankNicolson
from wx_factory.integrators.sdirk import SDIRKLstable
from wx_factory.solvers.fgmres import fgmres
from wx_factory.solvers.nonlin import newton_krylov


class ImplicitIntegratorTestCase(unittest.TestCase):
    def setUp(self):
        with patch.dict(os.environ, {"WX_FACTORY_DIFFERENTIABLE": "1"}):
            self.context = Context(MPI.COMM_SELF, "cpu")
        self.config = SimpleNamespace(tolerance=1e-10, verbose_solver=False, gmres_restart=20)
        self.matrix = -torch.diag(torch.linspace(0.5, 2.0, 32, dtype=torch.float64))
        self.matrix += 0.05 * torch.roll(torch.eye(32, dtype=torch.float64), 1, 1)
        self.state = torch.tensor([1.0, 0.5], dtype=torch.float64).repeat(16)

    def test_bdf2_startup_and_history(self):
        integrator = Bdf2(self.config, lambda state: self.matrix @ state, context=self.context)
        dt = 0.1
        initial = self.state.clone()
        first = integrator.step(self.state, dt)
        expected = torch.linalg.solve(torch.eye(32, dtype=torch.float64) - dt * self.matrix, initial)
        torch.testing.assert_close(first, expected, rtol=0, atol=1e-10)
        torch.testing.assert_close(self.state, initial, rtol=0, atol=0)
        self.state.zero_()
        second = integrator.step(first, dt)
        expected = torch.linalg.solve(
            torch.eye(32, dtype=torch.float64) - 2 * dt / 3 * self.matrix, 4 * first / 3 - initial / 3
        )
        torch.testing.assert_close(second, expected, rtol=0, atol=1e-10)
        self.assertEqual(second.dtype, initial.dtype)
        self.assertEqual(second.device, initial.device)

    def test_crank_nicolson_formula_and_second_order(self):
        errors = []
        exact = torch.matrix_exp(self.matrix) @ self.state
        for steps in (10, 20):
            integrator = CrankNicolson(self.config, lambda state: self.matrix @ state, context=self.context)
            state = self.state.clone()
            dt = 1 / steps
            for _ in range(steps):
                previous = state.clone()
                state = integrator.step(state, dt)
                expected = torch.linalg.solve(
                    torch.eye(32, dtype=torch.float64) - dt / 2 * self.matrix,
                    (torch.eye(32, dtype=torch.float64) + dt / 2 * self.matrix) @ previous,
                )
                torch.testing.assert_close(state, expected, rtol=0, atol=1e-10)
            errors.append((state - exact).norm().item())
        self.assertGreater(errors[0] / errors[1], 3.9)
        self.assertLess(errors[0] / errors[1], 4.1)

    def test_implicit_methods_have_the_expected_order(self):
        exact = torch.matrix_exp(self.matrix) @ self.state
        for method, order in ((BackwardEuler, 1), (Bdf2, 2), (SDIRKLstable, 2)):
            errors = []
            for steps in (80, 160):
                integrator = method(self.config, lambda state: self.matrix @ state, context=self.context)
                state = self.state.clone()
                for _ in range(steps):
                    state = integrator.step(state, 1 / steps)
                errors.append((state - exact).norm().item())
            self.assertGreater(errors[0] / errors[1], 0.95 * 2**order)
            self.assertLess(errors[0] / errors[1], 1.05 * 2**order)

    def test_mixed_krylov_basis_preserves_double_newton_residuals(self):
        target = torch.linspace(0.5, 2.0, 32, dtype=torch.float64)
        result, _, _ = newton_krylov(
            lambda state: state.square() - target,
            torch.ones_like(target),
            context=self.context,
            f_tol=1e-10,
            jacobian=lambda state: lambda vector: 2 * state * vector,
            krylov_dtype=torch.float32,
        )
        self.assertEqual(result.dtype, torch.float64)
        torch.testing.assert_close(result, target.sqrt(), rtol=0, atol=1e-10)

    def test_chunked_mixed_precision_krylov_products(self):
        diagonal = torch.linspace(0.5, 2.0, 65537, dtype=torch.float64)
        forcing = torch.ones_like(diagonal)
        result, _, _, _, flag, _ = fgmres(
            lambda vector: diagonal * vector,
            forcing,
            context=self.context,
            restart=8,
            maxiter=10,
            tol=1e-6,
            krylov_dtype=torch.float32,
        )
        self.assertEqual(flag, 0)
        self.assertLess((diagonal * result - forcing).norm().item() / forcing.norm().item(), 1e-6)

    def test_nonconvergence_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            newton_krylov(
                lambda state: state.square() - 2,
                self.state,
                context=self.context,
                f_tol=1e-10,
                maxiter=0,
            )

    def test_analytic_jacobian_is_refreshed_at_newton_iterates(self):
        target = torch.linspace(0.5, 2.0, 32, dtype=torch.float64).reshape(4, 8)
        iterates = []

        def linearize(state):
            iterates.append(state.clone())
            return lambda vector: 2 * state * vector.reshape_as(state)

        result, _, _ = newton_krylov(
            lambda state: state.square() - target,
            torch.ones_like(target),
            context=self.context,
            f_tol=1e-10,
            jacobian=linearize,
        )
        torch.testing.assert_close(result, target.sqrt(), rtol=0, atol=1e-10)
        self.assertGreater(len(iterates), 1)
        self.assertFalse(torch.equal(iterates[0], iterates[-1]))

    def test_integrators_select_the_analytic_rhs_jacobian(self):
        matrix = self.matrix
        calls = []

        class LinearRhs:
            def __call__(self, state):
                return matrix @ state

            def linearize(self, state):
                calls.append(state.clone())
                return lambda vector: matrix @ vector

        self.config.jacobian = "analytic"
        for method in (Bdf2, CrankNicolson, BackwardEuler, SDIRKLstable):
            integrator = method(self.config, LinearRhs(), context=self.context)
            integrator.step(self.state, 0.1)
            self.assertGreater(len(calls), 0)
            calls.clear()


if __name__ == "__main__":
    unittest.main()
