import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from mpi4py import MPI

from wx_factory.integrators.butcher import METHODS
from wx_factory.integrators.explicit_runge_kutta import global_rms_norm
from wx_factory.solvers.exode import ExodeStartup, exode
from wx_factory.solvers.exponential_solver import ExponentialSolverRequest, resolve_exponential_solver


class ExodeTestCase(unittest.TestCase):
    def test_phi_actions_match_augmented_matrix_exponential(self):
        matrix = torch.diag(torch.tensor([-2.0, 0.5], dtype=torch.float64))
        vectors = torch.tensor([[1.0, -2.0], [0.3, 0.1], [-0.5, 0.2], [0.1, -0.3]], dtype=torch.float64)
        context = SimpleNamespace(comm=MPI.COMM_SELF)
        for count, time, method in ((1, 1.0, "BS3(2)"), (4, 0.25, "DP5(4)"), (4, 1.5, "DP5(4)"), (4, 0.5, "M4(3)")):
            with self.subTest(count=count, time=time):
                u = vectors[:count].clone()
                size = matrix.shape[0]
                augmented = matrix.new_zeros((size + count - 1,) * 2)
                augmented[:size, :size] = matrix
                initial = matrix.new_zeros(augmented.shape[0])
                initial[:size] = u[0]
                if count > 1:
                    augmented[:size, size:] = u[1:].flip(0).T
                    for i in range(size, augmented.shape[0] - 1):
                        augmented[i, i + 1] = 1
                    initial[-1] = 1
                expected = (torch.matrix_exp(time * augmented) @ initial)[:size]
                actual, _ = exode(time, lambda x: matrix @ x, u, method=method, rtol=1e-9, atol=1e-11, context=context)
                torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-10)
                torch.testing.assert_close(u, vectors[:count], rtol=0, atol=0)

    def test_registry_uses_requested_time_and_context_communicator(self):
        context = SimpleNamespace(comm=MPI.COMM_SELF)
        vectors = torch.tensor([[1.0, 2.0], [0.3, -0.5], [-0.2, 0.1]], dtype=torch.float64)
        time = 0.25
        request = ExponentialSolverRequest(
            [time], lambda x: torch.zeros_like(x), vectors, 1e-8, 32, context, exode_controller="", announce=False
        )
        with patch("wx_factory.integrators.explicit_runge_kutta.global_rms_norm", wraps=global_rms_norm) as norm:
            result = resolve_exponential_solver("exode")(request)
        expected = vectors[0] + time * vectors[1] + time**2 / 2 * vectors[2]
        torch.testing.assert_close(result.value, expected, rtol=2e-14, atol=2e-14)
        self.assertTrue(norm.called)
        self.assertTrue(all(call.args[2] is context.comm for call in norm.call_args_list))

    def test_completed_solve_releases_operator_without_cyclic_collection(self):
        class Operator:
            def __call__(self, value):
                return -value

        enabled = gc.isenabled()
        gc.disable()
        try:
            operator = Operator()
            reference = weakref.ref(operator)
            exode(
                1.0,
                operator,
                torch.ones((1, 2), dtype=torch.float64),
                context=SimpleNamespace(comm=MPI.COMM_SELF),
            )
            del operator
            self.assertIsNone(reference())
        finally:
            if enabled:
                gc.enable()

    def test_failed_integration_does_not_return_a_partial_solution(self):
        solver = SimpleNamespace(status="running", t=0.0)
        solver.step = Mock(side_effect=lambda: setattr(solver, "status", "failed"))
        startup = ExodeStartup(0.1)
        with patch.dict(METHODS, {"BS3(2)": Mock(return_value=solver)}):
            with self.assertRaisesRegex(RuntimeError, "EXODE integration failed"):
                exode(1.0, lambda x: x, torch.ones((2, 2), dtype=torch.float64), startup=startup)
            self.assertEqual(startup.first_accepted, 0.1)

    def test_startup_reuses_first_accepted_step_and_is_scoped(self):
        cold_step = Mock(return_value=0.5)
        trials = []

        def make_solver(*args, **kwargs):
            trials.append(kwargs["first_step"])
            solver = SimpleNamespace(
                t=0.0,
                status="running",
                nfev=2,
                failed_steps=1,
                error_norm_old=0.1,
                h_previous=0.0,
                h=0.1,
                y=args[2],
                estimate_error=lambda: 0.0,
            )

            def step():
                solver.t += 0.2 if solver.t == 0 else 0.8
                if solver.t == 1.0:
                    solver.status = "finished"

            solver.step = step
            return solver

        solve = resolve_exponential_solver("exode")
        independent_solve = resolve_exponential_solver("exode")
        request = ExponentialSolverRequest(
            1.0,
            lambda x: x,
            torch.ones((1, 2)),
            1e-6,
            32,
            SimpleNamespace(comm=MPI.COMM_SELF),
            initial_step=cold_step,
            announce=False,
        )
        with patch.dict(METHODS, {"BS3(2)": make_solver}):
            for solver in (solve, solve, independent_solve):
                solver(request)
        self.assertEqual(trials, [0.5, 0.2, 0.5])
        self.assertEqual(cold_step.call_count, 2)

    def test_h211b_uses_previous_accepted_step_after_rejection(self):
        from wx_factory.integrators.explicit_runge_kutta import limiter

        solver = METHODS["BS3(2)"](
            lambda t, y: -y,
            0.0,
            torch.ones(1, dtype=torch.float64),
            10.0,
            first_step=0.1,
            controller=(0.25, 0.25, 0.25, 0.8),
        )
        with patch(
            "wx_factory.integrators.explicit_runge_kutta.global_rms_norm",
            side_effect=[0.2, 0.3, 2.0, 0.4],
        ):
            solver.step()
            first_step = solver.h_previous
            second_step = solver.h
            solver.step()
            expected_factor = 0.8**0.5 * (0.3 * 0.2) ** (-1 / 12) * (second_step / first_step) ** (-0.25)
            self.assertAlmostEqual(solver.h, second_step * limiter(expected_factor, 2))
            trial_step = solver.h
            solver.step()
        accepted_step = trial_step * limiter(0.8 * 2.0 ** (-1 / 3), 2)
        self.assertAlmostEqual(solver.h_previous, accepted_step)
        self.assertLessEqual(solver.h, accepted_step)
        self.assertEqual(solver.failed_steps, 1)

    def test_registry_tolerance_controls_relative_accuracy(self):
        import math

        context = SimpleNamespace(comm=MPI.COMM_SELF)
        errors = []
        for tolerance in (1e-4, 1e-7):
            request = ExponentialSolverRequest(
                [1.0],
                lambda x: -2 * x,
                torch.tensor([[1e6]], dtype=torch.float64),
                tolerance,
                32,
                context,
                announce=False,
            )
            result = resolve_exponential_solver("exode")(request)
            errors.append(abs(result.value.item() / (1e6 * math.exp(-2)) - 1))
        self.assertLess(errors[1], errors[0] / 100)
        self.assertLess(errors[1], 2e-6)

    def test_nonfinite_trial_is_retried_and_invalid_controls_are_rejected(self):
        import math

        def rhs(t, y):
            return torch.where(y.abs() > 2, math.nan, -10 * y)

        solver = METHODS["BS3(2)"](
            rhs,
            0.0,
            torch.ones(1, dtype=torch.float64),
            1.0,
            first_step=1.0,
            rtol=1e-6,
            atol=1e-8,
        )
        while solver.status == "running":
            solver.step()
        self.assertEqual(solver.status, "finished")
        self.assertGreater(solver.failed_steps, 0)
        torch.testing.assert_close(solver.y, torch.tensor([math.exp(-10)], dtype=torch.float64), rtol=1e-3, atol=1e-8)
        for controls in ({"rtol": -1}, {"atol": 0}, {"atol": math.nan}, {"max_step": 0}, {"first_step": math.nan}):
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                METHODS["BS3(2)"](rhs, 0.0, torch.ones(1, dtype=torch.float64), 1.0, **({"first_step": 0.1} | controls))

    def test_small_time_interval_has_no_absolute_step_floor(self):
        import math

        solver = METHODS["BS3(2)"](
            lambda t, y: -2e14 * y,
            0.0,
            torch.ones(1, dtype=torch.float64),
            1e-14,
            first_step=1e-14,
            rtol=1e-7,
            atol=1e-9,
        )
        while solver.status == "running":
            solver.step()
        self.assertEqual(solver.status, "finished")
        self.assertGreater(solver.failed_steps, 0)
        torch.testing.assert_close(solver.y, torch.tensor([math.exp(-2)], dtype=torch.float64), rtol=2e-6, atol=1e-9)
