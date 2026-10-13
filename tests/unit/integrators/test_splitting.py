import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from mpi4py import MPI

from wx_factory.context import Context
from wx_factory.integrators import resolve
from wx_factory.integrators.integrator import Integrator
from wx_factory.integrators.splitting import LieSplitting, OS22Splitting, StrangSplitting
from wx_factory.rhs.rhs_selector import RhsBundle


class LinearAction:
    def __init__(self, matrix):
        self.matrix = matrix

    def __call__(self, vector):
        return self.matrix @ vector.flatten()

    def solve(self, vector, dt):
        identity = torch.eye(self.matrix.shape[0], dtype=vector.dtype, device=vector.device)
        return torch.linalg.solve(identity - dt / 2 * self.matrix, vector.flatten())


class LinearPartition:
    def __init__(self, matrix, partition=None):
        self.matrix = matrix
        self.partition = partition
        self.metric = SimpleNamespace(volume=SimpleNamespace(G_cov=matrix))

    def __call__(self, state):
        return (self.matrix.to(state) @ state.flatten()).reshape_as(state)

    def linearize(self, state):
        return LinearAction(self.matrix.to(state))

    def double_precision_rhs(self):
        return LinearPartition(self.matrix.to(torch.float64), self.partition)


class FailedSubstep(Integrator):
    def __init__(self, config, context, *, raises=False):
        super().__init__(config, context=context)
        self.calls = 0
        self.raises = raises

    def __step__(self, state, dt):
        self.calls += 1
        self.failure_flag = 7
        if self.raises:
            raise RuntimeError("Substep failed")
        return state


class SplittingTestCase(unittest.TestCase):
    def setUp(self):
        self.default_device = torch.get_default_device()
        with patch.dict(os.environ, {"WX_FACTORY_DIFFERENTIABLE": "1"}):
            self.context = Context(MPI.COMM_SELF, "cpu")
        self.config = SimpleNamespace(
            tolerance=1e-11,
            verbose_solver=False,
            gmres_restart=20,
            krylov_mmax=32,
            exponential_solver="pmex",
            exode_method="BS3(2)",
            exode_controller="",
            jacobian="analytic",
            splitting_integrator_1="epi2",
            splitting_integrator_2="ros2",
            os22_parameter=0.25,
        )
        self.explicit = torch.tensor([[-0.1, 0.3], [-0.2, -0.1]], dtype=torch.float64)
        self.implicit = torch.tensor([[-1.0, 0], [0.2, -2.0]], dtype=torch.float64)
        self.state = torch.tensor([1.0, 0.5], dtype=torch.float64)

        def full(state):
            raise AssertionError("A splitting substep evaluated the full RHS")

        self.bundle = RhsBundle(
            full=full,
            shape=self.state.shape,
            explicit=LinearPartition(self.explicit),
            implicit=LinearPartition(self.implicit),
        )

    def tearDown(self):
        torch.set_default_device(self.default_device)

    def implicit_flow(self, state, dt):
        identity = torch.eye(state.numel(), dtype=state.dtype)
        return torch.linalg.solve(identity - dt / 2 * self.implicit, (identity + dt / 2 * self.implicit) @ state)

    def explicit_flow(self, state, dt):
        return torch.matrix_exp(dt * self.explicit) @ state

    def test_factory_bindings_and_composition_formulas(self):
        dt = 0.2
        for name in ("lie", "strang", "os22", "strang_epi2_ros2", "strang_ros2_epi2"):
            with self.subTest(name=name):
                method = resolve(name, self.config, self.bundle, None, self.context)
                with patch(
                    "wx_factory.integrators.ros2.FiniteDifferenceJacobian", side_effect=AssertionError("FD used")
                ):
                    actual = method.step(self.state, dt)
                if name == "lie":
                    expected = self.implicit_flow(self.explicit_flow(self.state, dt), dt)
                elif name in ("strang", "strang_epi2_ros2"):
                    expected = self.explicit_flow(
                        self.implicit_flow(self.explicit_flow(self.state, dt / 2), dt), dt / 2
                    )
                elif name == "strang_ros2_epi2":
                    expected = self.implicit_flow(
                        self.explicit_flow(self.implicit_flow(self.state, dt / 2), dt), dt / 2
                    )
                else:
                    expected = self.state
                    for a, b in ((1 / 3, 3 / 4), (2 / 3, 1 / 4)):
                        expected = self.implicit_flow(self.explicit_flow(expected, a * dt), b * dt)
                torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)

    def test_order_on_noncommuting_linear_partitions(self):
        exact = torch.matrix_exp(self.explicit + self.implicit) @ self.state
        for name, order in (("lie", 1), ("strang", 2), ("os22", 2)):
            errors = []
            for steps in (10, 20):
                method = resolve(name, self.config, self.bundle, None, self.context)
                state = self.state.clone()
                for step in range(steps):
                    state = method.step(state, 1 / steps)
                errors.append((state - exact).norm().item())
            with self.subTest(name=name):
                self.assertGreater(errors[0] / errors[1], 2**order * 0.85)
                self.assertLess(errors[0] / errors[1], 2**order * 1.15)

    def test_os22_endpoints_are_the_two_strang_orderings(self):
        for parameter, strang in ((0.0, "strang_epi2_ros2"), (0.5, "strang_ros2_epi2")):
            self.config.os22_parameter = parameter
            actual = resolve("os22", self.config, self.bundle, None, self.context).step(self.state, 0.2)
            expected = resolve(strang, self.config, self.bundle, None, self.context).step(self.state, 0.2)
            torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)

    def test_failures_stop_the_composition(self):
        for cls in (LieSplitting, StrangSplitting, OS22Splitting):
            for raises in (False, True):
                first = FailedSubstep(self.config, self.context, raises=raises)
                second = FailedSubstep(self.config, self.context)
                args = (self.config, first, second)
                if cls is OS22Splitting:
                    args += (0.0,)
                method = cls(*args, context=self.context)
                with self.assertRaises(RuntimeError):
                    method.step(self.state, 0.2)
                self.assertEqual(method.failure_flag, 7)
                self.assertEqual(second.calls, 0)

    def test_finite_difference_exponential_mode_keeps_analytic_column_solve(self):
        self.config.jacobian = "finite_difference"
        self.bundle.implicit.partition = "implicit"
        method = resolve("strang", self.config, self.bundle, None, self.context)
        with patch(
            "wx_factory.integrators.ros2.FiniteDifferenceJacobian", side_effect=AssertionError("FD column solve")
        ):
            actual = method.step(self.state, 0.2)
        expected = self.explicit_flow(self.implicit_flow(self.explicit_flow(self.state, 0.1), 0.2), 0.1)
        torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-8)

    def test_ros2_solve_failure_reaches_the_wrapper(self):
        method = resolve("lie", self.config, self.bundle, None, self.context)
        failed = (torch.zeros_like(self.state), 1.0, 1.0, 1, 4, [])
        with patch.object(LinearPartition, "linearize", return_value=lambda vector: vector), patch(
            "wx_factory.integrators.ros2.fgmres", return_value=failed
        ), self.assertRaises(RuntimeError):
            method.step(self.state, 0.2)
        self.assertEqual(method.failure_flag, 4)
        self.assertEqual(method.scheme2.failure_flag, 4)

    def test_invalid_parameters_and_subintegrators(self):
        for parameter in (None, float("nan"), -0.1, 0.75, 1.0):
            self.config.os22_parameter = parameter
            with self.assertRaises(ValueError):
                resolve("os22", self.config, self.bundle, None, self.context)
        for name in ("epi3", "bdf2", "strang", "partrosexp2"):
            self.config.splitting_integrator_1 = name
            with self.assertRaises(ValueError):
                resolve("lie", self.config, self.bundle, None, self.context)
        unavailable = RhsBundle(full=lambda state: state, shape=self.state.shape)
        for name in ("lie", "strang", "os22"):
            with self.assertRaises(ValueError):
                resolve(name, self.config, unavailable, None, self.context)


if __name__ == "__main__":
    unittest.main()
