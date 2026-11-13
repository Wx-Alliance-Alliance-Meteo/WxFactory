import unittest
from copy import copy
from unittest.mock import patch

import torch

from tests.unit.rhs.euler_frozen_fixture import frozen_problem
from wx_factory.common.matmul import apply_op
from wx_factory.geometry.metric5d import Metric5D
from wx_factory.jacobian.block_solve import BlockThomasFactor, blocks_matvec, cyclic_block_solve
from wx_factory.jacobian.column_layout import coupling_variable_rows, stiff_variable_rows
from wx_factory.jacobian.euler_vertical import EulerVerticalLinearization
from wx_factory.pde.euler_interfaces import pressure
from wx_factory.rhs.euler_partition import EulerRhsPartition
from wx_factory.rhs.rhs_euler import RhsEuler


class EulerPartitionTestCase(unittest.TestCase):
    def test_tracer_actions_and_column_solves(self):
        for periodic in (False, True):
            with self.subTest(periodic=periodic):
                rhs, state, delta = self.problem()
                rhs.periodic_vertical = periodic
                reference = rhs(state)
                config = copy(rhs.config)
                config.case_number = 410
                tracer_rhs = RhsEuler(rhs.geom, rhs.ops, rhs.metric, rhs.ptopo, config)
                tracer_rhs.periodic_vertical = periodic
                tracer_rhs.damping = rhs.damping
                rhs = tracer_rhs
                profile = torch.linspace(0, 1, state[0].numel(), dtype=state.dtype).reshape_as(state[0])
                state = torch.cat((state, torch.stack((state[4], state[0] * profile * 1e-6))))
                delta = torch.cat((delta, torch.stack((delta[4], state[0] * (0.2 - profile) * 1e-6))))
                torch.testing.assert_close(rhs(state)[:5], reference, rtol=2e-12, atol=2e-10)
                full = rhs.linearize(state)
                partitions = rhs.linearize_partitions(state)
                torch.testing.assert_close(
                    rhs(state), rhs.explicit(state) + rhs.implicit(state), rtol=2e-12, atol=2e-10
                )
                for function, action in (
                    (rhs, full),
                    (rhs.explicit, partitions.explicit),
                    (rhs.implicit, partitions.implicit),
                ):
                    epsilon = 1e-3
                    expected = (function(state + epsilon * delta) - function(state - epsilon * delta)) / (2 * epsilon)
                    actual = action(delta).reshape_as(state)
                    for row in range(state.shape[0]):
                        scale = max(actual[row].abs().max().item(), 1e-15)
                        self.assertLess((actual[row] - expected[row]).abs().max().item() / scale, 2e-5)
                solution = partitions.solve(delta, 0.1, column_batch_size=3)
                residual = solution - 0.05 * partitions.implicit(solution).reshape_as(state) - delta
                for row in range(state.shape[0]):
                    self.assertLess(residual[row].abs().max().item(), 1e-10 * delta[row].abs().max().item())

    def test_periodic_column_solve_matches_dense_system(self):
        generator = torch.Generator().manual_seed(18)
        for elements in (1, 2, 5):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(elements=elements, dtype=dtype):
                    blocks = [0.1 * torch.randn(2, elements, 3, 3, generator=generator, dtype=dtype) for _ in range(3)]
                    lower, diagonal, upper = blocks
                    matrix = torch.zeros(2, 3 * elements, 3 * elements, dtype=dtype)
                    for e in range(elements):
                        row = slice(3 * e, 3 * (e + 1))
                        for neighbour, block in (
                            ((e - 1) % elements, lower),
                            (e, diagonal),
                            ((e + 1) % elements, upper),
                        ):
                            matrix[:, row, 3 * neighbour : 3 * (neighbour + 1)] += block[:, e]
                    forcing = torch.randn(2, elements, 3, generator=generator, dtype=dtype)
                    torch.testing.assert_close(
                        blocks_matvec(*blocks, forcing, periodic=True).flatten(1),
                        (matrix @ forcing.flatten(1)[..., None])[..., 0],
                    )
                    expected = torch.linalg.solve(
                        torch.eye(3 * elements, dtype=dtype) - 0.05 * matrix, forcing.flatten(1)
                    )
                    actual = cyclic_block_solve(*blocks, forcing, 0.1)
                    torch.testing.assert_close(actual.flatten(1), expected)

    def problem(self, nodes=3, vertical_elements=3):
        rhs, state, *_ = frozen_problem(
            elements=2, nodes=nodes, vertical_elements=vertical_elements, terrain_slope=0.2, speed=2
        )
        generator = torch.Generator().manual_seed(62)
        state[[0, 4]] *= 1 + 0.002 * torch.rand(state[[0, 4]].shape, generator=generator, dtype=state.dtype)
        frame_derivatives = torch.tensor(
            [[0.0, 0.03, 0.02], [-0.08, 0.0, -0.04], [0.12, -0.07, 0.0]], dtype=state.dtype
        )
        rhs.metric.volume.dG_cov[:, 0, 1:4] = frame_derivatives.reshape((3, 3) + (1,) * state[0].ndim)
        state[2] = 2
        state[3] = 0.7
        rhs.damping = (torch.full_like(state[0], 0.01), state[1:4].clone())
        delta = torch.randn(state.shape, generator=generator, dtype=state.dtype)
        delta[0] *= 0.01
        delta[4] *= 3
        delta[1:4] *= 0.01
        return rhs, state, delta

    def test_callable_partitions_keep_analytic_actions_and_batched_column_solve(self):
        rhs, state, delta = self.problem()
        reference = rhs.linearize_partitions(state)
        explicit = EulerRhsPartition(rhs, "explicit")
        implicit = EulerRhsPartition(rhs, "implicit")
        torch.testing.assert_close(explicit(state), rhs.explicit(state))
        torch.testing.assert_close(implicit(state), rhs.implicit(state))
        torch.testing.assert_close(explicit.linearize(state)(delta), reference.explicit(delta))
        with patch.object(rhs, "linearize", side_effect=AssertionError("Implicit action constructed J2")):
            action = implicit.linearize(state)
            self.assertIsNone(action.full)
            actual = action.solve(delta, 0.1, column_batch_size=2)
            self.assertIsNone(action._implicit)
            torch.testing.assert_close(actual, reference.solve(delta, 0.1))
            torch.testing.assert_close(action(delta), reference.implicit(delta))

    def test_vertical_partition_needs_no_panel_exchange(self):
        rhs, state, _ = self.problem()
        with patch.object(rhs.ptopo, "start_exchange_euler_3d", side_effect=AssertionError("Unexpected exchange")):
            rhs.implicit(state)
            EulerVerticalLinearization(rhs, state)

    def test_full_rhs_and_partitions_match_component_equations(self):
        rhs, state, _ = self.problem(nodes=5)
        fields = rhs.metric.volume
        scalar_faces, pressure_faces, energy_faces, velocity_faces, theta_faces = rhs._interface_data(
            rhs._faces_with_neighbours(rhs._extrapolate(state))
        )

        def derivative(value, trace, direction):
            return apply_op(value, rhs.derivatives[direction]) + apply_op(trace, rhs.corrections[direction])

        velocity = fields.raise_velocity(state[1:4])
        energy = (state[1:4] * velocity).sum(dim=0) / 2 + fields.geopotential
        pressure_gradient = [derivative(pressure(state[4]), pressure_faces[i], i) for i in range(3)]
        energy_gradient = [derivative(energy, energy_faces[i], i) for i in range(3)]
        wind_gradient = [[derivative(state[j + 1], velocity_faces[i][j], i) for j in range(3)] for i in range(3)]
        curl = torch.stack(
            (
                velocity[1] * (wind_gradient[0][1] - wind_gradient[1][0])
                + velocity[2] * (wind_gradient[0][2] - wind_gradient[2][0]),
                velocity[0] * (wind_gradient[1][0] - wind_gradient[0][1])
                + velocity[2] * (wind_gradient[1][2] - wind_gradient[2][1]),
                velocity[0] * (wind_gradient[2][0] - wind_gradient[0][2])
                + velocity[1] * (wind_gradient[2][1] - wind_gradient[1][2]),
            )
        )
        full = torch.zeros_like(state)
        implicit = torch.zeros_like(state)
        for i in range(3):
            flux = fields.volume_jacobian * state[[0, 4]] * velocity[i]
            tendency = -derivative(flux, scalar_faces[i], i) / fields.volume_jacobian
            if i == 2:
                theta = state[4] / state[0]
                theta_gradient = derivative(theta, theta_faces[2], 2)
                tendency[1] = theta * tendency[0] - state[0] * velocity[2] * theta_gradient
                implicit[[0, 4]] = tendency
            full[[0, 4]] += tendency
            full[i + 1] = -pressure_gradient[i] / state[0] - energy_gradient[i] + curl[i]
            full[i + 1] += sum(velocity[j] * fields.frame_vorticity[i, j] for j in range(3))
        rate, reference = rhs.damping
        full[1:4] -= rate * (state[1:4] - reference)
        implicit[3] = -pressure_gradient[2] / state[0] - energy_gradient[2]
        implicit[3] += velocity[0] * wind_gradient[2][0] + velocity[1] * wind_gradient[2][1]
        implicit[3] += sum(velocity[a] * fields.dG_cov[2, 0, a + 1] for a in range(2))
        torch.testing.assert_close(rhs(state), full, rtol=2e-12, atol=2e-10)
        torch.testing.assert_close(rhs.implicit(state), implicit, rtol=2e-12, atol=2e-10)
        torch.testing.assert_close(rhs.explicit(state), full - implicit, rtol=2e-12, atol=2e-10)

    def test_partition_jacobians_match_directional_differences(self):
        rhs, state, delta = self.problem()
        action = rhs.linearize_partitions(state)
        for function, derivative in ((rhs.implicit, action.implicit), (rhs.explicit, action.explicit)):
            actual = derivative(delta).reshape_as(state)
            errors = []
            for epsilon in (2e-2, 1e-2, 5e-3):
                expected = (function(state + epsilon * delta) - function(state - epsilon * delta)) / (2 * epsilon)
                errors.append((actual - expected).abs().max().item())
            self.assertLess(errors[-1] / actual.abs().max().item(), 2e-7)
            self.assertLess(errors[-1], 0.4 * errors[0])
        torch.testing.assert_close(
            action.implicit(delta) + action.explicit(delta), action.full(delta), rtol=2e-12, atol=2e-10
        )

    def test_column_solve_includes_horizontal_velocity_coupling(self):
        for elements in (1, 3):
            rhs, state, delta = self.problem(nodes=5, vertical_elements=elements)
            action = EulerVerticalLinearization(rhs, state)
            forcing = torch.zeros_like(delta)
            forcing[1:3] = delta[1:3]
            forcing[1] *= state[1].abs().mean()
            dt = 0.1
            original = forcing.clone()
            solution = action.solve(forcing, dt)
            torch.testing.assert_close(forcing, original, rtol=0, atol=0)
            residual = solution - dt / 2 * action(solution).reshape_as(state) - forcing
            self.assertLess(residual.abs().max().item(), 2e-11)
            torch.testing.assert_close(solution[1:3], forcing[1:3], rtol=0, atol=0)
            self.assertGreater(solution[[0, 3, 4]].abs().max().item(), 1e-6)

    def test_column_assembly_is_deferred_until_after_the_exponential_action(self):
        rhs, state, delta = self.problem()
        with patch(
            "wx_factory.jacobian.euler_vertical.EulerVerticalLinearization", side_effect=AssertionError("Assembly")
        ):
            action = rhs.linearize_partitions(state)
            action.explicit(delta)
        action.release_explicit()
        self.assertIsNone(action.full)

    def test_batched_column_solve_matches_complete_column_solve(self):
        for nodes in (3, 5):
            rhs, state, delta = self.problem(nodes=nodes)
            action = rhs.linearize_partitions(state)
            reference = action.implicit.solve(delta, 0.1)
            action.release_explicit()
            original = delta.clone()
            actual = action.solve(delta, 0.1, column_batch_size=7)
            torch.testing.assert_close(delta, original, rtol=0, atol=0)
            torch.testing.assert_close(actual, reference, rtol=2e-12, atol=2e-11)

    def test_reused_column_factors_preserve_stage_equations_after_step_changes(self):
        for periodic in (False, True):
            with self.subTest(periodic=periodic):
                rhs, state, delta = self.problem(nodes=2, vertical_elements=2)
                rhs.periodic_vertical = periodic
                reference = rhs.linearize_partitions(state).implicit
                action = rhs.linearize_partitions(state).column_operator(column_batch_size=3)
                torch.testing.assert_close(action(delta), reference(delta), rtol=2e-12, atol=2e-10)
                original = delta.clone()
                with patch("torch.linalg.inv_ex", wraps=torch.linalg.inv_ex) as factor:
                    for index, dt in enumerate((0.1, 0.1, 0.025, 0.025, 0.1)):
                        previous_count = factor.call_count
                        forcing = (index + 1) * delta
                        solution = action.solve(forcing, dt)
                        if index in (1, 3):
                            self.assertEqual(factor.call_count, previous_count)
                        else:
                            self.assertGreater(factor.call_count, previous_count)
                        residual = solution - dt / 2 * reference(solution).reshape_as(state) - forcing
                        self.assertLess(residual.abs().max().item(), 2e-10)
                        torch.testing.assert_close(solution[1:3], forcing[1:3], rtol=0, atol=0)
                torch.testing.assert_close(delta, original, rtol=0, atol=0)

    def test_column_factors_match_dense_solve_and_detect_singular_blocks(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        generator = torch.Generator().manual_seed(89)
        for device in devices:
            for dtype in (torch.float32, torch.float64):
                with self.subTest(device=device, dtype=dtype):
                    blocks = [
                        (0.1 * torch.randn(2, 3, 3, 3, generator=generator, dtype=dtype)).to(device) for _ in range(3)
                    ]
                    matrix = torch.zeros(2, 9, 9, dtype=torch.float64, device=device)
                    for e in range(3):
                        row = slice(3 * e, 3 * (e + 1))
                        matrix[:, row, row] = blocks[1][:, e].double()
                        if e:
                            matrix[:, row, 3 * (e - 1) : 3 * e] = blocks[0][:, e].double()
                        if e < 2:
                            matrix[:, row, 3 * (e + 1) : 3 * (e + 2)] = blocks[2][:, e].double()
                    factor = BlockThomasFactor(*blocks, 0.2, refine=device == "cuda" and dtype == torch.float32)
                    for _ in range(2):
                        forcing = torch.randn(2, 3, 3, generator=generator, dtype=dtype).to(device)
                        expected = torch.linalg.solve(
                            torch.eye(9, dtype=torch.float64, device=device) - 0.1 * matrix,
                            forcing.double().flatten(1),
                        )
                        actual = factor.solve(forcing).flatten(1)
                        torch.testing.assert_close(actual, expected.to(dtype))
                    singular = torch.eye(3, dtype=dtype, device=device).expand(2, 3, 3, 3).clone()
                    zeros = torch.zeros_like(singular)
                    with self.assertRaises(torch.linalg.LinAlgError):
                        BlockThomasFactor(zeros, singular, zeros, 2.0)

    def test_stiff_and_coupling_rows_partition_the_column_variables(self):
        for num_var in (5, 7):
            with self.subTest(num_var=num_var):
                rows = torch.cat((stiff_variable_rows(3, "cpu", num_var), coupling_variable_rows(3, "cpu")))
                torch.testing.assert_close(rows.sort().values, torch.arange(3 * num_var))

    def test_frame_gradients_survive_offloading_and_precision_conversion(self):
        rhs, state, _ = self.problem()
        fields = rhs.metric.volume
        vertical = fields.dG_cov[2, 0, 1:3].clone()
        horizontal = fields.dG_cov[:2, 0, 3].clone()
        expected = rhs.implicit(state) + rhs.explicit(state)
        Metric5D.offload_derivatives(rhs.metric)
        torch.testing.assert_close(fields.vertical_frame_gradient, vertical, rtol=0, atol=0)
        torch.testing.assert_close(fields.horizontal_frame_gradient, horizontal, rtol=0, atol=0)
        torch.testing.assert_close(rhs.implicit(state) + rhs.explicit(state), expected, rtol=0, atol=0)
        Metric5D.to_runtime(rhs.metric, torch.float32)
        torch.testing.assert_close(fields.vertical_frame_gradient, vertical.float(), rtol=0, atol=0)
        torch.testing.assert_close(fields.horizontal_frame_gradient, horizontal.float(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
