import unittest
from unittest.mock import patch

import torch

from tests.unit.rhs.euler_frozen_fixture import frozen_problem
from wx_factory.common.matmul import apply_op
from wx_factory.geometry.metric5d import Metric5D
from wx_factory.jacobian.block_solve import blocks_matvec, cyclic_block_solve
from wx_factory.jacobian.euler_vertical import EulerVerticalLinearization
from wx_factory.pde.euler_interfaces import pressure
from wx_factory.rhs.euler_partition import EulerRhsPartition


class EulerPartitionTestCase(unittest.TestCase):
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
        scalar_faces, pressure_faces, energy_faces, velocity_faces = rhs._interface_data(
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
            full[[0, 4]] += tendency
            if i == 2:
                implicit[[0, 4]] = tendency
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
