import unittest

import torch

from tests.unit.rhs.euler_frozen_fixture import (
    central_shear_matrix,
    frozen_problem,
    transverse_direction,
)


class EulerOperatorTestCase(unittest.TestCase):
    def test_two_scalar_fluxes_preserve_constant_potential_temperature_transport(self):
        rhs, state, positions, _, _, _ = frozen_problem(elements=3)
        profile = torch.sin(positions / 3000).repeat(1, 16).reshape_as(state[0])
        state[0] = 1 + 0.01 * profile
        state[4] = 300 * state[0]
        faces = rhs._faces_with_neighbours(rhs._extrapolate(state))
        scalar_fluxes = rhs._interface_data(faces)[0]
        self.assertTrue(all(flux.shape[0] == 2 for flux in scalar_fluxes))
        result = rhs(state)
        self.assertGreater(result[0].abs().max().item(), 0)
        torch.testing.assert_close(result[4], 300 * result[0], rtol=2e-10, atol=2e-11)

    def test_curl_requires_common_riemann_velocity_traces(self):
        rhs, state, _, _, _, _ = frozen_problem(elements=3)
        velocity = rhs.metric.volume.raise_velocity(state[1:4])
        self.assertFalse(hasattr(rhs, "local_derivatives"))
        with self.assertRaises(TypeError):
            rhs.relative_covariant_curl(state[1:4])
        with self.assertRaisesRegex(ValueError, "Riemann solver"):
            rhs.relative_vorticity_tendency(state[1:4], velocity, None)
        with self.assertRaisesRegex(ValueError, "Riemann solver"):
            rhs.relative_vorticity_tendency(state[1:4], velocity, None, explicit=True)

    def test_double_jacobian_rhs_matches_original_operator_and_preserves_state(self):
        rhs, state, positions, _, _, _ = frozen_problem(elements=3)
        state += transverse_direction(torch.sin(2 * torch.pi * positions / 4000), state)
        working = state.to(torch.float32)
        saved = working.clone()
        expected = rhs(working.to(torch.float64))
        actual = rhs.jacobian_rhs(working)
        self.assertEqual(actual.dtype, torch.float64)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(working, saved, rtol=0, atol=0)
        cached = rhs._double_rhs
        rhs.jacobian_rhs(working)
        self.assertIs(rhs._double_rhs, cached)

    def test_assembled_transverse_symbol_matches_periodic_central_advection(self):
        for speed in (-20.0, 20.0):
            with self.subTest(speed=speed):
                rhs, state, positions, _, _, half_width = frozen_problem(elements=6, speed=speed)
                profile = torch.sin(2 * torch.pi * positions / 4000.0)
                direction = transverse_direction(profile, state)
                epsilon = 1e-2  # Avoid cancellation of the pressure contribution.
                action = (rhs(state + epsilon * direction) - rhs(state - epsilon * direction)) / (2 * epsilon)
                matrix = central_shear_matrix(rhs, 6, speed, half_width)
                expected = transverse_direction((matrix @ profile.flatten()).reshape_as(profile), state)[2]
                torch.testing.assert_close(action[2], expected, rtol=1e-9, atol=1e-10)
                for component in (0, 1, 3, 4):
                    torch.testing.assert_close(
                        action[component], torch.zeros_like(action[component]), rtol=0, atol=1e-9
                    )
                changed = direction.clone()
                changed[2, ..., -1, :] *= 2
                changed_action = (rhs(state + epsilon * changed) - rhs(state - epsilon * changed)) / (2 * epsilon)
                self.assertGreater((changed_action[2, ..., 0, :] - action[2, ..., 0, :]).abs().max().item(), 1e-5)

    def test_linearized_acoustic_operator_at_rest(self):
        from wx_factory.common.definitions import heat_capacity_ratio
        from wx_factory.pde.euler_interfaces import pressure

        rhs, state, positions, _, _, half_width = frozen_problem(elements=6, speed=0.0)
        state[4] = 1
        sound = torch.sqrt(heat_capacity_ratio * pressure(state[4, 0, 0, 0, 0])).item()
        profile = torch.sin(2 * torch.pi * positions / 4000.0)
        scalar = transverse_direction(profile, state)[2]
        left, right = profile @ rhs.ops.extrap_west, profile @ rhs.ops.extrap_east
        mean = (right + left.roll(-1, dims=0)) / 2
        cutoff = rhs.config.riemann_mach_cutoff
        pressure_scale = cutoff * (2 - cutoff)
        for sign in (-1, 1):
            direction = torch.zeros_like(state)
            direction[0] = scalar / sound
            direction[1] = sign * half_width * scalar
            direction[4] = scalar / sound
            velocity_face = sign * mean - (left.roll(-1, dims=0) - right) / 2
            pressure_face = sound * (mean - pressure_scale * sign * (left.roll(-1, dims=0) - right) / 2)
            pressure_faces = torch.stack((pressure_face.roll(1, dims=0), pressure_face), dim=-1)
            velocity_faces = torch.stack((velocity_face.roll(1, dims=0), velocity_face), dim=-1)
            divergence = (sign * profile @ rhs.ops.diff_solpt_tr + velocity_faces @ rhs.ops.correction_tr) / half_width
            gradient = sound * profile @ rhs.ops.diff_solpt_tr + pressure_faces @ rhs.ops.correction_tr
            expected = torch.zeros_like(state)
            expected[0] = transverse_direction(-divergence, state)[2]
            expected[1] = transverse_direction(-gradient, state)[2]
            expected[4] = expected[0]
            action = rhs.linearize(state)(direction).reshape_as(state)
            torch.testing.assert_close(action, expected, rtol=2e-10, atol=2e-7)

    def test_lifted_shear_is_bounded_and_converges_in_exact_semidiscrete_solution(self):
        errors = []
        for elements in (6, 12, 24):
            rhs, state, positions, _, weights, _ = frozen_problem(elements=elements, length=20000.0)
            initial = torch.sin(2 * torch.pi * positions / 4000.0)
            # Assemble the production Jacobian on transverse basis vectors; no
            # time integrator or test-built flux can mask a spatial instability.
            action = rhs.linearize(state)
            basis = torch.eye(initial.numel(), dtype=state.dtype)
            columns = []
            for profile in basis:
                direction = transverse_direction(profile.reshape_as(initial), state)
                tendency = action(direction).reshape_as(state)[2]
                columns.append(tendency.reshape(elements, -1, initial.shape[-1])[:, 0].flatten())
            matrix = torch.stack(columns, dim=1)
            mass_weights = weights.repeat(elements)
            weighted_matrix = mass_weights[:, None] * matrix
            torch.testing.assert_close(
                weighted_matrix + weighted_matrix.T, torch.zeros_like(matrix), atol=1e-12, rtol=0
            )
            evolved = (torch.matrix_exp(300.0 * matrix) @ initial.flatten()).reshape_as(initial)
            reference = torch.sin(2 * torch.pi * (positions - 6000.0) / 4000.0)
            norm = lambda value, weights=weights, elements=elements: torch.sqrt(
                (weights * value**2).sum() / (weights.sum() * elements)
            )
            self.assertLessEqual(norm(evolved).item(), norm(initial).item() * (1 + 1e-8))
            errors.append(norm(evolved - reference).item())
        self.assertLess(errors[1], errors[0] / 4)
        self.assertLess(errors[2], errors[1] / 4)
        self.assertLess(errors[2], 0.01)

    def test_complete_rhs_matches_smooth_polynomial_momentum(self):
        # Three interior points plus endpoints reproduce the degree-four kinetic energy.
        # Periodic velocity traces are continuous, so the curl is also exact.
        rhs, state, _, points, _, _ = frozen_problem(elements=1, nodes=3, length=2.0, speed=0.0)
        # Unit scalars give exactly constant log traces, isolating polynomial differentiation.
        state[4] = 1
        _, y, x = torch.meshgrid(points, points, points, indexing="ij")
        velocity_x = 0.2 + 0.1 * y * y
        velocity_y = 0.3 + 0.05 * x * x
        state[1] = velocity_x.flatten()
        state[2] = velocity_y.flatten()
        result = rhs(state)
        expected = torch.zeros_like(state)
        expected[1] = (-0.2 * y * velocity_y).flatten()
        expected[2] = (-0.1 * x * velocity_x).flatten()
        torch.testing.assert_close(result, expected, rtol=2e-9, atol=5e-10)
