import unittest
from unittest.mock import patch

import torch

from tests.unit.rhs.euler_frozen_fixture import frozen_problem
from wx_factory.jacobian.euler import AcousticLinearization, _apply_dfr_derivative
from wx_factory.pde.euler_interfaces import EulerInterfaceGeometry, acoustic_interface


class EulerJacobianTestCase(unittest.TestCase):
    def test_cpu_actions_do_not_load_compiler(self):
        rhs, state, *_ = frozen_problem(elements=2, nodes=2)
        with (
            patch("wx_factory.jacobian.euler._compiled_acoustic_action", side_effect=AssertionError("CPU compilation")),
            patch("wx_factory.jacobian.euler._compiled_volume_actions", side_effect=AssertionError("CPU compilation")),
            patch("wx_factory.jacobian.euler._compiled_dfr_derivative", side_effect=AssertionError("CPU compilation")),
        ):
            actual = rhs.linearize(state)(state)
            self.assertTrue(torch.isfinite(actual).all())
            actual = rhs.linearize(state).explicit(state)
            self.assertTrue(torch.isfinite(actual).all())

    def test_dfr_cache_follows_precision_promotion(self):
        rhs, _, *_ = frozen_problem(elements=2, nodes=3)
        rhs.derivatives = tuple(operator.float() for operator in rhs.derivatives)
        rhs.corrections = tuple(operator.float() for operator in rhs.corrections)
        original = rhs.dfr_derivatives(_apply_dfr_derivative)
        promoted = rhs.double_precision_rhs()
        for operator, previous in zip(promoted.dfr_derivatives(_apply_dfr_derivative), original):
            self.assertIsNot(operator, previous)
            self.assertEqual(operator.volume_operator.dtype, torch.float64)
            self.assertEqual(operator.face_operator.dtype, torch.float64)
            self.assertEqual(operator.weights.dtype, torch.float64)
            self.assertEqual(operator.face_weights.dtype, torch.float64)

    def test_zero_normal_flow_uses_central_tangent_and_left_scalar_branch(self):
        inverse = torch.eye(3, dtype=torch.float64)
        left = torch.tensor([1.0, 0.0, 2.0, 3.0, 300.0], dtype=torch.float64)
        right = left.clone()
        right[0], right[2] = 1.3, 5
        delta_left = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float64)
        delta_right = torch.tensor([0.2, -0.1, 0.6, -0.2, 0.3], dtype=torch.float64)
        for wall in (False, True):
            geometry = EulerInterfaceGeometry(
                None,
                None,
                inverse,
                torch.tensor(0.0),
                torch.tensor(1.0),
                torch.tensor(1.0),
                inverse[0],
                torch.tensor(wall),
            )
            action = AcousticLinearization(left, right, geometry, 0)
            self.assertTrue(action.upwind)
            normal, _, _, star, flux = action(delta_left, delta_right)
            torch.testing.assert_close(star[1:], (delta_left[2:4] + delta_right[2:4]) / 2, rtol=0, atol=0)
            torch.testing.assert_close(flux, normal * left[[0, 4]], rtol=0, atol=0)
            if wall:
                self.assertEqual(star[0].item(), 0.0)
                self.assertEqual(action(delta_left, delta_right)[0].item(), 0.0)

    def test_acoustic_action_matches_autograd_on_each_branch(self):
        inverse = torch.tensor([[1.2, 0.1, 0.05], [0.1, 0.9, 0.02], [0.05, 0.02, 1.1]], dtype=torch.float64)
        generator = torch.Generator().manual_seed(16)
        for speed in (20.0, -20.0, 0.0, 100.0, 800.0):
            for wall in (False, True):
                left = torch.tensor([1.2, speed, 3.0, -2.0, 360.0], dtype=torch.float64)
                right = torch.tensor([1.1, speed + 2, 2.0, -1.0, 330.0], dtype=torch.float64)
                if speed == 0:
                    left[1:4] = right[1:4] = 0
                delta_left = torch.randn(5, generator=generator, dtype=torch.float64)
                delta_right = torch.randn(5, generator=generator, dtype=torch.float64)
                scale = torch.sqrt(inverse[0, 0])
                row = inverse[0] / scale
                mask = torch.tensor(wall)
                geometry = EulerInterfaceGeometry(
                    None, None, inverse, torch.tensor(12.0), torch.tensor(1.0), scale, row, mask
                )
                linearization = AcousticLinearization(left, right, geometry, 0, mach_cutoff=0.1)

                def evaluate(left_state, right_state, mask=mask):
                    normal, pressure, energy, velocity = acoustic_interface(
                        left_state,
                        right_state,
                        inverse,
                        0,
                        torch.tensor(12.0),
                        wall=mask,
                        return_velocity=True,
                        mach_cutoff=0.1,
                    )
                    scalar = torch.where(normal >= 0, left_state, right_state)
                    return normal, pressure, energy, velocity, normal * scalar[[0, 4]]

                _, expected = torch.autograd.functional.jvp(evaluate, (left, right), (delta_left, delta_right))
                actual = linearization(delta_left, delta_right)
                velocity = linearization(delta_left, delta_right, velocity_only=True)
                torch.testing.assert_close(velocity, actual[3], rtol=0, atol=0)
                for actual_value, expected_value in zip(actual, expected):
                    torch.testing.assert_close(actual_value, expected_value, rtol=2e-13, atol=2e-10)

    def test_full_action_matches_centered_differences_and_is_linear(self):
        rhs, state, positions, _, _, _ = frozen_problem(elements=3, length=20000.0)
        positions = positions[:, None, :].expand(3, 16, 4).reshape(1, 1, 3, 64)
        state[2] = 2 + 0.2 * torch.sin(positions / 3000.0)
        state[3] = 0.5 + 0.1 * torch.cos(positions / 3000.0)
        state[0] *= 1 + 0.01 * torch.sin(positions / 3000.0)
        state[4] *= 1 + 0.01 * torch.cos(positions / 3000.0)
        generator = torch.Generator().manual_seed(33)
        direction = torch.randn(state.shape, generator=generator, dtype=torch.float64)
        direction[1:4] *= 0.01
        direction[0] *= 0.1
        direction[4] *= 10
        action = rhs.linearize(state)
        actual = action(direction).reshape_as(state)
        saved = actual.clone()
        action.explicit(-direction)
        torch.testing.assert_close(action(direction).reshape_as(state), saved, rtol=0, atol=0)
        torch.testing.assert_close(actual, saved, rtol=0, atol=0)
        errors = []
        for epsilon in (1.6e-3, 4e-4, 1e-4):
            expected = (rhs(state + epsilon * direction) - rhs(state - epsilon * direction)) / (2 * epsilon)
            errors.append((expected - actual).abs().max().item())
        self.assertLess(errors[-1] / actual.abs().max().item(), 1e-5)
        self.assertLess(errors[-1], 0.4 * errors[0])
        torch.testing.assert_close(action(2 * direction), 2 * action(direction), rtol=2e-13, atol=1e-12)
        torch.testing.assert_close(
            action(torch.zeros_like(direction)), torch.zeros_like(actual).flatten(), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
