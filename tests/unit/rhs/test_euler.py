import tempfile
import unittest

import torch

from wx_factory.common import Configuration, load_default_schema, readfile
from wx_factory.common.definitions import Rd, gravity, heat_capacity_ratio
from wx_factory.init.euler import deep_rest_pressure
from wx_factory.output.state import load_state, save_state
from wx_factory.pde.euler_interfaces import acoustic_interface, pressure


def principal_symbol(state: torch.Tensor, inverse_spatial: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """Return the frozen principal symbol for Q=(rho,v_i,rho*theta).

    The relative curl contributes to the principal part. ``normal`` is a
    covector with the same trailing point dimensions as state.
    """
    contravariant_velocity = torch.einsum("ij...,j...->i...", inverse_spatial, state[1:4])
    raised_normal = torch.einsum("ij...,j...->i...", inverse_spatial, normal)
    normal_advection_speed = (normal * contravariant_velocity).sum(dim=0)
    symbol = state.new_zeros((5, 5) + state.shape[1:])
    for i in range(5):
        symbol[i, i] = normal_advection_speed
    symbol[0, 1:4] = state[0] * raised_normal
    symbol[4, 1:4] = state[4] * raised_normal
    symbol[1:4, 4] = (heat_capacity_ratio * pressure(state[4]) / (state[0] * state[4])) * normal
    return symbol


class EulerTestCase(unittest.TestCase):
    def setUp(self):
        self.state = torch.tensor([1.1, 3.0, -2.0, 1.0, 330.0], dtype=torch.float64)
        basis = torch.tensor([[2.0, 0.3, 0.1], [0.0, 1.5, 0.2], [0.0, 0.0, 0.8]], dtype=torch.float64)
        self.basis = basis
        self.inverse = torch.linalg.inv(basis.T @ basis)

    def test_complete_principal_symbol_wave_speeds(self):
        normal = self.state.new_tensor([0.4, -0.7, 0.2])
        matrix = principal_symbol(self.state, self.inverse, normal)
        speed = normal @ self.inverse @ self.state[1:4]
        acoustic = torch.sqrt(
            heat_capacity_ratio * pressure(self.state[4]) / self.state[0] * (normal @ self.inverse @ normal)
        )
        expected = torch.stack([speed - acoustic, speed, speed, speed, speed + acoustic])
        eigenvalues = torch.linalg.eigvals(matrix)
        torch.testing.assert_close(eigenvalues.imag, torch.zeros_like(eigenvalues.imag), atol=1e-11, rtol=0)
        torch.testing.assert_close(eigenvalues.real.sort().values, expected, atol=1e-10, rtol=1e-12)

    def test_acoustic_consistency_and_wall(self):
        for direction in range(3):
            normal, traction, energy = acoustic_interface(
                self.state, self.state, self.inverse, direction, self.state.new_tensor(12.0)
            )
            torch.testing.assert_close(normal, (self.inverse @ self.state[1:4])[direction])
            torch.testing.assert_close(traction, pressure(self.state[4]))
            torch.testing.assert_close(energy, self.state[1:4] @ self.inverse @ self.state[1:4] / 2 + 12)
            normal, _, _ = acoustic_interface(
                self.state, self.state, self.inverse, direction, self.state.new_tensor(12.0), wall=torch.tensor(True)
            )
            self.assertEqual(normal.item(), 0.0)

    def test_coordinate_acoustic_characteristics(self):
        right = self.state.clone()
        right[0] *= 1.03
        right[1:4] += self.state.new_tensor([0.7, -0.4, 0.2])
        right[4] *= 1.001
        pressure_left, pressure_right = pressure(self.state[4]), pressure(right[4])
        sound_left = torch.sqrt(heat_capacity_ratio * pressure_left / self.state[0])
        sound_right = torch.sqrt(heat_capacity_ratio * pressure_right / right[0])
        impedance = (self.state[0] * sound_left + right[0] * sound_right) / 2
        mach = max(
            torch.sqrt(self.state[1:4] @ self.inverse @ self.state[1:4]) / sound_left,
            torch.sqrt(right[1:4] @ self.inverse @ right[1:4]) / sound_right,
        )
        for direction in range(3):
            covector = torch.zeros(3, dtype=self.state.dtype)
            covector[direction] = 1
            symbol = principal_symbol(self.state, self.inverse, covector)
            projection = torch.zeros((2, 5), dtype=self.state.dtype)
            projection[0, 1:4] = self.inverse[direction]
            projection[1, 4] = heat_capacity_ratio * pressure_left / self.state[4]
            velocity_left = (self.inverse @ self.state[1:4])[direction]
            velocity_right = (self.inverse @ right[1:4])[direction]
            acoustic = self.state.new_tensor(
                [
                    [velocity_left, self.inverse[direction, direction] / self.state[0]],
                    [heat_capacity_ratio * pressure_left, velocity_left],
                ]
            )
            torch.testing.assert_close(projection @ symbol, acoustic @ projection)
            velocity_common, pressure_common, _ = acoustic_interface(
                self.state, right, self.inverse, direction, self.state.new_tensor(12.0)
            )
            normal_scale = torch.sqrt(self.inverse[direction, direction])
            limited_mach = min(1.0, max(mach, 0.3))
            pressure_scale = limited_mach * (2 - limited_mach)
            torch.testing.assert_close(
                velocity_common,
                (velocity_left + velocity_right) / 2
                - normal_scale * (pressure_right - pressure_left) / (2 * impedance),
            )
            torch.testing.assert_close(
                pressure_common,
                (pressure_left + pressure_right) / 2
                - pressure_scale * impedance * (velocity_right - velocity_left) / (2 * normal_scale),
            )

    def test_low_mach_pressure_scaling_keeps_unscaled_normal_velocity(self):
        from wx_factory.common.definitions import Rd, p0

        inverse = torch.eye(3, dtype=torch.float64)
        corrections = []
        for mach in (1e-2, 1e-3, 1e-4):
            pressure_left = 1 / (heat_capacity_ratio * mach**2)
            left = torch.tensor([1.0, 0.8, 0.3, 0.0, 0.0], dtype=torch.float64)
            right = torch.tensor([1.0, 1.2, -0.2, 0.0, 0.0], dtype=torch.float64)
            left[4] = right[4] = p0 / Rd * (pressure_left / p0) ** (1 / heat_capacity_ratio)
            _, common_pressure, _ = acoustic_interface(left, right, inverse, 0, left.new_tensor(0), mach_cutoff=1e-6)
            pressure_correction = pressure(left[4]) - common_pressure
            right[4] = p0 / Rd * ((pressure_left + 0.1) / p0) ** (1 / heat_capacity_ratio)
            normal, _, _ = acoustic_interface(left, right, inverse, 0, left.new_tensor(0), mach_cutoff=1e-6)
            impedance = (
                torch.sqrt(heat_capacity_ratio * pressure(left[4]) * left[0])
                + torch.sqrt(heat_capacity_ratio * pressure(right[4]) * right[0])
            ) / 2
            expected_normal = (left[1] + right[1]) / 2 - (pressure(right[4]) - pressure(left[4])) / (2 * impedance)
            torch.testing.assert_close(normal, expected_normal, rtol=0, atol=0)
            corrections.append(pressure_correction)
        for correction in corrections[1:]:
            torch.testing.assert_close(correction, corrections[0], rtol=1e-2, atol=1e-8)

    def test_star_velocity_respects_metric_normal_tangent_and_energy(self):
        right = self.state.clone()
        right[1:4] += self.state.new_tensor([0.7, -0.4, 0.2])
        right[4] *= 1.001
        for direction in range(3):
            for wall in (False, True):
                normal, _, energy, star = acoustic_interface(
                    self.state,
                    right,
                    self.inverse,
                    direction,
                    self.state.new_tensor(12.0),
                    wall=torch.tensor(wall),
                    return_velocity=True,
                )
                torch.testing.assert_close((self.inverse @ star)[direction], normal)
                mean_velocity = (self.state[1:4] + right[1:4]) / 2
                # A correction along the normal covector preserves tangent projections.
                tangent = [i for i in range(3) if i != direction]
                torch.testing.assert_close(star[tangent], mean_velocity[tangent])
                unit_normal = self.basis @ self.inverse[:, direction] / torch.sqrt(self.inverse[direction, direction])
                physical_change = self.basis @ self.inverse @ (star - mean_velocity)
                tangential_change = physical_change - unit_normal * (unit_normal @ physical_change)
                torch.testing.assert_close(tangential_change, torch.zeros_like(tangential_change), atol=2e-14, rtol=0)
                torch.testing.assert_close(energy, star @ self.inverse @ star / 2 + 12.0)
                if wall:
                    self.assertEqual(normal.item(), 0.0)

    def test_common_velocity_and_energy_are_smooth_at_zero_normal_flow(self):
        from wx_factory.jacobian.euler import AcousticLinearization
        from wx_factory.pde.euler_interfaces import EulerInterfaceGeometry

        for direction in range(3):
            scale = self.inverse[direction, direction].sqrt()
            row = self.inverse[direction] / scale
            left = self.state.clone()
            left[1 + direction] -= (row @ left[1:4]) / scale
            right = left.clone()
            right[1:4] *= 2
            geometry = EulerInterfaceGeometry(
                None, None, self.inverse, left.new_tensor(0), left.new_tensor(1), scale, row, None
            )
            perturbation = torch.zeros_like(left)
            perturbation[1 + direction] = 1 / scale
            action = AcousticLinearization(left, right, geometry, direction)
            _, _, delta_energy, delta_velocity, _ = action(perturbation, perturbation)
            for epsilon in (1e-3, 1e-5):
                traces = []
                for sign in (-1, 1):
                    traces.append(
                        acoustic_interface(
                            left + sign * epsilon * perturbation,
                            right + sign * epsilon * perturbation,
                            self.inverse,
                            direction,
                            left.new_tensor(0),
                            return_velocity=True,
                        )
                    )
                energy_derivative = (traces[1][2] - traces[0][2]) / (2 * epsilon)
                velocity_derivative = (traces[1][3] - traces[0][3]) / (2 * epsilon)
                torch.testing.assert_close(energy_derivative, delta_energy, atol=2e-9, rtol=1e-8)
                torch.testing.assert_close(velocity_derivative, delta_velocity, atol=2e-9, rtol=1e-8)

    def test_deep_pressure_satisfies_hydrostatic_ode(self):
        with torch.inference_mode(False), torch.enable_grad():
            height = torch.tensor([0.0, 1000.0, 10000.0, 25000.0], dtype=torch.float64, requires_grad=True)
            radius = 6371220.0 / 125
            profile = deep_rest_pressure(height, radius)
            derivative = torch.autograd.grad(profile.sum(), height)[0]
            density = profile / (Rd * (300.0 - 0.0065 * height))
            torch.testing.assert_close(
                derivative, -density * gravity * (radius / (radius + height)) ** 2, atol=2e-13, rtol=2e-13
            )

    def test_restart_records_runtime_state_representation(self):
        for filename, representation in (
            ("config/dcmip21.ini", "relative_covariant"),
            ("config/case5.ini", "relative_covariant"),
        ):
            config = Configuration(readfile(filename), load_default_schema())
            with tempfile.NamedTemporaryFile() as saved:
                save_state(self.state, config, saved.name)
                restored, metadata = load_state(saved.name)
            torch.testing.assert_close(restored, self.state, rtol=0, atol=0)
            self.assertEqual(metadata.state_representation, representation)

    def test_previous_covariant_restart_is_accepted_and_conservative_euler_is_rejected(self):
        import numpy

        config = Configuration(readfile("config/dcmip21.ini"), load_default_schema())
        for version in (1, 2):
            with self.subTest(version=version), tempfile.NamedTemporaryFile() as saved:
                numpy.save(saved, self.state.numpy())
                saved.write(f"{version}\n".encode())
                if version == 2:
                    saved.write(b"lamb\n")
                saved.write(config.schema.raw_string.encode())
                saved.write(b"----- config_content ------\n")
                content = config.config_content
                if version == 2:
                    content = content.replace("jacobian = analytic", "lamb_jacobian = analytic")
                saved.write(content.encode())
                saved.flush()
                if version == 1:
                    with self.assertRaisesRegex(ValueError, "conservative momentum"):
                        load_state(saved.name)
                else:
                    restored, metadata = load_state(saved.name)
                    torch.testing.assert_close(restored, self.state, rtol=0, atol=0)
                    self.assertEqual(metadata.state_representation, "relative_covariant")
                    self.assertEqual(metadata.jacobian, "analytic")
