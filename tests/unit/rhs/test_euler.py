import tempfile
import unittest

import torch

from wx_factory.common import Configuration, load_default_schema, readfile
from wx_factory.common.definitions import Rd, gravity, heat_capacity_ratio
from wx_factory.init.euler import deep_rest_pressure
from wx_factory.output.state import load_state, save_state
from wx_factory.pde.euler_interfaces import acoustic_interface, pressure, principal_symbol


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
            coefficient = torch.sqrt(self.inverse[direction, direction]) / impedance
            torch.testing.assert_close(
                velocity_common + coefficient * pressure_common, velocity_left + coefficient * pressure_left
            )
            torch.testing.assert_close(
                velocity_common - coefficient * pressure_common, velocity_right - coefficient * pressure_right
            )

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
                upwind = self.state[1:4] if normal >= 0 else right[1:4]
                # A correction along the normal covector preserves tangent projections.
                tangent = [i for i in range(3) if i != direction]
                torch.testing.assert_close(star[tangent], upwind[tangent])
                unit_normal = self.basis @ self.inverse[:, direction] / torch.sqrt(self.inverse[direction, direction])
                physical_change = self.basis @ self.inverse @ (star - upwind)
                tangential_change = physical_change - unit_normal * (unit_normal @ physical_change)
                torch.testing.assert_close(tangential_change, torch.zeros_like(tangential_change), atol=2e-14, rtol=0)
                torch.testing.assert_close(energy, star @ self.inverse @ star / 2 + 12.0)
                if wall:
                    self.assertEqual(normal.item(), 0.0)

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
            ("config/case5.ini", "conservative"),
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
                    content = content.replace("riemann_solver = acoustic", "lamb_riemann_solver = acoustic")
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
                    self.assertEqual(metadata.riemann_solver, "acoustic")
