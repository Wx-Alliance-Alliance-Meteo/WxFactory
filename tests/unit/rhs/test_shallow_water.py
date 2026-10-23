import tempfile
import unittest

import numpy
import torch

from wx_factory.common import Configuration, load_default_schema, readfile
from wx_factory.geometry.metric2d import HorizontalMetric
from wx_factory.output.state import load_state
from wx_factory.pde.shallow_water_interfaces import GravityWaveInterface


class ShallowWaterInterfaceTestCase(unittest.TestCase):
    def test_gravity_wave_matching_and_central_tangential_trace(self):
        from wx_factory.common.definitions import gravity

        covariant = torch.tensor([[2.0, 0.6], [0.6, 1.5]], dtype=torch.float64)
        metric = HorizontalMetric(
            covariant,
            torch.linalg.inv(covariant),
            torch.sqrt(torch.det(covariant)),
            torch.tensor(0.0, dtype=torch.float64),
        )
        left = torch.tensor([12.0, 0.8, -0.3], dtype=torch.float64)
        right = torch.tensor([11.0, -0.2, 0.5], dtype=torch.float64)
        bottom = 2.0
        for direction in range(2):
            interface = GravityWaveInterface(left, right, metric, bottom, direction)
            c = interface.wave_speed
            for sign, state in ((1, left), (-1, right)):
                normal = (metric.contravariant[direction] @ state[1:]) / interface.normal_scale
                torch.testing.assert_close(
                    interface.normal_velocity + sign * gravity * interface.surface / c,
                    normal + sign * gravity * state[0] / c,
                )
            torch.testing.assert_close((interface.normal_row * interface.velocity).sum(), interface.normal_velocity)
            self.assertEqual(interface.velocity[1 - direction], (left[2 - direction] + right[2 - direction]) / 2)
            delta_left = left.new_tensor([0.2, -0.1, 0.3])
            delta_right = right.new_tensor([-0.1, 0.4, 0.1])
            analytic = interface.linearize(delta_left, delta_right)
            epsilon = 1e-4
            plus = GravityWaveInterface(
                left + epsilon * delta_left, right + epsilon * delta_right, metric, bottom, direction
            ).common_values()
            minus = GravityWaveInterface(
                left - epsilon * delta_left, right - epsilon * delta_right, metric, bottom, direction
            ).common_values()
            torch.testing.assert_close(analytic, (plus - minus) / (2 * epsilon), rtol=2e-8, atol=1e-9)

    def test_restart_rejects_legacy_momentum(self):
        config = Configuration(readfile("config/case5.ini"), load_default_schema())
        state = numpy.array([[12.0], [0.8], [-0.3]])
        with tempfile.NamedTemporaryFile() as saved:
            numpy.save(saved, state)
            saved.write(b"1\n")
            saved.write(config.schema.raw_string.encode())
            saved.write(b"----- config_content ------\n")
            saved.write(config.config_content.encode())
            saved.flush()
            with self.assertRaisesRegex(ValueError, "Legacy shallow-water"):
                load_state(saved.name)
