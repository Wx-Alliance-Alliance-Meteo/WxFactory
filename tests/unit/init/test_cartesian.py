import math
import unittest
from types import SimpleNamespace

import torch

from tests.unit.cartesian_fixture import cartesian_config
from wx_factory.common.definitions import gravity


class CartesianInitializationTestCase(unittest.TestCase):
    def test_density_current_temperature_anomaly_at_background_pressure(self):
        from wx_factory.common.definitions import Rd, cpd, heat_capacity_ratio, p0
        from wx_factory.init.initialize import initialize_cartesian3d

        x = torch.tensor([0.0, 2000.0, 4000.0, 10000.0], dtype=torch.float64)
        z = torch.full_like(x, 3000.0)
        geometry = SimpleNamespace(X1=x, X3=z, grid_shape_3d_new=x.shape)
        metric = SimpleNamespace(volume=SimpleNamespace(covariant_from_physical=lambda velocity: velocity))
        config = cartesian_config(4)
        state = initialize_cartesian3d(geometry, metric, config)
        exner = 1 - gravity * z / (cpd * config.bubble_theta)
        pressure = p0 * (Rd * state[4] / p0) ** heat_capacity_ratio
        temperature = pressure / (Rd * state[0])
        anomaly = torch.tensor([-15.0, -7.5, 0.0, 0.0], dtype=x.dtype)
        torch.testing.assert_close(pressure, p0 * exner ** (cpd / Rd), rtol=1e-13, atol=0)
        torch.testing.assert_close(temperature, config.bubble_theta * exner + anomaly, rtol=1e-13, atol=0)
        torch.testing.assert_close(state[4] / state[0], config.bubble_theta + anomaly / exner)
        self.assertTrue(bool((state[1:4] == 0).all()))

    def test_gresho_pressure_balance_and_equation_of_state(self):
        from wx_factory.common.definitions import Rd, heat_capacity_ratio, p0
        from wx_factory.geometry.cartesian_metric5d import CartesianMetric5D
        from wx_factory.init.initialize import initialize_cartesian3d

        config = cartesian_config(651)
        # Differentiate inside each smooth branch, away from the two break radii.
        radius = torch.tensor([0.1, 0.2, 0.3, 0.45], dtype=torch.float64, requires_grad=True)
        geometry = SimpleNamespace(
            X1=0.5 + radius,
            X3=torch.full_like(radius, 0.5),
            grid_shape_3d_new=radius.shape,
            delta_x1=1.0,
            delta_x2=0.1,
            ztop=1.0,
            num_elements_x3=1,
        )
        metric = CartesianMetric5D(geometry, config)
        metric.volume = metric.evaluate(geometry.X3)
        self.assertEqual(torch.count_nonzero(metric.volume.geopotential).item(), 0)
        self.assertEqual(torch.count_nonzero(metric.volume.dG_cov).item(), 0)
        for mach in (0.01, 0.1, 0.5):
            with self.subTest(mach=mach):
                config.gresho_mach = mach
                state = initialize_cartesian3d(geometry, metric, config)
                pressure = p0 * (Rd * state[4] / p0) ** heat_capacity_ratio
                gradient = torch.autograd.grad(pressure.sum(), radius)[0]
                wind = metric.volume.physical_velocity(state[1:4])
                torch.testing.assert_close(gradient, state[0] * wind[2] ** 2 / radius, atol=1e-11, rtol=1e-11)
                sound_speed = torch.sqrt(heat_capacity_ratio * pressure / state[0])
                maximum_mach = (torch.linalg.vector_norm(wind, dim=0) / sound_speed).max()
                torch.testing.assert_close(maximum_mach, maximum_mach.new_tensor(mach), atol=1e-12, rtol=1e-12)
                expected = (
                    1 / (heat_capacity_ratio * mach**2)
                    - 0.5
                    + torch.tensor(
                        [0.125, 0.5, 4 * math.log(1.5) + 4 - 6 + 1.125, 4 * math.log(2) - 2], dtype=pressure.dtype
                    )
                )
                torch.testing.assert_close(pressure, expected, atol=1e-11, rtol=1e-11)
        config.gresho_mach = 0.1
        geometry.X1 = torch.tensor([0.6, 0.5, 0.4, 0.5], dtype=torch.float64)
        geometry.X3 = torch.tensor([0.5, 0.6, 0.5, 0.4], dtype=torch.float64)
        metric.volume = metric.evaluate(geometry.X3)
        state = initialize_cartesian3d(geometry, metric, config)
        wind = metric.volume.physical_velocity(state[1:4])
        expected_wind = torch.tensor([[0, -0.5, 0, 0.5], [0, 0, 0, 0], [0.5, 0, -0.5, 0]], dtype=wind.dtype)
        torch.testing.assert_close(wind, expected_wind, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(state[4], state[4, :1].expand_as(state[4]))
        for invalid_mach in (0, -0.1, float("nan"), float("inf"), 2):
            config.gresho_mach = invalid_mach
            with self.subTest(invalid_mach=invalid_mach), self.assertRaisesRegex(ValueError, "gresho_mach"):
                initialize_cartesian3d(geometry, metric, config)
