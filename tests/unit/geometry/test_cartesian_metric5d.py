import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from wx_factory.common import Configuration, load_default_schema, readfile
from wx_factory.common.definitions import gravity
from wx_factory.common.matmul import apply_op
from wx_factory.pde.euler_interfaces import acoustic_interface
from wx_factory.simulation import Simulation


class CartesianEulerTestCase(unittest.TestCase):
    def test_density_current_temperature_anomaly_at_background_pressure(self):
        from wx_factory.common.definitions import Rd, cpd, heat_capacity_ratio, p0
        from wx_factory.init.initialize import initialize_cartesian3d

        x = torch.tensor([0.0, 2000.0, 4000.0, 10000.0], dtype=torch.float64)
        z = torch.full_like(x, 3000.0)
        geometry = SimpleNamespace(X1=x, X3=z, grid_shape_3d_new=x.shape)
        metric = SimpleNamespace(volume=SimpleNamespace(covariant_from_physical=lambda velocity: velocity))
        config = Configuration(readfile("config/density_current.ini"), load_default_schema())
        state = initialize_cartesian3d(geometry, metric, config)
        exner = 1 - gravity * z / (cpd * config.bubble_theta)
        pressure = p0 * (Rd * state[4] / p0)**heat_capacity_ratio
        temperature = pressure / (Rd * state[0])
        anomaly = torch.tensor([-15.0, -7.5, 0.0, 0.0], dtype=x.dtype)
        torch.testing.assert_close(pressure, p0 * exner**(cpd / Rd), rtol=1e-13, atol=0)
        torch.testing.assert_close(temperature, config.bubble_theta * exner + anomaly, rtol=1e-13, atol=0)
        torch.testing.assert_close(state[4] / state[0], config.bubble_theta + anomaly / exner)
        self.assertTrue(bool((state[1:4] == 0).all()))

    @classmethod
    def setUpClass(cls):
        config = Configuration(readfile("config/cold_bubble.ini"), load_default_schema())
        config.num_elements_horizontal = 2
        config.num_elements_vertical = 3
        config.pytorch_device = "cpu"
        config.output_freq = config.stat_freq = 0
        config.output_dir = "/tmp/wxfactory-euler-cartesian-tests"
        cls.default_device = torch.get_default_device()
        with patch.dict(os.environ, {"WX_FACTORY_DIFFERENTIABLE": "1"}):
            cls.sim = Simulation(config, quiet=True)

    @classmethod
    def tearDownClass(cls):
        torch.set_default_device(cls.default_device)

    def setUp(self):
        self.generator = torch.Generator().manual_seed(37)

    def random_like(self, value):
        return torch.randn(value.shape, generator=self.generator, dtype=value.dtype, device=value.device)

    def test_euler_filter_matches_conservative_operation(self):
        from copy import copy

        from wx_factory.geometry.operators import DFROperators

        metric = copy(self.sim.initial_state.metric)
        fields = copy(metric.volume)
        metric.volume = fields
        density = 1 + .1 * self.random_like(self.sim.Q[0])
        conservative = self.sim.Q.clone()
        conservative[0] = density
        conservative[1:4] = .01 * self.random_like(self.sim.Q[1:4])
        conservative[4] = 300 * density
        fields.volume_jacobian = 1 + .1 * self.random_like(density)
        fields.G_cov = fields.G_cov.clone()
        fields.G_contra = fields.G_contra.clone()
        scaling = 1 + .05 * self.random_like(density)
        fields.G_cov[1:4, 1:4] *= scaling
        fields.G_contra[1:4, 1:4] /= scaling
        state = conservative.clone()
        state[1:4] = fields.lower_velocity(conservative[1:4] / density)
        initial = state.clone()
        operators = self.sim.operators_real
        modal = operators.make_filter_3d(8, 4, .25, self.sim.geometry)
        expected = ((fields.volume_jacobian * conservative) @ modal) / fields.volume_jacobian
        actual = DFROperators.apply_filter_3d(state, metric, modal)
        recovered = actual.clone()
        recovered[1:4] = actual[0] * fields.raise_velocity(actual[1:4])
        torch.testing.assert_close(recovered, expected, atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(state, initial, atol=0, rtol=0)
        weights = self.sim.geometry.glweights
        weights = torch.einsum("i,j,k->ijk", weights, weights, weights).flatten()
        before = (fields.volume_jacobian * conservative * weights).sum(-1)
        after = (fields.volume_jacobian * recovered * weights).sum(-1)
        torch.testing.assert_close(before, after, atol=2e-12, rtol=2e-12)

    def test_metric_inverse_determinant_and_reference_scaling(self):
        metric = self.sim.initial_state.metric
        self.assertEqual(metric.scales, (250.0, 500.0, 1000 / 6))
        for fields in (metric.volume, *metric.faces):
            product = torch.einsum("ab...,bc...->ac...", fields.G_cov, fields.G_contra)
            identity = torch.eye(5, dtype=product.dtype).reshape((5, 5) + (1,) * fields.height.ndim)
            torch.testing.assert_close(product, identity.expand_as(product))
            determinant = torch.linalg.det(fields.G_cov.movedim((0, 1), (-2, -1)))
            torch.testing.assert_close(determinant, -fields.volume_jacobian**2)
            torch.testing.assert_close(fields.geopotential, gravity * fields.height)
            self.assertTrue(bool((fields.dG_cov[2, 0, 0] == -2 * gravity * metric.scales[2]).all()))
            gram = torch.einsum("ai...,aj...->ij...", fields.physical_basis, fields.physical_basis)
            torch.testing.assert_close(gram, fields.G_cov[1:4, 1:4])

    def test_geopotential_dfr_gradient_at_actual_faces(self):
        rhs = self.sim.rhs.full
        for direction, fields in enumerate(rhs.metric.faces):
            gradient = apply_op(rhs.metric.volume.geopotential, rhs.derivatives[direction])
            gradient += apply_op(fields.geopotential[rhs.mid_faces[direction]], rhs.corrections[direction])
            expected = torch.full_like(gradient, gravity * rhs.metric.scales[2] if direction == 2 else 0)
            torch.testing.assert_close(gradient, expected, rtol=1e-12, atol=1e-8)

    def test_physical_velocity_round_trip(self):
        fields = self.sim.initial_state.metric.volume
        physical = self.random_like(self.sim.Q[1:4])
        covariant = fields.covariant_from_physical(physical)
        torch.testing.assert_close(fields.physical_velocity(covariant), physical)
        for i, scale in enumerate(self.sim.initial_state.metric.scales):
            torch.testing.assert_close(covariant[i], scale * physical[i])

    def test_walls_and_closed_domain_scalar_conservation(self):
        rhs = self.sim.rhs.full
        state = self.sim.Q.clone()
        state[1:4] = 0.1 * self.random_like(state[1:4])
        faces = rhs._faces_with_neighbours(rhs._extrapolate(state))
        for direction, (geometry, face) in enumerate(zip(rhs._static_interfaces(), faces)):
            normal, _, _ = acoustic_interface(
                face[geometry.left],
                face[geometry.right],
                geometry.inverse_spatial,
                direction,
                geometry.geopotential,
                wall=geometry.wall,
            )
            self.assertTrue(bool((normal[geometry.wall] == 0).all()))
        value = rhs(state)
        weights = self.sim.geometry.glweights
        volume_weights = torch.einsum("k,j,i->kji", weights, weights, weights).flatten()
        total = (value[[0, 4]] * rhs.metric.volume.volume_jacobian * volume_weights).sum(dim=(1, 2, 3, 4))
        torch.testing.assert_close(total, torch.zeros_like(total), atol=1e-6, rtol=0)

    def test_periodic_horizontal_translation_and_gravity(self):
        config = Configuration(readfile("config/cold_bubble.ini"), load_default_schema())
        config.num_elements_horizontal = 2
        config.num_elements_vertical = 3
        config.pytorch_device = "cpu"
        config.lateral_boundary = "periodic"
        config.output_freq = config.stat_freq = 0
        config.output_dir = "/tmp/wxfactory-euler-cartesian-tests"
        with patch.dict(os.environ, {"WX_FACTORY_DIFFERENTIABLE": "1"}):
            sim = Simulation(config, quiet=True)
        state = torch.zeros_like(sim.Q)
        state[0], state[4] = 1, 300
        state[1] = 2 * sim.initial_state.metric.scales[0]
        expected = torch.zeros_like(state)
        expected[3] = -gravity * sim.initial_state.metric.scales[2]
        torch.testing.assert_close(sim.rhs.full(state), expected, rtol=1e-12, atol=2e-8)
        self.assertIsNone(sim.rhs.full._static_interfaces()[0].wall)
        self.assertIsNone(sim.rhs.full._static_interfaces()[1].wall)

    def test_gresho_pressure_balance_and_equation_of_state(self):
        from wx_factory.common.definitions import Rd, heat_capacity_ratio, p0
        from wx_factory.geometry.cartesian_metric5d import CartesianMetric5D
        from wx_factory.init.initialize import initialize_cartesian3d

        config = Configuration(readfile("config/gresho_vortex.ini"), load_default_schema())
        config.pytorch_device = "cpu"
        config.output_freq = config.stat_freq = 0
        config.output_dir = "/tmp/wxfactory-gresho-tests"
        with patch.dict(os.environ, {"WX_FACTORY_DIFFERENTIABLE": "1"}):
            sim = Simulation(config, quiet=True)
        for fields in (sim.initial_state.metric.volume, *sim.initial_state.metric.faces):
            self.assertEqual(torch.count_nonzero(fields.geopotential).item(), 0)
            self.assertEqual(torch.count_nonzero(fields.dG_cov).item(), 0)

        # Differentiate inside each smooth branch, away from the two break radii.
        from types import SimpleNamespace
        radius = torch.tensor([0.1, 0.3, 0.45], dtype=torch.float64, requires_grad=True)
        geometry = SimpleNamespace(X1=0.5 + radius, X3=torch.full_like(radius, 0.5),
                                   grid_shape_3d_new=radius.shape,
                                   delta_x1=1.0, delta_x2=0.1, ztop=1.0, num_elements_x3=1)
        metric = CartesianMetric5D(geometry, config)
        metric.volume = metric.evaluate(geometry.X3)
        state = initialize_cartesian3d(geometry, metric, config)
        pressure = p0 * (Rd * state[4] / p0)**heat_capacity_ratio
        gradient = torch.autograd.grad(pressure.sum(), radius)[0]
        wind = metric.volume.physical_velocity(state[1:4])
        torch.testing.assert_close(gradient, state[0] * wind[2]**2 / radius, atol=1e-12, rtol=1e-12)
        expected = 1 / (heat_capacity_ratio * 0.1**2) - 0.5 + torch.tensor(
            [0.125, 4 * math.log(1.5) + 4 - 6 + 1.125,
             4 * math.log(2) - 2], dtype=pressure.dtype)
        torch.testing.assert_close(pressure, expected, atol=1e-12, rtol=1e-12)

    def test_analytic_jacobian_and_partition(self):
        rhs = self.sim.rhs.full
        state = self.sim.Q.clone()
        state[1:4] = self.random_like(state[1:4])
        direction = self.random_like(state)
        direction[[0, 4]] *= 0.01
        epsilon = 1e-4
        torch.testing.assert_close(rhs(state), rhs.implicit(state) + rhs.explicit(state), atol=1e-8, rtol=1e-10)
        partition = rhs.linearize_partitions(state)
        explicit, implicit = partition.explicit, partition.implicit
        for function, linearization in (
            (rhs, rhs.linearize(state)),
            (rhs.explicit, explicit),
            (rhs.implicit, implicit),
        ):
            expected = (function(state + epsilon * direction) - function(state - epsilon * direction)) / (2 * epsilon)
            torch.testing.assert_close(linearization(direction).reshape_as(state), expected, atol=2e-4, rtol=2e-4)


if __name__ == "__main__":
    unittest.main()
