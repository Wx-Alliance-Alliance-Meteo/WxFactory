import torch
from mpi4py import MPI

from tests.unit.mpi_test import MpiTestCase
from wx_factory.common import Configuration, load_default_schema, readfile
from wx_factory.simulation import Simulation


class ShallowWaterRhsTestCase(MpiTestCase):
    def setUp(self):
        super().setUp()
        config = Configuration(readfile("config/case5.ini"), load_default_schema())
        config.lambda0, config.phi0, config.alpha0 = 0.2, 0.1, -0.15
        config.num_elements_horizontal = 3
        config.num_solpts = 4
        config.output_freq = 0
        config.save_state_freq = 0
        config.stat_freq = 0
        self.sim = Simulation(config, comm=self.comm, context=self.context)
        self.rhs = self.sim.rhs.full
        self.geometry = self.sim.geometry

    def test_lake_at_rest_over_topography(self):
        state = torch.zeros_like(self.sim.Q)
        state[0] = 6000 - self.rhs.bottom
        tendency = self.rhs(state)
        mass_error = self.comm.allreduce(tendency[0].abs().max().item(), op=MPI.MAX)
        acceleration = self.rhs.metric.volume.raise_velocity(tendency[1:])
        u, v = self.geometry.contra2wind(*acceleration)
        error = self.comm.allreduce(torch.maximum(u.abs(), v.abs()).max().item(), op=MPI.MAX)
        self.assertLess(mass_error, 1e-14)
        self.assertLess(error, 1e-14)

    def test_covariant_exchange_preserves_a_smooth_physical_wind(self):
        from wx_factory.geometry.metric5d import panel_frame, unit_sphere_basis

        geometry = self.geometry
        frame = panel_frame(geometry.lon_p, geometry.lat_p, geometry.angle_p, geometry.X)
        faces = []
        for coordinates in (geometry.coordVec_itf_i, geometry.coordVec_itf_j):
            radial, basis = unit_sphere_basis(coordinates, frame)
            basis[:, 0] *= geometry.earth_radius * geometry.delta_x1 / 2
            basis[:, 1] *= geometry.earth_radius * geometry.delta_x2 / 2
            physical = torch.stack((1 + radial[1], -2 + radial[2], 0.7 + radial[0]))
            covariant = torch.einsum("ai...,a...->i...", basis, physical)
            faces.append(torch.cat(((6000 + 100 * radial[2])[None], covariant)))
        width = geometry.num_solpts
        i, j = faces
        boundaries = (j[..., 1, :, :width], j[..., -2, :, width:], i[..., 1, :width], i[..., -2, width:])
        vectors = self.rhs.ptopo.start_exchange_vectors(
            *(tuple(face[1:]) for face in boundaries), geometry.boundary_sn, geometry.boundary_we, covariant=True
        )
        scalars = self.rhs.ptopo.start_exchange_scalars(
            *(face[0] for face in boundaries), boundary_shape=(geometry.num_elements_horizontal * width,)
        )
        faces = self.rhs._finish_exchange((faces, vectors, scalars))
        for direction, face in enumerate(faces):
            left, right = face[self.rhs.left[direction]], face[self.rhs.right[direction]]
            torch.testing.assert_close(left, right, rtol=2e-12, atol=2e-9)

    def test_full_jacobian_and_mass_conservation(self):
        state = self.sim.Q.clone()
        geometry = self.geometry
        generator = torch.Generator().manual_seed(183 + self.comm.rank)
        winds = torch.randn((2,) + state.shape[1:], generator=generator, dtype=state.dtype)
        state[1:] += self.rhs.metric.volume.lower_velocity(torch.stack(geometry.wind2contra(*winds)))
        profile = torch.cos(geometry.lat) * torch.sin(geometry.lon)
        perturbation = torch.empty_like(state)
        perturbation[0] = 2 * profile
        perturbation[1:] = self.rhs.metric.volume.lower_velocity(
            torch.stack(geometry.wind2contra(0.1 * profile, -0.2 * profile))
        )
        action = self.rhs.linearize(state)
        analytic = action(perturbation).reshape_as(state)
        epsilon = 1e-3
        reference = (self.rhs(state + epsilon * perturbation) - self.rhs(state - epsilon * perturbation)) / (
            2 * epsilon
        )
        for component in range(3):
            error = self.comm.allreduce((reference[component] - analytic[component]).abs().max().item(), op=MPI.MAX)
            scale = self.comm.allreduce(analytic[component].abs().max().item(), op=MPI.MAX)
            self.assertLess(error / scale, 2e-6)
        from wx_factory.output.diagnostic import global_integral_2d

        tendency = self.rhs(state)
        integral = global_integral_2d(
            tendency[0], self.sim.operators_real, self.rhs.metric, geometry.num_solpts, comm=self.comm
        )
        magnitude = global_integral_2d(
            tendency[0].abs(), self.sim.operators_real, self.rhs.metric, geometry.num_solpts, comm=self.comm
        )
        self.assertLess(abs(integral) / magnitude, 1e-10)
        other_layer = state.clone()
        other_layer[0] *= 1.03
        other_layer[1:] *= -0.4
        layers = torch.stack((state, other_layer), dim=1)
        expected = torch.stack((tendency, self.rhs(other_layer)), dim=1)
        torch.testing.assert_close(self.rhs(layers), expected, rtol=2e-12, atol=1e-9)

    def test_vorticity_diagnostic_in_reference_coordinates(self):
        from wx_factory.output.diagnostic import relative_vorticity

        points = self.geometry.solutionPoints
        xi = points.repeat(len(points))
        eta = points.repeat_interleave(len(points))
        ones = torch.ones_like(self.sim.Q[0])
        covariant = torch.stack((2 * eta * ones, 3 * xi * ones))
        velocity = self.rhs.metric.volume.raise_velocity(covariant)
        vorticity = relative_vorticity(*velocity, self.rhs.metric, self.sim.operators_real)
        torch.testing.assert_close(vorticity, self.rhs.metric.inv_sqrtG, rtol=1e-12, atol=0)

    def test_backward_euler_converges_at_zero_normal_velocity(self):
        config = Configuration(readfile("config/case5.ini"), load_default_schema())
        config.num_elements_horizontal = config.num_solpts = 3
        config.lambda0, config.phi0, config.alpha0 = 0, torch.pi / 4, 0
        config.precision = "double"
        config.time_integrator = "backward_euler"
        config.dt, config.t_end = 60, 180
        config.tolerance = 1e-8
        config.output_freq = config.stat_freq = config.save_state_freq = 0
        simulation = Simulation(config, comm=self.comm, context=self.context)
        for _ in range(3):
            previous = simulation.Q.clone()
            simulation.step()
            residual = (simulation.Q - previous) / config.dt - simulation.rhs.full(simulation.Q)
            error = self.comm.allreduce(residual.abs().max().item(), op=MPI.MAX)
            self.assertLessEqual(error, config.tolerance)
        self.assertTrue(bool(torch.isfinite(simulation.Q).all()))
        self.assertTrue(bool((simulation.Q[0] > 0).all()))
