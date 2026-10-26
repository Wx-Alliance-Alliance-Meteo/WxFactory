from unittest.mock import patch

import torch
from mpi4py import MPI

from tests.unit.mpi_test import MpiTestCase
from wx_factory.common import Configuration, load_default_schema, readfile
from wx_factory.common.definitions import heat_capacity_ratio
from wx_factory.geometry import CubedSphere3D, DFROperators, Metric5D
from wx_factory.init.euler import pack_euler_state
from wx_factory.jacobian.euler import _compiled_acoustic_action, _compiled_dfr_derivative, _compiled_volume_actions
from wx_factory.jacobian.euler_vertical import EulerVerticalLinearization
from wx_factory.pde.euler_interfaces import low_mach_scales, physical_mach_number, pressure
from wx_factory.process_topology import ProcessTopology
from wx_factory.rhs.rhs_euler import RhsEuler
from wx_factory.solvers.fgmres import fgmres


class EulerRhsTestCase(MpiTestCase):
    def test_baroclinic_stretch_tracers_and_reduced_planet_scaling(self):
        from wx_factory.init.initialize import initialize_euler

        reference = None
        for case in (410, 411, 412, 413):
            config = Configuration(readfile(f"config/dcmip{case}.ini"), load_default_schema())
            config.precision = "double"
            geometry = CubedSphere3D(
                1, 6, 5, self.topology.num_lines_per_panel, 0, 0, 0, config.ztop, self.topology, config
            )
            operators = DFROperators(geometry, self.context)
            metric = Metric5D(geometry, config)
            state, _ = initialize_euler(geometry, metric, operators, config)
            scaling = 10 ** (case - 410)
            self.assertAlmostEqual(geometry.earth_radius * scaling, 6371220)
            self.assertAlmostEqual(geometry.rotation_speed / scaling, 7.29212e-5)
            self.assertEqual(state.shape[0], 7)
            self.assertTrue(bool(torch.isfinite(state).all()))
            self.assertTrue(bool((state[6] >= 0).all()))
            torch.testing.assert_close(state[5], state[4], rtol=0, atol=0)
            torch.testing.assert_close(metric.volume.height, geometry.height_new, rtol=2e-14, atol=2e-10)
            self.assertTrue(bool((metric.faces[2].volume_jacobian > 0).all()))
            invariant = torch.stack((state[0], state[4], state[6] / scaling))
            if reference is None:
                reference = invariant
            else:
                torch.testing.assert_close(invariant, reference, rtol=3e-12, atol=2e-12)
            rhs = RhsEuler(geometry, operators, metric, self.topology, config)
            tendency = rhs(state)
            torch.testing.assert_close(tendency[5], tendency[4], rtol=2e-12, atol=2e-10)
            self.assertTrue(bool(torch.isfinite(tendency).all()))
            weights = torch.einsum("k,j,i->kji", geometry.glweights, geometry.glweights, geometry.glweights).reshape(-1)
            volume = metric.volume.volume_jacobian * weights
            for variable in (0, 4, 5, 6):
                integral = self.comm.allreduce((volume * tendency[variable]).sum().item(), op=MPI.SUM)
                magnitude = self.comm.allreduce((volume * tendency[variable]).abs().sum().item(), op=MPI.SUM)
                self.assertLess(abs(integral) / max(magnitude, 1), 2e-13)

    def test_compiled_actions_match_eager_full_and_partitioned_actions(self):
        self.config.riemann_mach_cutoff = 0.001
        profile = torch.sin(self.geom.polar[0]) * torch.cos(self.geom.polar[1])
        vector = torch.zeros_like(self.state)
        vector[0], vector[4] = 0.01 * self.state[0] * profile, 0.01 * self.state[4] * profile
        vector[1:4] = self.metric.covariant_winds(0.2 * profile, -0.1 * profile, 0.05 * profile)
        _compiled_acoustic_action.cache_clear()
        _compiled_volume_actions.cache_clear()
        _compiled_dfr_derivative.cache_clear()
        with patch("wx_factory.jacobian.euler._compile_action", side_effect=lambda action, component: action):
            full = self.rhs.linearize(self.state)(vector).clone()
            partition = self.rhs.linearize_partitions(self.state)
            references = (full, partition.implicit(vector).clone(), partition.explicit(vector).clone())
        _compiled_acoustic_action.cache_clear()
        _compiled_volume_actions.cache_clear()
        _compiled_dfr_derivative.cache_clear()
        with (
            patch("wx_factory.jacobian.euler._compiled_acoustic_action", wraps=_compiled_acoustic_action) as compiler,
            patch(
                "wx_factory.jacobian.euler._compiled_volume_actions", wraps=_compiled_volume_actions
            ) as volume_compiler,
            patch("wx_factory.jacobian.euler._compiled_dfr_derivative", wraps=_compiled_dfr_derivative) as dfr_compiler,
        ):
            full = self.rhs.linearize(self.state)(vector).clone()
            self.assertEqual(compiler.called, self.device_name == "cuda")
            self.assertEqual(volume_compiler.called, self.device_name == "cuda")
            self.assertEqual(dfr_compiler.called, self.device_name == "cuda")
            dfr_before = dfr_compiler.call_count
            volume_before = volume_compiler.call_count
            before = compiler.call_count
            partition = self.rhs.linearize_partitions(self.state)
            implicit = partition.implicit(vector).clone()
            self.assertEqual(compiler.call_count > before, self.device_name == "cuda")
            explicit = partition.explicit(vector).clone()
            self.assertEqual(volume_compiler.call_count > volume_before, self.device_name == "cuda")
            self.assertEqual(dfr_compiler.call_count > dfr_before, self.device_name == "cuda")
            for actual, expected in zip((full, implicit, explicit), references):
                torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-10)

        if self.device_name == "cuda":
            self.config.column_batch_size = 7
            action = self.rhs.linearize_partitions(self.state)
            reference = action.implicit.solve(vector, 0.1)
            action = self.rhs.linearize_partitions(self.state)
            action.release_explicit()
            with patch(
                "wx_factory.jacobian.euler_vertical.EulerVerticalLinearization", wraps=EulerVerticalLinearization
            ) as assembly:
                actual = action.solve(vector, 0.1)
            torch.testing.assert_close(actual, reference, rtol=2e-12, atol=2e-11)
            sizes = [call.kwargs["columns"].stop - call.kwargs["columns"].start for call in assembly.call_args_list]
            self.assertGreater(len(sizes), 1)
            self.assertLessEqual(max(sizes), self.config.column_batch_size)

    def test_integration_comparison_includes_failures_on_other_ranks(self):
        from tests.integration.test_integration_state import component_errors

        reference = torch.ones((5, 4), dtype=torch.float64, device=self.context.torch_device)
        actual = reference.clone()
        if self.comm.rank == self.comm.size - 1:
            actual[2] += 0.1
        errors = component_errors(reference, actual, self.comm)
        self.assertAlmostEqual(errors[2], 0.1 / self.comm.size**0.5)
        if self.comm.rank == self.comm.size - 1:
            actual[2, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            component_errors(reference, actual, self.comm)

    def test_gmres_stagnation_decision_is_collective(self):
        diagonal = torch.linspace(0.1, 100.0, 64, dtype=torch.float64)
        initial = torch.full_like(diagonal, 1e15 if self.comm.rank == 0 else 0)
        forcing = diagonal * initial + 1
        _, _, _, iterations, flag, _ = fgmres(
            lambda vector: diagonal * vector,
            forcing,
            x0=initial,
            context=self.context,
            restart=8,
            maxiter=2,
            tol=1e-18,
        )
        decisions = self.comm.allgather((iterations, flag))
        self.assertTrue(all(decision == decisions[0] for decision in decisions))

    def test_partition_actions_and_column_solve_on_terrain(self):
        for depth in ("shallow", "deep"):
            self.config.depth_approx = depth
            geometry = CubedSphere3D(
                2, 2, 3, 2 * self.topology.num_lines_per_panel, 0.2, 0.1, -0.15, 12000.0, self.topology, self.config
            )
            geometry.rotation_speed = 7.29212e-5
            metric = Metric5D(geometry, self.config)
            metric.build_metric()
            if depth == "deep":
                self.assertGreater(metric.volume.vertical_frame_gradient.abs().max().item(), 0)
            metric.offload_derivatives()
            operators = DFROperators(geometry, self.context)
            rhs = RhsEuler(geometry, operators, metric, self.topology, self.config)
            state = self.state.clone()
            state[1:4] = metric.covariant_winds(
                torch.ones_like(state[0]) * 2, torch.ones_like(state[0]) * -0.3, torch.ones_like(state[0]) * 0.1
            )
            full = rhs(state)
            torch.testing.assert_close(rhs.implicit(state) + rhs.explicit(state), full, rtol=2e-11, atol=2e-8)
            profile = torch.sin(geometry.polar[0]) * torch.cos(geometry.polar[1])
            delta = torch.zeros_like(state)
            delta[0], delta[4] = 0.01 * state[0] * profile, 0.01 * state[4] * profile
            delta[1:4] = metric.covariant_winds(0.2 * profile, -0.1 * profile, 0.05 * profile)
            action = rhs.linearize_partitions(state)
            for function, jacobian in ((rhs.implicit, action.implicit), (rhs.explicit, action.explicit)):
                analytic = jacobian(delta).reshape_as(state)
                epsilon = 0.001
                reference = (function(state + epsilon * delta) - function(state - epsilon * delta)) / (2 * epsilon)
                error = self.comm.allreduce((reference - analytic).abs().max().item(), op=MPI.MAX)
                scale = self.comm.allreduce(analytic.abs().max().item(), op=MPI.MAX)
                self.assertLess(error / scale, 2e-6)
            dt = 10
            reference = action.implicit.solve(delta, dt)
            action.release_explicit()
            solution = action.solve(delta, dt, column_batch_size=7)
            torch.testing.assert_close(solution, reference, rtol=2e-12, atol=2e-11)
            residual = solution - dt / 2 * action.implicit(solution).reshape_as(state) - delta
            error = self.comm.allreduce(residual.abs().max().item(), op=MPI.MAX)
            scale = self.comm.allreduce(delta.abs().max().item(), op=MPI.MAX)
            self.assertLess(error / scale, 2e-12)

    def test_common_interface_geometry_matches_both_face_copies(self):
        for depth in ("shallow", "deep"):
            self.config.depth_approx = depth
            geometry = CubedSphere3D(
                2, 2, 3, 2 * self.topology.num_lines_per_panel, 0.2, 0.1, -0.15, 12000.0, self.topology, self.config
            )
            metric = Metric5D(geometry, self.config)
            metric.build_metric()
            rhs = RhsEuler(geometry, self.ops, metric, self.topology, self.config)
            for fields, common in zip(metric.faces, rhs._static_interfaces()):
                for values, selected in (
                    (fields.G_contra[1:4, 1:4], common.inverse_spatial),
                    (fields.geopotential, common.geopotential),
                    (fields.volume_jacobian, common.jacobian),
                ):
                    torch.testing.assert_close(selected, values[common.left], rtol=0, atol=0)
                    tolerance = 64 * torch.finfo(values.dtype).eps
                    scale = values[common.left].abs().max().item()
                    torch.testing.assert_close(selected, values[common.right], rtol=tolerance, atol=tolerance * scale)

    def test_analytic_action_matches_full_rhs_across_panels_and_terrain(self):
        for depth in ("shallow", "deep"):
            self.config.depth_approx = depth
            geometry = CubedSphere3D(
                2, 2, 3, 2 * self.topology.num_lines_per_panel, 0.2, 0.1, -0.15, 12000.0, self.topology, self.config
            )
            metric = Metric5D(geometry, self.config)
            metric.build_metric()
            operators = DFROperators(geometry, self.context)
            rhs = RhsEuler(geometry, operators, metric, self.topology, self.config)
            state = self.state.clone()
            state[1:4] = metric.covariant_winds(
                torch.ones_like(state[0]) * 2, torch.ones_like(state[0]) * -0.3, torch.ones_like(state[0]) * 0.1
            )
            rhs.damping = (torch.ones_like(state[0]) * 0.04, state[1:4].clone())
            profile = torch.sin(geometry.polar[0]) * torch.cos(geometry.polar[1])
            direction = torch.zeros_like(state)
            direction[0] = 0.01 * state[0] * profile
            direction[4] = 0.01 * state[4] * profile
            direction[1:4] = metric.covariant_winds(0.2 * profile, -0.1 * profile, 0.05 * profile)
            action = rhs.linearize(state)
            exact = action(direction).reshape_as(state)
            errors = []
            for epsilon in (0.02, 0.01, 0.005):
                approximation = (rhs(state + epsilon * direction) - rhs(state - epsilon * direction)) / (2 * epsilon)
                local = (approximation - exact).abs().max().item()
                errors.append(self.comm.allreduce(local, op=MPI.MAX))
            scale = self.comm.allreduce(exact.abs().max().item(), op=MPI.MAX)
            self.assertLess(errors[-1] / scale, 2e-6)
            self.assertLess(errors[-1], 0.5 * errors[0])
            weights = torch.einsum("k,j,i->kji", geometry.glweights, geometry.glweights, geometry.glweights).reshape(-1)
            volume = metric.volume.volume_jacobian * weights
            for component in (0, 4):
                integral = self.comm.allreduce((volume * exact[component]).sum().item(), op=MPI.SUM)
                magnitude = self.comm.allreduce((volume * exact[component]).abs().sum().item(), op=MPI.SUM)
                self.assertLess(abs(integral) / max(magnitude, 1), 2e-13)

    def setUp(self):
        self.depth = "shallow"
        super().setUp()
        self.config = Configuration(readfile("config/dcmip201_ROSEXP.ini"), load_default_schema())
        self.config.depth_approx = self.depth
        self.topology = ProcessTopology(self.context)
        self.geom = CubedSphere3D(
            2, 2, 3, 2 * self.topology.num_lines_per_panel, 0.2, 0.1, -0.15, 12000.0, self.topology, self.config
        )
        self.metric = Metric5D(self.geom, self.config)
        self.metric.build_metric()
        self.ops = DFROperators(self.geom, self.context)
        self.rhs = RhsEuler(self.geom, self.ops, self.metric, self.topology, self.config)
        height = self.metric.volume.height
        rho = torch.exp(-height / 9000)
        wind = torch.stack((torch.ones_like(rho) * 2, torch.ones_like(rho) * -0.3, torch.zeros_like(rho)))
        covariant = self.metric.covariant_winds(*wind)
        self.state = pack_euler_state(
            rho,
            tuple(self.metric.volume.raise_velocity(covariant)),
            torch.ones_like(rho) * 300,
            self.metric,
            self.config,
        )

    def test_common_scalar_flux_from_coordinate_characteristics(self):
        traces = self.rhs._faces_with_neighbours(self.rhs._extrapolate(self.state))
        scalar_fluxes, pressure_traces, _, _ = self.rhs._interface_data(traces)
        for direction, (geometry, face) in enumerate(zip(self.rhs._static_interfaces(), traces)):
            left, right = face[geometry.left], face[geometry.right]
            pressure_left, pressure_right = pressure(left[4]), pressure(right[4])
            sound_left = torch.sqrt(heat_capacity_ratio * pressure_left / left[0])
            sound_right = torch.sqrt(heat_capacity_ratio * pressure_right / right[0])
            impedance = (left[0] * sound_left + right[0] * sound_right) / 2
            velocity_left = (geometry.inverse_spatial[direction] * left[1:4]).sum(dim=0)
            velocity_right = (geometry.inverse_spatial[direction] * right[1:4]).sum(dim=0)
            normal_scale = torch.sqrt(geometry.inverse_spatial[direction, direction])
            self.assertTrue(bool((velocity_left.abs() < sound_left * normal_scale).all()))
            self.assertTrue(bool((velocity_right.abs() < sound_right * normal_scale).all()))
            velocity_common = (velocity_left + velocity_right) / 2 - normal_scale * (pressure_right - pressure_left) / (
                2 * impedance
            )
            mach_left = physical_mach_number(left[1:4], geometry.inverse_spatial, sound_left)
            mach_right = physical_mach_number(right[1:4], geometry.inverse_spatial, sound_right)
            _, pressure_scale = low_mach_scales(mach_left, mach_right, self.rhs.config.riemann_mach_cutoff)
            pressure_common = (pressure_left + pressure_right) / 2 - pressure_scale * impedance * (
                velocity_right - velocity_left
            ) / (2 * normal_scale)
            if geometry.wall is not None:
                velocity_common = torch.where(geometry.wall, torch.zeros_like(velocity_common), velocity_common)
            scalars = torch.where(velocity_common[None] >= 0, left[[0, 4]], right[[0, 4]])
            common_flux = geometry.jacobian * velocity_common * scalars
            expected_flux = torch.zeros_like(face[[0, 4]])
            expected_flux[geometry.left] = expected_flux[geometry.right] = common_flux
            expected_pressure = torch.zeros_like(face[0])
            expected_pressure[geometry.left] = expected_pressure[geometry.right] = pressure_common
            torch.testing.assert_close(scalar_fluxes[direction], expected_flux[self.rhs.mid_faces[direction]])
            torch.testing.assert_close(pressure_traces[direction], expected_pressure[self.rhs.mid_faces[direction]])

    def test_scalar_conservation_and_impermeable_walls(self):
        result = self.rhs(self.state)
        weights = torch.einsum("k,j,i->kji", self.geom.glweights, self.geom.glweights, self.geom.glweights).reshape(-1)
        weighted = result[[0, 4]] * self.metric.volume.volume_jacobian * weights
        for slot in range(2):
            total = self.comm.allreduce(weighted[slot].sum().item(), op=MPI.SUM)
            scale = self.comm.allreduce(weighted[slot].abs().sum().item(), op=MPI.SUM)
            self.assertLess(abs(total), 2e-12 * scale)
        vertical = self.rhs.diagnostics["interface_normal_winds"][2]
        self.assertEqual(vertical[[0, -1]].abs().max().item(), 0.0)
        self.assertTrue(torch.isfinite(result).all())

    def test_common_riemann_curl_matches_rhs_and_does_no_work(self):
        velocity = self.metric.volume.raise_velocity(self.state[1:4])
        faces = self.rhs._faces_with_neighbours(self.rhs._extrapolate(self.state))
        velocity_traces = self.rhs._interface_data(faces)[3]
        curl = self.rhs.relative_vorticity_tendency(self.state[1:4], velocity, velocity_traces)
        self.assertLess((curl * velocity).sum(dim=0).abs().max().item(), 1e-12)
        self.rhs(self.state)
        torch.testing.assert_close(curl, self.rhs.diagnostics["relative_vorticity"], rtol=0, atol=0)

    def test_lifted_curl_in_deep_atmosphere_is_finite_and_does_no_work(self):
        self.config.depth_approx = "deep"
        geometry = CubedSphere3D(
            2,
            2,
            3,
            2 * self.topology.num_lines_per_panel,
            0.2,
            0.1,
            -0.15,
            12000.0,
            self.topology,
            self.config,
        )
        metric = Metric5D(geometry, self.config)
        metric.build_metric()
        operators = DFROperators(geometry, self.context)
        rhs = RhsEuler(geometry, operators, metric, self.topology, self.config)
        height = metric.volume.height
        wind = torch.stack((torch.ones_like(height) * 2, torch.ones_like(height) * -0.3, torch.zeros_like(height)))
        covariant = metric.covariant_winds(*wind)
        state = pack_euler_state(
            torch.exp(-height / 9000),
            tuple(metric.volume.raise_velocity(covariant)),
            torch.ones_like(height) * 300,
            metric,
            self.config,
        )
        result = rhs(state)
        velocity = metric.volume.raise_velocity(state[1:4])
        work = (rhs.diagnostics["relative_vorticity"] * velocity).sum(dim=0)
        self.assertTrue(torch.isfinite(result).all())
        self.assertLess(work.abs().max().item(), 1e-12)
        self.assertEqual(rhs.diagnostics["interface_normal_winds"][2][[0, -1]].abs().max().item(), 0.0)

    def test_plain_log_scalar_extrapolation(self):
        from wx_factory.common.matmul import apply_op

        traces = self.rhs._extrapolate(self.state)
        for trace, extrapolate in zip(traces, self.rhs.extrapolations):
            torch.testing.assert_close(
                trace[[0, 4]], torch.exp(apply_op(torch.log(self.state[[0, 4]]), extrapolate)), rtol=0, atol=0
            )

    def test_epi2_pmex_step_is_finite_and_positive(self):
        # The existing finite-difference Jacobian uses COMM_WORLD; do not run
        # this integration check on a subset of a larger MPI test launch.
        if MPI.COMM_WORLD.size != self.comm.size:
            self.skipTest("EPI finite-difference norms currently use COMM_WORLD")
        from wx_factory.integrators.epi import Epi

        self.config.exponential_solver = "pmex"
        self.config.tolerance = 1e-8
        self.config.krylov_mmax = 32
        self.config.jacobian = "finite_difference"
        integrator = Epi(self.config, 2, self.rhs, context=self.context)
        updated = integrator.step(self.state, 0.001)
        self.assertTrue(torch.isfinite(updated).all())
        self.assertTrue((updated[[0, 4]] > 0).all())
        self.config.jacobian = "analytic"
        analytic = Epi(self.config, 2, self.rhs, context=self.context).step(self.state, 0.001)
        torch.testing.assert_close(analytic[[0, 4]], updated[[0, 4]], rtol=2e-9, atol=2e-7)
        # Compare physical winds so tolerances do not depend on element size.
        for actual, expected in zip(
            self.metric.physical_winds(analytic[1:4]), self.metric.physical_winds(updated[1:4])
        ):
            torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-8)

    def test_full_euler_payload_preserves_analytic_face_winds(self):
        from wx_factory.pde.euler_interfaces import interface_slices

        traces = []
        for fields, interior in zip(self.metric.faces, self.rhs.mid_faces):
            radial = fields.radial
            physical = torch.stack((1 + radial[1], -2 + radial[2], 0.7 + radial[0]))
            trace = torch.empty((5,) + fields.height.shape, dtype=torch.float64)
            trace[0] = 1
            trace[4] = 300
            trace[1:4] = fields.covariant_from_physical(physical)
            traces.append(trace[interior])
        # Complete the exchange before checking any rank's result.
        joined = self.rhs._faces_with_neighbours(tuple(traces))
        for direction, fields in enumerate(self.metric.faces[:2]):
            radial = fields.radial
            expected = torch.stack((1 + radial[1], -2 + radial[2], 0.7 + radial[0]))
            actual = fields.physical_velocity(joined[direction][1:4])
            left, right = interface_slices(direction, self.geom.num_solpts)
            for side in (left, right):
                torch.testing.assert_close(actual[side], expected[side], rtol=2e-13, atol=2e-13)
            torch.testing.assert_close(fields.G_cov[left], fields.G_cov[right], rtol=0, atol=0)

    def test_manufactured_curl_in_terrain_reference_coordinates(self):
        from wx_factory.common.matmul import apply_op

        z, y, x = torch.meshgrid(
            self.geom.solutionPoints, self.geom.solutionPoints, self.geom.solutionPoints, indexing="ij"
        )
        covariant = torch.stack((y * y + x * z, x * x + y * z, x * y + z * z)).flatten(1)
        covariant = covariant[:, None, None, None, :].expand_as(self.state[1:4]).clone()
        velocity = self.metric.volume.raise_velocity(covariant)
        x, y = x.flatten(), y.flatten()
        expected = torch.stack(
            (
                (2 * x - 2 * y) * velocity[1] + (y - x) * velocity[2],
                -(2 * x - 2 * y) * velocity[0] + (x - y) * velocity[2],
                -(y - x) * velocity[0] - (x - y) * velocity[1],
            )
        )
        state = self.state.clone()
        state[1:4] = covariant
        faces = self.rhs._faces_with_neighbours(self.rhs._extrapolate(state))
        velocity_traces = self.rhs._interface_data(faces)[3]
        corrections = torch.stack(
            [
                apply_op(trace - apply_op(covariant, extrapolate), correction)
                for trace, extrapolate, correction in zip(
                    velocity_traces, self.rhs.extrapolations, self.rhs.corrections
                )
            ]
        )
        expected += torch.einsum("j...,ij...->i...", velocity, corrections - corrections.transpose(0, 1))
        actual = self.rhs.relative_vorticity_tendency(covariant, velocity, velocity_traces)
        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=1e-20)
