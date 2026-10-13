import os
from unittest.mock import patch

import torch
import xarray as xr
from mpi4py import MPI

from tests.unit.mpi_test import MpiTestCase
from wx_factory.common import Configuration, load_default_schema, readfile
from wx_factory.common.definitions import Rd, gravity
from wx_factory.init.dcmip import dcmip_T11_update_winds, dcmip_T12_update_winds
from wx_factory.simulation import Simulation


class PrescribedTransportTestCase(MpiTestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"WX_FACTORY_DIFFERENTIABLE": "1"})
        self.environment.start()
        self.mode = torch.inference_mode(False)
        self.mode.__enter__()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        self.mode.__exit__(None, None, None)
        self.environment.stop()

    def simulation(self, case):
        config = Configuration(readfile("config/dcmip21.ini"), load_default_schema())
        config.case_number = case
        config.advection_only = case in (11, 12, 13)
        config.enable_schar_mountain = False
        config.num_elements_horizontal = 1 if case == 41 else 2
        config.num_elements_vertical = 3
        config.num_solpts = 5
        config.precision = "double"
        config.pytorch_device = self.device_name
        config.output_freq = config.stat_freq = config.save_state_freq = 0
        config.output_dir = "/tmp/wxfactory-euler-prescribed-tests"
        config.time_integrator = "epi2"
        config.expfilter_apply = False
        return Simulation(config, context=self.context, quiet=True)

    def test_prescribed_winds_and_tracer_jacobian(self):
        for case in (11, 12, 13):
            sim = self.simulation(case)
            rhs = sim.rhs.full
            state = sim.Q.clone()
            for clock in (0.0, 0.137, 0.52):
                winds = rhs.wind(rhs.metric.volume, clock)
                if case in (11, 12):
                    function = dcmip_T11_update_winds if case == 11 else dcmip_T12_update_winds
                    reference = torch.stack(
                        function(sim.geometry, rhs.metric, sim.operators_real, sim.config, time=clock * rhs.period)
                    )
                    torch.testing.assert_close(winds, reference, atol=2e-12, rtol=2e-12)
                augmented = torch.cat((state[5:], torch.full_like(state[:1], clock)))
                torch.manual_seed(31 + self.comm.rank)
                direction = torch.randn_like(augmented)
                direction[-1] *= 0.01
                action = rhs.linearize(augmented)(direction)
                epsilon = 1e-6
                difference = (rhs(augmented + epsilon * direction) - rhs(augmented - epsilon * direction)) / (
                    2 * epsilon
                )
                error = self.comm.allreduce((action - difference).norm().item(), op=MPI.MAX)
                scale = self.comm.allreduce(action.norm().item(), op=MPI.MAX)
                self.assertLess(error / max(scale, 1e-20), 2e-6)
            initial = state.clone()
            evolved = sim.integrator.step(state, 2)
            torch.testing.assert_close(state, initial, atol=0, rtol=0)
            torch.testing.assert_close(evolved[:5], rhs.restore(state, 2)[:5], atol=0, rtol=0)
            self.assertFalse(sim.rhs.has_partition)

    def test_global_tracer_conservation_and_prescribed_filter(self):
        from wx_factory.step_hooks.dcmip import ExponentialFilterHook

        for case in (11, 12, 13):
            sim = self.simulation(case)
            rhs = sim.rhs.full
            augmented = torch.cat((sim.Q[5:], torch.full_like(sim.Q[:1], 0.137)))
            rate = rhs(augmented)[:-1]
            weights = sim.geometry.glweights
            weights = torch.einsum("i,j,k->ijk", weights, weights, weights).flatten()
            volume = rhs.metric.volume.volume_jacobian * weights
            total = self.comm.allreduce((rate * volume).sum((1, 2, 3, 4)).cpu().numpy())
            absolute = self.comm.allreduce((rate.abs() * volume).sum((1, 2, 3, 4)).cpu().numpy())
            self.assertLess(max(abs(total) / absolute.clip(min=1e-20)), 2e-12)
            sim.config.expfilter_strength, sim.config.expfilter_order, sim.config.expfilter_cutoff = 1.0, 4, 0.5
            hook = ExponentialFilterHook(sim.geometry, rhs.metric, sim.operators_real, sim.config)
            filtered = hook.process(sim.Q, 0)
            torch.testing.assert_close(filtered[:5], sim.Q[:5], atol=0, rtol=0)
            torch.testing.assert_close(
                (filtered[5:] * volume).sum(-1), (sim.Q[5:] * volume).sum(-1), rtol=2e-12, atol=1e-5
            )

    def test_dynamic_initial_conditions_and_polar_metric(self):
        for case in (41, 77):
            sim = self.simulation(case)
            fields = sim.initial_state.metric.volume
            torch.testing.assert_close(fields.height, sim.geometry.height_new, atol=1e-8, rtol=1e-12)
            self.assertTrue(bool(torch.isfinite(fields.dG_cov).all()))
            self.assertTrue(bool(torch.isfinite(sim.Q).all()))
            evolved = sim.integrator.step(sim.Q, 2)
            self.assertTrue(bool(torch.isfinite(evolved).all()))
            self.assertTrue(bool((evolved[[0, 4]] > 0).all()))

    def test_era5_terrain_and_physical_units(self):
        import numpy as np

        levels = [1000, 500, 100]
        names = ("geopotential", "u_component_of_wind", "v_component_of_wind", "vertical_velocity", "temperature")
        features = [f"{name}_h{level}" for name in names for level in levels]
        values = np.zeros((1, len(features), 5, 9))
        for index, name in enumerate(names):
            for offset, level in enumerate(levels):
                value = {
                    "geopotential": -Rd * 300 * np.log(level / 1000),
                    "u_component_of_wind": 12.0,
                    "v_component_of_wind": -3.0,
                    "vertical_velocity": -0.1,
                    "temperature": 300.0,
                }[name]
                values[:, 3 * index + offset] = value
        dataset = xr.Dataset(
            {"data": (("time", "features", "latitude", "longitude"), values)},
            coords={
                "time": np.array(["2020-01-01"], dtype="datetime64[ns]"),
                "features": features,
                "latitude": np.linspace(-90, 90, 5),
                "longitude": np.arange(0, 360, 40),
            },
        )

        def surface(filename, geometry):
            angles = geometry.radians
            field = 200 + 30 * angles[1] ** 2 + 40 * angles[0] ** 3 + 20 * angles[0] * angles[1]
            return geometry.get_floor(field[None])[0]

        with (
            patch("wx_factory.output.input_manager.InputManager.read_mountain", side_effect=surface),
            patch("xarray.open_zarr", side_effect=lambda *args, **kwargs: dataset.copy()),
        ):
            config = Configuration(readfile("config/dcmip21.ini"), load_default_schema())
            config.case_number = -1
            config.initial_conditions = "synthetic.zarr"
            config.topography_file = "synthetic.fst"
            config.time_start = "2020-01-01"
            config.num_elements_horizontal = 2
            config.num_elements_vertical = 3
            config.enable_schar_mountain = False
            config.pytorch_device = self.device_name
            config.precision = "double"
            config.output_freq = config.stat_freq = config.save_state_freq = 0
            config.output_dir = "/tmp/wxfactory-euler-era5-tests"
            sim = Simulation(config, context=self.context, quiet=True)
        fields = sim.initial_state.metric.volume
        torch.testing.assert_close(fields.height, sim.geometry.height_new, atol=1e-9, rtol=1e-12)
        u, v, w = sim.initial_state.metric.physical_winds(sim.Q[1:4])
        torch.testing.assert_close(u, torch.full_like(u, 12), atol=1e-10, rtol=1e-12)
        torch.testing.assert_close(v, torch.full_like(v, -3), atol=1e-10, rtol=1e-12)
        torch.testing.assert_close(w, 0.1 / (sim.Q[0] * gravity), atol=1e-10, rtol=1e-12)
        rhs = sim.rhs.full
        weights = sim.geometry.glweights
        weights = torch.einsum("i,j,k->ijk", weights, weights, weights).flatten()
        volume = fields.volume_jacobian * weights
        rate = rhs(sim.Q)[0]
        total = self.comm.allreduce((volume * rate).sum().item())
        absolute = self.comm.allreduce((volume * rate.abs()).sum().item())
        self.assertLess(abs(total) / max(absolute, 1e-20), 1e-11)
        self.assertTrue(bool(torch.isfinite(sim.Q).all()))
        self.assertTrue(bool((sim.Q[[0, 4]] > 0).all()))
