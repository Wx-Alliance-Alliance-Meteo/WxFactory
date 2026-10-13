import os
from unittest.mock import patch

import torch

from tests.unit.mpi_test import MpiTestCase
from wx_factory.common import Configuration, load_default_schema, readfile
from wx_factory.simulation import Simulation
from wx_factory.step_hooks import ScharMountainHook


class TerrainRampTestCase(MpiTestCase):
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

    def simulation(self, case, terrain_steps=0, integrator="epi2"):
        config = Configuration(readfile("config/dcmip21.ini"), load_default_schema())
        config.case_number = case
        config.num_elements_horizontal = 2
        config.num_elements_vertical = 3
        config.num_solpts = 5
        config.precision = "double"
        config.pytorch_device = self.device_name
        config.output_freq = config.stat_freq = config.save_state_freq = 0
        config.output_dir = "/tmp/wxfactory-euler-terrain-tests"
        config.enable_schar_mountain = True
        config.schar_mountain_step = terrain_steps
        config.time_integrator = integrator
        config.expfilter_apply = False
        return Simulation(config, context=self.context, quiet=True)

    def test_frozen_grid_terrain_growth(self):
        for case, integrator in ((21, "epi3"), (22, "partrosexp2")):
            with self.subTest(case=case, integrator=integrator):
                sim = self.simulation(case, terrain_steps=2, integrator=integrator)
                hook = sim.step_hooks[ScharMountainHook]
                self.assertEqual(hook.metric.height_map.terrain_ratio, 0)
                original = sim.Q.clone()
                promoted = sim.rhs.full.double_precision_rhs()
                sim.rhs.full._static_interfaces()
                self.assertIs(hook.process(sim.Q, sim.t), sim.Q)
                torch.testing.assert_close(sim.Q, original, rtol=0, atol=0)
                self.assertEqual(hook.metric.height_map.terrain_ratio, 0.5)
                sim.rhs.full.refresh_geometry()
                self.assertIsNone(sim.rhs.full._interface_geometry)
                self.assertIsNot(sim.rhs.full.double_precision_rhs(), promoted)
                del promoted
                hook.step = 0
                hook.apply(0)
                sim.rhs.full.refresh_geometry()
                for step in range(3):
                    previous = sim.integrator
                    self.assertTrue(sim.step())
                    self.assertEqual(sim.integrator is not previous, step < 2)
                    self.assertTrue(torch.isfinite(sim.Q).all())
                    self.assertTrue((sim.Q[[0, 4]] > 0).all())
                self.assertEqual(hook.step, 2)
                self.assertEqual(hook.metric.height_map.terrain_ratio, 1)
                metric = sim.initial_state.metric
                for actual, coordinates in zip((metric.volume, *metric.faces), metric._construction_coordinates):
                    expected = metric.evaluate(coordinates)
                    torch.testing.assert_close(actual.G_cov, expected.G_cov.to(actual.G_cov.dtype))
        with patch.object(Simulation, "_determine_starting_state", lambda sim: (sim.initial_state.Q, 1)):
            restarted = self.simulation(21, terrain_steps=2)
        hook = restarted.step_hooks[ScharMountainHook]
        self.assertEqual(hook.step, 1)
        self.assertEqual(hook.metric.height_map.terrain_ratio, 0.5)
        self.assertTrue(restarted.step())
        self.assertEqual(hook.step, 2)
        self.assertEqual(hook.metric.height_map.terrain_ratio, 1)
