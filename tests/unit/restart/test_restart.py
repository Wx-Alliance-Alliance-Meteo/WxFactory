import copy
import os
import unittest

import numpy

from common import Configuration, ConfigurationSchema, default_schema_path, load_default_schema, readfile
from simulation import Simulation
from wx_mpi import do_once, SingleProcess, Conditional

from mpi_test import MpiTestCase


class Euler2DRestartTestCase(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.schema = load_default_schema()
        self.base_config = Configuration(readfile(os.path.join(dir_path, "euler2d.ini")), self.schema)
        self.base_sim = Simulation(self.base_config)

    def test_gen_restart(self):
        restart_file = self.base_sim.output.state_file_name(1)
        if os.path.exists(restart_file):
            os.remove(restart_file)

        self.assertFalse(os.path.exists(restart_file))
        self.base_sim.step()
        self.assertTrue(os.path.exists(restart_file))

    def test_read_restart(self):
        restart_file = self.base_sim.output.state_file_name(1)

        new_sim = Simulation(self.base_config)
        new_sim.step()

        self.assertTrue(os.path.exists(restart_file))

        config = copy.deepcopy(self.base_config)
        config.starting_step = 1

        sim = Simulation(config)
        self.assertEqual(sim.starting_step, 1, f"Starting step is not 1! {sim.starting_step}")

        diff = sim.initial_Q - new_sim.Q
        diff_norm = numpy.linalg.norm(diff)
        self.assertTrue(diff_norm == 0.0, f"Restart state is not the same as computed")


class MultiProcRestartTestCase(MpiTestCase):
    def __init__(self, num_procs, config_file, methodName, optional=False):
        super().__init__(num_procs, methodName, optional)
        self.config_file = config_file

    def setUp(self):
        super().setUp()

        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(dir_path, self.config_file)

        self.schema = ConfigurationSchema(do_once(readfile, default_schema_path, comm=self.comm))
        self.base_config = Configuration(do_once(readfile, config_path, comm=self.comm), self.schema)
        self.base_sim = Simulation(self.base_config)

    def test_read_restart(self):
        restart_file = self.base_sim.output.state_file_name(1)
        if os.path.exists(restart_file):
            os.remove(restart_file)

        self.assertFalse(os.path.exists(restart_file))
        self.base_sim.step()
        self.assertTrue(os.path.exists(restart_file))

        config = copy.deepcopy(self.base_config)
        config.starting_step = 1

        sim = Simulation(config)
        self.assertEqual(sim.starting_step, 1, f"Starting step is not 1! {sim.starting_step}")

        diff = sim.initial_Q - self.base_sim.Q
        diff_norm = numpy.linalg.norm(diff)
        self.assertTrue(diff_norm == 0.0, f"Restart state is not the same as computed, diff = {diff_norm:.2e}")


class ShallowWaterRestartTestCase(MultiProcRestartTestCase):
    def __init__(self, num_procs, methodName, optional=False):
        super().__init__(num_procs, "shallow_water.ini", methodName, optional)


class Euler3DRestartTestCase(MultiProcRestartTestCase):
    def __init__(self, num_procs, methodName, optional=False):
        super().__init__(num_procs, "euler3d.ini", methodName, optional)
