import copy
import os

import numpy
from mpi_test import MpiTestCase, WxTestCase

from wx_factory.common import (
    Configuration,
    ConfigurationSchema,
    default_schema_path,
    load_default_schema,
    readfile,
)
from wx_factory.output.state import load_state
from wx_factory.simulation import Simulation
from wx_factory.wx_mpi import do_once


class Euler2DRestartTestCase(WxTestCase):

    def setUp(self) -> None:
        super().setUp()

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.schema = load_default_schema()
        self.base_config = Configuration(readfile(os.path.join(dir_path, "euler2d.ini")), self.schema)
        self.base_sim = Simulation(self.base_config)

    def test_gen_restart(self):
        """Verify that when running with the save_state option activated, a state is acually generated."""
        restart_file = self.base_sim.output.state_file_name(1)
        if os.path.exists(restart_file):
            os.remove(restart_file)

        self.assertFalse(os.path.exists(restart_file))
        self.base_sim.step()
        self.assertTrue(os.path.exists(restart_file))

    def test_read_restart(self):
        """Verify that we read the state file and that it's the same as if we had run the problem up to that step."""
        restart_file = self.base_sim.output.state_file_name(1)

        # Running up to step 1 will ensure there is a restart file
        new_sim = Simulation(self.base_config)
        new_sim.step()

        self.assertTrue(os.path.exists(restart_file))

        # Creating a Simulation with starting_step=1 will require it to load the restart file
        config = copy.deepcopy(self.base_config)
        config.starting_step = 1
        sim = Simulation(config)
        self.assertEqual(sim.starting_step, 1, f"Starting step is not 1! {sim.starting_step}")

        # Verify that the state just read is the same as was saved
        diff = sim.initial_state.Q - new_sim.Q
        diff_norm = numpy.linalg.norm(diff)
        self.assertTrue(diff_norm == 0.0, "Restart state is not the same as computed")


class MultiProcRestartTestCase(MpiTestCase):
    def __init__(self, num_procs: int, config_file: str, methodName: str, device_name: str, optional: bool = False):
        super().__init__(num_procs, methodName, device_name, optional)
        self.config_file = config_file

    def setUp(self):
        super().setUp()

        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(dir_path, self.config_file)

        self.schema = ConfigurationSchema(do_once(readfile, default_schema_path, comm=self.comm))
        self.base_config = Configuration(do_once(readfile, config_path, comm=self.comm), self.schema)
        self.base_config.pytorch_device = self.device_name
        self.base_sim = Simulation(self.base_config, comm=self.comm, context=self.context)

        self.smaller_comm = self.comm.Split(self.comm.rank < 6, self.comm.rank)
        if self.comm.rank >= 6:
            self.smaller_comm = None
        else:
            self.smaller_sim = Simulation(self.base_config, comm=self.smaller_comm)

    def test_read_restart(self):
        """Verify that we read the state file and that it's the same as if we had run the problem up to that step."""
        restart_file = self.base_sim.output.state_file_name(1)

        # Remove existing restart file, if any
        if self.comm.rank == 0:
            if os.path.exists(restart_file):
                os.remove(restart_file)

            self.assertFalse(os.path.exists(restart_file))

        # Run 1 step to generate restart file
        self.base_sim.step()

        if self.comm.rank == 0:
            self.assertTrue(os.path.exists(restart_file))

        # Create Simulation with starting_step=1 to load the restart file
        config = copy.deepcopy(self.base_config)
        config.starting_step = 1
        config.pytorch_device = self.device_name
        sim = Simulation(config, comm=self.comm)
        self.assertEqual(sim.starting_step, 1, f"Starting step is not 1! {sim.starting_step}")

        self.assertIn(self.device_name, str(self.base_sim.Q.device), "Wrong device type for base_sim.Q")
        self.assertIn(self.device_name, str(sim.initial_state.Q.device), "Wrong device type for sim.initial_state.Q")

        # Verify that the loaded state is the same as the simulated one
        diff = sim.initial_state.Q - self.base_sim.Q
        diff_norm = numpy.linalg.norm(diff)
        self.assertTrue(diff_norm == 0.0, f"Restart state is not the same as computed, diff = {diff_norm:.2e}")

    def test_multisize(self):
        """Verify that we can use the same restart file for different processor counts"""

        self.assertGreaterEqual(
            self.comm.size, 24, f"We need at least 24 processors for this test, but we only have {self.comm.size}"
        )

        restart_file = self.base_sim.output.state_file_name(1)
        restart_file_ref = f"{restart_file}.ref"

        # Make sure there is not restart file
        if self.comm.rank == 0:
            if os.path.exists(restart_file):
                os.remove(restart_file)

            self.assertFalse(os.path.exists(restart_file))

        # Generate the restart, then move it to use as a reference
        self.base_sim.step()

        if self.comm.rank == 0:
            self.assertTrue(os.path.exists(restart_file))
            os.rename(restart_file, restart_file_ref)

            self.assertFalse(os.path.exists(restart_file))

        # With a smaller number of processors, use the same config to generate a restart
        # If both restart files are identical, this means different processor counts can generate/read the
        # same file
        if self.smaller_comm is not None:
            self.smaller_sim.step()

            # Compare the two restart files
            if self.smaller_comm.rank == 0:
                self.assertTrue(os.path.exists(restart_file))
                q1, _ = load_state(restart_file_ref)
                q2, _ = load_state(restart_file)

                diff = q1 - q2
                rel_norm = numpy.linalg.norm(diff) / numpy.linalg.norm(q1)

                self.assertLessEqual(rel_norm, 1e-15, "Result should be the same with different proc counts")

        self.comm.barrier()


class ShallowWaterRestartTestCase(MultiProcRestartTestCase):
    def __init__(self, num_procs: int, methodName: str, device_name: str, optional=False):
        super().__init__(num_procs, "shallow_water.ini", methodName, device_name, optional)


class Euler3DRestartTestCase(MultiProcRestartTestCase):
    def __init__(self, num_procs: int, methodName: str, device_name: str, optional=False):
        super().__init__(num_procs, "euler3d.ini", methodName, device_name, optional)
