import copy
import os
import unittest

from mpi4py import MPI

from common import Configuration
from device import Device
from output import InputManager
from simulation import Simulation
from wx_mpi import SingleProcess, Conditional

from mpi_test import MpiTestCase


class OperatorsExtrapGenericTestCase(MpiTestCase):
    def __init__(self, num_procs, state_dir, methodName, optional=False):
        super().__init__(num_procs, methodName, optional)
        self.state_dir = state_dir

    def test_extrap_kernel(self, device: str):
        state_files = [os.path.join(self.state_dir, f) for f in os.listdir(self.state_dir)]

        if self.comm.rank == 0:
            print(f"Device: {device}", flush=True)

        for state_file in state_files:
            config, global_state = InputManager.read_config_from_save_file(state_file, self.comm)
            config.desired_device = device
            sim = Simulation(config, comm=self.comm)
            local_state = sim.process_topo.distribute_cube(global_state, 4)
            local_state = sim.device.array(local_state)  # Copy to GPU, if needed

            itf_shape = local_state.shape[:4] + (2 * config.num_solpts**2,)
            xp = sim.device.xp
            x1_py, x2_py, x3_py, x1_code, x2_code, x3_code = [xp.zeros(itf_shape, dtype=float) for _ in range(6)]

            sim.rhs.full.extrap_3d_py(local_state, x1_py, x2_py, x3_py)
            sim.rhs.full.extrap_3d_code(local_state, x1_code, x2_code, x3_code)

            diff1 = x1_py - x1_code
            diff2 = x2_py - x2_code
            diff3 = x3_py - x3_code
            diff1_norm = xp.linalg.norm(diff1) / xp.linalg.norm(x1_py)
            diff2_norm = xp.linalg.norm(diff2) / xp.linalg.norm(x2_py)
            diff3_norm = xp.linalg.norm(diff3) / xp.linalg.norm(x3_py)

            if diff1_norm > 1e-15:
                if self.comm.rank == 0:
                    print(
                        f"Expected \n{x1_py[0, 0, 0, 0]}\n"
                        f"Got      \n{x1_code[0, 0, 0, 0]}\n"
                        f"diff     \n{diff1[0, 0, 0, 0]}"
                        f"                       ",
                        flush=True,
                    )

            self.assertLessEqual(diff1_norm, 4e-16)
            self.assertLessEqual(diff2_norm, 4e-16)
            self.assertLessEqual(diff3_norm, 4e-16)

    def test_extrap_kernel_cpu(self):
        self.test_extrap_kernel("cpp")

    def test_extrap_kernel_gpu(self):
        if not Device.cuda_available():
            self.skipTest(f"Need CUDA for this test")

        self.test_extrap_kernel("cuda")
        self.test_extrap_kernel("omp")


class OperatorsExtrapEuler3DTestCase(OperatorsExtrapGenericTestCase):

    def __init__(self, num_procs, methodName, optional=False):
        super().__init__(num_procs, "tests/data/unit/sample_state_vectors/euler3d", methodName, optional)
