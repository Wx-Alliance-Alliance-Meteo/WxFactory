import os

import numpy

from device import Device
from output import InputManager
from simulation import Simulation
from rhs.rhs_dfr import RHSDirecFluxReconstruction_mpi

from mpi_test import MpiTestCase


class PdeRusanovGenericTestCase(MpiTestCase):
    def __init__(self, num_procs, state_dir, methodName, optional=False):
        super().__init__(num_procs, methodName, optional)
        self.state_dir = state_dir
        self.state_files = [os.path.join(self.state_dir, f) for f in os.listdir(self.state_dir)]

    def test_rusanov_kernel(self, device: str):

        for state_file in self.state_files:
            config, global_state = InputManager.read_config_from_save_file(state_file, self.comm)
            config.desired_device = "cpp" if device == "cpu" else "cuda"
            sim = Simulation(config, comm=self.comm)
            local_state = sim.process_topo.distribute_cube(global_state, 4)
            local_state = sim.device.array(local_state)  # Copy to GPU, if needed

            xp = sim.device.xp

            if not isinstance(sim.rhs.full, RHSDirecFluxReconstruction_mpi):
                raise ValueError(f"Incorrect RHS type {type(sim.rhs.full)}. Expected {RHSDirecFluxReconstruction_mpi}")

            sim.rhs.full.allocate_arrays(local_state)
            sim.rhs.full.solution_extrapolation(local_state)
            sim.rhs.full.start_communication()
            sim.rhs.full.pointwise_fluxes(local_state)
            sim.rhs.full.flux_divergence_partial()
            sim.rhs.full.end_communication()
            sim.rhs.full.riemann_fluxes()  # Need that to properly initialize input arrays

            outputs_code = [
                xp.zeros_like(sim.rhs.full.q_itf_full_x1),  # flux x1
                xp.zeros_like(sim.rhs.full.q_itf_full_x2),  # flux x2
                xp.zeros_like(sim.rhs.full.q_itf_full_x3),  # flux x3
                xp.zeros_like(sim.rhs.full.q_itf_full_x1[0]),  # Pressure x1
                xp.zeros_like(sim.rhs.full.q_itf_full_x2[0]),  # Pressure x2
                xp.zeros_like(sim.rhs.full.q_itf_full_x3[0]),  # Pressure x3
                xp.zeros_like(sim.rhs.full.q_itf_full_x1[0]),  # wflux adv x1
                xp.zeros_like(sim.rhs.full.q_itf_full_x1[0]),  # wflux pres x1
                xp.zeros_like(sim.rhs.full.q_itf_full_x2[0]),  # wflux adv x2
                xp.zeros_like(sim.rhs.full.q_itf_full_x2[0]),  # wflux pres x2
                xp.zeros_like(sim.rhs.full.q_itf_full_x3[0]),  # wflux adv x3
                xp.zeros_like(sim.rhs.full.q_itf_full_x3[0]),  # wflux pres x3
            ]
            outputs_py = [xp.zeros_like(a) for a in outputs_code]

            sim.rhs.full.pde.riemann_fluxes_py(
                sim.rhs.full.q_itf_full_x1,
                sim.rhs.full.q_itf_full_x2,
                sim.rhs.full.q_itf_full_x3,
                outputs_py[0],
                outputs_py[1],
                outputs_py[2],
                outputs_py[3],
                outputs_py[4],
                outputs_py[5],
                outputs_py[6],
                outputs_py[7],
                outputs_py[8],
                outputs_py[9],
                outputs_py[10],
                outputs_py[11],
                sim.rhs.full.metric,
            )
            sim.rhs.full.pde.riemann_fluxes_code(
                sim.rhs.full.q_itf_full_x1,
                sim.rhs.full.q_itf_full_x2,
                sim.rhs.full.q_itf_full_x3,
                outputs_code[0],
                outputs_code[1],
                outputs_code[2],
                outputs_code[3],
                outputs_code[4],
                outputs_code[5],
                outputs_code[6],
                outputs_code[7],
                outputs_code[8],
                outputs_code[9],
                outputs_code[10],
                outputs_code[11],
                sim.rhs.full.metric,
            )

            diffs = [b - a for a, b in zip(outputs_py, outputs_code)]
            diff_norms = xp.array([xp.linalg.norm(diff) for diff in diffs])
            for i, ref in enumerate(outputs_py):
                norm = xp.linalg.norm(ref)
                if norm > 0.0:
                    diff_norms[i] /= norm

            threshold = 3e-16
            if xp.any(diff_norms > threshold):
                if self.comm.rank == 0:
                    numpy.set_printoptions(precision=2)
                    print(
                        f"Rank {self.comm.rank} differences: {xp.count_nonzero(diff_norms > threshold)}\n{diff_norms}",
                        flush=True,
                    )
                    # print(f"q itf full x3: \n{sim.rhs.full.q_itf_full_x3[4]}\n", flush=True)
                    # print(
                    #     f"pressure py: \n{outputs_py[5]}\n"
                    #     f"pressure code: \n{outputs_code[5]}\n"
                    #     f"pressure diff: \n{diffs[5]}",
                    #     flush=True,
                    # )

                self.fail(f"Difference is too large")

    def test_rusanov_kernel_cpu(self):
        self.test_rusanov_kernel("cpu")

    def test_rusanov_kernel_gpu(self):
        if not Device.cuda_available():
            self.skipTest(f"Need CUDA for this test")

        self.test_rusanov_kernel("gpu")


class PdeRusanov3DTestCase(PdeRusanovGenericTestCase):

    def __init__(self, num_procs, methodName, optional=False):
        super().__init__(num_procs, "tests/data/unit/sample_state_vectors/euler3d", methodName, optional)
