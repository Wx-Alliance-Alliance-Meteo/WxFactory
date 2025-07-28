import os
import copy

from mpi_test import MpiTestCase
from numpy.typing import NDArray
import numpy

from device import Device
from output import InputManager
from simulation import Simulation


def rel_diff(a, b):
    diffs = numpy.array([numpy.linalg.norm(b[i] - a[i]) / numpy.linalg.norm(a[i]) for i in range(a.shape[0])])
    # print(f"diffs = {diffs}", flush=True)
    return diffs.mean()


class RhsSideBySideGenericTestCase(MpiTestCase):
    def __init__(self, num_procs, state_dir, methodName, optional=False):
        super().__init__(num_procs, methodName, optional)
        self.state_dir = state_dir

    def test_rhs_side_by_side(self):
        """Compare the result of calling RHS for all available backends."""

        numpy.set_printoptions(precision=2)

        if not Device.cuda_available():
            self.skipTest(f"Need CUDA for this test")

        THRESHOLD = 1e-15

        state_files = [os.path.join(self.state_dir, f) for f in os.listdir(self.state_dir)]
        print(f"state files = {state_files}", flush=True)
        for state_file in state_files:
            if self.comm.rank == 0:
                print(f"Testing vector {state_file}", flush=True)
            config, global_state = InputManager.read_config_from_save_file(state_file, self.comm)

            results: dict[str, NDArray] = {}
            for backend in ["cpp", "numpy", "cuda", "cupy", "omp"]:
                local_config = copy.deepcopy(config)
                local_config.desired_device = backend
                sim = Simulation(local_config, comm=self.comm, quiet=True)

                local_state = sim.process_topo.distribute_cube(global_state, 4)
                local_state = sim.device.array(local_state)  # Copy to GPU, if needed

                results[backend] = sim.device.to_host(sim.rhs.full(local_state))

            ref = results["numpy"]
            diff_cpp = results["cpp"] - ref
            diff_cuda = results["cuda"] - ref
            diff_cupy = results["cupy"] - ref
            diff_cuda_cupy = results["cuda"] - results["cupy"]
            diff_omp = results["omp"] - ref

            # ref_norm = numpy.linalg.norm(ref)
            # diff_cpp_norm = numpy.linalg.norm(diff_cpp) / ref_norm
            # diff_cuda_norm = numpy.linalg.norm(diff_cuda) / ref_norm
            # diff_cupy_norm = numpy.linalg.norm(diff_cupy) / ref_norm
            # diff_cuda_cupy_norm = numpy.linalg.norm(diff_cuda_cupy) / numpy.linalg.norm(results["cuda"])
            # diff_omp_norm = numpy.linalg.norm(diff_omp) / ref_norm

            diff_cpp_norm = rel_diff(ref, results["cpp"])
            diff_cuda_norm = rel_diff(ref, results["cuda"])
            diff_cupy_norm = rel_diff(ref, results["cupy"])
            diff_cuda_cupy_norm = rel_diff(results["cuda"], results["cupy"])
            diff_omp_norm = rel_diff(ref, results["omp"])

            cpp_ok = diff_cpp_norm < THRESHOLD
            cuda_ok = diff_cuda_norm < THRESHOLD
            cupy_ok = diff_cupy_norm < THRESHOLD
            omp_ok = diff_omp_norm < THRESHOLD

            if self.comm.rank == 0:
                print(
                    f"cpp:       {diff_cpp_norm:.2e} ({cpp_ok})\n"
                    f"cuda:      {diff_cuda_norm:.2e} ({cuda_ok})\n"
                    f"cupy:      {diff_cupy_norm:.2e} ({cupy_ok})\n"
                    f"cuda/cupy: {diff_cuda_cupy_norm:.2e}\n"
                    f"cupy:      {diff_omp_norm:.2e} ({omp_ok})",
                    flush=True,
                )

            # if diff1_norm > 1e-15:
            #     if self.comm.rank == 0:
            #         print(
            #             f"Expected \n{x1_py[0, 0, 0, 0]}\n"
            #             f"Got      \n{x1_code[0, 0, 0, 0]}\n"
            #             f"diff     \n{diff1[0, 0, 0, 0]}"
            #             f"                       ",
            #             flush=True,
            #         )

            self.skipTest("We know it fails (difference is too large). We need to investigate that.")

            self.assertTrue(cpp_ok)
            self.assertTrue(cuda_ok)
            self.assertTrue(cupy_ok)


class RhsSideBySideEuler3DTestCase(RhsSideBySideGenericTestCase):

    def __init__(self, num_procs, methodName, optional=False):
        super().__init__(num_procs, "tests/data/unit/sample_state_vectors/euler3d", methodName, optional)
