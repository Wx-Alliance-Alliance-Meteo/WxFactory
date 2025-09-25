import os
import copy

from mpi_test import MpiTestCase
from numpy.typing import NDArray
import numpy

from device import Device
from output import InputManager
from simulation import Simulation
from rhs.rhs_dfr import RHSDirecFluxReconstruction_mpi


def rel_diff(a: NDArray, b: NDArray) -> float:
    vars = []
    for i in range(a.shape[0]):
        vars.append(numpy.linalg.norm(b[i] - a[i]))
        ref_norm = numpy.linalg.norm(a[i])
        if ref_norm > 0.0:
            vars[i] /= ref_norm
    # print(f"diffs = {diffs}", flush=True)
    diffs = numpy.array(vars)
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

            backends = ["numpy", "cpp", "cuda", "cupy"]
            results: dict[str, NDArray] = {}

            simulations: list[Simulation] = []
            local_states: list[NDArray] = []
            rhss: list[RHSDirecFluxReconstruction_mpi] = []

            for backend in backends:
                local_config = copy.deepcopy(config)
                local_config.desired_device = backend
                sim = Simulation(local_config, comm=self.comm, quiet=True)

                local_state = sim.process_topo.distribute_cube(global_state, 4)
                local_state = sim.device.array(local_state)  # Copy to GPU, if needed

                simulations.append(sim)
                local_states.append(local_state)
                rhss.append(sim.rhs.full)

                # results[backend] = sim.device.to_host(sim.rhs.full(local_state))

            for i in range(len(backends)):
                rhs = rhss[i]
                q = local_states[i]
                rhs.allocate_arrays(q)
                rhs.solution_extrapolation(q)

            x1_diff = [rel_diff(rhss[0].q_itf_x1, simulations[i].device.to_host(rhss[i].q_itf_x1)) for i in range(1, 4)]
            x2_diff = [rel_diff(rhss[0].q_itf_x2, simulations[i].device.to_host(rhss[i].q_itf_x2)) for i in range(1, 4)]
            x3_diff = [rel_diff(rhss[0].q_itf_x3, simulations[i].device.to_host(rhss[i].q_itf_x3)) for i in range(1, 4)]

            print(f"{self.comm.rank} differences \nx1 {x1_diff}, \nx2 {x2_diff}, \nx3 {x3_diff}", flush=True)

            res_tmp = [None, None, None, None]
            out: list[NDArray] = [None, None, None, None]
            for i in [0, 3]:
                sim = simulations[i]
                rhs = rhss[i]
                xp = sim.device.xp
                arrays = [
                    rhs.f_x1,
                    rhs.f_x2,
                    rhs.f_x3,
                    rhs.pressure,
                    rhs.wflux_adv_x1,
                    rhs.wflux_adv_x2,
                    rhs.wflux_adv_x3,
                    rhs.wflux_pres_x1,
                    rhs.wflux_pres_x2,
                    rhs.wflux_pres_x3,
                    rhs.log_p,
                ]
                inputs = [xp.zeros_like(x) for x in arrays]

                r = rhs.pde.pointwise_fluxes_py(
                    local_states[i],
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    inputs[8],
                    inputs[9],
                    inputs[10],
                )

                res_tmp[i] = inputs
                out[i] = sim.device.to_host(r)

            # for i in range(len(backends)):
            #     rhss[i].start_communication()
            #     rhss[i].pointwise_fluxes(local_states[i])

            # f1_diff = [rel_diff(rhss[0].f_x1, simulations[i].device.to_host(rhss[i].f_x1)) for i in range(1, 4)]
            # f2_diff = [rel_diff(rhss[0].f_x2, simulations[i].device.to_host(rhss[i].f_x2)) for i in range(1, 4)]
            # f3_diff = [rel_diff(rhss[0].f_x3, simulations[i].device.to_host(rhss[i].f_x3)) for i in range(1, 4)]

            # print(f"{self.comm.rank} differences \nfx1 {f1_diff}, \nfx2 {f2_diff}, \nfx3 {f3_diff}", flush=True)

            if self.comm.rank == 0:
                a = out[0]
                b = out[3]
                diff = (b - a) / numpy.linalg.norm(a)
                print(
                    f"numpy f1 =                           \n{a}\n"
                    f"cupy f1 =                            \n{b}\n"
                    f"diff f1 = \n{diff}\n"
                    f"max = {diff.max()}",
                    flush=True,
                )

            # dx1_diff = [rel_diff(rhss[0].df1_dx1, simulations[i].device.to_host(rhss[i].df1_dx1)) for i in range(1, 4)]
            # dx2_diff = [rel_diff(rhss[0].df2_dx2, simulations[i].device.to_host(rhss[i].df2_dx2)) for i in range(1, 4)]
            # dx3_diff = [rel_diff(rhss[0].df3_dx3, simulations[i].device.to_host(rhss[i].df3_dx3)) for i in range(1, 4)]

            # print(f"{self.comm.rank} differences \ndx1 {dx1_diff}, \ndx2 {dx2_diff}, \ndx3 {dx3_diff}", flush=True)
            self.comm.Barrier()

            ref = results["numpy"]
            diff_cpp = results["cpp"] - ref
            diff_cuda = results["cuda"] - ref
            diff_cupy = results["cupy"] - ref
            diff_cuda_cupy = results["cuda"] - results["cupy"]
            # diff_omp = results["omp"] - ref

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
            # diff_omp_norm = rel_diff(ref, results["omp"])

            cpp_ok = diff_cpp_norm < THRESHOLD
            cuda_ok = diff_cuda_norm < THRESHOLD
            cupy_ok = diff_cupy_norm < THRESHOLD
            # omp_ok = diff_omp_norm < THRESHOLD

            if self.comm.rank == 0:
                print(
                    f"cpp:       {diff_cpp_norm:.2e} ({cpp_ok})\n"
                    f"cuda:      {diff_cuda_norm:.2e} ({cuda_ok})\n"
                    f"cupy:      {diff_cupy_norm:.2e} ({cupy_ok})\n"
                    f"cuda/cupy: {diff_cuda_cupy_norm:.2e}\n",
                    # f"cupy:      {diff_omp_norm:.2e} ({omp_ok})",
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

            # self.skipTest("We know it fails (difference is too large). We need to investigate that.")

            self.assertTrue(cpp_ok)
            self.assertTrue(cuda_ok)
            self.assertTrue(cupy_ok)
            # self.assertTrue(omp_ok)


class RhsSideBySideEuler3DTestCase(RhsSideBySideGenericTestCase):

    def __init__(self, num_procs, methodName, optional=False):
        super().__init__(num_procs, "tests/data/unit/sample_state_vectors/euler3d", methodName, optional)
