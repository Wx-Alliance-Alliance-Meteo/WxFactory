import os
import time

import numpy

from mpi_test import MpiTestCase
from device import Device
from output import InputManager
from simulation import Simulation
from rhs.rhs import RHS
from pde import PDE


class FakePDE(PDE):
    def __init__(self):
        pass

    def pointwise_fluxes(self, *params):
        pass

    def riemann_fluxes(self, *params):
        pass


class PDERiemannFluxGenericTestCase(MpiTestCase):
    def __init__(self, num_procs, state_dir, methodName, optional=False):
        super().__init__(num_procs, methodName, optional)
        self.state_dir = state_dir

    def rel_diff(self, v1, v2, xp) -> float:
        rel_norm = xp.linalg.norm(v1)
        if rel_norm == 0:
            return xp.linalg.norm(v2)

        return xp.linalg.norm(v1 - v2) / xp.linalg.norm(v1)

    def test_riemann_flux_kernel(self, device: str):
        state_files = [os.path.join(self.state_dir, f) for f in os.listdir(self.state_dir)]

        for state_file in state_files:
            config, global_state = InputManager.read_config_from_save_file(state_file, self.comm)
            config.desired_device = "cpp" if device == "cpu" else "cuda"
            sim = Simulation(config, comm=self.comm)
            local_state = sim.process_topo.distribute_cube(global_state, 4)
            local_state = sim.device.array(local_state)

            xp = sim.device.xp
            sim.rhs.full.allocate_arrays(local_state)
            sim.rhs.full.solution_extrapolation(local_state)
            sim.rhs.full.start_communication()
            sim.rhs.full.pointwise_fluxes(local_state)
            sim.rhs.full.flux_divergence_partial()
            sim.rhs.full.end_communication()

            pde = sim.rhs.full.pde

            sim.rhs.full.pde = FakePDE()
            q_itf_full_x1 = xp.ones_like(sim.rhs.full.q_itf_full_x1)
            q_itf_full_x2 = xp.ones_like(sim.rhs.full.q_itf_full_x2)
            q_itf_full_x3 = xp.ones_like(sim.rhs.full.q_itf_full_x3)

            mid_i = xp.s_[..., 1:-1, :]
            mid_j = xp.s_[..., 1:-1, :, :]
            mid_k = xp.s_[..., 1:-1, :, :, :]

            itf_size = sim.rhs.full.geom.itf_size
            s = numpy.s_[..., 0, :, itf_size:]
            n = numpy.s_[..., -1, :, :itf_size]
            w = numpy.s_[..., 0, itf_size:]
            e = numpy.s_[..., -1, :itf_size]
            b = numpy.s_[..., 0, :, :, itf_size:]
            t = numpy.s_[..., -1, :, :, :itf_size]

            q_itf_full_x1[mid_i] = sim.rhs.full.q_itf_x1
            q_itf_full_x2[mid_j] = sim.rhs.full.q_itf_x2
            q_itf_full_x3[mid_k] = sim.rhs.full.q_itf_x3

            # Element interfaces from neighboring tiles
            q_itf_full_x1[w] = sim.rhs.full.q_itf_w
            q_itf_full_x1[e] = sim.rhs.full.q_itf_e
            q_itf_full_x2[s] = sim.rhs.full.q_itf_s
            q_itf_full_x2[n] = sim.rhs.full.q_itf_n

            # Top + bottom layers
            q_itf_full_x3[b] = q_itf_full_x3[..., 1, :, :, :itf_size]
            q_itf_full_x3[t] = q_itf_full_x3[..., -2, :, :, itf_size:]

            sim.rhs.full.riemann_fluxes()
            q_itf_full_x1 = sim.rhs.full.q_itf_full_x1.copy()
            q_itf_full_x2 = sim.rhs.full.q_itf_full_x2.copy()
            q_itf_full_x3 = sim.rhs.full.q_itf_full_x3.copy()
            # diff1 = q_itf_full_x1 - sim.rhs.full.q_itf_full_x1
            # diff2 = q_itf_full_x2 - sim.rhs.full.q_itf_full_x2
            # diff3 = q_itf_full_x3 - sim.rhs.full.q_itf_full_x3

            # rel1 = self.rel_diff(q_itf_full_x1, sim.rhs.full.q_itf_full_x1, xp)
            # rel2 = self.rel_diff(q_itf_full_x2, sim.rhs.full.q_itf_full_x2, xp)
            # rel3 = self.rel_diff(q_itf_full_x3, sim.rhs.full.q_itf_full_x3, xp)

            # print(f"Rel diffs: {rel1:.2e} {rel2:.2e} {rel3:.2e}", flush=True)

            q_itf_full_x1_py, q_itf_full_x1_code = [sim.rhs.full.q_itf_full_x1.copy() for _ in range(2)]
            q_itf_full_x2_py, q_itf_full_x2_code = [sim.rhs.full.q_itf_full_x2.copy() for _ in range(2)]
            q_itf_full_x3_py, q_itf_full_x3_code = [sim.rhs.full.q_itf_full_x3.copy() for _ in range(2)]
            f_itf_full_x1_py, f_itf_full_x1_code = [xp.zeros_like(sim.rhs.full.f_itf_full_x1) for _ in range(2)]
            f_itf_full_x2_py, f_itf_full_x2_code = [xp.zeros_like(sim.rhs.full.f_itf_full_x2) for _ in range(2)]
            f_itf_full_x3_py, f_itf_full_x3_code = [xp.zeros_like(sim.rhs.full.f_itf_full_x3) for _ in range(2)]

            pressure_itf_full_x1_py, pressure_itf_full_x1_code = [
                sim.rhs.full.pressure_itf_full_x1.copy() for _ in range(2)
            ]
            pressure_itf_full_x2_py, pressure_itf_full_x2_code = [
                sim.rhs.full.pressure_itf_full_x2.copy() for _ in range(2)
            ]
            pressure_itf_full_x3_py, pressure_itf_full_x3_code = [
                sim.rhs.full.pressure_itf_full_x3.copy() for _ in range(2)
            ]

            wflux_adv_itf_full_x1_py, wflux_adv_itf_full_x1_code = [
                sim.rhs.full.wflux_adv_itf_full_x1.copy() for _ in range(2)
            ]
            wflux_adv_itf_full_x2_py, wflux_adv_itf_full_x2_code = [
                sim.rhs.full.wflux_adv_itf_full_x2.copy() for _ in range(2)
            ]
            wflux_adv_itf_full_x3_py, wflux_adv_itf_full_x3_code = [
                sim.rhs.full.wflux_adv_itf_full_x3.copy() for _ in range(2)
            ]

            wflux_pres_itf_full_x1_py, wflux_pres_itf_full_x1_code = [
                sim.rhs.full.wflux_pres_itf_full_x1.copy() for _ in range(2)
            ]
            wflux_pres_itf_full_x2_py, wflux_pres_itf_full_x2_code = [
                sim.rhs.full.wflux_pres_itf_full_x2.copy() for _ in range(2)
            ]
            wflux_pres_itf_full_x3_py, wflux_pres_itf_full_x3_code = [
                sim.rhs.full.wflux_pres_itf_full_x3.copy() for _ in range(2)
            ]

            pde.riemann_fluxes_py(
                q_itf_full_x1_py,
                q_itf_full_x2_py,
                q_itf_full_x3_py,
                f_itf_full_x1_py,
                f_itf_full_x2_py,
                f_itf_full_x3_py,
                pressure_itf_full_x1_py,
                pressure_itf_full_x2_py,
                pressure_itf_full_x3_py,
                wflux_adv_itf_full_x1_py,
                wflux_pres_itf_full_x1_py,
                wflux_adv_itf_full_x2_py,
                wflux_pres_itf_full_x2_py,
                wflux_adv_itf_full_x3_py,
                wflux_pres_itf_full_x3_py,
                sim.rhs.full.metric,
            )

            pde.riemann_fluxes_code(
                q_itf_full_x1_code,
                q_itf_full_x2_code,
                q_itf_full_x3_code,
                f_itf_full_x1_code,
                f_itf_full_x2_code,
                f_itf_full_x3_code,
                pressure_itf_full_x1_code,
                pressure_itf_full_x2_code,
                pressure_itf_full_x3_code,
                wflux_adv_itf_full_x1_code,
                wflux_pres_itf_full_x1_code,
                wflux_adv_itf_full_x2_code,
                wflux_pres_itf_full_x2_code,
                wflux_adv_itf_full_x3_code,
                wflux_pres_itf_full_x3_code,
                sim.rhs.full.metric,
            )

            if self.rel_diff(f_itf_full_x3_py, f_itf_full_x3_code, xp) > 1e-5:
                numpy.set_printoptions(precision=2)
                print(
                    f"Ahhhhh too large error (rank {sim.device.comm.rank})\n",
                    # f"py \n{f_itf_full_x3_py[3]}\n"
                    # f"new \n{f_itf_full_x3_code[3]}\n"
                    # f"diff\n{((f_itf_full_x3_py - f_itf_full_x3_code)/f_itf_full_x3_py)[3]}",
                    flush=True,
                )
                if sim.device.comm.rank == 0:
                    print(
                        # f"py \n{f_itf_full_x3_py[3]}\n"
                        # f"new \n{f_itf_full_x3_code[3]}\n"
                        f"diff\n{((f_itf_full_x3_py - f_itf_full_x3_code)/f_itf_full_x3_py)[3]}",
                        flush=True,
                    )

            # time.sleep(2)

            self.assertLessEqual(self.rel_diff(f_itf_full_x1_py, f_itf_full_x1_code, xp), 4e-16)
            self.assertLessEqual(self.rel_diff(f_itf_full_x2_py, f_itf_full_x2_code, xp), 4e-16)
            self.assertLessEqual(self.rel_diff(f_itf_full_x3_py, f_itf_full_x3_code, xp), 4e-16)
            self.assertLessEqual(self.rel_diff(pressure_itf_full_x1_py, pressure_itf_full_x1_code, xp), 4e-16)
            self.assertLessEqual(self.rel_diff(pressure_itf_full_x2_py, pressure_itf_full_x2_code, xp), 4e-16)
            self.assertLessEqual(self.rel_diff(pressure_itf_full_x3_py, pressure_itf_full_x3_code, xp), 4e-16)
            self.assertLessEqual(self.rel_diff(wflux_pres_itf_full_x1_py, wflux_pres_itf_full_x1_code, xp), 4e-16)
            self.assertLessEqual(self.rel_diff(wflux_pres_itf_full_x2_py, wflux_pres_itf_full_x2_code, xp), 4e-16)
            self.assertLessEqual(self.rel_diff(wflux_pres_itf_full_x3_py, wflux_pres_itf_full_x3_code, xp), 4e-16)
            self.assertLessEqual(self.rel_diff(wflux_adv_itf_full_x1_py, wflux_adv_itf_full_x1_code, xp), 4e-16)
            self.assertLessEqual(self.rel_diff(wflux_adv_itf_full_x2_py, wflux_adv_itf_full_x2_code, xp), 4e-16)
            self.assertLessEqual(self.rel_diff(wflux_adv_itf_full_x3_py, wflux_adv_itf_full_x3_code, xp), 4e-16)

    def test_riemann_flux_kernel_cpu(self):
        self.test_riemann_flux_kernel("cpu")

    def test_riemann_flux_kernel_gpu(self):
        if not Device.cuda_available():
            self.skipTest(f"Need CUDA for this test")

        self.test_riemann_flux_kernel("gpu")


class PDERiemannFlux3DTestCase(PDERiemannFluxGenericTestCase):
    def __init__(self, num_procs, methodName, optional=False):
        super().__init__(num_procs, "tests/data/unit/sample_state_vectors/euler3d", methodName, optional)
