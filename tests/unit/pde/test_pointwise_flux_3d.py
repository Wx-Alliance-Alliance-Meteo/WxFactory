import os

from mpi_test import MpiTestCase
from device import Device
from output import InputManager
from simulation import Simulation

class PDEPointwiseFluxGenericTestCase(MpiTestCase):
    def __init__(self, num_procs, state_dir, methodName, optional=False):
        super().__init__(num_procs, methodName, optional)
        self.state_dir = state_dir

    def rel_diff(self, v1, v2, xp) -> float:
        rel_norm = xp.linalg.norm(v1)
        if rel_norm == 0:
            return xp.linalg.norm(v2)

        return xp.linalg.norm(v1 - v2) / xp.linalg.norm(v1)

    def test_pointwise_flux_kernel(self, device: str):
        state_files = [os.path.join(self.state_dir, f) for f in os.listdir(self.state_dir)]

        for state_file in state_files:
            config, global_state = InputManager.read_config_from_save_file(state_file, self.comm)
            config.desired_device = "cpp" if device == "cpu" else "cuda"
            sim = Simulation(config, comm=self.comm)
            local_state = sim.process_topo.distribute_cube(global_state, 4)
            local_state = sim.device.array(local_state)  # Copy to GPU, if needed

            itf_shape = local_state.shape[:4] + (4 * config.num_solpts**2,)
            pressure_shape = local_state.shape[1:4] + (4 * config.num_solpts**2,)
            
            xp = sim.device.xp
            pressure_py, logp_py, wflux_pres_x1_py, wflux_pres_x2_py, wflux_pres_x3_py, wflux_adv_x1_py, wflux_adv_x2_py, wflux_adv_x3_py = [xp.zeros(pressure_shape, dtype=float) for _ in range(8)]
            flux_x1_py, flux_x2_py, flux_x3_py = [xp.zeros(itf_shape, dtype=float) for _ in range(3)]

            pressure_code, logp_code, wflux_pres_x1_code, wflux_pres_x2_code, wflux_pres_x3_code, wflux_adv_x1_code, wflux_adv_x2_code, wflux_adv_x3_code = [xp.zeros(pressure_shape, dtype=float) for _ in range(8)]
            flux_x1_code, flux_x2_code, flux_x3_code = [xp.zeros(itf_shape, dtype=float) for _ in range(3)]

            sim.rhs.full.pde.pointwise_fluxes_py(local_state, flux_x1_py, flux_x2_py, flux_x3_py, pressure_py, wflux_adv_x1_py, wflux_adv_x2_py, wflux_adv_x3_py, wflux_pres_x1_py, wflux_pres_x2_py, wflux_pres_x3_py, logp_py)
            sim.rhs.full.pde.pointwise_fluxes_code(local_state, flux_x1_code, flux_x2_code, flux_x3_code, pressure_code, wflux_adv_x1_code, wflux_adv_x2_code, wflux_adv_x3_code, wflux_pres_x1_code, wflux_pres_x2_code, wflux_pres_x3_code, logp_code)

            self.assertLessEqual(self.rel_diff(flux_x1_py, flux_x1_code, xp), 4e-16)
            self.assertLessEqual(self.rel_diff(flux_x2_py, flux_x2_code, xp), 4e-16)
            self.assertLessEqual(self.rel_diff(flux_x3_py, flux_x3_code, xp), 4e-16)

            self.assertLessEqual(self.rel_diff(pressure_py, pressure_code, xp), 4e-16)
            self.assertLessEqual(self.rel_diff(logp_py, logp_code, xp), 4e-16)

            self.assertLessEqual(self.rel_diff(wflux_adv_x1_py, wflux_adv_x1_code, xp), 4e-16)
            self.assertLessEqual(self.rel_diff(wflux_adv_x2_py, wflux_adv_x2_code, xp), 4e-16)
            self.assertLessEqual(self.rel_diff(wflux_adv_x3_py, wflux_adv_x3_code, xp), 4e-16)

            self.assertLessEqual(self.rel_diff(wflux_pres_x1_py, wflux_pres_x1_code, xp), 4e-16)
            self.assertLessEqual(self.rel_diff(wflux_pres_x2_py, wflux_pres_x2_code, xp), 4e-16)
            self.assertLessEqual(self.rel_diff(wflux_pres_x3_py, wflux_pres_x3_code, xp), 4e-16)
            
    def test_pointwise_flux_kernel_cpu(self):
        self.test_pointwise_flux_kernel("cpu")

    def test_pointwise_flux_kernel_gpu(self):
        if not Device.cuda_available():
            self.skipTest(f"Need CUDA for this test")
        
        self.test_pointwise_flux_kernel("gpu")

class PDEPointWiseFlux3DTestCase(PDEPointwiseFluxGenericTestCase):
    def __init__(self, num_procs, methodName, optional=False):
        super().__init__(num_procs, "tests/data/unit/sample_state_vectors/euler3d", methodName, optional)