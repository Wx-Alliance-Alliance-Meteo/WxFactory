import os

from mpi_test import MpiTestCase
from device import Device
from output import InputManager
from simulation import Simulation

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
            local_state = sim.device.array(local_state)  # Copy to GPU, if needed

            itf_shape = local_state.shape[:4] + (4 * config.num_solpts**2,)
            pressure_shape = local_state.shape[1:4] + (4 * config.num_solpts**2,)
            
            xp = sim.device.xp
            