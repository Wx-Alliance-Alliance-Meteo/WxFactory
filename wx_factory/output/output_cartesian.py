"""Image output for the flat cartesian slab (Cartesian3D).

A cartesian slab has no cube panels to assemble into a NetCDF file, so it is visualised directly as
x-z contour plots (the classic 2D bubble/current pictures). The slab is a thin y-invariant 3D grid,
so any y-plane is representative; we take the middle one and plot it with :func:`image_field`.
"""

import torch

from ..common.definitions import idx_rho, idx_rho_theta, idx_rho_u1, idx_rho_u3
from ..common.graphx import image_field
from .output_manager import OutputManager


class OutputCartesian(OutputManager):
    def __write_result__(self, Q, step_id):
        filename = f"{self.output_dir}/euler_cartesian_{self.config.case_number}_{step_id:08d}"

        block = self.geometry.to_single_block(Q)  # (nvar, nk, nj, ni)
        j = block.shape[2] // 2  # y-invariant slab: pick a representative y-plane
        q_xz = block[:, :, j, :]  # (nvar, nk, ni)

        rho = q_xz[idx_rho]
        theta = q_xz[idx_rho_theta] / rho
        w = q_xz[idx_rho_u3] / rho

        if self.config.case_number == 0:
            image_field(self.geometry, w, filename, -1, 1, 25, label="w (m/s)", colormap="bwr")
        elif self.config.case_number <= 2:
            image_field(self.geometry, theta, filename, 303.1, 303.7, 7)
        elif self.config.case_number == 3:
            image_field(self.geometry, theta, filename, 303.0, 303.7, 8)
        elif self.config.case_number == 4:
            image_field(self.geometry, theta, filename, 290.0, 300.0, 10)
        else:
            image_field(self.geometry, theta, filename, float(theta.min()), float(theta.max()), 20)

    def __blockstats__(self, Q, step_id):
        rho = Q[idx_rho]
        theta = Q[idx_rho_theta] / rho
        u1 = Q[idx_rho_u1] / rho
        u3 = Q[idx_rho_u3] / rho
        if self.comm.rank == 0:
            print("==============================================", flush=True)
            print(f" Blockstats for timestep {step_id}", flush=True)
            for name, f in (("rho", rho), ("u1", u1), ("u3", u3), ("theta", theta)):
                print(f"   {name:6s} min {float(torch.min(f)):+.6e}  max {float(torch.max(f)):+.6e}", flush=True)
