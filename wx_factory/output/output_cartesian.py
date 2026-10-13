"""The Cartesian slab is y-invariant, so a middle y-plane represents its x-z output."""

import torch

from ..common.definitions import idx_rho, idx_rho_theta
from ..common.graphx import image_field
from .output_manager import OutputManager


class OutputCartesian(OutputManager):
    def __init__(self, config, geometry, operators, context, metric=None):
        super().__init__(config, geometry, operators, context)
        self.metric = metric

    def _physical_winds(self, Q):
        return self.metric.physical_winds(Q[1:4])

    def __write_result__(self, Q, step_id):
        filename = f"{self.output_dir}/euler_cartesian_{self.config.case_number}_{step_id:08d}"

        block = self.geometry.to_single_block(Q)
        j = block.shape[2] // 2
        q_xz = block[:, :, j, :]

        rho = q_xz[idx_rho]
        theta = q_xz[idx_rho_theta] / rho
        u1, _u2, u3 = (self.geometry.to_single_block(wind)[:, j, :] for wind in self._physical_winds(Q))
        speed = torch.sqrt(u1**2 + u3**2)
        w = u3

        if self.config.case_number == 0:
            image_field(self.geometry, w, filename, -1, 1, 25, label="w (m/s)", colormap="bwr")
        elif self.config.case_number == 651:
            image_field(
                self.geometry,
                speed,
                filename,
                0.,
                1.,
                20,
                label="|u| (m/s)",
                colormap="plasma"
            )
        elif self.config.case_number <= 2:
            image_field(self.geometry, theta, filename, 303.1, 303.7, 7)
        elif self.config.case_number == 3:
            image_field(self.geometry, theta, filename, 303.0, 303.7, 8)
        elif self.config.case_number == 4:
            background_theta = self.config.bubble_theta
            # Roundoff at the background value must not trigger the over-range color.
            theta = torch.where(torch.abs(theta - background_theta) < 1e-10, background_theta, theta)
            image_field(self.geometry, theta, filename, background_theta - 18.0, background_theta, 19, label="θ (K)")
        else:
            image_field(self.geometry, theta, filename, float(theta.min()), float(theta.max()), 20)

    def __blockstats__(self, Q, step_id):
        rho = Q[idx_rho]
        theta = Q[idx_rho_theta] / rho
        u1, u2, u3 = self._physical_winds(Q)
        if self.comm.rank == 0:
            print("==============================================", flush=True)
            print(f" Blockstats for timestep {step_id}", flush=True)
            for name, f in (("rho", rho), ("u1", u1), ("u2", u2), ("u3", u3), ("theta", theta)):
                print(f"   {name:6s} min {float(torch.min(f)):+.6e}  max {float(torch.max(f)):+.6e}", flush=True)
