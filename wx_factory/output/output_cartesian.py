from mpi4py import MPI
import numpy

from common.definitions import (
    idx_2d_rho as RHO,
    idx_2d_rho_u as RHO_U,
    idx_2d_rho_w as RHO_W,
    idx_2d_rho_theta as RHO_THETA,
)
from common.graphx import image_field

from .output_manager import OutputManager


class OutputCartesian(OutputManager):
    def __write_result__(self, Q, step_id):
        filename = f"{self.output_dir}/euler2D_{self.config.case_number}_{step_id:08d}"
        Q_cartesian = self.geometry.to_single_block(Q)

        if self.config.case_number == 0:
            image_field(self.geometry, (Q_cartesian[RHO_W, ...]), filename, -1, 1, 25, label="w (m/s)", colormap="bwr")
        elif self.config.case_number <= 2:
            image_field(self.geometry, (Q_cartesian[RHO_THETA, ...] / Q_cartesian[RHO, ...]), filename, 303.1, 303.7, 7)
        elif self.config.case_number == 3:
            image_field(self.geometry, (Q_cartesian[RHO_THETA, ...] / Q_cartesian[RHO, ...]), filename, 303.0, 303.7, 8)
        elif self.config.case_number == 4:
            image_field(
                self.geometry, (Q_cartesian[RHO_THETA, ...] / Q_cartesian[RHO, ...]), filename, 290.0, 300.0, 10
            )

    def __blockstats__(self, Q: numpy.ndarray, step_id: int):

        geom = self.geometry

        def get_stats(field, geom):
            f_minloc = numpy.unravel_index(field.argmin(), field.shape)
            f_maxloc = numpy.unravel_index(field.argmax(), field.shape)
            f_mincoord = (geom.X1[f_minloc], geom.X3[f_minloc])
            f_maxcoord = (geom.X1[f_maxloc], geom.X3[f_maxloc])
            f_min = field[f_minloc]
            f_max = field[f_maxloc]
            f_avg = field.mean()

            return f_mincoord, f_maxcoord, f_min, f_max, f_avg

        rho = Q[RHO]
        rho_mincoord, rho_maxcoord, rho_min, rho_max, rho_avg = get_stats(rho, geom)

        u = Q[RHO_U] / rho
        u_mincoord, u_maxcoord, u_min, u_max, u_avg = get_stats(u, geom)

        w = Q[RHO_W] / rho
        w_mincoord, w_maxcoord, w_min, w_max, w_avg = get_stats(w, geom)

        theta = Q[RHO_THETA] / rho
        theta_mincoord, theta_maxcoord, theta_min, theta_max, theta_avg = get_stats(theta, geom)

        if self.comm.rank == 0:
            print("==============================================")
            print(f" Blockstats for timestep {step_id}")
            print(f"   Var        Min        Max        Mean")
            print(
                f"  ρ        {rho_min:9.2e}  {rho_max:9.2e}  {rho_avg:9.2e}   "
                # f"({rho_mincoord[0]:7.0f}, {rho_mincoord[1]:7.0f})  "
                # f"({rho_maxcoord[0]:7.0f}, {rho_maxcoord[1]:7.0f})"
            )
            print(
                f"  u        {u_min:9.2e}  {u_max:9.2e}  {u_avg:9.2e}   "
                # f"({u_mincoord[0]:7.0f}, {u_mincoord[1]:7.0f})  "
                # f"({u_maxcoord[0]:7.0f}, {u_maxcoord[1]:7.0f})"
            )
            print(
                f"  w        {w_min:9.2e}  {w_max:9.2e}  {w_avg:9.2e}   "
                # f"({w_mincoord[0]:7.0f}, {w_mincoord[1]:7.0f})  "
                # f"({w_maxcoord[0]:7.0f}, {w_maxcoord[1]:7.0f})"
            )
            print(
                f"  θ        {theta_min:9.2e}  {theta_max:9.2e}  {theta_avg:9.2e}   "
                # f"({theta_mincoord[0]:7.0f}, {theta_mincoord[1]:7.0f})  "
                # f"({theta_maxcoord[0]:7.0f}, {theta_maxcoord[1]:7.0f})"
            )
            print("==============================================")
