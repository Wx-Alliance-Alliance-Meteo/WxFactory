from pde.pde import PDE
from numpy.typing import NDArray

from pde.kernels.interface import pointwise_fluxes, riemann_solver
from common.definitions import idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta, gravity


class PDEEulerCartesian(PDE):

    def pointwise_fluxes(self, q, flux_x1, flux_x2, flux_x3):

        # Compute the kernel inputs
        nb_elements = self.config.nb_elements_horizontal * self.config.nb_elements_vertical
        nb_solpts = self.config.nbsolpts

        # Call CPU kernel
        pointwise_fluxes(q, flux_x1, flux_x3, nb_elements, nb_solpts)

    def riemann_fluxes(self, q_itf_x1: NDArray, q_itf_x2: NDArray, q_itf_x3: NDArray,
                       fluxes_itf_x1: NDArray, flux_itf_x2: NDArray, fluxes_itf_x3: NDArray) -> None:

        # Compute the kernel inputs
        nb_elements_x = self.config.nb_elements_horizontal
        nb_elements_z = self.config.nb_elements_vertical
        nb_solpts = self.config.nbsolpts

        riemann_solver(q_itf_x1, q_itf_x3, fluxes_itf_x1,
                       fluxes_itf_x3, nb_elements_x, nb_elements_z, nb_solpts)

    def forcing_terms(self, rhs, q):
        rhs[idx_2d_rho_w, :, :] -= q[idx_2d_rho, :, :] * gravity
