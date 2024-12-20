from numpy.typing import NDArray

from .pde import PDE
from common.definitions import idx_2d_rho, idx_2d_rho_w, gravity
from geometry import Cartesian2D


class PDEEulerCartesian(PDE):

    def __init__(self, geometry: Cartesian2D, config, metric):

        libmodule = geometry.device.libmodule
        super().__init__(
            geometry,
            config,
            metric,
            2,
            4,
            config.num_elements_horizontal * config.num_elements_vertical,
            pointwise_func=libmodule.pointwise_eulercartesian_2d,
            riemann_func=libmodule.riemann_eulercartesian_ausm_2d,
        )

    def pointwise_fluxes(self, q: NDArray, flux_x1: NDArray, flux_x2: NDArray, flux_x3: NDArray) -> None:
        num_elem_x1 = self.config.num_elements_horizontal
        num_elem_x3 = self.config.num_elements_vertical
        num_solpts_tot = self.config.num_solpts**2

        self.pointwise_func(q, flux_x1, flux_x3, num_elem_x1, num_elem_x3, num_solpts_tot)

    def riemann_fluxes(
        self,
        q_itf_x1: NDArray,
        q_itf_x2: NDArray,
        q_itf_x3: NDArray,
        flux_itf_x1: NDArray,
        flux_itf_x2: NDArray,
        flux_itf_x3: NDArray,
    ) -> None:

        num_elem_x1 = self.config.num_elements_horizontal
        num_elem_x3 = self.config.num_elements_vertical
        num_solpts = self.config.num_solpts

        self.riemann_func(q_itf_x1, q_itf_x3, flux_itf_x1, flux_itf_x3, num_elem_x1, num_elem_x3, num_solpts)

    def forcing_terms(self, rhs, q):
        rhs[idx_2d_rho_w, :, :] -= q[idx_2d_rho, :, :] * gravity
