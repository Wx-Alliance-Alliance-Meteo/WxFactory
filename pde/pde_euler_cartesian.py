import os

from numpy import ndarray

from pde.pde import PDE
from common.definitions import idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta, gravity
from common.device import CudaDevice


class PDEEulerCartesian(PDE):

    def _setup_cpu_kernels(self):
        # Define here the kernels that will be used based on dimension, geometry and riemann solver choice

        from lib.pde.kernels.libkernels import (
            pointwise_eulercartesian_2d_wrapper,
            riemann_eulercartesian_ausm_2d_wrapper,
            boundary_eulercartesian_2d_wrapper,
        )

        self.pointwise_func = pointwise_eulercartesian_2d_wrapper
        self.riemann_func = riemann_eulercartesian_ausm_2d_wrapper
        self.boundary_func = boundary_eulercartesian_2d_wrapper

    def _setup_cuda_kernels(self):
        # Compile speficic kernels
        self.pointwise_func = self.pointwise_module.get_function("euler_flux")
        self.riemann_func = self.riemann_module.get_function("ausm_solver")
        self.boundary_func = self.boundary_module.get_function("boundary_flux")

    def pointwise_fluxes_cuda(self, q: ndarray, flux_x1: ndarray, flux_x2: ndarray, flux_x3: ndarray) -> None:

        if self.num_dim == 2:
            self._kernel_call(self.pointwise_func, self.blocks_pointwise, q, flux_x1, flux_x3, self.nthread_pointwise)

    def pointwise_fluxes_cpu(self, q: ndarray, flux_x1: ndarray, flux_x2: ndarray, flux_x3: ndarray) -> None:

        # Compute the kernel inputs
        nb_elements_x1 = self.config.nb_elements_horizontal
        nb_elements_x3 = self.config.nb_elements_vertical
        nb_solpts = self.config.nbsolpts**2

        # Call CPU kernel
        self.pointwise_func(q, flux_x1, flux_x3, nb_elements_x1, nb_elements_x3, nb_solpts)

    def riemann_fluxes_cuda(
        self,
        q_itf_x1: ndarray,
        q_itf_x2: ndarray,
        q_itf_x3: ndarray,
        fluxes_itf_x1: ndarray,
        flux_itf_x2: ndarray,
        fluxes_itf_x3: ndarray,
    ) -> None:

        stride = self.riem_stride
        nb_var = self.nb_var

        if self.num_dim == 2:
            args_riem_x1 = (q_itf_x1, fluxes_itf_x1, nb_var, 0, self.indxl, self.indxr, stride, self.nthread_riem_x1)

            self._kernel_call(self.riemann_func, self.blocks_riem_x1, *args_riem_x1)

            args_riem_x3 = (q_itf_x3, fluxes_itf_x3, nb_var, 1, self.indzl, self.indzr, stride, self.nthread_riem_x3)

            self._kernel_call(self.riemann_func, self.blocks_riem_x3, *args_riem_x3)

            args_bound_x1 = (q_itf_x1, fluxes_itf_x1, self.indbx, 0, stride, self.nthread_bound_x1)

            self._kernel_call(self.boundary_func, self.blocks_bound_x1, *args_bound_x1)

            args_bound_x3 = (q_itf_x3, fluxes_itf_x3, self.indbz, 1, stride, self.nthread_bound_x3)

            self._kernel_call(self.boundary_func, self.blocks_bound_x3, *args_bound_x3)

    def riemann_fluxes_cpu(
        self,
        q_itf_x1: ndarray,
        q_itf_x2: ndarray,
        q_itf_x3: ndarray,
        fluxes_itf_x1: ndarray,
        flux_itf_x2: ndarray,
        fluxes_itf_x3: ndarray,
    ) -> None:

        # Compute the kernel inputs
        nb_elements_x = self.config.nb_elements_horizontal
        nb_elements_z = self.config.nb_elements_vertical
        nb_solpts = self.config.nbsolpts
        nvar = q_itf_x1.shape[0]
        stride = q_itf_x1[0].size

        # Compute the Riemann fluxes
        self.riemann_func(
            q_itf_x1, q_itf_x3, fluxes_itf_x1, fluxes_itf_x3, nb_elements_x, nb_elements_z, nb_solpts, nvar, stride
        )

        # Update the boundary fluxes
        self.boundary_func(
            q_itf_x1, q_itf_x3, fluxes_itf_x1, fluxes_itf_x3, nb_elements_x, nb_elements_z, nb_solpts, nvar, stride
        )

    def forcing_terms(self, rhs, q):
        rhs[idx_2d_rho_w, :, :] -= q[idx_2d_rho, :, :] * gravity
