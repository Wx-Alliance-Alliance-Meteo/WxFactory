import os

from pde.pde import PDE
from numpy import ndarray

from common.definitions import idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta, gravity


class PDEEulerCartesian(PDE):

    cu_file_name = 'kernels/euler_cartesian.cu'

    def __init__(self, config, device):

        super().__init__(config, device)

        print("entering here", self.num_dim)
        if self.num_dim == 2:
            self.nb_elements = self.config.nb_elements_horizontal \
                * self.config.nb_elements_vertical

        self.nb_var = config.nb_var

        print(self.config.device)
        if self.config.device == 'cuda':
            self.pointwise_fluxes = self.pointwise_fluxes_cuda
            self.riemann_fluxes = self.riemann_fluxes_cuda

            kernel_file = os.path.join(
                os.path.dirname(__file__), self.cu_file_name)

            with open(kernel_file, 'r') as f:
                kernel_code = f.read()

            self.module = device.xp.RawModule(code=kernel_code)

            print("loaded self.module")
            # Compile speficic kernels
            self.flux_func = self.module.get_function('euler_flux')
            self.riemm_func = self.module.get_function('ausm_solver')
            self.bound_func = self.module.get_function('boundary_flux')

            # Specify the number of threads needed for pointwise and riemann fluxes
            self.nt_flux = self.nb_elements * self.config.nbsolpts * self.config.nbsolpts
            self.nt_riem_x1 = len(self.indxl)
            self.nt_riem_x3 = len(self.indzl)
            self.nt_fbound_x1 = len(self.indbx)
            self.nt_fbound_x3 = len(self.indbz)

            # Compute the number of launch blocks
            self.threads_per_block = 256
            self.blocks_flux = (self.nt_flux + self.threads_per_block - 1) \
                // self.threads_per_block

            self.blocks_riem_x1 = (self.nt_riem_x1 + self.threads_per_block - 1) \
                // self.threads_per_block
            self.blocks_riem_x3 = (self.nt_riem_x3 + self.threads_per_block - 1) \
                // self.threads_per_block
            self.blocks_riemb_x1 = (self.nt_fbound_x1 + self.threads_per_block - 1) \
                // self.threads_per_block
            self.blocks_riemb_x3 = (self.nt_fbound_x3 + self.threads_per_block - 1) \
                // self.threads_per_block

            if self.num_dim == 3:
                self.nt_riem_x2 = len(self.indyl)
                self.nt_fbound_x2 = len(self.indby)
                self.blocks_riem_x2 = (self.nt_riem_x2 + self.threads_per_block - 1) \
                    // self.threads_per_block
                self.blocks_riemb_x2 = (self.nt_fbound_x2 + self.threads_per_block - 1) \
                    // self.threads_per_block

            self.stride = self.nb_elements * self.config.nbsolpts * 2
        else:
            self.pointwise_fluxes = self.pointwise_fluxes_cpu
            self.riemann_fluxes = self.riemann_fluxes_cpu

    def pointwise_fluxes_cuda(self, q: ndarray,
                              flux_x1: ndarray, flux_x2: ndarray, flux_x3: ndarray):

        if self.num_dim == 2:
            self.flux_func((self.blocks_flux,), (self.threads_per_block,),
                           (q, flux_x1, flux_x3, self.nt_flux))

    def pointwise_fluxes_cpu(self, q: ndarray,
                             flux_x1: ndarray, flux_x2: ndarray, flux_x3: ndarray):
        from pde.kernels.interface import pointwise_fluxes

        # Compute the kernel inputs
        nb_elements = self.config.nb_elements_horizontal * self.config.nb_elements_vertical
        nb_solpts = self.config.nbsolpts

        # Call CPU kernel
        pointwise_fluxes(q, flux_x1, flux_x3, nb_elements, nb_solpts)

    def riemann_fluxes_cuda(self, q_itf_x1: ndarray, q_itf_x2: ndarray, q_itf_x3: ndarray,
                            fluxes_itf_x1: ndarray, flux_itf_x2: ndarray, fluxes_itf_x3: ndarray) -> None:

        self.riemm_func((self.blocks_riem_x1,), (self.threads_per_block,),
                        (q_itf_x1, fluxes_itf_x1, self.nb_var, 0, self.indxl,
                         self.indxr, self.stride, self.nt_riem_x1))

        self.riemm_func((self.blocks_riem_x3,), (self.threads_per_block,),
                        (q_itf_x3, fluxes_itf_x3, self.nb_var, 1, self.indzl,
                         self.indzr, self.stride, self.nt_riem_x3))

        # Set the boundary fluxes
        self.bound_func((self.blocks_riemb_x1,), (self.threads_per_block,),
                        (q_itf_x1, fluxes_itf_x1, self.indbx, 0,  self.stride, self.nt_fbound_x1))

        self.bound_func((self.blocks_riemb_x3,), (self.threads_per_block,),
                        (q_itf_x3, fluxes_itf_x3, self.indbz, 1,  self.stride, self.nt_fbound_x3))

    def riemann_fluxes_cpu(self, q_itf_x1: ndarray, q_itf_x2: ndarray, q_itf_x3: ndarray,
                           fluxes_itf_x1: ndarray, flux_itf_x2: ndarray, fluxes_itf_x3: ndarray) -> None:

        from pde.kernels.interface import riemann_solver

        # Compute the kernel inputs
        nb_elements_x = self.config.nb_elements_horizontal
        nb_elements_z = self.config.nb_elements_vertical
        nb_solpts = self.config.nbsolpts

        riemann_solver(q_itf_x1, q_itf_x3, fluxes_itf_x1,
                       fluxes_itf_x3, nb_elements_x, nb_elements_z, nb_solpts)

    def forcing_terms(self, rhs, q):
        rhs[idx_2d_rho_w, :, :] -= q[idx_2d_rho, :, :] * gravity
