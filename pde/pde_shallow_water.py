from mpi4py import MPI

from pde.pde import PDE
from numpy import ndarray

from common.definitions import idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta, gravity


class PDEShallowWater(PDE):

    cu_file_name = 'kernels/sw_cubedsphere.cu'

    def __init__(self, config, device, geom, metric):

        super().__init__(config, device, geom, metric)

        if self.num_dim == 2:
            self.nb_elements = self.config.nb_elements_horizontal \
                * self.config.nb_elements_vertical

        self.nb_var = config.nb_var

        self.pointwise_fluxes = self.pointwise_fluxes_cpu
        self.riemann_fluxes = self.riemann_fluxes_cpu

    def pointwise_fluxes_cuda(self, q: ndarray,
                              flux_x1: ndarray, flux_x2: ndarray, flux_x3: ndarray):

        pass

    def pointwise_fluxes_cpu(self, q: ndarray,
                             flux_x1: ndarray, flux_x2: ndarray, flux_x3: ndarray):
        from pde.kernels.interface import pointwise_fluxes

        # Compute the kernel inputs
        nb_elements = self.config.nb_elements_horizontal * self.config.nb_elements_vertical
        nb_solpts = self.config.nbsolpts

        if MPI.COMM_WORLD.rank == 0:
            print(self.sqrt_g)
            print(q[0])
        exit()
        
        # Call CPU kernel
        pointwise_fluxes(q, flux_x1, flux_x2, flux_x3, self.metrics, self.sqrt_g,  nb_elements, nb_solpts, self.num_dim)

    def riemann_fluxes_cuda(self, q_itf_x1: ndarray, q_itf_x2: ndarray, q_itf_x3: ndarray,
                            fluxes_itf_x1: ndarray, flux_itf_x2: ndarray, fluxes_itf_x3: ndarray) -> None:

        pass

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
