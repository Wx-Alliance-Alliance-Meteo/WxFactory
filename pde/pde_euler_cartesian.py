import os
import ctypes

from numpy import ndarray

from pde.pde import PDE
from common.definitions import idx_2d_rho, idx_2d_rho_w, gravity


class PDEEulerCartesian(PDE):

    def pointwise_fluxes(self, q: ndarray, flux_x1: ndarray, flux_x2: ndarray, flux_x3: ndarray) -> None:
                
        nb_elem_x1 = self.config.nb_elements_horizontal
        nb_elem_x3 = self.config.nb_elements_vertical
        nb_solpts_tot = self.config.nbsolpts**2

        # Extract pointer from cpu/gpu arrays
        q_ptr = self.get_pointer(q, self.c_type)
        f_x1_ptr = self.get_pointer(flux_x1, self.c_type)
        f_x3_ptr = self.get_pointer(flux_x3, self.c_type)

        self.pointwise_func(q_ptr, f_x1_ptr, f_x3_ptr, nb_elem_x1, nb_elem_x3, nb_solpts_tot)


    def riemann_fluxes(self, q_itf_x1: ndarray, q_itf_x2: ndarray, q_itf_x3: ndarray, 
                       flux_itf_x1: ndarray, flux_itf_x2: ndarray, flux_itf_x3: ndarray) -> None:
        
        nb_elem_x1 = self.config.nb_elements_horizontal
        nb_elem_x3 = self.config.nb_elements_vertical
        nb_solpts = self.config.nbsolpts

        # Extract pointers from cpu/gpu arrays
        q_x1_ptr = self.get_pointer(q_itf_x1, self.c_type)
        q_x3_ptr = self.get_pointer(q_itf_x3, self.c_type)
        f_x1_ptr = self.get_pointer(flux_itf_x1, self.c_type)
        f_x3_ptr = self.get_pointer(flux_itf_x3, self.c_type)

        self.riemann_func(q_x1_ptr, q_x3_ptr, f_x1_ptr, f_x3_ptr, nb_elem_x1, nb_elem_x3, nb_solpts)
        
    def forcing_terms(self, rhs, q):
        rhs[idx_2d_rho_w, :, :] -= q[idx_2d_rho, :, :] * gravity
