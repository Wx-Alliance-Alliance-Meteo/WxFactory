from numpy.typing import NDArray

from .pde import PDE
from common.definitions import idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, gravity
from geometry import Cartesian2D
from init.entropy_vars import conservative_to_entropy


class PDEEulerCartesian(PDE):

    def __init__(self, geometry: Cartesian2D, config, metric):

        pde = geometry.device.pde
        super().__init__(
            geometry,
            config,
            metric,
            num_dim=2,
            num_var=4,
            num_elem=geometry.num_elements_horizontal * geometry.num_elements_vertical,
            pointwise_func=pde.pointwise_eulercartesian_2d,
            riemann_func=self.get_riemann_solver(pde, "ausm"),
        )

    @staticmethod
    def get_riemann_solver(pde, name):
        if name == "ausm":
            return pde.riemann_eulercartesian_ausm_2d

    def pointwise_fluxes(self, q: NDArray, flux_x1: NDArray, flux_x2: NDArray, flux_x3: NDArray) -> None:
        num_elem_x1 = self.geometry.num_elements_horizontal
        num_elem_x3 = self.geometry.num_elements_vertical
        num_solpts_tot = self.geometry.num_solpts**2

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

        num_elem_x1 = self.geometry.num_elements_horizontal
        num_elem_x3 = self.geometry.num_elements_vertical
        num_solpts = self.geometry.num_solpts

        self.riemann_func(q_itf_x1, q_itf_x3, flux_itf_x1, flux_itf_x3, num_elem_x1, num_elem_x3, num_solpts)

    def forcing_terms(self, rhs, q):
        rhs[idx_2d_rho_w, :, :] -= q[idx_2d_rho, :, :] * gravity
        
    def entropy_average(
        self,
        q_itf_x1: NDArray,
        q_itf_x3: NDArray,
        bc = "other"
    ):
        """Computes averages of entropy variables. No-slip boundary conditions."""
        xp = self.device.xp
        
        num_elements_horizontal = self.geometry.num_elements_horizontal
        num_elements_vertical = self.geometry.num_elements_vertical
        num_solpts = self.geometry.num_solpts
        
        # Indices to get extrapolated values on the west/down and east/up within one element
        west_indices = slice(0,num_solpts)
        east_indices = slice(num_solpts,2*num_solpts)
        down_indices = west_indices
        up_indices = east_indices
        
        
        # 1. Add ghost cells
        q_itf_ghost_x1 = xp.pad(q_itf_x1, ((0, 0), (0, 0), (1, 1), (0, 0)), mode='constant',constant_values=0) # (num_eqs, num_el_vertical, num_el_horizontal+2, 2*num_solpts)
        q_itf_ghost_x3 = xp.pad(q_itf_x3, ((0, 0), (1, 1), (0, 0), (0, 0)), mode='constant',constant_values=0) # (num_eqs, num_el_vertical+2, num_el_horizontal, 2*num_solpts)

        if bc == "periodic":
            # Copy the values at the boundaries to the ghost cells
            q_itf_ghost_x1[:, :, 0, east_indices] =  q_itf_ghost_x1[:, :, -2, east_indices]
            q_itf_ghost_x1[:, :, -1, west_indices] =  q_itf_ghost_x1[:, :, 1, west_indices]
        
            # # from bottom to top
            q_itf_ghost_x3[:, 0, :, up_indices] =  q_itf_ghost_x3[:, -2, :, up_indices]
            q_itf_ghost_x3[:, -1, :, down_indices] =  q_itf_ghost_x3[:, 1, :, down_indices]
        else:     
            # Copy the values at the boundaries to the ghost cells
            q_itf_ghost_x1[:, :, 0, east_indices] =  q_itf_ghost_x1[:, :, 1, west_indices]
            q_itf_ghost_x1[:, :, -1, west_indices] =  q_itf_ghost_x1[:, :, -2, east_indices]
        
            # # from bottom to top
            q_itf_ghost_x3[:, 0, :, up_indices] =  q_itf_ghost_x3[:, 1, :, down_indices]
            q_itf_ghost_x3[:, -1, :, down_indices] =  q_itf_ghost_x3[:, -2, :, up_indices]
        
            # # # from top to bottom
            # q_itf_ghost_x3[:, 0, :, down_indices] =  q_itf_ghost_x3[:, 1, :, up_indices]
            # q_itf_ghost_x3[:, -1, :, up_indices] =  q_itf_ghost_x3[:, -2, :, down_indices]
            
            # Enforce no-slip boundary conditions (uu=0,ww=0) at ghost cells (set uu, ww to appropriate negative values at ghost cells)
            q_itf_ghost_x1[idx_2d_rho_u:idx_2d_rho_w+1, :, 0, east_indices] =  -q_itf_ghost_x1[idx_2d_rho_u:idx_2d_rho_w+1, :, 0, east_indices]
            q_itf_ghost_x1[idx_2d_rho_u:idx_2d_rho_w+1, :, -1, west_indices] =  -q_itf_ghost_x1[idx_2d_rho_u:idx_2d_rho_w+1, :, -1, west_indices]
            
            # # from bottom to top
            q_itf_ghost_x3[idx_2d_rho_u:idx_2d_rho_w+1,  0, :, up_indices] =  -q_itf_ghost_x3[idx_2d_rho_u:idx_2d_rho_w+1,  0, :, up_indices]
            q_itf_ghost_x3[idx_2d_rho_u:idx_2d_rho_w+1, -1, :, down_indices] =  -q_itf_ghost_x3[idx_2d_rho_u:idx_2d_rho_w+1, -1, :, down_indices]
            # # # # # from top to bottom
            # q_itf_ghost_x3[idx_2d_rho_u:idx_2d_rho_w+1,  0, :, down_indices] =  -q_itf_ghost_x3[idx_2d_rho_u:idx_2d_rho_w+1,  0, :, down_indices]
            # q_itf_ghost_x3[idx_2d_rho_u:idx_2d_rho_w+1, -1, :, up_indices] =  -q_itf_ghost_x3[idx_2d_rho_u:idx_2d_rho_w+1, -1, :, up_indices]
        
        
        # 2. Compute q- and q+
        # q- and q+ to compute avg on the western boundary of the element 
        q_minus_west = q_itf_ghost_x1[:, :, 1:num_elements_horizontal+1, west_indices] # (num_eqs, num_el_vertical, num_el_horizontal, num_solpts)
        q_plus_west = q_itf_ghost_x1[:, :, 0:num_elements_horizontal, east_indices]
        
        # q- and q+ to compute avg on the western boundary of the element
        q_minus_east = q_itf_ghost_x1[:, :, 1:num_elements_horizontal+1, east_indices] 
        q_plus_east = q_itf_ghost_x1[:, :, 2:num_elements_horizontal+2, west_indices]
        
        # # from bottom to top
        # q- and q+ to compute avg on the lower boundary of the element 
        q_minus_down = q_itf_ghost_x3[:, 1:num_elements_vertical+1, :, down_indices] 
        q_plus_down = q_itf_ghost_x3[:, 0:num_elements_vertical, :, up_indices]
        
        # q- and q+ to compute avg on the western boundary of the element
        q_minus_up = q_itf_ghost_x3[:, 1:num_elements_vertical+1, :, up_indices] 
        q_plus_up = q_itf_ghost_x3[:, 2:num_elements_vertical+2, :, down_indices]
        
        # from top to bottom
        # q- and q+ to compute avg on the lower boundary of the element 
        # q_minus_down = q_itf_ghost_x3[:, 1:num_elements_vertical+1, :, down_indices] 
        # q_plus_down = q_itf_ghost_x3[:, 2:num_elements_vertical+2, :, up_indices]
        
        # # q- and q+ to compute avg on the western boundary of the element
        # q_minus_up = q_itf_ghost_x3[:, 1:num_elements_vertical+1, :, up_indices] 
        # q_plus_up = q_itf_ghost_x3[:, 0:num_elements_vertical, :, down_indices]
    
        # 3. Compute v(u-) and v(u+)
        v_minus_west = conservative_to_entropy(q_minus_west, self.geometry, self.config)
        v_plus_west = conservative_to_entropy(q_plus_west, self.geometry, self.config)
        
        v_minus_east = conservative_to_entropy(q_minus_east, self.geometry, self.config)
        v_plus_east = conservative_to_entropy(q_plus_east, self.geometry, self.config)
        
        v_minus_down = conservative_to_entropy(q_minus_down, self.geometry, self.config) 
        v_plus_down = conservative_to_entropy(q_plus_down, self.geometry, self.config)
        
        v_minus_up = conservative_to_entropy(q_minus_up, self.geometry, self.config) 
        v_plus_up = conservative_to_entropy(q_plus_up, self.geometry, self.config) 
        
        # 4. Compute the average: {v} = 1/2 * [v(u-) + v(u+)]
        v_avg_west = 0.5 * (v_minus_west + v_plus_west)
        v_avg_east = 0.5 * (v_minus_east + v_plus_east)
        
        v_avg_down = 0.5 * (v_minus_down + v_plus_down)
        v_avg_up = 0.5 * (v_minus_up + v_plus_up)
        
        # 5. Concantenate the arays
        v_avg_x1 = xp.concatenate([v_avg_west, v_avg_east], axis=3) # (num_eqs, num_elements_vertical, num_elements_horizontal, 2*num_solpts)
        v_avg_x3 = xp.concatenate([v_avg_down, v_avg_up], axis=3)
        
        return v_avg_x1,v_avg_x3
    
    def viscous_flux_average(
        self,
        g1_itf_x1: NDArray,
        g3_itf_x3: NDArray,
        bc = "other"
    ):
        """Computes averages of viscous flux . Flux equals 0 at the boundary."""
        xp = self.device.xp
        
        num_elements_horizontal = self.geometry.num_elements_horizontal
        num_elements_vertical = self.geometry.num_elements_vertical
        num_solpts = self.geometry.num_solpts
        
        # Indices to get extrapolated values on the west/down and east/up within one element
        west_indices = slice(0,num_solpts)
        east_indices = slice(num_solpts,2*num_solpts)
        down_indices = west_indices
        up_indices = east_indices
        
        # 1. Add ghost cells. 
        g1_itf_ghost_x1 = xp.pad(g1_itf_x1, ((0, 0), (0, 0), (1, 1), (0, 0)), mode='constant',constant_values=0) # (num_eqs, num_el_vertical, num_el_horizontal+2, 2*num_solpts)
        g3_itf_ghost_x3 = xp.pad(g3_itf_x3, ((0, 0), (1, 1), (0, 0), (0, 0)), mode='constant',constant_values=0) # (num_eqs, num_el_vertical+2, num_el_horizontal, 2*num_solpts)
        
        if bc == "periodic":
            # Enforce wall conditions (g=0) by setting all values at the ghost cells to appropriate negative values
            g1_itf_ghost_x1[:, :, 0, east_indices] =  g1_itf_ghost_x1[:, :, -2, east_indices]
            g1_itf_ghost_x1[:, :, -1, west_indices] =  g1_itf_ghost_x1[:, :, 1, west_indices]

            # # from bottom to top
            g3_itf_ghost_x3[:, 0, :, up_indices] =  g3_itf_ghost_x3[:, -2, :, up_indices]
            g3_itf_ghost_x3[:, -1, :, down_indices] =  g3_itf_ghost_x3[:, 1, :, down_indices]
        else:
            # Enforce wall conditions (g=0) by setting all values at the ghost cells to appropriate negative values
            g1_itf_ghost_x1[:, :, 0, east_indices] =  -g1_itf_ghost_x1[:, :, 1, west_indices]
            g1_itf_ghost_x1[:, :, -1, west_indices] =  -g1_itf_ghost_x1[:, :, -2, east_indices]

            # # from bottom to top
            g3_itf_ghost_x3[:, 0, :, up_indices] =  -g3_itf_ghost_x3[:, 1, :, down_indices]
            g3_itf_ghost_x3[:, -1, :, down_indices] =  -g3_itf_ghost_x3[:, -2, :, up_indices]
            #
            # ##from top to bottom
            # g3_itf_ghost_x3[:, 0, :, down_indices] =  -g3_itf_ghost_x3[:, 1, :, up_indices]
            # g3_itf_ghost_x3[:, -1, :, up_indices] =  -g3_itf_ghost_x3[:, -2, :, down_indices]
        
        
        # 2. Compute g- and g+ for each edge
        # g- and g+ to compute avg on the western boundary of the element 
        g1_minus_west = g1_itf_ghost_x1[:, :, 1:num_elements_horizontal+1, west_indices] # (num_eqs, num_el_vertical, num_el_horizontal, num_solpts)
        g1_plus_west = g1_itf_ghost_x1[:, :, 0:num_elements_horizontal, east_indices]
        
        # g- and g+ to compute avg on the western boundary of the element
        g1_minus_east = g1_itf_ghost_x1[:, :, 1:num_elements_horizontal+1, east_indices] 
        g1_plus_east = g1_itf_ghost_x1[:, :, 2:num_elements_horizontal+2, west_indices]
        
        # from bottom to top
        # #g- and g+ to compute avg on the lower boundary of the element 
        g3_minus_down = g3_itf_ghost_x3[:, 1:num_elements_vertical+1, :, down_indices] 
        g3_plus_down = g3_itf_ghost_x3[:, 0:num_elements_vertical, :, up_indices]
        
        # g- and g+ to compute avg on the western boundary of the element
        g3_minus_up = g3_itf_ghost_x3[:, 1:num_elements_vertical+1, :, up_indices] 
        g3_plus_up = g3_itf_ghost_x3[:, 2:num_elements_vertical+2, :, down_indices]
        
        #
        # from top to bottom
        # g- and g+ to compute avg on the lower boundary of the element 
        # g3_minus_down = g3_itf_ghost_x3[:, 1:num_elements_vertical+1, :, down_indices] 
        # g3_plus_down = g3_itf_ghost_x3[:, 2:num_elements_vertical+2, :, up_indices]
        
        # # g- and g+ to compute avg on the western boundary of the element
        # g3_minus_up = g3_itf_ghost_x3[:, 1:num_elements_vertical+1, :, up_indices] 
        # g3_plus_up = g3_itf_ghost_x3[:, 0:num_elements_vertical, :, down_indices]
    
        
        # 3. Compute the average: {g} = 1/2 * [g(u-) + g(u+)]
        g1_avg_west = 0.5 * (g1_minus_west + g1_plus_west)
        g1_avg_east = 0.5 * (g1_minus_east + g1_plus_east)
        
        g3_avg_down = 0.5 * (g3_minus_down + g3_plus_down)
        g3_avg_up = 0.5 * (g3_minus_up + g3_plus_up)


        # 4. Concantenate the arays
        g1_avg_x1 = xp.concatenate([g1_avg_west, g1_avg_east], axis=3) # (num_eqs, num_elements_vertical, num_elements_horizontal, 2*num_solpts)
        g3_avg_x3 = xp.concatenate([g3_avg_down, g3_avg_up], axis=3)
        
        # Try setting the bc explicitly
        # Enforce wall conditions (g=0) by setting all values at the ghost cells to appropriate negative values
        # g1_avg_x1[:, :, 0, east_indices] =  0
        # g1_avg_x1[:, :, -1, west_indices] =  0

        # # # from bottom to top
        # g3_avg_x3[:, 0, :, up_indices] =  0
        # g3_avg_x3[:, -1, :, down_indices] =  0
        
        return g1_avg_x1,g3_avg_x3
