from numpy import ndarray

from rhs.rhs import RHS


class RHS_FV(RHS):

    def __init__(self, pde_name, geometry, operators, metric, topology, process_topo, config, device):
        super().__init__(pde_name, geometry, operators, metric, topology, process_topo, config, device)

        xp = device.xp
        num_dim = config.num_dim

        # Initialize arrays that will be used in this RHS
        nb_solpts = 1
        nb_var = config.nb_var
        nb_elems = config.nb_elements_horizontal * config.nb_elements_vertical

        nb_itf_solpts_x1 = 2
        nb_itf_solpts_x2 = 2
        nb_itf_solpts_x3 = 2

        if num_dim == 3:
            nb_elems *= config.nb_elements_relief_layer

        # Pointwise fluxes are not used in FV discretization
        self.f_x1 = None
        self.f_x2 = None
        self.f_x3 = None

        # Allocate interface arrays
        self.q_itf_x1 = xp.empty((nb_var, nb_elems, nb_itf_solpts_x1))
        self.q_itf_x2 = None
        self.q_itf_x3 = xp.empty((nb_var, nb_elems, nb_itf_solpts_x3))

        # Allocate Riemann flux arrays
        self.f_itf_x1 = xp.empty_like(self.q_itf_x1)
        self.f_itf_x2 = None
        self.f_itf_x3 = xp.empty_like(self.q_itf_x3)

        # Allocate derivative arrays
        self.df1_dx1 = xp.empty_like(self.f_x1)
        self.df2_dx2 = None
        self.df3_dx3 = xp.empty_like(self.f_x1)

        # Add third-dimension arrays if needed
        if num_dim == 3:
            self.q_itf_x2 = xp.empty((nb_var, nb_elems, nb_itf_solpts_x2))
            self.f_itf_x2 = xp.empty_like(self.q_itf_x2)
            self.df2_dx2 = xp.empty_like(self.f_x2)

        # Initialize rhs matrix
        self.rhs = xp.empty((nb_var, nb_elems, nb_solpts))

    def solution_extrapolation(self, q: ndarray) -> None:

        # The interface solution are the cell centered values
        self.q_itf_x1[:, :, 0] = q.squeeze()
        self.q_itf_x3[:, :, 1] = q.squeeze()

        self.q_itf_x3 = self.q_itf_x1
        self.q_itf_x2 = self.q_itf_x1

    def pointwise_fluxes(self, q: ndarray) -> None:
        # Not applicable in finite volume
        pass

    def flux_divergence_partial(self) -> ndarray:
        # Not applicable in finite volume
        pass

    def flux_divergence(self):
        xp = self.device.xp

        if self.num_dim == 2:
            # Compute FD/FV derivative (structured)
            self.df1_dx1 = -(self.f_itf_x1[:, :, 1] - self.f_itf_x1[:, :, 0]) / self.geom.Δx1
            self.df3_dx3 = -(self.f_itf_x3[:, :, 1] - self.f_itf_x3[:, :, 0]) / self.geom.Δx3
        else:
            raise Exception("3D not implemented yet!")

        xp.add(self.df1_dx1, self.df3_dx3, out=self.rhs)
