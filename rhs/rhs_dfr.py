from numpy import ndarray

from rhs.rhs import RHS


class RHS_DFR(RHS):

    def __init__(self, pde_name, geometry, operators, metric, topology, process_topo, config, device):
        super().__init__(pde_name, geometry, operators, metric, topology, process_topo, config, device)

        xp = device.xp
        num_dim = config.num_dim

        # Initialize arrays that will be used in this RHS
        nb_solpts = operators.extrap_x.shape[0]
        nb_var = config.nb_var

        nb_elems_x1 = config.nb_elements_horizontal
        nb_elems_x2 = config.nb_elements_vertical

        nb_itf_solpts_x1 = operators.extrap_x.shape[1]
        nb_itf_solpts_x3 = operators.extrap_z.shape[1]

        # Assume two-dimensions first
        self.f_x1 = xp.empty((nb_var, nb_elems_x2, nb_elems_x1, nb_solpts))
        self.f_x3 = xp.empty((nb_var, nb_elems_x2, nb_elems_x1, nb_solpts))

        self.q_itf_x1 = xp.empty((nb_var, nb_elems_x2, nb_elems_x1, nb_itf_solpts_x1))
        self.q_itf_x3 = xp.empty((nb_var, nb_elems_x2, nb_elems_x1, nb_itf_solpts_x3))

        self.f_itf_x1 = xp.empty_like(self.q_itf_x1)
        self.f_itf_x3 = xp.empty_like(self.q_itf_x3)

        self.df1_dx1 = xp.empty_like(self.f_x1)
        self.df3_dx3 = xp.empty_like(self.f_x3)

        self.f_x2 = None
        self.q_itf_x2 = None
        self.f_itf_x2 = None
        self.df2_dx2 = None

        # Initialize rhs matrix
        self.rhs = xp.empty_like(self.f_x1)

    def solution_extrapolation(self, q: ndarray) -> None:
        xp = self.device.xp

        # Extrapolate the solution to element boundaries
        # if self.num_dim == 2:
        # Investigate why this is slower since no reallocation is needed
        #     xp.matmul(q, self.ops.extrap_x, out=self.q_itf_x1)
        #     xp.matmul(q, self.ops.extrap_z, out=self.q_itf_x3)

        self.q_itf_x1 = q @ self.ops.extrap_x
        self.q_itf_x3 = q @ self.ops.extrap_z

    def pointwise_fluxes(self, q: ndarray) -> None:
        self.pde.pointwise_fluxes(q, self.f_x1, self.f_x2, self.f_x3)

    def flux_divergence_partial(self) -> ndarray:
        xp = self.device.xp

        # Compute derivatives, with correction from boundaries
        if self.num_dim == 2:

            # Investigate why this is slower
            # xp.matmul(self.f_x1, self.ops.derivative_x, out=self.df1_dx1)
            # xp.matmul(self.f_x3, self.ops.derivative_z, out=self.df3_dx3)
            self.df1_dx1 = self.f_x1 @ self.ops.derivative_x
            self.df3_dx3 = self.f_x3 @ self.ops.derivative_z
        else:
            raise Exception("3D not implemented yet!")

    def flux_divergence(self):
        xp = self.device.xp

        if self.num_dim == 2:
            self.df1_dx1 += self.f_itf_x1 @ self.ops.correction_WE
            self.df1_dx1 *= -2.0 / self.geom.Δx1

            self.df3_dx3 += self.f_itf_x3 @ self.ops.correction_DU
            self.df3_dx3 *= -2.0 / self.geom.Δx3
        else:
            raise Exception("3D not implemented yet!")

        xp.add(self.df1_dx1, self.df3_dx3, out=self.rhs)
