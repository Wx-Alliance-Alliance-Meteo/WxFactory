from numpy.typing import NDArray

from rhs.rhs import RHS


class RHS_DFR(RHS):

    def solution_extrapolation(self, q: NDArray) -> None:
        xp = self.device.xp

        # Extrapolate the solution to element boundaries
        if self.num_dim == 2:
            xp.matmul(q, self.ops.extrap_x, out=self.q_itf_x1)
            xp.matmul(q, self.ops.extrap_z, out=self.q_itf_x3)
        else:
            raise Exception("3D not implemented yet!")
        return  # extrapolation

    def pointwise_fluxes(self, q) -> None:
        self.pde.pointwise_fluxes(q, self.f_x1, self.f_x2, self.f_x3)

    def flux_divergence_partial(self) -> NDArray:
        xp = self.device.xp

        # Compute derivatives, with correction from boundaries
        if self.num_dim == 2:
            xp.matmul(self.f_x1, self.ops.derivative_x, out=self.df1_dx1)
            xp.matmul(self.f_x3, self.ops.derivative_z, out=self.df3_dx3)
        else:
            raise Exception("3D not implemented yet!")

    def flux_correction(self):
        xp = self.device.xp

        if self.num_dim == 2:
            self.df1_dx1 += self.f_itf_x1 @ self.ops.correction_WE
            self.df1_dx1 *= (-2.0 / self.geom.Δx1)

            self.df3_dx3 += self.f_itf_x3 @ self.ops.correction_DU
            self.df3_dx3 *= (-2.0 / self.geom.Δx3)
        else:
            raise Exception("3D not implemented yet!")

        xp.add(self.df1_dx1, self.df3_dx3, out=self.rhs)
