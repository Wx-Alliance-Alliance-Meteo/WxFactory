from numpy import ndarray

from rhs.rhs import RHS


class RHS_FV(RHS):

    def __init__(self, *args):
        super().__init__(*args)

        print("here!!!!!")
        self.f_x1 = None
        self.f_x2 = None
        self.f_x3 = None

        self.q_itf_x1 = None
        self.q_itf_x2 = None
        self.q_itf_x3 = None

        self.f_itf_x1 = None
        self.f_itf_x2 = None
        self.f_itf_x3 = None

        self.df1_dx1 = None
        self.df2_dx2 = None
        self.df3_dx3 = None

        self.rhs = None

    def solution_extrapolation(self, q: ndarray) -> None:

        if self.q_itf_x1 is None or self.q_itf_x1.dtype != q.dtype:
            xp = self.device.xp
            self.q_itf_x1 = xp.empty((*self.q.shape, 2), dtype=q.dtype)
            self.f_itf_x1 = xp.empty_like(self.q_itf_x1)
            self.f_itf_x3 = xp.empty_like(self.q_itf_x3)

        # The interface solution are the cell centered values
        self.q_itf_x1[..., 0] = q.squeeze()
        self.q_itf_x1[..., 1] = q.squeeze()

        # These are the same values used in all directions
        self.q_itf_x3 = self.q_itf_x1
        self.q_itf_x2 = self.q_itf_x1

    def pointwise_fluxes(self, q: ndarray) -> None:
        # Not applicable in finite volume
        pass

    def flux_divergence_partial(self) -> None:
        # Not applicable in finite volume
        pass

    def flux_divergence(self) -> None:
        xp = self.device.xp

        if self.num_dim == 2:
            # Compute FD/FV derivative (structured)
            self.df1_dx1 = -(self.f_itf_x1[:, :, 1] - self.f_itf_x1[:, :, 0]) / self.geom.Δx1
            self.df3_dx3 = -(self.f_itf_x3[:, :, 1] - self.f_itf_x3[:, :, 0]) / self.geom.Δx3
        else:
            raise Exception("3D not implemented yet!")

        if self.rhs is None or self.rhs.dtype != self.df1_dx1.dtype:
            self.rhs = xp.empty_like(self.df1_dx1)

        xp.add(self.df1_dx1, self.df3_dx3, out=self.rhs)
