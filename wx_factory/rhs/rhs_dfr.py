import numpy
from numpy.typing import NDArray

from common.definitions import idx_rho, idx_rho_u1, idx_rho_u2, idx_rho_w, idx_rho_theta
from rhs.rhs import RHS


class RHSDirecFluxReconstruction(RHS):

    def solution_extrapolation(self, q: NDArray) -> None:
        # Extrapolate the solution to element boundaries
        # if self.num_dim == 2:
        # Investigate why this is slower since no reallocation is needed
        #     xp.matmul(q, self.ops.extrap_x, out=self.q_itf_x1)
        #     xp.matmul(q, self.ops.extrap_z, out=self.q_itf_x3)

        self.q_itf_x1 = q @ self.ops.extrap_x
        self.q_itf_x3 = q @ self.ops.extrap_z
        if hasattr(self.ops, "extrap_y"):
            self.q_itf_x2 = q @ self.ops.extrap_y

    def pointwise_fluxes(self, q: NDArray) -> None:
        self.pde.pointwise_fluxes(q, self.f_x1, self.f_x2, self.f_x3)

    def flux_divergence_partial(self) -> NDArray:
        xp = self.device.xp

        # Compute derivatives, with correction from boundaries
        # Investigate why this is slower
        # xp.matmul(self.f_x1, self.ops.derivative_x, out=self.df1_dx1)
        # xp.matmul(self.f_x3, self.ops.derivative_z, out=self.df3_dx3)

        self.df1_dx1 = self.f_x1 @ self.ops.derivative_x
        self.df3_dx3 = self.f_x3 @ self.ops.derivative_z

    def flux_divergence(self):
        xp = self.device.xp

        self.df1_dx1 += self.f_itf_x1 @ self.ops.correction_WE
        self.df1_dx1 *= -2.0 / self.geom.Δx1

        self.df3_dx3 += self.f_itf_x3 @ self.ops.correction_DU
        self.df3_dx3 *= -2.0 / self.geom.Δx3

        xp.add(self.df1_dx1, self.df3_dx3, out=self.rhs)


class RHSDirecFluxReconstruction_mpi(RHSDirecFluxReconstruction):

    def solution_extrapolation(self, q: NDArray) -> None:
        # Extrapolate the solution to element boundaries
        # if self.num_dim == 2:
        # Investigate why this is slower since no reallocation is needed
        #     xp.matmul(q, self.ops.extrap_x, out=self.q_itf_x1)
        #     xp.matmul(q, self.ops.extrap_z, out=self.q_itf_x3)
        xp = self.device.xp

        self.q_itf_x1 = q @ self.ops.extrap_x
        self.q_itf_x2 = q @ self.ops.extrap_y
        self.q_itf_x3 = q @ self.ops.extrap_z

        self.log_rho_p = xp.log(q[idx_rho])
        self.log_rho_theta = xp.log(q[idx_rho_theta])

        # TODO clean this up (avoid overwriting previous computation)
        self.q_itf_x1[idx_rho] = xp.exp(self.log_rho_p @ self.ops.extrap_x)
        self.q_itf_x1[idx_rho_theta] = xp.exp(self.log_rho_theta @ self.ops.extrap_x)
        self.q_itf_x2[idx_rho] = xp.exp(self.log_rho_p @ self.ops.extrap_y)
        self.q_itf_x2[idx_rho_theta] = xp.exp(self.log_rho_theta @ self.ops.extrap_y)
        self.q_itf_x3[idx_rho] = xp.exp(self.log_rho_p @ self.ops.extrap_z)
        self.q_itf_x3[idx_rho_theta] = xp.exp(self.log_rho_theta @ self.ops.extrap_z)

    def pointwise_fluxes(self, q: NDArray) -> None:
        self.pde.pointwise_fluxes(
            q,
            self.f_x1,
            self.f_x2,
            self.f_x3,
            self.pressure,
            self.wflux_adv_x1,
            self.wflux_adv_x2,
            self.wflux_adv_x3,
            self.wflux_pres_x1,
            self.wflux_pres_x2,
            self.wflux_pres_x3,
            self.log_p,
        )

    def flux_divergence_partial(self):
        super().flux_divergence_partial()

        self.df2_dx2 = self.f_x2 @ self.ops.derivative_y

        self.w_df1_dx1_adv = self.wflux_adv_x1 @ self.ops.derivative_x
        self.w_df1_dx1_presa = self.wflux_pres_x1 @ self.ops.derivative_x
        self.w_df1_dx1_presb = self.log_p @ self.ops.derivative_x

        self.w_df2_dx2_adv = self.wflux_adv_x2 @ self.ops.derivative_y
        self.w_df2_dx2_presa = self.wflux_pres_x2 @ self.ops.derivative_y
        self.w_df2_dx2_presb = self.log_p @ self.ops.derivative_y

        self.w_df3_dx3_adv = self.wflux_adv_x3 @ self.ops.derivative_z
        self.w_df3_dx3_presa = self.wflux_pres_x3 @ self.ops.derivative_z
        self.w_df3_dx3_presb = self.log_p @ self.ops.derivative_z

    def flux_divergence(self):
        xp = self.device.xp

        self.df1_dx1 += self.f_itf_x1 @ self.ops.correction_WE
        self.df2_dx2 += self.f_itf_x2 @ self.ops.correction_SN
        self.df3_dx3 += self.f_itf_x3 @ self.ops.correction_DU

        logp_bdy_i = xp.log(self.pressure_itf_x1)
        logp_bdy_j = xp.log(self.pressure_itf_x2)
        logp_bdy_k = xp.log(self.pressure_itf_x3)

        self.w_df1_dx1_adv += self.wflux_adv_itf_x1 @ self.ops.correction_WE
        self.w_df1_dx1_presa += self.wflux_pres_itf_x1 @ self.ops.correction_WE
        self.w_df1_dx1_presa *= self.pressure
        self.w_df1_dx1_presb += logp_bdy_i @ self.ops.correction_WE
        self.w_df1_dx1_presb *= self.pressure * self.wflux_pres_x1
        self.w_df1_dx1[...] = self.w_df1_dx1_adv + self.w_df1_dx1_presa + self.w_df1_dx1_presb

        self.w_df2_dx2_adv += self.wflux_adv_itf_x2 @ self.ops.correction_SN
        self.w_df2_dx2_presa += self.wflux_pres_itf_x2 @ self.ops.correction_SN
        self.w_df2_dx2_presa *= self.pressure
        self.w_df2_dx2_presb += logp_bdy_j @ self.ops.correction_SN
        self.w_df2_dx2_presb *= self.pressure * self.wflux_pres_x2
        self.w_df2_dx2[...] = self.w_df2_dx2_adv + self.w_df2_dx2_presa + self.w_df2_dx2_presb

        self.w_df3_dx3_adv += self.wflux_adv_itf_x3 @ self.ops.correction_DU
        self.w_df3_dx3_presa += self.wflux_pres_itf_x3 @ self.ops.correction_DU
        self.w_df3_dx3_presa *= self.pressure
        self.w_df3_dx3_presb += logp_bdy_k @ self.ops.correction_DU
        self.w_df3_dx3_presb *= self.pressure * self.wflux_pres_x3
        self.w_df3_dx3 = self.w_df3_dx3_adv + self.w_df3_dx3_presa + self.w_df3_dx3_presb

        self.rhs[...] = -self.metric.inv_sqrtG_new * (self.df1_dx1 + self.df2_dx2 + self.df3_dx3)
        self.rhs[idx_rho_w] = -self.metric.inv_sqrtG_new * (self.w_df1_dx1 + self.w_df2_dx2 + self.w_df3_dx3)

    def start_communication(self):
        s2_ = numpy.s_[..., 0, :, : self.geom.itf_size]
        n2_ = numpy.s_[..., -1, :, self.geom.itf_size :]
        w2_ = numpy.s_[..., 0, : self.geom.itf_size]
        e2_ = numpy.s_[..., -1, self.geom.itf_size :]

        self.transfer_s = self.q_itf_x2[s2_]
        self.req_r = self.ptopo.start_exchange_scalars(
            self.q_itf_x2[idx_rho][s2_],
            self.q_itf_x2[idx_rho][n2_],
            self.q_itf_x1[idx_rho][w2_],
            self.q_itf_x1[idx_rho][e2_],
            boundary_shape=self.geom.halo_side_shape,
            flip_dim=(-3, -1),
        )
        self.req_u = self.ptopo.start_exchange_vectors(
            (self.q_itf_x2[idx_rho_u1][s2_], self.q_itf_x2[idx_rho_u2][s2_], self.q_itf_x2[idx_rho_w][s2_]),
            (self.q_itf_x2[idx_rho_u1][n2_], self.q_itf_x2[idx_rho_u2][n2_], self.q_itf_x2[idx_rho_w][n2_]),
            (self.q_itf_x1[idx_rho_u1][w2_], self.q_itf_x1[idx_rho_u2][w2_], self.q_itf_x1[idx_rho_w][w2_]),
            (self.q_itf_x1[idx_rho_u1][e2_], self.q_itf_x1[idx_rho_u2][e2_], self.q_itf_x1[idx_rho_w][e2_]),
            self.geom.boundary_sn_new,
            self.geom.boundary_we_new,
            flip_dim=(-3, -1),
        )
        self.req_t = self.ptopo.start_exchange_scalars(
            self.q_itf_x2[idx_rho_theta][s2_],
            self.q_itf_x2[idx_rho_theta][n2_],
            self.q_itf_x1[idx_rho_theta][w2_],
            self.q_itf_x1[idx_rho_theta][e2_],
            boundary_shape=self.geom.halo_side_shape,
            flip_dim=(-3, -1),
        )

    def end_communication(self):
        xp = self.device.xp
        dtype = self.q_itf_x1.dtype
        if self.q_itf_s is None or self.q_itf_w.dtype != dtype:
            sh = (self.num_var,) + self.req_r.shape
            self.q_itf_s = xp.zeros(sh, dtype=dtype)
            self.q_itf_n = xp.zeros(sh, dtype=dtype)
            self.q_itf_w = xp.zeros(sh, dtype=dtype)
            self.q_itf_e = xp.zeros(sh, dtype=dtype)

        (
            self.q_itf_s[idx_rho],
            self.q_itf_n[idx_rho],
            self.q_itf_w[idx_rho],
            self.q_itf_e[idx_rho],
        ) = self.req_r.wait()
        (
            (self.q_itf_s[idx_rho_u1], self.q_itf_s[idx_rho_u2], self.q_itf_s[idx_rho_w]),
            (self.q_itf_n[idx_rho_u1], self.q_itf_n[idx_rho_u2], self.q_itf_n[idx_rho_w]),
            (self.q_itf_w[idx_rho_u1], self.q_itf_w[idx_rho_u2], self.q_itf_w[idx_rho_w]),
            (self.q_itf_e[idx_rho_u1], self.q_itf_e[idx_rho_u2], self.q_itf_e[idx_rho_w]),
        ) = self.req_u.wait()
        (
            self.q_itf_s[idx_rho_theta],
            self.q_itf_n[idx_rho_theta],
            self.q_itf_w[idx_rho_theta],
            self.q_itf_e[idx_rho_theta],
        ) = self.req_t.wait()

    def riemann_fluxes(self) -> None:
        xp = self.device.xp
        dtype = self.q_itf_x1.dtype

        itf_i_shape = (self.num_var,) + self.geom.itf_i_shape
        itf_j_shape = (self.num_var,) + self.geom.itf_j_shape
        itf_k_shape = (self.num_var,) + self.geom.itf_k_shape

        mid_i = xp.s_[..., 1:-1, :]
        mid_j = xp.s_[..., 1:-1, :, :]
        mid_k = xp.s_[..., 1:-1, :, :, :]

        s = numpy.s_[..., 0, :, self.geom.itf_size :]
        n = numpy.s_[..., -1, :, : self.geom.itf_size]
        w = numpy.s_[..., 0, self.geom.itf_size :]
        e = numpy.s_[..., -1, : self.geom.itf_size]
        b = numpy.s_[..., 0, :, :, self.geom.itf_size :]
        t = numpy.s_[..., -1, :, :, : self.geom.itf_size]

        if self.f_itf_x1 is None or self.f_itf_x1.dtype != dtype:
            self.f_itf_x1 = xp.zeros_like(self.q_itf_x1)
            self.f_itf_x2 = xp.zeros_like(self.q_itf_x2)
            self.f_itf_x3 = xp.zeros_like(self.q_itf_x3)

            self.pressure_itf_x1 = xp.zeros_like(self.f_itf_x1[0])
            self.pressure_itf_x2 = xp.zeros_like(self.f_itf_x2[0])
            self.pressure_itf_x3 = xp.zeros_like(self.f_itf_x3[0])

            self.wflux_adv_itf_x1 = xp.zeros_like(self.f_itf_x1[0])
            self.wflux_pres_itf_x1 = xp.zeros_like(self.f_itf_x1[0])
            self.wflux_adv_itf_x2 = xp.zeros_like(self.f_itf_x2[0])
            self.wflux_pres_itf_x2 = xp.zeros_like(self.f_itf_x2[0])
            self.wflux_adv_itf_x3 = xp.zeros_like(self.f_itf_x3[0])
            self.wflux_pres_itf_x3 = xp.zeros_like(self.f_itf_x3[0])

            self.q_itf_full_x1 = xp.ones(itf_i_shape, dtype=dtype)
            self.q_itf_full_x2 = xp.ones(itf_j_shape, dtype=dtype)
            self.q_itf_full_x3 = xp.ones(itf_k_shape, dtype=dtype)

            self.f_itf_full_x1 = xp.zeros_like(self.q_itf_full_x1)
            self.f_itf_full_x2 = xp.zeros_like(self.q_itf_full_x2)
            self.f_itf_full_x3 = xp.zeros_like(self.q_itf_full_x3)

            self.pressure_itf_full_x1 = xp.zeros_like(self.q_itf_full_x1[0])
            self.pressure_itf_full_x2 = xp.zeros_like(self.q_itf_full_x2[0])
            self.pressure_itf_full_x3 = xp.zeros_like(self.q_itf_full_x3[0])

            self.wflux_adv_itf_full_x1 = xp.zeros_like(self.q_itf_full_x1[0])
            self.wflux_pres_itf_full_x1 = xp.zeros_like(self.q_itf_full_x1[0])
            self.wflux_adv_itf_full_x2 = xp.zeros_like(self.q_itf_full_x2[0])
            self.wflux_pres_itf_full_x2 = xp.zeros_like(self.q_itf_full_x2[0])
            self.wflux_adv_itf_full_x3 = xp.zeros_like(self.q_itf_full_x3[0])
            self.wflux_pres_itf_full_x3 = xp.zeros_like(self.q_itf_full_x3[0])

        self.q_itf_full_x1[mid_i] = self.q_itf_x1
        self.q_itf_full_x2[mid_j] = self.q_itf_x2
        self.q_itf_full_x3[mid_k] = self.q_itf_x3

        self.q_itf_full_x1[w] = self.q_itf_w
        self.q_itf_full_x1[e] = self.q_itf_e
        self.q_itf_full_x2[s] = self.q_itf_s
        self.q_itf_full_x2[n] = self.q_itf_n
        self.q_itf_full_x3[b] = self.q_itf_full_x3[..., 1, :, :, : self.geom.itf_size]
        self.q_itf_full_x3[..., 0, :, :, : self.geom.itf_size] = self.q_itf_full_x3[b]
        self.q_itf_full_x3[t] = self.q_itf_full_x3[..., -2, :, :, self.geom.itf_size :]
        self.q_itf_full_x3[..., -1, :, :, self.geom.itf_size :] = self.q_itf_full_x3[t]

        self.pde.riemann_fluxes(
            self.q_itf_full_x1,
            self.q_itf_full_x2,
            self.q_itf_full_x3,
            self.f_itf_full_x1,
            self.f_itf_full_x2,
            self.f_itf_full_x3,
            self.pressure_itf_full_x1,
            self.pressure_itf_full_x2,
            self.pressure_itf_full_x3,
            self.wflux_adv_itf_full_x1,
            self.wflux_pres_itf_full_x1,
            self.wflux_adv_itf_full_x2,
            self.wflux_pres_itf_full_x2,
            self.wflux_adv_itf_full_x3,
            self.wflux_pres_itf_full_x3,
            self.metric,
        )

        self.f_itf_x1[...] = self.f_itf_full_x1[mid_i]
        self.f_itf_x2[...] = self.f_itf_full_x2[mid_j]
        self.f_itf_x3[...] = self.f_itf_full_x3[mid_k]

        self.pressure_itf_x1[...] = self.pressure_itf_full_x1[mid_i]
        self.pressure_itf_x2[...] = self.pressure_itf_full_x2[mid_j]
        self.pressure_itf_x3[...] = self.pressure_itf_full_x3[mid_k]

        self.wflux_adv_itf_x1[...] = self.wflux_adv_itf_full_x1[mid_i]
        self.wflux_pres_itf_x1[...] = self.wflux_pres_itf_full_x1[mid_i]
        self.wflux_adv_itf_x2[...] = self.wflux_adv_itf_full_x2[mid_j]
        self.wflux_pres_itf_x2[...] = self.wflux_pres_itf_full_x2[mid_j]
        self.wflux_adv_itf_x3[...] = self.wflux_adv_itf_full_x3[mid_k]
        self.wflux_pres_itf_x3[...] = self.wflux_pres_itf_full_x3[mid_k]

    def forcing_terms(self, q: NDArray) -> None:
        self.pde.forcing_terms(self.rhs, q, self.pressure, self.metric, self.ops, self.forcing)

        # For pure advection problems, we do not update the dynamical variables
        if self.pde.advection_only:
            self.rhs[idx_rho] = 0.0
            self.rhs[idx_rho_u1] = 0.0
            self.rhs[idx_rho_u2] = 0.0
            self.rhs[idx_rho_w] = 0.0
            self.rhs[idx_rho_theta] = 0.0
