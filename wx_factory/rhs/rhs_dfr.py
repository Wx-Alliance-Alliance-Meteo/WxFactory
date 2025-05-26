import numpy
from numpy.typing import NDArray

from common.definitions import idx_rho, idx_rho_u1, idx_rho_u2, idx_rho_w, idx_rho_theta
from geometry import CubedSphere
from rhs.rhs import RHS
from wx_mpi import SingleProcess, Conditional


def apply_op(vec: NDArray, op: NDArray):
    sh = vec.shape
    return (vec.reshape(-1, sh[-1]) @ op).reshape(*sh[:-1], -1)
    # return vec @ op


class RHSDirecFluxReconstruction(RHS):

    def solution_extrapolation(self, q: NDArray) -> None:
        # Extrapolate the solution to element boundaries
        # if self.num_dim == 2:
        # Investigate why this is slower since no reallocation is needed
        #     xp.matmul(q, self.ops.extrap_x, out=self.q_itf_x1)
        #     xp.matmul(q, self.ops.extrap_z, out=self.q_itf_x3)

        self.q_itf_x1 = apply_op(q, self.ops.extrap_x)
        self.q_itf_x3 = apply_op(q, self.ops.extrap_z)
        if hasattr(self.ops, "extrap_y"):
            self.q_itf_x2 = apply_op(q, self.ops.extrap_y)

    def pointwise_fluxes(self, q: NDArray) -> None:
        self.pde.pointwise_fluxes(q, self.f_x1, self.f_x2, self.f_x3)

    def flux_divergence_partial(self) -> NDArray:
        xp = self.device.xp

        # Compute derivatives, with correction from boundaries
        # Investigate why this is slower
        # xp.matmul(self.f_x1, self.ops.derivative_x, out=self.df1_dx1)
        # xp.matmul(self.f_x3, self.ops.derivative_z, out=self.df3_dx3)

        self.df1_dx1 = apply_op(self.f_x1, self.ops.derivative_x)
        self.df3_dx3 = apply_op(self.f_x3, self.ops.derivative_z)

    def flux_divergence(self):
        xp = self.device.xp

        self.df1_dx1 += self.f_itf_x1 @ self.ops.correction_WE
        self.df1_dx1 *= -2.0 / self.geom.Δx1

        self.df3_dx3 += self.f_itf_x3 @ self.ops.correction_DU
        self.df3_dx3 *= -2.0 / self.geom.Δx3

        xp.add(self.df1_dx1, self.df3_dx3, out=self.rhs)


class RHSDirecFluxReconstruction_mpi(RHSDirecFluxReconstruction):
    def __init__(
        self,
        pde,
        geometry: CubedSphere,
        operators,
        metric,
        topography,
        process_topo,
        config,
        expected_shape,
        debug=False,
    ):
        super().__init__(pde, geometry, operators, metric, topography, process_topo, config, expected_shape, debug)
        self.extrap_3d = self.extrap_3d_code
        if config.desired_device in ["numpy", "cupy"]:
            self.extrap_3d = self.extrap_3d_py

    def allocate_arrays(self, q):
        super().allocate_arrays(q)

        xp = self.device.xp
        dtype = self.q_itf_x1.dtype

        itf_i_shape = (self.num_var,) + self.geom.itf_i_shape
        itf_j_shape = (self.num_var,) + self.geom.itf_j_shape
        itf_k_shape = (self.num_var,) + self.geom.itf_k_shape

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

            # Set to ones, because uninitialized values will be used in a log
            # TODO separate two interface sides so that we don't need to do these useless calculations
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

    def extrap_3d_py(self, q: NDArray, itf_x1: NDArray, itf_x2: NDArray, itf_x3: NDArray) -> None:
        itf_x1[...] = q @ self.ops.extrap_x
        itf_x2[...] = q @ self.ops.extrap_y
        itf_x3[...] = q @ self.ops.extrap_z

    def extrap_3d_code(self, q: NDArray, itf_x1: NDArray, itf_x2: NDArray, itf_x3: NDArray) -> None:
        # xp = self.device.xp
        # itf_x1[...] = xp.arange(self.q_itf_x1.size).reshape(self.q_itf_x1.shape)
        # itf_x2[...] = xp.arange(1000, self.q_itf_x1.size + 1000).reshape(self.q_itf_x1.shape)
        # itf_x3[...] = xp.arange(500000, self.q_itf_x1.size + 500000).reshape(self.q_itf_x1.shape)
        _, nz, ny, nx = q.shape[:4]
        self.device.operators.extrap_all_3d(
            q,
            itf_x1,
            itf_x2,
            itf_x3,
            nx,
            ny,
            nz,
            self.geom.num_solpts,
            # 0 if self.device.comm.rank != 0 else 1,
            0,
        )

    def solution_extrapolation(self, q: NDArray) -> None:
        # Extrapolate the solution to element boundaries
        # if self.num_dim == 2:
        # Investigate why this is slower since no reallocation is needed
        #     xp.matmul(q, self.ops.extrap_x, out=self.q_itf_x1)
        #     xp.matmul(q, self.ops.extrap_z, out=self.q_itf_x3)
        xp = self.device.xp

        self.extrap_3d(q, self.q_itf_x1, self.q_itf_x2, self.q_itf_x3)

        self.log_rho_p = xp.log(q[idx_rho])
        self.log_rho_theta = xp.log(q[idx_rho_theta])

        # TODO clean this up (avoid overwriting previous computation)
        self.q_itf_x1[idx_rho] = xp.exp(apply_op(self.log_rho_p, self.ops.extrap_x))
        self.q_itf_x1[idx_rho_theta] = xp.exp(apply_op(self.log_rho_theta, self.ops.extrap_x))
        self.q_itf_x2[idx_rho] = xp.exp(apply_op(self.log_rho_p, self.ops.extrap_y))
        self.q_itf_x2[idx_rho_theta] = xp.exp(apply_op(self.log_rho_theta, self.ops.extrap_y))
        self.q_itf_x3[idx_rho] = xp.exp(apply_op(self.log_rho_p, self.ops.extrap_z))
        self.q_itf_x3[idx_rho_theta] = xp.exp(apply_op(self.log_rho_theta, self.ops.extrap_z))

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

        self.df1_dx1 = apply_op(self.f_x1, self.ops.derivative_x)

        xp = self.device.xp
        res = xp.zeros_like(self.df1_dx1)
        num_fields, nz, ny, nx = self.f_x1.shape[:4]
        num_solpts = self.geom.num_solpts
        self.device.operators.deriv_x_3d(
            self.f_x1,
            self.ops.derivative_x,
            res,
            num_fields,
            nx,
            ny,
            nz,
            num_solpts,
            0 if self.device.comm.rank != 0 else 1,
            # 0,
        )

        self.df2_dx2 = apply_op(self.f_x2, self.ops.derivative_y)
        self.df3_dx3 = apply_op(self.f_x3, self.ops.derivative_z)

        self.w_df1_dx1_adv = apply_op(self.wflux_adv_x1, self.ops.derivative_x)
        self.w_df1_dx1_presa = apply_op(self.wflux_pres_x1, self.ops.derivative_x)
        self.w_df1_dx1_presb = apply_op(self.log_p, self.ops.derivative_x)

        self.w_df2_dx2_adv = apply_op(self.wflux_adv_x2, self.ops.derivative_y)
        self.w_df2_dx2_presa = apply_op(self.wflux_pres_x2, self.ops.derivative_y)
        self.w_df2_dx2_presb = apply_op(self.log_p, self.ops.derivative_y)

        self.w_df3_dx3_adv = apply_op(self.wflux_adv_x3, self.ops.derivative_z)
        self.w_df3_dx3_presa = apply_op(self.wflux_pres_x3, self.ops.derivative_z)
        self.w_df3_dx3_presb = apply_op(self.log_p, self.ops.derivative_z)

    def flux_divergence(self):
        xp = self.device.xp

        self.df1_dx1 += apply_op(self.f_itf_x1, self.ops.correction_WE)
        self.df2_dx2 += apply_op(self.f_itf_x2, self.ops.correction_SN)
        self.df3_dx3 += apply_op(self.f_itf_x3, self.ops.correction_DU)

        logp_bdy_i = xp.log(self.pressure_itf_x1)
        logp_bdy_j = xp.log(self.pressure_itf_x2)
        logp_bdy_k = xp.log(self.pressure_itf_x3)

        self.w_df1_dx1_adv += apply_op(self.wflux_adv_itf_x1, self.ops.correction_WE)
        self.w_df1_dx1_presa += apply_op(self.wflux_pres_itf_x1, self.ops.correction_WE)
        self.w_df1_dx1_presa *= self.pressure
        self.w_df1_dx1_presb += apply_op(logp_bdy_i, self.ops.correction_WE)
        self.w_df1_dx1_presb *= self.pressure * self.wflux_pres_x1
        self.w_df1_dx1[...] = self.w_df1_dx1_adv + self.w_df1_dx1_presa + self.w_df1_dx1_presb

        self.w_df2_dx2_adv += apply_op(self.wflux_adv_itf_x2, self.ops.correction_SN)
        self.w_df2_dx2_presa += apply_op(self.wflux_pres_itf_x2, self.ops.correction_SN)
        self.w_df2_dx2_presa *= self.pressure
        self.w_df2_dx2_presb += apply_op(logp_bdy_j, self.ops.correction_SN)
        self.w_df2_dx2_presb *= self.pressure * self.wflux_pres_x2
        self.w_df2_dx2[...] = self.w_df2_dx2_adv + self.w_df2_dx2_presa + self.w_df2_dx2_presb

        self.w_df3_dx3_adv += apply_op(self.wflux_adv_itf_x3, self.ops.correction_DU)
        self.w_df3_dx3_presa += apply_op(self.wflux_pres_itf_x3, self.ops.correction_DU)
        self.w_df3_dx3_presa *= self.pressure
        self.w_df3_dx3_presb += apply_op(logp_bdy_k, self.ops.correction_DU)
        self.w_df3_dx3_presb *= self.pressure * self.wflux_pres_x3
        self.w_df3_dx3[...] = self.w_df3_dx3_adv + self.w_df3_dx3_presa + self.w_df3_dx3_presb

        self.rhs[...] = -self.metric.inv_sqrtG_new * (self.df1_dx1 + self.df2_dx2 + self.df3_dx3)
        self.rhs[idx_rho_w] = -self.metric.inv_sqrtG_new * (self.w_df1_dx1 + self.w_df2_dx2 + self.w_df3_dx3)

    def start_communication(self):

        self.req_all = self.ptopo.start_exchange_euler_3d(
            self.q_itf_x2[..., 0, :, : self.geom.itf_size],
            self.q_itf_x2[..., -1, :, self.geom.itf_size :],
            self.q_itf_x1[..., 0, : self.geom.itf_size],
            self.q_itf_x1[..., -1, self.geom.itf_size :],
            self.geom.boundary_sn_new,
            self.geom.boundary_we_new,
            flip_dim=(-3, -1),
        )

    def end_communication(self):
        # xp = self.device.xp
        # dtype = self.q_itf_x1.dtype
        # if self.q_itf_s is None or self.q_itf_w.dtype != dtype:
        #     sh = (self.num_var,) + self.req_all.shape
        #     self.q_itf_s = xp.zeros(sh, dtype=dtype)
        #     self.q_itf_n = xp.zeros(sh, dtype=dtype)
        #     self.q_itf_w = xp.zeros(sh, dtype=dtype)
        #     self.q_itf_e = xp.zeros(sh, dtype=dtype)

        # self.q_itf_s[...], self.q_itf_n[...], self.q_itf_w[...], self.q_itf_e[...] = self.req_all.wait()
        self.q_itf_s, self.q_itf_n, self.q_itf_w, self.q_itf_e = self.req_all.wait()

    def riemann_fluxes(self) -> None:
        xp = self.device.xp

        mid_i = xp.s_[..., 1:-1, :]
        mid_j = xp.s_[..., 1:-1, :, :]
        mid_k = xp.s_[..., 1:-1, :, :, :]

        s = numpy.s_[..., 0, :, self.geom.itf_size :]
        n = numpy.s_[..., -1, :, : self.geom.itf_size]
        w = numpy.s_[..., 0, self.geom.itf_size :]
        e = numpy.s_[..., -1, : self.geom.itf_size]
        b = numpy.s_[..., 0, :, :, self.geom.itf_size :]
        t = numpy.s_[..., -1, :, :, : self.geom.itf_size]

        self.q_itf_full_x1[mid_i] = self.q_itf_x1
        self.q_itf_full_x2[mid_j] = self.q_itf_x2
        self.q_itf_full_x3[mid_k] = self.q_itf_x3

        # Element interfaces from neighboring tiles
        self.q_itf_full_x1[w] = self.q_itf_w
        self.q_itf_full_x1[e] = self.q_itf_e
        self.q_itf_full_x2[s] = self.q_itf_s
        self.q_itf_full_x2[n] = self.q_itf_n

        # Top + bottom layers
        self.q_itf_full_x3[b] = self.q_itf_full_x3[..., 1, :, :, : self.geom.itf_size]
        # self.q_itf_full_x3[..., 0, :, :, : self.geom.itf_size] = self.q_itf_full_x3[b]
        self.q_itf_full_x3[t] = self.q_itf_full_x3[..., -2, :, :, self.geom.itf_size :]
        # self.q_itf_full_x3[..., -1, :, :, self.geom.itf_size :] = self.q_itf_full_x3[t]

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
