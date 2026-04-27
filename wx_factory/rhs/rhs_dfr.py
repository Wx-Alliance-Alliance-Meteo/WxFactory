import numpy
from numpy.typing import NDArray

from common.definitions import idx_rho, idx_rho_u1, idx_rho_u2, idx_rho_w, idx_rho_theta
from common.matmul import apply_op
from geometry import CubedSphere, DFROperators
from rhs.rhs import RHS
from wx_mpi import SingleProcess, Conditional


mid_i = numpy.s_[..., 1:-1, :]
mid_j = numpy.s_[..., 1:-1, :, :]
mid_k = numpy.s_[..., 1:-1, :, :, :]


class RHSDirecFluxReconstruction(RHS):

    def solution_extrapolation(self, q: NDArray) -> None:
        # Extrapolate the solution to element boundaries
        # if self.num_dim == 2:
        # Investigate why this is slower since no reallocation is needed
        #     xp.matmul(q, self.ops.extrap_x, out=self.q_itf_x1)
        #     xp.matmul(q, self.ops.extrap_z, out=self.q_itf_x3)

        xp = self.device.xp

        op_extrap_x = self.ops.extrap_x if not xp.iscomplexobj(q) else self.ops.extrap_x_complex
        op_extrap_z = self.ops.extrap_z if not xp.iscomplexobj(q) else self.ops.extrap_z_complex
        op_extrap_y = self.ops.extrap_y if not xp.iscomplexobj(q) else self.ops.extrap_y_complex

        self.q_itf_x1 = apply_op(q, op_extrap_x)
        self.q_itf_x3 = apply_op(q, op_extrap_z)
        if hasattr(self.ops, "extrap_y"):
            self.q_itf_x2 = apply_op(q, op_extrap_y)

    def pointwise_fluxes(self, q: NDArray) -> None:
        self.pde.pointwise_fluxes(q, self.f_x1, self.f_x2, self.f_x3)

    def flux_divergence_partial(self) -> NDArray:
        xp = self.device.xp

        # Compute derivatives, with correction from boundaries
        # Investigate why this is slower
        # xp.matmul(self.f_x1, self.ops.derivative_x, out=self.df1_dx1)
        # xp.matmul(self.f_x3, self.ops.derivative_z, out=self.df3_dx3)

        op_dx = self.ops.derivative_x if not xp.iscomplexobj(self.f_x1) else self.ops.derivative_x_complex
        op_dz = self.ops.derivative_z if not xp.iscomplexobj(self.f_x3) else self.ops.derivative_z_complex

        self.df1_dx1 = apply_op(self.f_x1, op_dx)
        self.df3_dx3 = apply_op(self.f_x3, op_dz)

    def flux_divergence(self):
        xp = self.device.xp

        op_correction_WE = (
            self.ops.correction_WE if not xp.iscomplexobj(self.f_itf_x1) else self.ops.correction_WE_complex
        )
        op_correction_DU = (
            self.ops.correction_DU if not xp.iscomplexobj(self.f_itf_x3) else self.ops.correction_DU_complex
        )

        self.df1_dx1 += apply_op(self.f_itf_x1, op_correction_WE)
        self.df1_dx1 *= -2.0 / self.geom.Δx1

        self.df3_dx3 += apply_op(self.f_itf_x3, op_correction_DU)
        self.df3_dx3 *= -2.0 / self.geom.Δx3

        xp.add(self.df1_dx1, self.df3_dx3, out=self.rhs)


class RHSDirecFluxReconstruction_mpi(RHSDirecFluxReconstruction):
    def __init__(
        self,
        pde,
        geometry: CubedSphere,
        operators_real: DFROperators,
        operators_complex: DFROperators,
        metric,
        topography,
        process_topo,
        config,
        expected_shape,
        debug=False,
    ):
        super().__init__(
            pde,
            geometry,
            operators_real,
            operators_complex,
            metric,
            topography,
            process_topo,
            config,
            expected_shape,
            debug,
        )
        self.extrap_3d = self.extrap_3d_code
        if config.desired_device in ["numpy", "cupy", "torch"]:
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

        op_extrap_x = self.ops.extrap_x if not xp.iscomplexobj(q) else self.ops.extrap_x_complex
        op_extrap_z = self.ops.extrap_z if not xp.iscomplexobj(q) else self.ops.extrap_z_complex
        op_extrap_y = self.ops.extrap_y if not xp.iscomplexobj(q) else self.ops.extrap_y_complex

        self.extrap_3d(q, self.q_itf_x1, self.q_itf_x2, self.q_itf_x3)

        self.log_rho_p = xp.log(q[idx_rho])
        self.log_rho_theta = xp.log(q[idx_rho_theta])

        # TODO clean this up (avoid overwriting previous computation)
        self.q_itf_x1[idx_rho] = xp.exp(apply_op(self.log_rho_p, op_extrap_x))
        self.q_itf_x1[idx_rho_theta] = xp.exp(apply_op(self.log_rho_theta, op_extrap_x))
        self.q_itf_x2[idx_rho] = xp.exp(apply_op(self.log_rho_p, op_extrap_y))
        self.q_itf_x2[idx_rho_theta] = xp.exp(apply_op(self.log_rho_theta, op_extrap_y))
        self.q_itf_x3[idx_rho] = xp.exp(apply_op(self.log_rho_p, op_extrap_z))
        self.q_itf_x3[idx_rho_theta] = xp.exp(apply_op(self.log_rho_theta, op_extrap_z))

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
        xp = self.device.xp

        op_dx = self.ops.derivative_x if not xp.iscomplexobj(self.f_x1) else self.ops.derivative_x_complex
        op_dy = self.ops.derivative_y if not xp.iscomplexobj(self.f_x2) else self.ops.derivative_y_complex
        op_dz = self.ops.derivative_z if not xp.iscomplexobj(self.f_x3) else self.ops.derivative_z_complex

        self.df1_dx1 = apply_op(self.f_x1, op_dx)
        self.df2_dx2 = apply_op(self.f_x2, op_dy)
        self.df3_dx3 = apply_op(self.f_x3, op_dz)

        self.w_df1_dx1_adv = apply_op(self.wflux_adv_x1, op_dx)
        self.w_df1_dx1_presa = apply_op(self.wflux_pres_x1, op_dx)
        self.w_df1_dx1_presb = apply_op(self.log_p, op_dx)

        self.w_df2_dx2_adv = apply_op(self.wflux_adv_x2, op_dy)
        self.w_df2_dx2_presa = apply_op(self.wflux_pres_x2, op_dy)
        self.w_df2_dx2_presb = apply_op(self.log_p, op_dy)

        self.w_df3_dx3_adv = apply_op(self.wflux_adv_x3, op_dz)
        self.w_df3_dx3_presa = apply_op(self.wflux_pres_x3, op_dz)
        self.w_df3_dx3_presb = apply_op(self.log_p, op_dz)

    def flux_divergence(self):
        xp = self.device.xp

        op_correction_WE = (
            self.ops.correction_WE if not xp.iscomplexobj(self.f_itf_x1) else self.ops.correction_WE_complex
        )
        op_correction_SN = (
            self.ops.correction_SN if not xp.iscomplexobj(self.f_itf_x2) else self.ops.correction_SN_complex
        )
        op_correction_DU = (
            self.ops.correction_DU if not xp.iscomplexobj(self.f_itf_x3) else self.ops.correction_DU_complex
        )

        self.df1_dx1 += apply_op(self.f_itf_x1, op_correction_WE)
        self.df2_dx2 += apply_op(self.f_itf_x2, op_correction_SN)
        self.df3_dx3 += apply_op(self.f_itf_x3, op_correction_DU)

        logp_bdy_i = xp.log(self.pressure_itf_x1)
        logp_bdy_j = xp.log(self.pressure_itf_x2)
        logp_bdy_k = xp.log(self.pressure_itf_x3)

        self.w_df1_dx1_adv += apply_op(self.wflux_adv_itf_x1, op_correction_WE)
        self.w_df1_dx1_presa += apply_op(self.wflux_pres_itf_x1, op_correction_WE)
        self.w_df1_dx1_presa *= self.pressure
        self.w_df1_dx1_presb += apply_op(logp_bdy_i, op_correction_WE)
        self.w_df1_dx1_presb *= self.pressure * self.wflux_pres_x1
        self.w_df1_dx1[...] = self.w_df1_dx1_adv + self.w_df1_dx1_presa + self.w_df1_dx1_presb

        self.w_df2_dx2_adv += apply_op(self.wflux_adv_itf_x2, op_correction_SN)
        self.w_df2_dx2_presa += apply_op(self.wflux_pres_itf_x2, op_correction_SN)
        self.w_df2_dx2_presa *= self.pressure
        self.w_df2_dx2_presb += apply_op(logp_bdy_j, op_correction_SN)
        self.w_df2_dx2_presb *= self.pressure * self.wflux_pres_x2
        self.w_df2_dx2[...] = self.w_df2_dx2_adv + self.w_df2_dx2_presa + self.w_df2_dx2_presb

        self.w_df3_dx3_adv += apply_op(self.wflux_adv_itf_x3, op_correction_DU)
        self.w_df3_dx3_presa += apply_op(self.wflux_pres_itf_x3, op_correction_DU)
        self.w_df3_dx3_presa *= self.pressure
        self.w_df3_dx3_presb += apply_op(logp_bdy_k, op_correction_DU)
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
        self.q_itf_s, self.q_itf_n, self.q_itf_w, self.q_itf_e = self.req_all.wait()

    def _riemann_fluxes_prepare(self) -> None:
        itf_size = self.geom.itf_size

        s = numpy.s_[..., 0, :, itf_size:]
        n = numpy.s_[..., -1, :, :itf_size]
        w = numpy.s_[..., 0, itf_size:]
        e = numpy.s_[..., -1, :itf_size]
        b = numpy.s_[..., 0, :, :, itf_size:]
        t = numpy.s_[..., -1, :, :, :itf_size]

        self.q_itf_full_x1[mid_i] = self.q_itf_x1
        self.q_itf_full_x2[mid_j] = self.q_itf_x2
        self.q_itf_full_x3[mid_k] = self.q_itf_x3

        # Element interfaces from neighboring tiles
        self.q_itf_full_x1[w] = self.q_itf_w
        self.q_itf_full_x1[e] = self.q_itf_e
        self.q_itf_full_x2[s] = self.q_itf_s
        self.q_itf_full_x2[n] = self.q_itf_n

        # Top + bottom layers
        self.q_itf_full_x3[b] = self.q_itf_full_x3[..., 1, :, :, :itf_size]
        self.q_itf_full_x3[t] = self.q_itf_full_x3[..., -2, :, :, itf_size:]

    def riemann_fluxes(self) -> None:
        self._riemann_fluxes_prepare()

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


class RHSDirecFluxReconstruction_mpi_v2(RHSDirecFluxReconstruction):
    def __init__(
        self,
        pde,
        geometry: CubedSphere,
        operators_real: DFROperators,
        operators_complex: DFROperators,
        metric,
        topography,
        process_topo,
        config,
        expected_shape,
        debug=False,
    ):
        super().__init__(
            pde,
            geometry,
            operators_real,
            operators_complex,
            metric,
            topography,
            process_topo,
            config,
            expected_shape,
            debug,
        )
        self.extrap_3d = self.extrap_3d_code
        if config.desired_device in ["numpy", "cupy", "torch"]:
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

        op_extrap_x = self.ops.extrap_x if not xp.iscomplexobj(q) else self.ops.extrap_x_complex
        op_extrap_z = self.ops.extrap_z if not xp.iscomplexobj(q) else self.ops.extrap_z_complex
        op_extrap_y = self.ops.extrap_y if not xp.iscomplexobj(q) else self.ops.extrap_y_complex

        self.extrap_3d(q, self.q_itf_x1, self.q_itf_x2, self.q_itf_x3)

        self.log_rho_p = xp.log(q[idx_rho])
        self.log_rho_theta = xp.log(q[idx_rho_theta])

        # TODO clean this up (avoid overwriting previous computation)
        self.q_itf_x1[idx_rho] = xp.exp(apply_op(self.log_rho_p, op_extrap_x))
        self.q_itf_x1[idx_rho_theta] = xp.exp(apply_op(self.log_rho_theta, op_extrap_x))
        self.q_itf_x2[idx_rho] = xp.exp(apply_op(self.log_rho_p, op_extrap_y))
        self.q_itf_x2[idx_rho_theta] = xp.exp(apply_op(self.log_rho_theta, op_extrap_y))
        self.q_itf_x3[idx_rho] = xp.exp(apply_op(self.log_rho_p, op_extrap_z))
        self.q_itf_x3[idx_rho_theta] = xp.exp(apply_op(self.log_rho_theta, op_extrap_z))

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
        xp = self.device.xp

        op_dx = self.ops.derivative_x if not xp.iscomplexobj(self.f_x1) else self.ops.derivative_x_complex
        op_dy = self.ops.derivative_y if not xp.iscomplexobj(self.f_x2) else self.ops.derivative_y_complex
        op_dz = self.ops.derivative_z if not xp.iscomplexobj(self.f_x3) else self.ops.derivative_z_complex

        # Accumulate uncorrected derivatives into self.rhs directly
        apply_op(self.f_x1, op_dx, out=self.rhs, beta=0.0)
        apply_op(self.f_x2, op_dy, out=self.rhs, beta=1.0)
        apply_op(self.f_x3, op_dz, out=self.rhs, beta=1.0)

        # Accumulate advective terms into common self.w_df1_dx1 variable
        # Accumulate pressure A terms into common self.w_presa variable
        # Maintain separate pressure B terms for each dimension
        apply_op(self.wflux_adv_x1, op_dx, out=self.w_df1_dx1, beta=0.0)
        apply_op(self.wflux_adv_x2, op_dy, out=self.w_df1_dx1, beta=1.0)
        apply_op(self.wflux_adv_x3, op_dz, out=self.w_df1_dx1, beta=1.0)

        self.w_presa = apply_op(self.wflux_pres_x1, op_dx)
        apply_op(self.wflux_pres_x2, op_dy, out=self.w_presa, beta=1.0)
        apply_op(self.wflux_pres_x3, op_dz, out=self.w_presa, beta=1.0)

        self.w_df1_dx1_presb = apply_op(self.log_p, op_dx)
        self.w_df2_dx2_presb = apply_op(self.log_p, op_dy)
        self.w_df3_dx3_presb = apply_op(self.log_p, op_dz)

    def flux_divergence(self):
        xp = self.device.xp

        op_correction_WE = (
            self.ops.correction_WE if not xp.iscomplexobj(self.f_itf_x1) else self.ops.correction_WE_complex
        )
        op_correction_SN = (
            self.ops.correction_SN if not xp.iscomplexobj(self.f_itf_x2) else self.ops.correction_SN_complex
        )
        op_correction_DU = (
            self.ops.correction_DU if not xp.iscomplexobj(self.f_itf_x3) else self.ops.correction_DU_complex
        )

        # Accumulate correction terms into self.rhs
        apply_op(self.f_itf_x1, op_correction_WE, out=self.rhs, beta=1.0)
        apply_op(self.f_itf_x2, op_correction_SN, out=self.rhs, beta=1.0)
        apply_op(self.f_itf_x3, op_correction_DU, out=self.rhs, beta=1.0)

        logp_bdy_i = xp.log(self.pressure_itf_x1)
        logp_bdy_j = xp.log(self.pressure_itf_x2)
        logp_bdy_k = xp.log(self.pressure_itf_x3)

        # Accumulate correction for advective terms into common self.w_df1_dx1 variable
        # Accumulate correction for pressure A terms into common self.w_presa variable
        # Maintain separate pressure B terms for each dimension
        apply_op(self.wflux_adv_itf_x1, op_correction_WE, out=self.w_df1_dx1, beta=1.0)
        apply_op(self.wflux_adv_itf_x2, op_correction_SN, out=self.w_df1_dx1, beta=1.0)
        apply_op(self.wflux_adv_itf_x3, op_correction_DU, out=self.w_df1_dx1, beta=1.0)

        apply_op(self.wflux_pres_itf_x1, op_correction_WE, out=self.w_presa, beta=1.0)
        apply_op(self.wflux_pres_itf_x2, op_correction_SN, out=self.w_presa, beta=1.0)
        apply_op(self.wflux_pres_itf_x3, op_correction_DU, out=self.w_presa, beta=1.0)

        apply_op(logp_bdy_i, op_correction_WE, out=self.w_df1_dx1_presb, beta=1.0)
        self.w_df1_dx1_presb *= self.wflux_pres_x1

        apply_op(logp_bdy_j, op_correction_SN, out=self.w_df2_dx2_presb, beta=1.0)
        self.w_df2_dx2_presb *= self.wflux_pres_x2

        apply_op(logp_bdy_k, op_correction_DU, out=self.w_df3_dx3_presb, beta=1.0)
        self.w_df3_dx3_presb *= self.wflux_pres_x3

        self.rhs[idx_rho_w] = self.w_df1_dx1 + self.pressure * (
            self.w_presa + self.w_df1_dx1_presb + self.w_df2_dx2_presb + self.w_df3_dx3_presb
        )
        self.rhs *= -self.metric.inv_sqrtG_new

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
        self.q_itf_s, self.q_itf_n, self.q_itf_w, self.q_itf_e = self.req_all.wait()

    def _riemann_fluxes_prepare(self) -> None:
        itf_size = self.geom.itf_size

        s = numpy.s_[..., 0, :, itf_size:]
        n = numpy.s_[..., -1, :, :itf_size]
        w = numpy.s_[..., 0, itf_size:]
        e = numpy.s_[..., -1, :itf_size]
        b = numpy.s_[..., 0, :, :, itf_size:]
        t = numpy.s_[..., -1, :, :, :itf_size]

        self.q_itf_full_x1[mid_i] = self.q_itf_x1
        self.q_itf_full_x2[mid_j] = self.q_itf_x2
        self.q_itf_full_x3[mid_k] = self.q_itf_x3

        # Element interfaces from neighboring tiles
        self.q_itf_full_x1[w] = self.q_itf_w
        self.q_itf_full_x1[e] = self.q_itf_e
        self.q_itf_full_x2[s] = self.q_itf_s
        self.q_itf_full_x2[n] = self.q_itf_n

        # Top + bottom layers
        self.q_itf_full_x3[b] = self.q_itf_full_x3[..., 1, :, :, :itf_size]
        self.q_itf_full_x3[t] = self.q_itf_full_x3[..., -2, :, :, itf_size:]

    def riemann_fluxes(self) -> None:
        self._riemann_fluxes_prepare()

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
