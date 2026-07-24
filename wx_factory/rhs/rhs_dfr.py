import numpy
from numpy.typing import NDArray

from ..common.definitions import idx_rho, idx_rho_u1, idx_rho_u2, idx_rho_u3, idx_rho_theta, gravity
from ..common.matmul import apply_op
from ..geometry import CubedSphere, DFROperators
from ..rhs.rhs import RHS
from ..wx_mpi import SingleProcess, Conditional

mid_i = numpy.s_[..., 1:-1, :]
mid_j = numpy.s_[..., 1:-1, :, :]
mid_k = numpy.s_[..., 1:-1, :, :, :]


class RHSDirecFluxReconstruction(RHS):

    def allocate_arrays(self, q: NDArray) -> None:
        super().allocate_arrays(q)
        xp = self.device.xp

        if self.q_itf_x1 is None or self.q_itf_x1.dtype != q.dtype:
            itf_shape = q.shape[:4] + (2 * self.geom.num_solpts**2,)
            self.q_itf_x1 = xp.empty(itf_shape, dtype=q.dtype)
            self.q_itf_x2 = xp.empty_like(self.q_itf_x1)
            self.q_itf_x3 = xp.empty_like(self.q_itf_x1)

    def solution_extrapolation(self, q: NDArray) -> None:
        # Extrapolate the solution to element boundaries
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
            self.pressure = xp.zeros_like(q[0])
            self.log_p = xp.zeros_like(q[0])

            self.wflux_adv_x1 = xp.zeros_like(q[0])
            self.wflux_pres_x1 = xp.zeros_like(q[0])
            self.wflux_adv_x2 = xp.zeros_like(q[0])
            self.wflux_pres_x2 = xp.zeros_like(q[0])
            self.wflux_adv_x3 = xp.zeros_like(q[0])
            self.wflux_pres_x3 = xp.zeros_like(q[0])

            self.w_df1_dx1 = xp.zeros_like(q[0])
            self.w_df2_dx2 = xp.zeros_like(q[0])
            self.w_df3_dx3 = xp.zeros_like(q[0])

            self.forcing = xp.zeros_like(q)

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
        self.device.operators.extrap_all_3d(
            q,
            itf_x1,
            itf_x2,
            itf_x3,
            0,
        )

    def solution_extrapolation(self, q: NDArray) -> None:
        # Extrapolate the solution to element boundaries
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
        self.rhs[idx_rho_u3] = -self.metric.inv_sqrtG_new * (self.w_df1_dx1 + self.w_df2_dx2 + self.w_df3_dx3)

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
            self.rhs[idx_rho_u3] = 0.0
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
            self.pressure = xp.zeros_like(q[0])
            self.log_p = xp.zeros_like(q[0])

            self.wflux_adv_x1 = xp.zeros_like(q[0])
            self.wflux_pres_x1 = xp.zeros_like(q[0])
            self.wflux_adv_x2 = xp.zeros_like(q[0])
            self.wflux_pres_x2 = xp.zeros_like(q[0])
            self.wflux_adv_x3 = xp.zeros_like(q[0])
            self.wflux_pres_x3 = xp.zeros_like(q[0])

            self.w_df1_dx1 = xp.zeros_like(q[0])
            self.w_df2_dx2 = xp.zeros_like(q[0])
            self.w_df3_dx3 = xp.zeros_like(q[0])

            self.forcing = xp.zeros_like(q)

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
        self.device.operators.extrap_all_3d(
            q,
            itf_x1,
            itf_x2,
            itf_x3,
            0,
        )

    def solution_extrapolation(self, q: NDArray) -> None:
        # Extrapolate the solution to element boundaries
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

        self.rhs[idx_rho_u3] = self.w_df1_dx1 + self.pressure * (
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
            self.rhs[idx_rho_u3] = 0.0
            self.rhs[idx_rho_theta] = 0.0

    def implicit(self, q: NDArray) -> NDArray:
        """Vertically-stiff partition f1 for PartRosExp2 (Option B: exact vertical part of ``full``).

        Reproduces the x3-only part of the full right-hand side, so that f2 = full - f1 (taken by
        difference in the integrator) contains only horizontal operators. The four rows
        rho, rho u1, rho u2, rho theta use the plain vertical flux divergence; the rho_w row uses the
        same well-balanced split as ``full`` (advective divergence + p * central metric divergence +
        (sqrtG h33) p * central log-pressure divergence, design note eq:wbrhow), and gravity is the
        same high-order-filtered source as ``full`` (the -= sign comes from ``rhs -= forcing`` there).
        Christoffel/Coriolis forcing and the Rayleigh sponge are horizontal/source terms and stay in
        f2. The vertical operator couples only within a column, so no horizontal halo exchange is
        needed; the horizontal interface traces are filled locally purely so the shared Riemann
        routine runs (its x1/x2 outputs are discarded)."""
        xp = self.device.xp
        given_shape = q.shape
        self.ops = self.ops_complex if xp.iscomplexobj(q) else self.ops_real
        self.allocate_arrays(q)

        # 1. Extrapolate to element boundaries (vertical traces are the ones we use).
        self.solution_extrapolation(q)

        # No horizontal exchange: feed local face values so the (shared) Riemann routine runs. Only the
        # x3 fluxes are used; the x1/x2 results computed from these local traces are discarded.
        itf_size = self.geom.itf_size
        self.q_itf_w = self.q_itf_x1[..., 0, :itf_size].copy()
        self.q_itf_e = self.q_itf_x1[..., -1, itf_size:].copy()
        self.q_itf_s = self.q_itf_x2[..., 0, :, :itf_size].copy()
        self.q_itf_n = self.q_itf_x2[..., -1, :, itf_size:].copy()

        # 2. Pointwise fluxes: f_x3 (all rows) plus the well-balanced pieces wflux_adv_x3, wflux_pres_x3,
        #    log_p and the pressure.
        self.pointwise_fluxes(q)

        # 3. Interior vertical derivative d3(sqrtG F3) for all five rows (rho_w overwritten below).
        op_dz = self.ops.derivative_z if not xp.iscomplexobj(self.f_x3) else self.ops.derivative_z_complex
        apply_op(self.f_x3, op_dz, out=self.rhs, beta=0.0)

        # 4. Vertical Rusanov interface fluxes (fills f_itf_x3, wflux_adv/pres_itf_x3, pressure_itf_x3).
        self.riemann_fluxes()

        # 5. Boundary correction for the four plain rows.
        op_corr = self.ops.correction_DU if not xp.iscomplexobj(self.f_itf_x3) else self.ops.correction_DU_complex
        apply_op(self.f_itf_x3, op_corr, out=self.rhs, beta=1.0)

        # 5b. Well-balanced rho_w row (x3 only), mirroring the vertical part of ``full``:
        #     R_w = D3^R[sqrtG w rho_w] + p * ( D3^c_g[sqrtG h33] + (sqrtG h33) * D3^c[log p] ).
        w_df3 = apply_op(self.wflux_adv_x3, op_dz)
        apply_op(self.wflux_adv_itf_x3, op_corr, out=w_df3, beta=1.0)

        w_presa = apply_op(self.wflux_pres_x3, op_dz)
        apply_op(self.wflux_pres_itf_x3, op_corr, out=w_presa, beta=1.0)

        logp_bdy_k = xp.log(self.pressure_itf_x3)
        w_presb = apply_op(self.log_p, op_dz)
        apply_op(logp_bdy_k, op_corr, out=w_presb, beta=1.0)
        w_presb *= self.wflux_pres_x3

        self.rhs[idx_rho_u3] = w_df3 + self.pressure * (w_presa + w_presb)

        # 6. Outer 1/sqrtG factor.
        self.rhs *= -self.metric.inv_sqrtG_new

        # 7. Gravity source in the vertical-momentum row (same filtered form as ``full``; the sign is
        #    that of ``rhs -= forcing``).
        self.rhs[idx_rho_u3] -= (
            self.metric.inv_dzdeta_new
            * gravity
            * self.metric.inv_sqrtG_new
            * ((self.metric.sqrtG_new * q[idx_rho]) @ self.ops.highfilter_k)
        )

        return self.rhs.reshape(given_shape).copy()

    def explicit(self, q: NDArray) -> NDArray:
        """Horizontal partition f2 = full - f1 for PartRosExp2, computed DIRECTLY (not as full - f1).

        Forming full - f1 in single precision loses almost all of f2: f1 nearly equals full in the
        theta-scaled rho_theta row (the vertical flux dominates), so the difference is a catastrophic
        cancellation of large float32 numbers, which then makes the exponential propagator blow up.
        Here the horizontal (x1, x2) flux divergences, the well-balanced rho_w horizontal terms and
        the Christoffel/Coriolis/Rayleigh forcing are accumulated directly; gravity is removed since
        it belongs to f1, so f1 + f2 = full exactly."""
        xp = self.device.xp
        given_shape = q.shape
        self.ops = self.ops_complex if xp.iscomplexobj(q) else self.ops_real
        self.allocate_arrays(q)

        self.solution_extrapolation(q)
        self.start_communication()
        self.pointwise_fluxes(q)

        cplx = xp.iscomplexobj(self.f_x1)
        op_dx = self.ops.derivative_x if not cplx else self.ops.derivative_x_complex
        op_dy = self.ops.derivative_y if not cplx else self.ops.derivative_y_complex

        # 1. Interior horizontal derivatives (x1, x2 only).
        apply_op(self.f_x1, op_dx, out=self.rhs, beta=0.0)
        apply_op(self.f_x2, op_dy, out=self.rhs, beta=1.0)
        apply_op(self.wflux_adv_x1, op_dx, out=self.w_df1_dx1, beta=0.0)
        apply_op(self.wflux_adv_x2, op_dy, out=self.w_df1_dx1, beta=1.0)
        self.w_presa = apply_op(self.wflux_pres_x1, op_dx)
        apply_op(self.wflux_pres_x2, op_dy, out=self.w_presa, beta=1.0)
        self.w_df1_dx1_presb = apply_op(self.log_p, op_dx)
        self.w_df2_dx2_presb = apply_op(self.log_p, op_dy)

        self.end_communication()
        self.riemann_fluxes()

        op_corr_WE = self.ops.correction_WE if not cplx else self.ops.correction_WE_complex
        op_corr_SN = self.ops.correction_SN if not cplx else self.ops.correction_SN_complex

        # 2. Boundary corrections (x1, x2 only).
        apply_op(self.f_itf_x1, op_corr_WE, out=self.rhs, beta=1.0)
        apply_op(self.f_itf_x2, op_corr_SN, out=self.rhs, beta=1.0)
        apply_op(self.wflux_adv_itf_x1, op_corr_WE, out=self.w_df1_dx1, beta=1.0)
        apply_op(self.wflux_adv_itf_x2, op_corr_SN, out=self.w_df1_dx1, beta=1.0)
        apply_op(self.wflux_pres_itf_x1, op_corr_WE, out=self.w_presa, beta=1.0)
        apply_op(self.wflux_pres_itf_x2, op_corr_SN, out=self.w_presa, beta=1.0)

        logp_bdy_i = xp.log(self.pressure_itf_x1)
        logp_bdy_j = xp.log(self.pressure_itf_x2)
        apply_op(logp_bdy_i, op_corr_WE, out=self.w_df1_dx1_presb, beta=1.0)
        self.w_df1_dx1_presb *= self.wflux_pres_x1
        apply_op(logp_bdy_j, op_corr_SN, out=self.w_df2_dx2_presb, beta=1.0)
        self.w_df2_dx2_presb *= self.wflux_pres_x2

        # 3. Well-balanced rho_w horizontal row, then the outer 1/sqrtG factor.
        self.rhs[idx_rho_u3] = self.w_df1_dx1 + self.pressure * (
            self.w_presa + self.w_df1_dx1_presb + self.w_df2_dx2_presb
        )
        self.rhs *= -self.metric.inv_sqrtG_new

        # 4. Forcing (Christoffel / Coriolis / Rayleigh), obtained as the full forcing with gravity
        #    added back (gravity is in f1).
        self.forcing_terms(q)
        self.rhs[idx_rho_u3] += (
            self.metric.inv_dzdeta_new
            * gravity
            * self.metric.inv_sqrtG_new
            * ((self.metric.sqrtG_new * q[idx_rho]) @ self.ops.highfilter_k)
        )

        return self.rhs.reshape(given_shape).copy()

    def horizontal_flux_div(self, q: NDArray) -> NDArray:
        """Plain horizontal (x1, x2) flux divergence for all five rows, no well-balanced rho_w split
        and no forcing. Used to split f2 into a stiff (acoustic) part -- whose analytic Jacobian is
        cheap and exact -- and a non-stiff remainder (well-balanced rho_w correction + forcing) that
        PartRosExp2 differentiates by finite differences. Populates the horizontal traces / exchanged
        neighbours / pressure that the analytic Jacobian reuses."""
        xp = self.device.xp
        given_shape = q.shape
        self.ops = self.ops_complex if xp.iscomplexobj(q) else self.ops_real
        self.allocate_arrays(q)

        self.solution_extrapolation(q)
        self.start_communication()
        self.pointwise_fluxes(q)

        cplx = xp.iscomplexobj(self.f_x1)
        op_dx = self.ops.derivative_x if not cplx else self.ops.derivative_x_complex
        op_dy = self.ops.derivative_y if not cplx else self.ops.derivative_y_complex
        apply_op(self.f_x1, op_dx, out=self.rhs, beta=0.0)
        apply_op(self.f_x2, op_dy, out=self.rhs, beta=1.0)

        self.end_communication()
        self.riemann_fluxes()

        op_corr_WE = self.ops.correction_WE if not cplx else self.ops.correction_WE_complex
        op_corr_SN = self.ops.correction_SN if not cplx else self.ops.correction_SN_complex
        apply_op(self.f_itf_x1, op_corr_WE, out=self.rhs, beta=1.0)
        apply_op(self.f_itf_x2, op_corr_SN, out=self.rhs, beta=1.0)
        self.rhs *= -self.metric.inv_sqrtG_new
        return self.rhs.reshape(given_shape).copy()

    def forcing_only(self, q: NDArray) -> NDArray:
        """Just the f2 forcing (Christoffel / Coriolis / Rayleigh), no flux, no gravity. This is the
        non-stiff remainder of f2 that PartRosExp2 differentiates by finite differences (the stiff
        flux divergence has an analytic Jacobian). Sign matches ``rhs -= forcing`` with gravity added
        back (gravity is in f1)."""
        xp = self.device.xp
        given_shape = q.shape
        self.ops = self.ops_complex if xp.iscomplexobj(q) else self.ops_real
        self.allocate_arrays(q)
        self.pointwise_fluxes(q)  # sets self.pressure
        self.rhs[...] = 0.0
        self.forcing_terms(q)  # rhs -= (Christoffel/Coriolis/gravity/Rayleigh)
        self.rhs[idx_rho_u3] += (
            self.metric.inv_dzdeta_new
            * gravity
            * self.metric.inv_sqrtG_new
            * ((self.metric.sqrtG_new * q[idx_rho]) @ self.ops.highfilter_k)
        )
        return self.rhs.reshape(given_shape).copy()
