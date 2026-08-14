import numpy
import torch
from torch import Tensor

from ..common.definitions import (
    idx_rho,
    idx_rho_theta,
    idx_rho_u1,
    idx_rho_u2,
    idx_rho_u3,
)
from ..common.matmul import apply_op
from ..geometry import CubedSphere, CubedSphere3D, DFROperators, Metric3DTopo
from ..rhs.rhs import RHS

mid_i = numpy.s_[..., 1:-1, :]
mid_j = numpy.s_[..., 1:-1, :, :]
mid_k = numpy.s_[..., 1:-1, :, :, :]


class RHSDirecFluxReconstruction(RHS):
    def allocate_arrays(self, q: Tensor) -> None:
        super().allocate_arrays(q)
        if self.q_itf_x1 is None or self.q_itf_x1.dtype != q.dtype:
            itf_shape = q.shape[:4] + (2 * self.geom.num_solpts**2,)
            self.q_itf_x1 = torch.empty(itf_shape, dtype=q.dtype)
            self.q_itf_x2 = torch.empty_like(self.q_itf_x1)
            self.q_itf_x3 = torch.empty_like(self.q_itf_x1)

    def solution_extrapolation(self, q: Tensor) -> None:
        op_extrap_x = self.ops.extrap_x
        op_extrap_z = self.ops.extrap_z
        op_extrap_y = self.ops.extrap_y

        self.q_itf_x1 = apply_op(q, op_extrap_x)
        self.q_itf_x3 = apply_op(q, op_extrap_z)
        if hasattr(self.ops, "extrap_y"):
            self.q_itf_x2 = apply_op(q, op_extrap_y)

    def pointwise_fluxes(self, q: Tensor) -> None:
        self.pde.pointwise_fluxes(q, self.f_x1, self.f_x2, self.f_x3)

    def flux_divergence_partial(self):
        op_dx = self.ops.derivative_x
        op_dz = self.ops.derivative_z

        self.df1_dx1 = apply_op(self.f_x1, op_dx)
        self.df3_dx3 = apply_op(self.f_x3, op_dz)

    def flux_divergence(self):
        op_correction_WE = self.ops.correction_WE
        op_correction_DU = self.ops.correction_DU

        self.df1_dx1 += apply_op(self.f_itf_x1, op_correction_WE)
        self.df1_dx1 *= -2.0 / self.geom.Δx1

        self.df3_dx3 += apply_op(self.f_itf_x3, op_correction_DU)
        self.df3_dx3 *= -2.0 / self.geom.Δx3

        torch.add(self.df1_dx1, self.df3_dx3, out=self.rhs)


class RHSDirecFluxReconstruction_mpi(RHSDirecFluxReconstruction):
    geom: CubedSphere3D
    metric: Metric3DTopo

    q_itf_s: Tensor

    def __init__(
        self,
        pde,
        geometry: CubedSphere,
        operators_real: DFROperators,
        metric: Metric3DTopo,
        topography,
        process_topo,
        config,
        debug=False,
    ):
        super().__init__(
            pde,
            geometry,
            operators_real,
            metric,
            topography,
            process_topo,
            config,
            debug,
        )
        self.extrap_3d = self.extrap_3d_py

    def allocate_arrays(self, q):
        super().allocate_arrays(q)

        dtype = self.q_itf_x1.dtype

        itf_i_shape = (self.num_var,) + self.geom.itf_i_shape
        itf_j_shape = (self.num_var,) + self.geom.itf_j_shape
        itf_k_shape = (self.num_var,) + self.geom.itf_k_shape

        if self.f_itf_x1 is None or self.f_itf_x1.dtype != dtype:
            self.pressure = torch.zeros_like(q[0])
            self.log_p = torch.zeros_like(q[0])

            self.wflux_adv_x1 = torch.zeros_like(q[0])
            self.wflux_pres_x1 = torch.zeros_like(q[0])
            self.wflux_adv_x2 = torch.zeros_like(q[0])
            self.wflux_pres_x2 = torch.zeros_like(q[0])
            self.wflux_adv_x3 = torch.zeros_like(q[0])
            self.wflux_pres_x3 = torch.zeros_like(q[0])

            self.w_df1_dx1 = torch.zeros_like(q[0])
            self.w_df2_dx2 = torch.zeros_like(q[0])
            self.w_df3_dx3 = torch.zeros_like(q[0])

            self.forcing = torch.zeros_like(q)

            self.f_itf_x1 = torch.zeros_like(self.q_itf_x1)
            self.f_itf_x2 = torch.zeros_like(self.q_itf_x2)
            self.f_itf_x3 = torch.zeros_like(self.q_itf_x3)

            self.pressure_itf_x1 = torch.zeros_like(self.f_itf_x1[0])
            self.pressure_itf_x2 = torch.zeros_like(self.f_itf_x2[0])
            self.pressure_itf_x3 = torch.zeros_like(self.f_itf_x3[0])

            self.wflux_adv_itf_x1 = torch.zeros_like(self.f_itf_x1[0])
            self.wflux_pres_itf_x1 = torch.zeros_like(self.f_itf_x1[0])
            self.wflux_adv_itf_x2 = torch.zeros_like(self.f_itf_x2[0])
            self.wflux_pres_itf_x2 = torch.zeros_like(self.f_itf_x2[0])
            self.wflux_adv_itf_x3 = torch.zeros_like(self.f_itf_x3[0])
            self.wflux_pres_itf_x3 = torch.zeros_like(self.f_itf_x3[0])

            # Padding enters logarithms before every slot is filled.
            self.q_itf_full_x1 = torch.ones(itf_i_shape, dtype=dtype)
            self.q_itf_full_x2 = torch.ones(itf_j_shape, dtype=dtype)
            self.q_itf_full_x3 = torch.ones(itf_k_shape, dtype=dtype)

            self.f_itf_full_x1 = torch.zeros_like(self.q_itf_full_x1)
            self.f_itf_full_x2 = torch.zeros_like(self.q_itf_full_x2)
            self.f_itf_full_x3 = torch.zeros_like(self.q_itf_full_x3)

            self.pressure_itf_full_x1 = torch.zeros_like(self.q_itf_full_x1[0])
            self.pressure_itf_full_x2 = torch.zeros_like(self.q_itf_full_x2[0])
            self.pressure_itf_full_x3 = torch.zeros_like(self.q_itf_full_x3[0])

            self.wflux_adv_itf_full_x1 = torch.zeros_like(self.q_itf_full_x1[0])
            self.wflux_pres_itf_full_x1 = torch.zeros_like(self.q_itf_full_x1[0])
            self.wflux_adv_itf_full_x2 = torch.zeros_like(self.q_itf_full_x2[0])
            self.wflux_pres_itf_full_x2 = torch.zeros_like(self.q_itf_full_x2[0])
            self.wflux_adv_itf_full_x3 = torch.zeros_like(self.q_itf_full_x3[0])
            self.wflux_pres_itf_full_x3 = torch.zeros_like(self.q_itf_full_x3[0])

    def extrap_3d_py(self, q: Tensor, itf_x1: Tensor, itf_x2: Tensor, itf_x3: Tensor) -> None:
        itf_x1[...] = q @ self.ops.extrap_x
        itf_x2[...] = q @ self.ops.extrap_y
        itf_x3[...] = q @ self.ops.extrap_z

    def solution_extrapolation(self, q: Tensor) -> None:
        op_extrap_x = self.ops.extrap_x
        op_extrap_z = self.ops.extrap_z
        op_extrap_y = self.ops.extrap_y

        self.extrap_3d(q, self.q_itf_x1, self.q_itf_x2, self.q_itf_x3)

        self.log_rho_p = torch.log(q[idx_rho])
        self.log_rho_theta = torch.log(q[idx_rho_theta])

        # Preserve positivity by extrapolating density variables in logarithmic form.
        self.q_itf_x1[idx_rho] = torch.exp(apply_op(self.log_rho_p, op_extrap_x))
        self.q_itf_x1[idx_rho_theta] = torch.exp(apply_op(self.log_rho_theta, op_extrap_x))
        self.q_itf_x2[idx_rho] = torch.exp(apply_op(self.log_rho_p, op_extrap_y))
        self.q_itf_x2[idx_rho_theta] = torch.exp(apply_op(self.log_rho_theta, op_extrap_y))
        self.q_itf_x3[idx_rho] = torch.exp(apply_op(self.log_rho_p, op_extrap_z))
        self.q_itf_x3[idx_rho_theta] = torch.exp(apply_op(self.log_rho_theta, op_extrap_z))

    def pointwise_fluxes(self, q: Tensor) -> None:
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
        op_dx = self.ops.derivative_x
        op_dy = self.ops.derivative_y
        op_dz = self.ops.derivative_z

        # Accumulate volume derivatives.
        apply_op(self.f_x1, op_dx, out=self.rhs, beta=0.0)
        apply_op(self.f_x2, op_dy, out=self.rhs, beta=1.0)
        apply_op(self.f_x3, op_dz, out=self.rhs, beta=1.0)

        # Accumulate the split vertical-momentum terms.
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
        op_correction_WE = self.ops.correction_WE
        op_correction_SN = self.ops.correction_SN
        op_correction_DU = self.ops.correction_DU

        # Add interface corrections.
        apply_op(self.f_itf_x1, op_correction_WE, out=self.rhs, beta=1.0)
        apply_op(self.f_itf_x2, op_correction_SN, out=self.rhs, beta=1.0)
        apply_op(self.f_itf_x3, op_correction_DU, out=self.rhs, beta=1.0)

        logp_bdy_i = torch.log(self.pressure_itf_x1)
        logp_bdy_j = torch.log(self.pressure_itf_x2)
        logp_bdy_k = torch.log(self.pressure_itf_x3)

        # Correct the split vertical-momentum terms.
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

        # Horizontal neighbours.
        self.q_itf_full_x1[w] = self.q_itf_w
        self.q_itf_full_x1[e] = self.q_itf_e
        self.q_itf_full_x2[s] = self.q_itf_s
        self.q_itf_full_x2[n] = self.q_itf_n

        # Vertical boundaries.
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

    def forcing_terms(self, q: Tensor) -> None:
        self.pde.forcing_terms(self.rhs, q, self.pressure, self.metric, self.ops, self.forcing)

        # Freeze the dynamical variables in tracer-only tests.
        if self.config.advection_only:
            self.rhs[idx_rho] = 0.0
            self.rhs[idx_rho_u1] = 0.0
            self.rhs[idx_rho_u2] = 0.0
            self.rhs[idx_rho_u3] = 0.0
            self.rhs[idx_rho_theta] = 0.0

    def implicit(self, q: Tensor) -> Tensor:
        """Return the column-local stiff partition.

        It contains the vertical mass, vertical-momentum and thermodynamic fluxes, plus gravity.
        Horizontal-momentum tendencies remain in ``explicit`` to preserve their terrain balance.
        """
        given_shape = q.shape
        self.ops = self.operators_for(q)
        self.allocate_arrays(q)

        self.solution_extrapolation(q)

        # Supply local horizontal traces; only the vertical Riemann flux is retained.
        itf_size = self.geom.itf_size
        self.q_itf_w = self.q_itf_x1[..., 0, :itf_size].clone()
        self.q_itf_e = self.q_itf_x1[..., -1, itf_size:].clone()
        self.q_itf_s = self.q_itf_x2[..., 0, :, :itf_size].clone()
        self.q_itf_n = self.q_itf_x2[..., -1, :, itf_size:].clone()

        self.pointwise_fluxes(q)

        # The well-balanced vertical-momentum row is replaced below.
        op_dz = self.ops.derivative_z
        apply_op(self.f_x3, op_dz, out=self.rhs, beta=0.0)

        self.riemann_fluxes()

        op_corr = self.ops.correction_DU
        apply_op(self.f_itf_x3, op_corr, out=self.rhs, beta=1.0)

        # Well-balanced vertical-momentum residual.
        w_df3 = apply_op(self.wflux_adv_x3, op_dz)
        apply_op(self.wflux_adv_itf_x3, op_corr, out=w_df3, beta=1.0)

        w_presa = apply_op(self.wflux_pres_x3, op_dz)
        apply_op(self.wflux_pres_itf_x3, op_corr, out=w_presa, beta=1.0)

        logp_bdy_k = torch.log(self.pressure_itf_x3)
        w_presb = apply_op(self.log_p, op_dz)
        apply_op(logp_bdy_k, op_corr, out=w_presb, beta=1.0)
        w_presb *= self.wflux_pres_x3

        self.rhs[idx_rho_u3] = w_df3 + self.pressure * (w_presa + w_presb)

        self.rhs *= -self.metric.inv_sqrtG_new

        # Gravity uses the same filtered density as the full RHS.
        self.rhs[idx_rho_u3] -= (
            self.metric.inv_dzdeta_new
            * self.metric.gravity_new
            * self.metric.inv_sqrtG_new
            * ((self.metric.sqrtG_new * q[idx_rho]) @ self.ops.highfilter_k)
        )

        # Keep the complete terrain pressure balance in f2; splitting it would couple f1 horizontally.
        self.rhs[idx_rho_u1] = 0.0
        self.rhs[idx_rho_u2] = 0.0

        self.pin_y_momentum()

        return self.rhs.reshape(given_shape).clone()

    def explicit(self, q: Tensor) -> Tensor:
        """Return the complementary partition f2.

        It includes horizontal fluxes and forcing, plus the vertical fluxes of horizontal momentum
        needed to keep the terrain pressure balance within one partition.
        """
        given_shape = q.shape
        self.ops = self.operators_for(q)
        self.allocate_arrays(q)

        self.solution_extrapolation(q)
        self.start_communication()
        self.pointwise_fluxes(q)

        op_dx = self.ops.derivative_x
        op_dy = self.ops.derivative_y

        # Horizontal volume derivatives.
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

        op_corr_WE = self.ops.correction_WE
        op_corr_SN = self.ops.correction_SN

        # Horizontal interface corrections.
        apply_op(self.f_itf_x1, op_corr_WE, out=self.rhs, beta=1.0)
        apply_op(self.f_itf_x2, op_corr_SN, out=self.rhs, beta=1.0)
        apply_op(self.wflux_adv_itf_x1, op_corr_WE, out=self.w_df1_dx1, beta=1.0)
        apply_op(self.wflux_adv_itf_x2, op_corr_SN, out=self.w_df1_dx1, beta=1.0)
        apply_op(self.wflux_pres_itf_x1, op_corr_WE, out=self.w_presa, beta=1.0)
        apply_op(self.wflux_pres_itf_x2, op_corr_SN, out=self.w_presa, beta=1.0)

        logp_bdy_i = torch.log(self.pressure_itf_x1)
        logp_bdy_j = torch.log(self.pressure_itf_x2)
        apply_op(logp_bdy_i, op_corr_WE, out=self.w_df1_dx1_presb, beta=1.0)
        self.w_df1_dx1_presb *= self.wflux_pres_x1
        apply_op(logp_bdy_j, op_corr_SN, out=self.w_df2_dx2_presb, beta=1.0)
        self.w_df2_dx2_presb *= self.wflux_pres_x2

        # Keep the vertical horizontal-momentum fluxes with their terrain pressure balance in f2.
        op_dz = self.ops.derivative_z
        op_corr_DU = self.ops.correction_DU
        for row in (idx_rho_u1, idx_rho_u2):
            apply_op(self.f_x3[row], op_dz, out=self.rhs[row], beta=1.0)
            apply_op(self.f_itf_x3[row], op_corr_DU, out=self.rhs[row], beta=1.0)

        # Well-balanced horizontal contribution to vertical momentum.
        self.rhs[idx_rho_u3] = self.w_df1_dx1 + self.pressure * (
            self.w_presa + self.w_df1_dx1_presb + self.w_df2_dx2_presb
        )
        self.rhs *= -self.metric.inv_sqrtG_new

        # Remove gravity, which belongs to f1, from the full forcing.
        self.forcing_terms(q)
        self.rhs[idx_rho_u3] += (
            self.metric.inv_dzdeta_new
            * self.metric.gravity_new
            * self.metric.inv_sqrtG_new
            * ((self.metric.sqrtG_new * q[idx_rho]) @ self.ops.highfilter_k)
        )

        self.pin_y_momentum()

        return self.rhs.reshape(given_shape).clone()

    def horizontal_flux_div(self, q: Tensor) -> Tensor:
        """Return the plain horizontal flux divergence and prepare J2 trace data."""
        given_shape = q.shape
        self.ops = self.operators_for(q)
        self.allocate_arrays(q)

        self.solution_extrapolation(q)
        self.start_communication()
        self.pointwise_fluxes(q)

        op_dx = self.ops.derivative_x
        op_dy = self.ops.derivative_y
        apply_op(self.f_x1, op_dx, out=self.rhs, beta=0.0)
        apply_op(self.f_x2, op_dy, out=self.rhs, beta=1.0)

        self.end_communication()
        self.riemann_fluxes()

        op_corr_WE = self.ops.correction_WE
        op_corr_SN = self.ops.correction_SN
        apply_op(self.f_itf_x1, op_corr_WE, out=self.rhs, beta=1.0)
        apply_op(self.f_itf_x2, op_corr_SN, out=self.rhs, beta=1.0)
        self.rhs *= -self.metric.inv_sqrtG_new
        return self.rhs.reshape(given_shape).clone()

    def forcing_only(self, q: Tensor) -> Tensor:
        """Return the non-gravitational forcing in f2."""
        given_shape = q.shape
        self.ops = self.operators_for(q)
        self.allocate_arrays(q)
        self.pointwise_fluxes(q)  # sets self.pressure
        self.rhs[...] = 0.0
        self.forcing_terms(q)  # rhs -= (Christoffel/Coriolis/gravity/Rayleigh)
        self.rhs[idx_rho_u3] += (
            self.metric.inv_dzdeta_new
            * self.metric.gravity_new
            * self.metric.inv_sqrtG_new
            * ((self.metric.sqrtG_new * q[idx_rho]) @ self.ops.highfilter_k)
        )
        return self.rhs.reshape(given_shape).clone()
