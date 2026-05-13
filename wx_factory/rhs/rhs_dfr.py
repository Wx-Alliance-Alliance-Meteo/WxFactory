import numpy
from numpy.typing import NDArray

from common.definitions import idx_rho, idx_rho_u1, idx_rho_u2, idx_rho_w, idx_rho_theta
from rhs.rhs import RHS
from wx_mpi import SingleProcess, Conditional
from init.entropy_vars import (
    conservative_to_entropy,
    du_dv,
    entropy_potential,
    entropy_to_conservative,
    jacobian_complex_field,
    jacobian_fd_field,
)
from common.graphx import image_field


def apply_op(vec: NDArray, op: NDArray):
    sh = vec.shape
    return (vec.reshape(-1, sh[-1]) @ op).reshape(*sh[:-1], -1)


class RHSDirectFluxReconstruction_ESAV(RHS):

    def solution_extrapolation(self, q: NDArray) -> None:
        # Extrapolate the solution to element boundaries
        xp = self.device.xp
        self.q_itf_x1 = apply_op(q, self.ops.extrap_x)
        self.q_itf_x3 = apply_op(q, self.ops.extrap_z)

    def pointwise_fluxes(self, q: NDArray) -> None:
        xp = self.device.xp
        self.pde.pointwise_fluxes(q, self.f_x1, self.f_x2, self.f_x3)

    def flux_divergence_partial(self) -> NDArray:
        xp = self.device.xp

        self.df1_dx1 = apply_op(self.f_x1, self.ops.derivative_x)
        self.df3_dx3 = apply_op(self.f_x3, self.ops.derivative_z)

    def flux_divergence(self):
        xp = self.device.xp

        self.df1_dx1 += self.f_itf_x1 @ self.ops.correction_WE
        self.df1_dx1 *= -2.0 / self.geom.Δx1

        self.df3_dx3 += self.f_itf_x3 @ self.ops.correction_DU
        self.df3_dx3 *= -2.0 / self.geom.Δx3

        xp.add(self.df1_dx1, self.df3_dx3, out=self.rhs)

    #
    # ESAV Methods
    #

    def solution_extrapolation_entropy(self, v: NDArray) -> None:
        # Extrapolate the entropy variables to element boundaries

        self.v_itf_x1 = apply_op(v, self.ops.extrap_x)
        self.v_itf_x3 = apply_op(v, self.ops.extrap_z)

    def entropy_gradient_partial(self, v: NDArray) -> None:
        """Gradient for v - discontinuous part, no boundary terms"""
        xp = self.device.xp

        self.dv_dx1_volume = apply_op(v, self.ops.derivative_x_esav)
        self.dv_dx3_volume = apply_op(v, self.ops.derivative_z_esav)

        self.dv_dx1_volume *= 2.0 / self.geom.Δx1
        self.dv_dx3_volume *= 2.0 / self.geom.Δx3

        self.dv_dx1 = apply_op(v, self.ops.derivative_x)
        self.dv_dx3 = apply_op(v, self.ops.derivative_z)

    def entropy_average(self) -> None:
        """Entropy average"""

        self.v_avg_x1, self.v_avg_x3 = self.pde.entropy_average(self.q_itf_x1, self.q_itf_x3)

    def entropy_gradient(self) -> None:
        """Compute derivatives of v, with correction from boundaries"""
        xp = self.device.xp

        self.dv_dx1 += self.v_avg_x1 @ self.ops.correction_WE
        self.dv_dx1 *= 2.0 / self.geom.Δx1

        self.dv_dx3 += self.v_avg_x3 @ self.ops.correction_DU
        self.dv_dx3 *= 2.0 / self.geom.Δx3

    def volume_integral(self, h):
        xp = self.device.xp
        return xp.einsum("vhp,p->vh", h, self.ops.weights_volume_integral)

    def boundary_integral(self, h):
        xp = self.device.xp
        num_solpts = self.geom.num_solpts

        negative = xp.einsum("vhp,p->vh", h[..., :num_solpts], self.ops.weights_boundary_integral)  # West/Down
        positive = xp.einsum("vhp,p->vh", h[..., num_solpts:], self.ops.weights_boundary_integral)  # East/Up

        return xp.stack([negative, positive], axis=-1)

    def dot_product(self, a, b):
        # Takes 4D vectors
        xp = self.device.xp

        return xp.einsum("pijk,pijk->ijk", a, b)

    def entropy_residual(self, print_results=False):
        xp = self.device.xp

        vol_int1 = (
            self.geom.Δx1
            / 2.0
            * self.geom.Δx3
            / 2.0
            * self.volume_integral(self.dot_product(self.f_x1, self.dv_dx1_volume))
        )

        vol_int2 = (
            self.geom.Δx1
            / 2.0
            * self.geom.Δx3
            / 2.0
            * self.volume_integral(self.dot_product(self.f_x3, self.dv_dx3_volume))
        )

        # Precompute entropy potentials
        psi1_itf_x1, _ = entropy_potential(self.q_itf_x1)
        _, psi3_itf_x3 = entropy_potential(self.q_itf_x3)

        # Boundary terms
        # psi_1
        boundary_int1_WE = self.geom.Δx3 / 2.0 * self.boundary_integral(psi1_itf_x1)
        # boundary_int1_DU = self.geom.Δx1 / 2.0 * self.boundary_integral(psi1_itf_x3)

        # psi_3
        # boundary_int2_WE = self.geom.Δx3 / 2.0 * self.boundary_integral(psi3_itf_x1)
        boundary_int2_DU = self.geom.Δx1 / 2.0 * self.boundary_integral(psi3_itf_x3)

        # Compute entropy residual
        sigma = (
            -vol_int1
            - vol_int2
            - boundary_int1_WE[:, :, 0]
            + boundary_int1_WE[:, :, 1]
            - boundary_int2_DU[:, :, 0]
            + boundary_int2_DU[:, :, 1]
        )
        return sigma

    def denominator_viscosity_coeff(self):
        xp = self.device.xp

        # Kdv1_dx1 = xp.einsum("abijk,bijk->aijk", self.K, self.dv_dx1)
        # Kdv3_dx3 = xp.einsum("abijk,bijk->aijk", self.K, self.dv_dx3)


        Kdv1_dx1 = xp.einsum("abij,bijp->aijp", self.K, self.dv_dx1)
        Kdv3_dx3 = xp.einsum("abij,bijp->aijp", self.K, self.dv_dx3)

        # Volume terms
        vol_int1 = (
            self.geom.Δx1 / 2.0 * self.geom.Δx3 / 2.0 * self.volume_integral(self.dot_product(Kdv1_dx1, self.dv_dx1))
        )
        vol_int2 = (
            self.geom.Δx1 / 2.0 * self.geom.Δx3 / 2.0 * self.volume_integral(self.dot_product(Kdv3_dx3, self.dv_dx3))
        )

        denominator = vol_int1 + vol_int2
        return denominator

    def approx_division(self, a, b, tol=1e-14):
        return (a * b) / (tol + b**2)

    def viscosity_coeff(self, q: NDArray) -> None:
        """Computes the elementwise constant viscosity coefficient"""
        # TODO: implement the entropy preserving viscosity coeffs
        xp = self.device.xp
        entropy_stable_coeff = True

        if entropy_stable_coeff:
            sigma = self.entropy_residual()
            a = -xp.minimum(0, sigma)
            b = self.denominator_viscosity_coeff()

            self.epsilon = self.approx_division(a, b)
        else:
            epsilon_val = 0
            num_equations = 4
            # shape = (num_equations, self.config.num_elements_vertical, self.config.num_elements_horizontal)
            shape = (self.config.num_elements_vertical, self.config.num_elements_horizontal)
            self.epsilon = xp.full(shape, epsilon_val, dtype=q.dtype)

    def viscous_fluxes(self) -> None:
        """Computes the viscous flux g_m = \sum_n epsilon K_mn dv_dxn"""
        xp = self.device.xp

        Kdg1_dx1 = xp.einsum("abij,bijp->aijp", self.K, self.dv_dx1)
        Kdg3_dx3 = xp.einsum("abij,bijp->aijp", self.K, self.dv_dx3)

        self.g_x1 = self.epsilon[..., None] * Kdg1_dx1
        self.g_x3 = self.epsilon[..., None] * Kdg3_dx3

    def compute_K(self, q: NDArray) -> None:
        """Computes K= du/dv"""
        xp = self.device.xp
        # q_bar = xp.mean(q, axis=-1)

        # print("q_bar",q_bar.shape)

        # self.K = du_dv(q_bar,self.geom,self.config)

        # print("K shape",self.K.shape)
        # self.K = du_dv(q, self.geom, self.config)

        # q shape: (nvar, ne_z, ne_x, npts)
        w = self.ops.weights_volume_integral
        wsum = xp.sum(w)

        q_bar = xp.einsum("aijp,p->aij", q, w) / wsum

        self.K = du_dv(q_bar, self.geom, self.config)


        # # K shape: (nvar, nvar, ne_z, ne_x)
        # self.K = jacobian_complex_field(
        #     entropy_to_conservative,
        #     v_bar,
        #     self.geom,
        #     self.config,
        # )

        # self.K = jacobian_complex_field(entropy_to_conservative, self.v, self.geom, self.config)
        # self.K2 = jacobian_fd_field(entropy_to_conservative, self.v, self.geom, self.config)

        # print(self.K[:, 0, 0, 0])
        # print(self.K1[:, 0, 0, 0])
        # print(self.K2[:, 0, 0, 0])
        # exit()
        # print("K1[:,i1,j1,0]\n",self.K1[:,:,self.i1,self.j1,0])
        # print("K[:,i1,j1,0]\n",self.K[:,:,self.i1,self.j1,0])

    def viscous_flux_divergence_partial(self) -> None:
        """Part of the divergence for g - discontinuous part, no boundary terms"""

        self.dg1_dx1 = apply_op(self.g_x1, self.ops.derivative_x)
        self.dg3_dx3 = apply_op(self.g_x3, self.ops.derivative_z)

    def viscous_flux_average(self) -> None:
        """Entropy average"""

        g1_itf_x1 = apply_op(self.g_x1, self.ops.extrap_x)
        g3_itf_x3 = apply_op(self.g_x3, self.ops.extrap_z)

        self.g_avg_x1, self.g_avg_x3 = self.pde.viscous_flux_average(g1_itf_x1, g3_itf_x3)

    def viscous_flux_divergence(self) -> None:
        """Compute derivatives of g, with correction from boundaries"""
        xp = self.device.xp

        self.dg1_dx1 += self.g_avg_x1 @ self.ops.correction_WE
        self.dg1_dx1 *= 2.0 / self.geom.Δx1

        self.dg3_dx3 += self.g_avg_x3 @ self.ops.correction_DU
        self.dg3_dx3 *= 2.0 / self.geom.Δx3

        xp.add(self.rhs, self.dg1_dx1, out=self.rhs)
        xp.add(self.rhs, self.dg3_dx3, out=self.rhs)


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
    def __init__(self, pde, geometry, operators, metric, topography, process_topo, config, expected_shape, debug=False):
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
        itf_size = self.geom.itf_size

        mid_i = xp.s_[..., 1:-1, :]
        mid_j = xp.s_[..., 1:-1, :, :]
        mid_k = xp.s_[..., 1:-1, :, :, :]

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
        # Boundary conditions
        self.q_itf_full_x3[idx_rho_w, 0, :, :, :itf_size] = 0.0
        self.q_itf_full_x3[idx_rho_w, 0, :, :, itf_size:] = -self.q_itf_full_x3[idx_rho_w, 1, :, :, :itf_size]
        self.q_itf_full_x3[idx_rho_w, -1, :, :, itf_size:] = 0.0
        self.q_itf_full_x3[idx_rho_w, -1, :, :, :itf_size] = -self.q_itf_full_x3[idx_rho_w, -2, :, :, itf_size:]

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
