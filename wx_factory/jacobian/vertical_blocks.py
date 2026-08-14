"""Assemble the block-tridiagonal Jacobian of the vertical partition.

Blocks have shape ``(num_columns, num_elem_z, block_size, block_size)``.
"""

from typing import NamedTuple

import numpy
import torch

from ..common.definitions import (
    heat_capacity_ratio,
    idx_rho,
    idx_rho_theta,
    idx_rho_u2,
    idx_rho_u3,
)
from ..common.matmul import apply_op
from .column_layout import (
    column_dims,
    horizontal_momentum_rows,
    scalar_interface_to_columns,
    scalar_to_columns,
    state_interface_to_columns,
    state_to_columns,
    stiff_variable_rows,
)
from .flux_jacobian import equation_of_state, flux_jacobian_matrix


class ImplicitBaseState(NamedTuple):
    """Base-state fields retained for the J1 linearization."""

    ops: object  # the DFR operator set matching the working precision of q
    pressure: object
    q_itf_x3: object  # vertical traces, before ghost padding
    q_itf_full_x3: object  # vertical traces, padded with the reflected wall ghosts
    wflux_pres_x3: object  # sqrt(G) h^{33} p, the vertical pressure flux of the rho u^3 row
    wflux_pres_itf_x3: object
    log_p: object
    pressure_itf_x3: object


def j1_prepare(rhsobj, q) -> ImplicitBaseState:
    """Evaluate f1 once and snapshot the base-state fields its Jacobian needs."""
    rhsobj.implicit(q)
    return ImplicitBaseState(
        ops=rhsobj.ops,
        pressure=rhsobj.pressure.copy(),
        q_itf_x3=rhsobj.q_itf_x3.copy(),
        q_itf_full_x3=rhsobj.q_itf_full_x3.copy(),
        wflux_pres_x3=rhsobj.wflux_pres_x3.copy(),
        wflux_pres_itf_x3=rhsobj.wflux_pres_itf_x3.copy(),
        log_p=rhsobj.log_p.copy(),
        pressure_itf_x3=rhsobj.pressure_itf_x3.copy(),
    )


def _h_contra_itf_k(metric, i: int):
    """Return the contravariant metric component h^{i3} on the vertical interfaces."""
    return metric.h_contra_itf_k_new[i, 2]


class _VerticalBlockAssembly:
    """Build the three block diagonals of J1 for one base state."""

    def __init__(self, rhsobj, q):
        self.rhsobj = rhsobj
        self.metric = rhsobj.metric
        self.dims = column_dims(rhsobj, q)
        self.dtype = q.dtype
        self.base = j1_prepare(rhsobj, q)

        self._reference_operators()
        self._volume_data(q)
        self._interface_data()
        self._riemann_speed()
        self._riemann_speed_derivative()
        self._linearised_extrapolation(q)

    # ------------------------------------------------------------------ base state

    def _reference_operators(self):
        """Extract the one-dimensional vertical operators."""
        ops = self.base.ops
        self.deriv_1d = ops.diff_solpt  # D, derivative at the solution points
        self.extrap_down = ops.extrap_down  # extrapolation to the bottom face
        self.extrap_up = ops.extrap_up  # extrapolation to the top face
        self.corr_down = ops.correction[:, 0]  # boundary correction from the bottom face
        self.corr_up = ops.correction[:, 1]  # boundary correction from the top face
        self.highfilter = ops.highfilter
        self.eye_solpts = torch.eye(self.dims.num_solpts, dtype=self.dtype)

    def _volume_data(self, q):
        """Gather the volume fields in column layout and the pointwise vertical flux Jacobian."""
        dims = self.dims
        metric = self.metric

        self.q_col = state_to_columns(self.rhsobj, q).reshape(
            dims.num_columns, dims.num_elem_z, dims.num_var, dims.num_solpts
        )
        self.pressure_col = scalar_to_columns(self.base.pressure, dims)
        self.sqrtG_col = scalar_to_columns(metric.sqrtG_new, dims)
        self.inv_sqrtG_col = scalar_to_columns(metric.inv_sqrtG_new, dims)
        self.inv_dzdeta_col = scalar_to_columns(metric.inv_dzdeta_new, dims)
        self.gravity_col = scalar_to_columns(metric.gravity_new, dims)

        # dF_3/dq at every volume solution point, shape (num_columns, num_elem_z, num_solpts, 5, 5).
        self.flux_jac_vol = flux_jacobian_matrix(
            torch.permute(self.q_col, (2, 0, 1, 3)),
            self.pressure_col,
            2,
            scalar_to_columns(metric.h_contra_new[0, 2], dims),
            scalar_to_columns(metric.h_contra_new[1, 2], dims),
            scalar_to_columns(metric.h_contra_new[2, 2], dims),
        )

    def _interface_data(self):
        """Gather the ghost-padded vertical traces in column layout and their flux Jacobian."""
        dims = self.dims
        metric = self.metric

        self.q_itf_full_col = state_interface_to_columns(self.base.q_itf_full_x3, dims)
        self.sqrtG_itf_col = scalar_interface_to_columns(metric.sqrtG_itf_k_new, dims)
        self.h33_itf_col = scalar_interface_to_columns(_h_contra_itf_k(metric, 2), dims)
        self.pressure_itf_col = equation_of_state(self.q_itf_full_col[..., idx_rho_theta])

        # Evaluate wall flux Jacobians on the reflected ghost states.
        q_itf_reflected = self.q_itf_full_col.copy()
        q_itf_reflected[:, 0, :, idx_rho_u3] *= -1.0
        q_itf_reflected[:, -1, :, idx_rho_u3] *= -1.0

        self.flux_jac_itf = flux_jacobian_matrix(
            torch.permute(q_itf_reflected, (3, 0, 1, 2)),
            self.pressure_itf_col,
            2,
            scalar_interface_to_columns(_h_contra_itf_k(metric, 0), dims),
            scalar_interface_to_columns(_h_contra_itf_k(metric, 1), dims),
            self.h33_itf_col,
        )

    def _riemann_speed(self):
        """Form the Rusanov Jacobians from both traces at each interface."""
        num_elem_z = self.dims.num_elem_z
        self.below = numpy.s_[:, : num_elem_z + 1, 1]
        self.above = numpy.s_[:, 1:, 0]
        below, above = self.below, self.above

        q_itf = self.q_itf_full_col
        self.sqrtG_below, self.sqrtG_above = self.sqrtG_itf_col[below], self.sqrtG_itf_col[above]
        self.rho_below, self.rho_above = q_itf[below][..., idx_rho], q_itf[above][..., idx_rho]
        self.w_below = q_itf[below][..., idx_rho_u3] / self.rho_below
        self.w_above = q_itf[above][..., idx_rho_u3] / self.rho_above
        self.p_below, self.p_above = self.pressure_itf_col[below], self.pressure_itf_col[above]
        self.rho_theta_below = q_itf[below][..., idx_rho_theta]
        self.rho_theta_above = q_itf[above][..., idx_rho_theta]

        # Acoustic part of the wave speed, c_s sqrt(h^33), on each trace.
        self.sound_below = torch.sqrt(self.h33_itf_col[below] * heat_capacity_ratio * self.p_below / self.rho_below)
        self.sound_above = torch.sqrt(self.h33_itf_col[above] * heat_capacity_ratio * self.p_above / self.rho_above)
        self.speed_below = torch.abs(self.w_below) + self.sound_below
        self.speed_above = torch.abs(self.w_above) + self.sound_above
        self.rusanov_speed = torch.maximum(self.speed_below, self.speed_above)

        # Derivative of fhat with respect to each of the two traces, at frozen lambda:
        #   d fhat / d q_below = 1/2 ( sqrt(G) dF3/dq|below + lambda I )
        #   d fhat / d q_above = 1/2 ( sqrt(G) dF3/dq|above - lambda I )
        eye_var = torch.eye(self.dims.num_var, dtype=self.dtype).reshape(1, 1, self.dims.num_var, self.dims.num_var)
        damping = (self.rusanov_speed * self.sqrtG_below)[..., None, None]
        self.jac_wrt_below = 0.5 * (self.sqrtG_below[..., None, None] * self.flux_jac_itf[below] + damping * eye_var)
        self.jac_wrt_above = 0.5 * (self.sqrtG_above[..., None, None] * self.flux_jac_itf[above] - damping * eye_var)

    def _riemann_speed_derivative(self):
        """Add the derivative of the selected Rusanov wave speed."""
        below_wins = self.speed_below >= self.speed_above
        w_win = torch.where(below_wins, self.w_below, self.w_above)
        rho_win = torch.where(below_wins, self.rho_below, self.rho_above)
        rho_theta_win = torch.where(below_wins, self.rho_theta_below, self.rho_theta_above)
        sound_win = torch.where(below_wins, self.sound_below, self.sound_above)
        sign_w = torch.sgn(w_win)

        # Differentiate lambda at the trace selected by torch.maximum.
        d_speed = torch.zeros_like(self.q_itf_full_col[self.below])
        d_speed[..., idx_rho] = -sign_w * w_win / rho_win - sound_win / (2.0 * rho_win)
        d_speed[..., idx_rho_u3] = sign_w / rho_win
        d_speed[..., idx_rho_theta] = heat_capacity_ratio * sound_win / (2.0 * rho_theta_win)

        q_jump = self.q_itf_full_col[self.above] - self.q_itf_full_col[self.below]
        speed_term = (-0.5 * self.sqrtG_below)[..., None, None] * q_jump[..., :, None] * d_speed[..., None, :]
        no_term = torch.zeros_like(speed_term)
        keep = below_wins[..., None, None]
        self.jac_wrt_below = self.jac_wrt_below + torch.where(keep, speed_term, no_term)
        self.jac_wrt_above = self.jac_wrt_above + torch.where(keep, no_term, speed_term)

        self.below_wins = below_wins
        self.d_speed = d_speed
        self.q_jump = q_jump

    def _linearised_extrapolation(self, q):
        """Build extrapolation derivatives for each variable."""
        dims = self.dims
        q_itf = state_interface_to_columns(self.base.q_itf_x3, dims)
        shape = (dims.num_columns, dims.num_elem_z, dims.num_var, dims.num_solpts)
        self.extrap_down_lin = torch.zeros(shape, dtype=self.dtype) + self.extrap_down
        self.extrap_up_lin = torch.zeros(shape, dtype=self.dtype) + self.extrap_up
        for var in (idx_rho, idx_rho_theta):
            self.extrap_down_lin[:, :, var, :] = self.extrap_down * (
                q_itf[:, :, 0, var][..., None] / self.q_col[:, :, var, :]
            )
            self.extrap_up_lin[:, :, var, :] = self.extrap_up * (
                q_itf[:, :, 1, var][..., None] / self.q_col[:, :, var, :]
            )
        self.q_itf_col = q_itf

    # ------------------------------------------------------------------ assembly

    def assemble(self):
        """Return the three block diagonals of J1."""
        self._conservative_rows()
        self._vertical_momentum_row()
        self._apply_metric_factor()
        return self._finish()

    def _element_slices(self):
        """Return aligned slices for interfaces and adjacent elements."""
        num_elem_z = self.dims.num_elem_z
        return slice(1, num_elem_z), slice(1, num_elem_z), slice(0, num_elem_z - 1)

    def _conservative_rows(self):
        """Assemble the conservative flux-divergence rows."""
        dims = self.dims
        shape = (dims.num_columns, dims.num_elem_z, dims.num_var, dims.num_solpts, dims.num_var, dims.num_solpts)
        self.lower = torch.zeros(shape, dtype=self.dtype)
        self.diag = torch.zeros(shape, dtype=self.dtype)
        self.upper = torch.zeros(shape, dtype=self.dtype)

        # Volume term: D applied to sqrt(G) F_3(q), differentiated pointwise.
        self.diag += torch.einsum("os,ces,cesij->ceiojs", self.deriv_1d, self.sqrtG_col, self.flux_jac_vol)

        # Each interior interface contributes to both adjacent block rows.
        interior, elem_above, elem_below = self._element_slices()
        self.lower[:, elem_above] += torch.einsum(
            "o,ceij,cejs->ceiojs", self.corr_down, self.jac_wrt_below[:, interior], self.extrap_up_lin[:, elem_below]
        )
        self.diag[:, elem_above] += torch.einsum(
            "o,ceij,cejs->ceiojs", self.corr_down, self.jac_wrt_above[:, interior], self.extrap_down_lin[:, elem_above]
        )
        self.diag[:, elem_below] += torch.einsum(
            "o,ceij,cejs->ceiojs", self.corr_up, self.jac_wrt_below[:, interior], self.extrap_up_lin[:, elem_below]
        )
        self.upper[:, elem_below] += torch.einsum(
            "o,ceij,cejs->ceiojs", self.corr_up, self.jac_wrt_above[:, interior], self.extrap_down_lin[:, elem_above]
        )

        # Differentiate each reflected wall trace against its interior trace.
        num_elem_z = dims.num_elem_z
        reflection = torch.ones(dims.num_var, dtype=self.dtype)
        reflection[idx_rho_u3] = -1.0
        self.diag[:, 0] += torch.einsum(
            "o,cij,cjs->ciojs",
            self.corr_down,
            self.jac_wrt_below[:, 0] * reflection + self.jac_wrt_above[:, 0],
            self.extrap_down_lin[:, 0],
        )
        self.diag[:, num_elem_z - 1] += torch.einsum(
            "o,cij,cjs->ciojs",
            self.corr_up,
            self.jac_wrt_below[:, num_elem_z] + self.jac_wrt_above[:, num_elem_z] * reflection,
            self.extrap_up_lin[:, num_elem_z - 1],
        )

    def _vertical_momentum_row(self):
        """Assemble the derivative of the well-balanced vertical-momentum row."""
        dims = self.dims
        num_elem_z = dims.num_elem_z
        interior, elem_above, elem_below = self._element_slices()

        self.lower[:, :, idx_rho_u3] = 0.0
        self.diag[:, :, idx_rho_u3] = 0.0
        self.upper[:, :, idx_rho_u3] = 0.0

        rho_col = self.q_col[:, :, idx_rho, :]
        rho_theta_col = self.q_col[:, :, idx_rho_theta, :]
        w_col = self.q_col[:, :, idx_rho_u3, :] / rho_col

        # --- Advective part, sqrt(G) rho (u^3)^2, in the element volume.
        self.diag[:, :, idx_rho_u3, :, idx_rho_u3, :] += torch.einsum(
            "os,ces->ceos", self.deriv_1d, self.sqrtG_col * 2.0 * w_col
        )
        self.diag[:, :, idx_rho_u3, :, idx_rho, :] += torch.einsum(
            "os,ces->ceos", self.deriv_1d, -self.sqrtG_col * w_col**2
        )

        # --- Advective part at the interfaces: the same Rusanov flux, restricted to this row.
        wadv_wrt_below = torch.zeros((dims.num_columns, num_elem_z + 1, dims.num_var), dtype=self.dtype)
        wadv_wrt_above = torch.zeros((dims.num_columns, num_elem_z + 1, dims.num_var), dtype=self.dtype)
        wadv_wrt_below[..., idx_rho_u3] = 0.5 * (
            2.0 * self.sqrtG_below * self.w_below + self.rusanov_speed * self.sqrtG_below
        )
        wadv_wrt_below[..., idx_rho] = -0.5 * self.sqrtG_below * self.w_below**2
        wadv_wrt_above[..., idx_rho_u3] = 0.5 * (
            2.0 * self.sqrtG_above * self.w_above - self.rusanov_speed * self.sqrtG_below
        )
        wadv_wrt_above[..., idx_rho] = -0.5 * self.sqrtG_above * self.w_above**2

        # The variation of lambda applies to this row too.
        speed_term = (-0.5 * self.sqrtG_below * self.q_jump[..., idx_rho_u3])[..., None] * self.d_speed
        no_term = torch.zeros_like(speed_term)
        keep = self.below_wins[..., None]
        wadv_wrt_below = wadv_wrt_below + torch.where(keep, speed_term, no_term)
        wadv_wrt_above = wadv_wrt_above + torch.where(keep, no_term, speed_term)

        self.lower[:, elem_above, idx_rho_u3] += torch.einsum(
            "o,cej,cejs->ceojs", self.corr_down, wadv_wrt_below[:, interior], self.extrap_up_lin[:, elem_below]
        )
        self.diag[:, elem_above, idx_rho_u3] += torch.einsum(
            "o,cej,cejs->ceojs", self.corr_down, wadv_wrt_above[:, interior], self.extrap_down_lin[:, elem_above]
        )
        self.diag[:, elem_below, idx_rho_u3] += torch.einsum(
            "o,cej,cejs->ceojs", self.corr_up, wadv_wrt_below[:, interior], self.extrap_up_lin[:, elem_below]
        )
        self.upper[:, elem_below, idx_rho_u3] += torch.einsum(
            "o,cej,cejs->ceojs", self.corr_up, wadv_wrt_above[:, interior], self.extrap_down_lin[:, elem_above]
        )
        # At the walls the reflected trace makes the advective flux, and its derivative, exactly zero.

        # Pressure variation multiplying the base-state pressure operators.
        wflux_pres_x3 = self.base.wflux_pres_x3
        op_dz = self.base.ops.derivative_z
        op_corr = self.base.ops.correction_DU
        w_presa_base = apply_op(wflux_pres_x3, op_dz)
        apply_op(self.base.wflux_pres_itf_x3, op_corr, out=w_presa_base, beta=1.0)
        w_presb_base = apply_op(self.base.log_p, op_dz)
        apply_op(torch.log(self.base.pressure_itf_x3), op_corr, out=w_presb_base, beta=1.0)
        w_presb_base = w_presb_base * wflux_pres_x3

        dp_drho_theta = heat_capacity_ratio * self.pressure_col / rho_theta_col
        self.diag[:, :, idx_rho_u3, :, idx_rho_theta, :] += torch.einsum(
            "os,ces->ceos",
            self.eye_solpts,
            scalar_to_columns(w_presa_base + w_presb_base, dims) * dp_drho_theta,
        )

        # Linearize the pressure-flux ratio for each face and trace.
        sqrtG_h33_below = self.sqrtG_below * self.h33_itf_col[self.below]
        sqrtG_h33_above = self.sqrtG_above * self.h33_itf_col[self.above]
        dp_below = heat_capacity_ratio * self.p_below / self.rho_theta_below
        dp_above = heat_capacity_ratio * self.p_above / self.rho_theta_above
        down_wrt_below = 0.5 * sqrtG_h33_below / self.p_above * dp_below
        down_wrt_above = -0.5 * sqrtG_h33_below / self.p_above * (self.p_below / self.p_above) * dp_above
        up_wrt_above = 0.5 * sqrtG_h33_above / self.p_below * dp_above
        up_wrt_below = -0.5 * sqrtG_h33_above / self.p_below * (self.p_above / self.p_below) * dp_below

        pres_corr_down = self.pressure_col * self.corr_down
        pres_corr_up = self.pressure_col * self.corr_up
        self.lower[:, elem_above, idx_rho_u3, :, idx_rho_theta, :] += torch.einsum(
            "ceo,ce,ces->ceos",
            pres_corr_down[:, elem_above],
            down_wrt_below[:, interior],
            self.extrap_up_lin[:, elem_below, idx_rho_theta],
        )
        self.diag[:, elem_above, idx_rho_u3, :, idx_rho_theta, :] += torch.einsum(
            "ceo,ce,ces->ceos",
            pres_corr_down[:, elem_above],
            down_wrt_above[:, interior],
            self.extrap_down_lin[:, elem_above, idx_rho_theta],
        )
        self.diag[:, elem_below, idx_rho_u3, :, idx_rho_theta, :] += torch.einsum(
            "ceo,ce,ces->ceos",
            pres_corr_up[:, elem_below],
            up_wrt_below[:, interior],
            self.extrap_up_lin[:, elem_below, idx_rho_theta],
        )
        self.upper[:, elem_below, idx_rho_u3, :, idx_rho_theta, :] += torch.einsum(
            "ceo,ce,ces->ceos",
            pres_corr_up[:, elem_below],
            up_wrt_above[:, interior],
            self.extrap_down_lin[:, elem_above, idx_rho_theta],
        )
        # At a wall both traces are the same interior state, so the two coefficients simply add.
        self.diag[:, 0, idx_rho_u3, :, idx_rho_theta, :] += torch.einsum(
            "co,c,cs->cos",
            pres_corr_down[:, 0],
            down_wrt_below[:, 0] + down_wrt_above[:, 0],
            self.extrap_down_lin[:, 0, idx_rho_theta],
        )
        self.diag[:, num_elem_z - 1, idx_rho_u3, :, idx_rho_theta, :] += torch.einsum(
            "co,c,cs->cos",
            pres_corr_up[:, num_elem_z - 1],
            up_wrt_below[:, num_elem_z] + up_wrt_above[:, num_elem_z],
            self.extrap_up_lin[:, num_elem_z - 1, idx_rho_theta],
        )

        # Pressure contribution through the perturbation of log(p).
        pres_wflux = self.pressure_col * scalar_to_columns(wflux_pres_x3, dims)
        self.diag[:, :, idx_rho_u3, :, idx_rho_theta, :] += torch.einsum(
            "ceo,os,ces->ceos", pres_wflux, self.deriv_1d, heat_capacity_ratio / rho_theta_col
        )
        self.diag[:, :, idx_rho_u3, :, idx_rho_theta, :] += torch.einsum(
            "ceo,o,ces->ceos",
            pres_wflux,
            self.corr_down,
            heat_capacity_ratio
            * self.extrap_down_lin[:, :, idx_rho_theta, :]
            / self.q_itf_col[:, :, 0, idx_rho_theta][..., None],
        )
        self.diag[:, :, idx_rho_u3, :, idx_rho_theta, :] += torch.einsum(
            "ceo,o,ces->ceos",
            pres_wflux,
            self.corr_up,
            heat_capacity_ratio
            * self.extrap_up_lin[:, :, idx_rho_theta, :]
            / self.q_itf_col[:, :, 1, idx_rho_theta][..., None],
        )

    def _apply_metric_factor(self):
        """Apply the outer ``-1/sqrt(G)`` factor and add the filtered gravity term."""
        scale = (-self.inv_sqrtG_col)[:, :, None, :, None, None]
        self.lower *= scale
        self.diag *= scale
        self.upper *= scale

        # Add the filtered gravity derivative to the vertical-momentum row.
        self.diag[:, :, idx_rho_u3, :, idx_rho, :] -= torch.einsum(
            "ceo,os,ces->ceos",
            self.inv_dzdeta_col * self.gravity_col * self.inv_sqrtG_col,
            self.highfilter,
            self.sqrtG_col,
        )

    def _finish(self):
        """Collapse the (variable, point) index pairs into single block indices."""
        dims = self.dims
        shape = (dims.num_columns, dims.num_elem_z, dims.block_size, dims.block_size)
        lower = self.lower.reshape(shape)
        diag = self.diag.reshape(shape)
        upper = self.upper.reshape(shape)

        if self.rhsobj.y_invariant_slab:
            # The y-momentum tendency is pinned, so its rows are identically zero.
            y_rows = numpy.s_[..., idx_rho_u2 * dims.num_solpts : (idx_rho_u2 + 1) * dims.num_solpts, :]
            lower[y_rows] = 0.0
            diag[y_rows] = 0.0
            upper[y_rows] = 0.0

        return lower, diag, upper


def assemble_vertical_blocks(rhsobj, q):
    """Assemble all rows of the vertical block-tridiagonal Jacobian."""
    return _VerticalBlockAssembly(rhsobj, q).assemble()


def split_vertical_blocks(rhsobj, lower, diag, upper):
    """Return the horizontal-momentum rows and stiff-variable sub-blocks."""
    num_solpts = rhsobj.geom.num_solpts
    rows = horizontal_momentum_rows(num_solpts)
    stiff = stiff_variable_rows(num_solpts, lower.device)
    horizontal_momentum = tuple(b[..., rows, :].clone() for b in (lower, diag, upper))
    stiff_blocks = tuple(b[..., stiff, :][..., :, stiff].clone() for b in (lower, diag, upper))
    return horizontal_momentum, stiff_blocks


def assemble_j1_blocks(rhsobj, q):
    """Assemble J1 in the five-variable layout used by validation tests."""
    lower, diag, upper = assemble_vertical_blocks(rhsobj, q)
    rows = horizontal_momentum_rows(rhsobj.geom.num_solpts)
    for blocks in (lower, diag, upper):
        blocks[..., rows, :] = 0.0
    return lower, diag, upper
