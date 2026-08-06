"""Matrix-free Jacobian of the PartRosExp2 non-stiff partition."""

import numpy
import torch

from ..common.definitions import (
    heat_capacity_ratio,
    idx_rho,
    idx_rho_theta,
    idx_rho_u1,
    idx_rho_u2,
    idx_rho_u3,
)
from ..common.matmul import apply_op
from ..rhs.rhs_dfr import mid_i, mid_j
from .block_solve import horizontal_momentum_matvec
from .flux_jacobian import MOMENTUM_ROWS, equation_of_state, flux_jacobian_matvec
from .vertical_blocks import assemble_vertical_blocks, split_vertical_blocks


def _riemann_flux_jvp(direction, q_itf_full, dq_itf_full, sqrtG_itf, h_contra_itf, left, right):
    """Differentiate a horizontal Rusanov flux, including its wave speed."""
    pressure_itf = equation_of_state(q_itf_full[idx_rho_theta])
    mom_row = MOMENTUM_ROWS[direction]
    u_right = q_itf_full[mom_row][right] / q_itf_full[idx_rho][right]
    u_left = q_itf_full[mom_row][left] / q_itf_full[idx_rho][left]
    h_dd = h_contra_itf[direction, direction]
    sound_left = torch.sqrt(h_dd[left] * heat_capacity_ratio * pressure_itf[left] / q_itf_full[idx_rho][left])
    sound_right = torch.sqrt(h_dd[right] * heat_capacity_ratio * pressure_itf[right] / q_itf_full[idx_rho][right])
    speed_left = torch.abs(u_left) + sound_left
    speed_right = torch.abs(u_right) + sound_right
    rusanov_speed = torch.maximum(speed_left, speed_right)

    dflux_left = sqrtG_itf[left] * flux_jacobian_matvec(
        dq_itf_full[left],
        q_itf_full[left],
        pressure_itf[left],
        direction,
        h_contra_itf[direction, 0][left],
        h_contra_itf[direction, 1][left],
        h_contra_itf[direction, 2][left],
    )
    dflux_right = sqrtG_itf[right] * flux_jacobian_matvec(
        dq_itf_full[right],
        q_itf_full[right],
        pressure_itf[right],
        direction,
        h_contra_itf[direction, 0][right],
        h_contra_itf[direction, 1][right],
        h_contra_itf[direction, 2][right],
    )

    # Differentiate the wave speed on the side selected by torch.maximum.
    left_wins = speed_left >= speed_right
    rho_win = torch.where(left_wins, q_itf_full[idx_rho][left], q_itf_full[idx_rho][right])
    rho_theta_win = torch.where(left_wins, q_itf_full[idx_rho_theta][left], q_itf_full[idx_rho_theta][right])
    u_win = torch.where(left_wins, u_left, u_right)
    sound_win = torch.where(left_wins, sound_left, sound_right)
    d_rho_win = torch.where(left_wins, dq_itf_full[idx_rho][left], dq_itf_full[idx_rho][right])
    d_mom_win = torch.where(left_wins, dq_itf_full[mom_row][left], dq_itf_full[mom_row][right])
    d_rho_theta_win = torch.where(left_wins, dq_itf_full[idx_rho_theta][left], dq_itf_full[idx_rho_theta][right])
    du_win = (d_mom_win - u_win * d_rho_win) / rho_win
    dsound_win = 0.5 * sound_win * (heat_capacity_ratio * d_rho_theta_win / rho_theta_win - d_rho_win / rho_win)
    d_speed = torch.sgn(u_win) * du_win + dsound_win

    out = torch.zeros_like(q_itf_full)
    out[left] = 0.5 * (
        dflux_left
        + dflux_right
        - sqrtG_itf[left]
        * (rusanov_speed * (dq_itf_full[right] - dq_itf_full[left]) + d_speed * (q_itf_full[right] - q_itf_full[left]))
    )
    out[right] = out[left]
    return out


def _well_balanced_riemann_jvp(direction, q_itf_full, dq_itf_full, sqrtG_itf, h_contra_itf, left, right):
    """Differentiate the advective and pressure interface terms for ``rho u^3``."""
    pressure = equation_of_state(q_itf_full[idx_rho_theta])
    dpressure = heat_capacity_ratio * pressure * dq_itf_full[idx_rho_theta] / q_itf_full[idx_rho_theta]
    mom_row = MOMENTUM_ROWS[direction]
    u_left = q_itf_full[mom_row][left] / q_itf_full[idx_rho][left]
    u_right = q_itf_full[mom_row][right] / q_itf_full[idx_rho][right]
    du_left = (dq_itf_full[mom_row][left] - u_left * dq_itf_full[idx_rho][left]) / q_itf_full[idx_rho][left]
    du_right = (dq_itf_full[mom_row][right] - u_right * dq_itf_full[idx_rho][right]) / q_itf_full[idx_rho][right]

    h_dd = h_contra_itf[direction, direction]
    sound_left = torch.sqrt(h_dd[left] * heat_capacity_ratio * pressure[left] / q_itf_full[idx_rho][left])
    sound_right = torch.sqrt(h_dd[right] * heat_capacity_ratio * pressure[right] / q_itf_full[idx_rho][right])
    speed_left = torch.abs(u_left) + sound_left
    speed_right = torch.abs(u_right) + sound_right
    left_wins = speed_left >= speed_right
    rusanov_speed = torch.maximum(speed_left, speed_right)
    d_speed_left = torch.sgn(u_left) * du_left + 0.5 * sound_left * (
        heat_capacity_ratio * dq_itf_full[idx_rho_theta][left] / q_itf_full[idx_rho_theta][left]
        - dq_itf_full[idx_rho][left] / q_itf_full[idx_rho][left]
    )
    d_speed_right = torch.sgn(u_right) * du_right + 0.5 * sound_right * (
        heat_capacity_ratio * dq_itf_full[idx_rho_theta][right] / q_itf_full[idx_rho_theta][right]
        - dq_itf_full[idx_rho][right] / q_itf_full[idx_rho][right]
    )
    d_speed = torch.where(left_wins, d_speed_left, d_speed_right)

    rho_w_left, rho_w_right = q_itf_full[idx_rho_u3][left], q_itf_full[idx_rho_u3][right]
    drho_w_left, drho_w_right = dq_itf_full[idx_rho_u3][left], dq_itf_full[idx_rho_u3][right]
    dadvective = 0.5 * (
        sqrtG_itf[left] * (du_left * rho_w_left + u_left * drho_w_left)
        + sqrtG_itf[right] * (du_right * rho_w_right + u_right * drho_w_right)
        - sqrtG_itf[left] * (rusanov_speed * (drho_w_right - drho_w_left) + d_speed * (rho_w_right - rho_w_left))
    )

    # Interface pressure flux, averaged over the two sides then divided by each trace pressure.
    sqrtG_h_left = sqrtG_itf[left] * h_contra_itf[direction, 2][left]
    sqrtG_h_right = sqrtG_itf[right] * h_contra_itf[direction, 2][right]
    numerator = 0.5 * (sqrtG_h_left * pressure[left] + sqrtG_h_right * pressure[right])
    dnumerator = 0.5 * (sqrtG_h_left * dpressure[left] + sqrtG_h_right * dpressure[right])
    dpres_left = (dnumerator * pressure[left] - numerator * dpressure[left]) / pressure[left] ** 2
    dpres_right = (dnumerator * pressure[right] - numerator * dpressure[right]) / pressure[right] ** 2

    advective = torch.zeros_like(q_itf_full[idx_rho])
    pressure_part = torch.zeros_like(advective)
    advective[left], advective[right] = dadvective, dadvective
    pressure_part[left], pressure_part[right] = dpres_left, dpres_right
    return advective, pressure_part


def j2_prepare(rhsobj, q, momentum_blocks=None):
    """Prepare the base-state fields and vertical momentum blocks used by J2."""
    if momentum_blocks is None:
        lower, diag, upper = assemble_vertical_blocks(rhsobj, q)
        # Retain the horizontal-momentum rows for J2.
        momentum_blocks, _ = split_vertical_blocks(rhsobj, lower, diag, upper)
    rhsobj.horizontal_flux_div(q)
    return (
        rhsobj.ops,
        rhsobj.pressure.copy(),
        rhsobj.q_itf_x1.copy(),
        rhsobj.q_itf_x2.copy(),
        rhsobj.q_itf_full_x1.copy(),
        rhsobj.q_itf_full_x2.copy(),
        momentum_blocks,
    )


def j2_flux_matvec(rhsobj, q, dq, base=None):
    """Apply the flux contribution of J2 to a perturbation ``dq``."""
    metric = rhsobj.metric
    if base is None:
        base = j2_prepare(rhsobj, q)
    ops, pressure, q_itf_x1, q_itf_x2, q_itf_full_x1, q_itf_full_x2, momentum_blocks = base
    num_solpts_2d = rhsobj.geom.num_solpts**2
    itf_size = rhsobj.geom.itf_size
    h_contra = metric.h_contra_new

    # Volume flux derivative, in both horizontal directions.
    df_x1 = metric.sqrtG_new * flux_jacobian_matvec(dq, q, pressure, 0, h_contra[0, 0], h_contra[0, 1], h_contra[0, 2])
    df_x2 = metric.sqrtG_new * flux_jacobian_matvec(dq, q, pressure, 1, h_contra[1, 0], h_contra[1, 1], h_contra[1, 2])
    out = apply_op(df_x1, ops.derivative_x)
    apply_op(df_x2, ops.derivative_y, out=out, beta=1.0)

    # Linearize the logarithmic extrapolation of rho and rho theta.
    dq_itf_x1 = apply_op(dq, ops.extrap_x)
    dq_itf_x2 = apply_op(dq, ops.extrap_y)
    for dq_itf, op_extrap, q_itf in ((dq_itf_x1, ops.extrap_x, q_itf_x1), (dq_itf_x2, ops.extrap_y, q_itf_x2)):
        dq_itf[idx_rho] = q_itf[idx_rho] * apply_op(dq[idx_rho] / q[idx_rho], op_extrap)
        dq_itf[idx_rho_theta] = q_itf[idx_rho_theta] * apply_op(dq[idx_rho_theta] / q[idx_rho_theta], op_extrap)

    # The perturbation needs the same halo exchange as the state itself.
    request = rhsobj.ptopo.start_exchange_euler_3d(
        dq_itf_x2[..., 0, :, :itf_size],
        dq_itf_x2[..., -1, :, itf_size:],
        dq_itf_x1[..., 0, :itf_size],
        dq_itf_x1[..., -1, itf_size:],
        rhsobj.geom.boundary_sn_new,
        rhsobj.geom.boundary_we_new,
        flip_dim=(-3, -1),
    )
    dq_south, dq_north, dq_west, dq_east = request.wait()

    dq_itf_full_x1 = torch.zeros_like(q_itf_full_x1)
    dq_itf_full_x2 = torch.zeros_like(q_itf_full_x2)
    dq_itf_full_x1[mid_i] = dq_itf_x1
    dq_itf_full_x2[mid_j] = dq_itf_x2
    dq_itf_full_x1[..., 0, itf_size:] = dq_west
    dq_itf_full_x1[..., -1, :itf_size] = dq_east
    dq_itf_full_x2[..., 0, :, itf_size:] = dq_south
    dq_itf_full_x2[..., -1, :, :itf_size] = dq_north

    # The two traces meeting at each interface.  The name says which element the trace comes from:
    # `from_west` is the east face of the element to the west of the interface, and so on.
    from_west = numpy.s_[..., :-1, num_solpts_2d:]
    from_east = numpy.s_[..., 1:, :num_solpts_2d]
    from_south = numpy.s_[..., :-1, :, num_solpts_2d:]
    from_north = numpy.s_[..., 1:, :, :num_solpts_2d]

    df_itf_x1 = _riemann_flux_jvp(
        0, q_itf_full_x1, dq_itf_full_x1, metric.sqrtG_itf_i_new, metric.h_contra_itf_i_new, from_west, from_east
    )
    df_itf_x2 = _riemann_flux_jvp(
        1, q_itf_full_x2, dq_itf_full_x2, metric.sqrtG_itf_j_new, metric.h_contra_itf_j_new, from_south, from_north
    )

    apply_op(df_itf_x1[mid_i], ops.correction_WE, out=out, beta=1.0)
    apply_op(df_itf_x2[mid_j], ops.correction_SN, out=out, beta=1.0)

    out *= -metric.inv_sqrtG_new

    # The vertical blocks already include the outer metric factor.
    out += horizontal_momentum_matvec(rhsobj, momentum_blocks, dq)

    # ---- Replace the plain vertical-momentum row by its well-balanced differential.
    sqrtG_h13 = metric.sqrtG_new * h_contra[0, 2]
    sqrtG_h23 = metric.sqrtG_new * h_contra[1, 2]
    u1 = q[idx_rho_u1] / q[idx_rho]
    u2 = q[idx_rho_u2] / q[idx_rho]
    du1 = (dq[idx_rho_u1] - u1 * dq[idx_rho]) / q[idx_rho]
    du2 = (dq[idx_rho_u2] - u2 * dq[idx_rho]) / q[idx_rho]

    # Advective part: the horizontal transport of rho u^3.
    dadv_x1 = metric.sqrtG_new * (du1 * q[idx_rho_u3] + u1 * dq[idx_rho_u3])
    dadv_x2 = metric.sqrtG_new * (du2 * q[idx_rho_u3] + u2 * dq[idx_rho_u3])
    dadv = apply_op(dadv_x1, ops.derivative_x)
    apply_op(dadv_x2, ops.derivative_y, out=dadv, beta=1.0)

    dwadv_x1, dwpres_x1 = _well_balanced_riemann_jvp(
        0, q_itf_full_x1, dq_itf_full_x1, metric.sqrtG_itf_i_new, metric.h_contra_itf_i_new, from_west, from_east
    )
    dwadv_x2, dwpres_x2 = _well_balanced_riemann_jvp(
        1, q_itf_full_x2, dq_itf_full_x2, metric.sqrtG_itf_j_new, metric.h_contra_itf_j_new, from_south, from_north
    )
    apply_op(dwadv_x1[mid_i], ops.correction_WE, out=dadv, beta=1.0)
    apply_op(dwadv_x2[mid_j], ops.correction_SN, out=dadv, beta=1.0)

    # Pressure part.  The RHS writes it as p * (base-state operator applied to log p), so the
    # derivative has one term through p and one through log p.
    p_itf_x1 = equation_of_state(q_itf_full_x1[idx_rho_theta])
    p_itf_x2 = equation_of_state(q_itf_full_x2[idx_rho_theta])
    wflux_pres_itf_x1 = 0.5 * (
        metric.sqrtG_itf_i_new[from_west] * metric.h_contra_itf_i_new[0, 2][from_west] * p_itf_x1[from_west]
        + metric.sqrtG_itf_i_new[from_east] * metric.h_contra_itf_i_new[0, 2][from_east] * p_itf_x1[from_east]
    )
    wflux_pres_itf_x2 = 0.5 * (
        metric.sqrtG_itf_j_new[from_south] * metric.h_contra_itf_j_new[1, 2][from_south] * p_itf_x2[from_south]
        + metric.sqrtG_itf_j_new[from_north] * metric.h_contra_itf_j_new[1, 2][from_north] * p_itf_x2[from_north]
    )
    wpres_over_p_x1 = torch.zeros_like(q_itf_full_x1[idx_rho])
    wpres_over_p_x2 = torch.zeros_like(q_itf_full_x2[idx_rho])
    wpres_over_p_x1[from_west] = wflux_pres_itf_x1 / p_itf_x1[from_west]
    wpres_over_p_x1[from_east] = wflux_pres_itf_x1 / p_itf_x1[from_east]
    wpres_over_p_x2[from_south] = wflux_pres_itf_x2 / p_itf_x2[from_south]
    wpres_over_p_x2[from_north] = wflux_pres_itf_x2 / p_itf_x2[from_north]

    # Base-state coefficient multiplying the perturbation of p.
    log_p = torch.log(pressure)
    w_pres_base = apply_op(sqrtG_h13, ops.derivative_x)
    apply_op(sqrtG_h23, ops.derivative_y, out=w_pres_base, beta=1.0)
    apply_op(wpres_over_p_x1[mid_i], ops.correction_WE, out=w_pres_base, beta=1.0)
    apply_op(wpres_over_p_x2[mid_j], ops.correction_SN, out=w_pres_base, beta=1.0)
    log_p_itf_x1 = torch.log(p_itf_x1)
    log_p_itf_x2 = torch.log(p_itf_x2)
    dlog_p_itf_x1 = heat_capacity_ratio * dq_itf_full_x1[idx_rho_theta] / q_itf_full_x1[idx_rho_theta]
    dlog_p_itf_x2 = heat_capacity_ratio * dq_itf_full_x2[idx_rho_theta] / q_itf_full_x2[idx_rho_theta]
    w_presb_x1 = apply_op(log_p, ops.derivative_x)
    w_presb_x2 = apply_op(log_p, ops.derivative_y)
    apply_op(log_p_itf_x1[mid_i], ops.correction_WE, out=w_presb_x1, beta=1.0)
    apply_op(log_p_itf_x2[mid_j], ops.correction_SN, out=w_presb_x2, beta=1.0)
    w_pres_base += sqrtG_h13 * w_presb_x1 + sqrtG_h23 * w_presb_x2

    # Perturbation of the same operator, at frozen p.
    dlog_p = heat_capacity_ratio * dq[idx_rho_theta] / q[idx_rho_theta]
    dw_pres = apply_op(dwpres_x1[mid_i], ops.correction_WE)
    apply_op(dwpres_x2[mid_j], ops.correction_SN, out=dw_pres, beta=1.0)
    dw_presb_x1 = apply_op(dlog_p, ops.derivative_x)
    dw_presb_x2 = apply_op(dlog_p, ops.derivative_y)
    apply_op(dlog_p_itf_x1[mid_i], ops.correction_WE, out=dw_presb_x1, beta=1.0)
    apply_op(dlog_p_itf_x2[mid_j], ops.correction_SN, out=dw_presb_x2, beta=1.0)
    dw_pres += sqrtG_h13 * dw_presb_x1 + sqrtG_h23 * dw_presb_x2

    dp = heat_capacity_ratio * pressure * dq[idx_rho_theta] / q[idx_rho_theta]
    out[idx_rho_u3] = -metric.inv_sqrtG_new * (dadv + dp * w_pres_base + pressure * dw_pres)

    if rhsobj.y_invariant_slab:
        out[idx_rho_u2] = 0.0  # The pinned tendency has zero derivative.

    return out
