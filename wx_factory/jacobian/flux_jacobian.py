"""Pointwise Jacobian of the directional Euler flux."""

import torch

from ..common.definitions import (
    Rd,
    cpd,
    cvd,
    heat_capacity_ratio,
    idx_rho,
    idx_rho_theta,
    idx_rho_u1,
    idx_rho_u2,
    idx_rho_u3,
    p0,
)

#: Momentum row for each coordinate direction.
MOMENTUM_ROWS = (idx_rho_u1, idx_rho_u2, idx_rho_u3)

NUM_VAR = 5


def equation_of_state(rho_theta):
    """Return pressure from ``rho theta``."""
    return p0 * torch.exp((cpd / cvd) * torch.log(rho_theta * (Rd / p0)))


def flux_jacobian_matvec(dq, q, pressure, direction, h_1d, h_2d, h_3d):
    """Apply the directional flux Jacobian without the ``sqrt(G)`` factor."""
    rho = q[idx_rho]
    u = (q[idx_rho_u1] / rho, q[idx_rho_u2] / rho, q[idx_rho_u3] / rho)
    theta = q[idx_rho_theta] / rho
    mom_d = MOMENTUM_ROWS[direction]
    u_d = u[direction]
    # dp / d(rho theta), which is the squared sound speed divided by theta.
    cs2_over_theta = heat_capacity_ratio * pressure / q[idx_rho_theta]
    h_d = (h_1d, h_2d, h_3d)

    d_rho = dq[idx_rho]
    d_mom_d = dq[mom_d]
    d_rho_theta = dq[idx_rho_theta]

    out = torch.zeros_like(dq)
    out[idx_rho] = d_mom_d
    for i in range(3):
        out[MOMENTUM_ROWS[i]] = (
            u_d * dq[MOMENTUM_ROWS[i]] + u[i] * d_mom_d - u[i] * u_d * d_rho + h_d[i] * cs2_over_theta * d_rho_theta
        )
    out[idx_rho_theta] = u_d * d_rho_theta + theta * d_mom_d - u_d * theta * d_rho
    return out


def flux_jacobian_matrix(q, pressure, direction, h_1d, h_2d, h_3d):
    """Build the directional 5-by-5 flux Jacobian without ``sqrt(G)``."""
    rho = q[idx_rho]
    u = (q[idx_rho_u1] / rho, q[idx_rho_u2] / rho, q[idx_rho_u3] / rho)
    theta = q[idx_rho_theta] / rho
    mom_d = MOMENTUM_ROWS[direction]
    u_d = u[direction]
    cs2_over_theta = heat_capacity_ratio * pressure / q[idx_rho_theta]
    h_d = (h_1d, h_2d, h_3d)

    jac = torch.zeros(rho.shape + (NUM_VAR, NUM_VAR), dtype=q.dtype)
    jac[..., idx_rho, mom_d] = 1.0
    for i in range(3):
        mom_i = MOMENTUM_ROWS[i]
        jac[..., mom_i, mom_i] += u_d
        jac[..., mom_i, mom_d] += u[i]
        jac[..., mom_i, idx_rho] += -u[i] * u_d
        jac[..., mom_i, idx_rho_theta] += h_d[i] * cs2_over_theta
    jac[..., idx_rho_theta, idx_rho_theta] += u_d
    jac[..., idx_rho_theta, mom_d] += theta
    jac[..., idx_rho_theta, idx_rho] += -u_d * theta
    return jac
