"""Pointwise Jacobian of the non-stiff forcing."""

import torch

from ..common.definitions import (
    heat_capacity_ratio,
    idx_rho,
    idx_rho_theta,
    idx_rho_u1,
    idx_rho_u2,
    idx_rho_u3,
)
from .flux_jacobian import MOMENTUM_ROWS, equation_of_state


def forcing_jac_prepare(rhsobj, q):
    """Precompute the base-state coefficients :func:`forcing_jvp` needs."""
    rho = q[idx_rho]
    u = (q[idx_rho_u1] / rho, q[idx_rho_u2] / rho, q[idx_rho_u3] / rho)
    rho_theta = q[idx_rho_theta]
    pressure = equation_of_state(rho_theta)
    dp_drho_theta = heat_capacity_ratio * pressure / rho_theta

    rayleigh = None
    case = getattr(rhsobj.pde, "case_number", None)
    if case in (21, 22):
        from ..init.dcmip import dcmip_schar_damping_coeffs

        rate, u1_ref, u2_ref, u3_ref = dcmip_schar_damping_coeffs(
            rhsobj.metric, rhsobj.pde.geometry, shear=(case == 22)
        )
        rayleigh = (rate, (u1_ref, u2_ref, u3_ref))

    return u, dp_drho_theta, rayleigh


def forcing_jvp(rhsobj, q, v, base=None):
    """Apply the pointwise Jacobian of ``rhs.forcing_only`` to a perturbation ``v``."""
    metric = rhsobj.metric
    if base is None:
        base = forcing_jac_prepare(rhsobj, q)
    u, dp_drho_theta, rayleigh = base
    u1, u2, u3 = u
    # christoffel[d] holds the 9 independent symbols of direction d: the three Gamma^d_{0i}
    # followed by the upper triangle of the symmetric Gamma^d_{ij}.
    christoffel = metric.christoffel
    h_contra = metric.h_contra_new
    h11, h12, h13 = h_contra[0, 0], h_contra[0, 1], h_contra[0, 2]
    h22, h23, h33 = h_contra[1, 1], h_contra[1, 2], h_contra[2, 2]

    d_rho = v[idx_rho]
    d_mom = (v[idx_rho_u1], v[idx_rho_u2], v[idx_rho_u3])
    d_rho_theta = v[idx_rho_theta]

    out = torch.zeros_like(v)
    for direction, row in enumerate(MOMENTUM_ROWS):
        c01, c02, c03 = christoffel[direction, 0], christoffel[direction, 1], christoffel[direction, 2]
        c11, c12, c13 = christoffel[direction, 3], christoffel[direction, 4], christoffel[direction, 5]
        c22, c23, c33 = christoffel[direction, 6], christoffel[direction, 7], christoffel[direction, 8]

        # Differentiate the geometric source with respect to the conserved variables.
        dF_drho = -(
            c11 * u1 * u1
            + 2.0 * c12 * u1 * u2
            + 2.0 * c13 * u1 * u3
            + c22 * u2 * u2
            + 2.0 * c23 * u2 * u3
            + c33 * u3 * u3
        )
        dF_dmom1 = 2.0 * c01 + 2.0 * (c11 * u1 + c12 * u2 + c13 * u3)
        dF_dmom2 = 2.0 * c02 + 2.0 * (c12 * u1 + c22 * u2 + c23 * u3)
        dF_dmom3 = 2.0 * c03 + 2.0 * (c13 * u1 + c23 * u2 + c33 * u3)
        dF_drho_theta = (
            c11 * h11 + 2.0 * c12 * h12 + 2.0 * c13 * h13 + c22 * h22 + 2.0 * c23 * h23 + c33 * h33
        ) * dp_drho_theta
        if direction == 2:
            # The vertical pressure source belongs to the implicit partition.
            dF_drho_theta = torch.zeros_like(dF_drho_theta)

        dF = (
            dF_drho * d_rho
            + dF_dmom1 * d_mom[0]
            + dF_dmom2 * d_mom[1]
            + dF_dmom3 * d_mom[2]
            + dF_drho_theta * d_rho_theta
        )

        if rayleigh is not None:
            rate, u_ref = rayleigh
            # Linearize the relaxation toward the reference wind.
            dF = dF + rate * (d_mom[direction] - u_ref[direction] * d_rho)

        out[row] = -dF

    if rhsobj.y_invariant_slab:
        out[idx_rho_u2] = 0.0  # The pinned tendency has zero derivative.

    return out
