"""Riemann solver for the interface fluxes of the 3D Euler equations."""

import torch
from numpy.typing import NDArray

from ..common.definitions import (
    heat_capacity_ratio,
    idx_rho,
    idx_rho_u1,
    idx_rho_u2,
    idx_rho_u3,
)
from ..geometry import Metric3DTopo

#: Momentum rows in coordinate order.
MOMENTUM_ROWS = (idx_rho_u1, idx_rho_u2, idx_rho_u3)


def interface_slices(direction: int, num_solpts: int) -> tuple[tuple, tuple]:
    """Return paired high and low traces on interfaces normal to ``direction``."""
    num_solpts_2d = num_solpts**2
    trailing = (slice(None),) * direction
    left = (Ellipsis, slice(None, -1)) + trailing + (slice(num_solpts_2d, None),)
    right = (Ellipsis, slice(1, None)) + trailing + (slice(None, num_solpts_2d),)
    return left, right


def outward_faces(direction: int, num_solpts: int) -> tuple[tuple, tuple]:
    """Return the unused outer halo faces normal to ``direction``."""
    num_solpts_2d = num_solpts**2
    trailing = (slice(None),) * direction
    first = (Ellipsis, 0) + trailing + (slice(None, num_solpts_2d),)
    last = (Ellipsis, -1) + trailing + (slice(num_solpts_2d, None),)
    return first, last


def rusanov_3d(
    direction: int,
    velocity_itf: NDArray,
    variables_itf: NDArray,
    pressure_itf: NDArray,
    metric: Metric3DTopo,
    advection_only: bool,
    flux_itf: NDArray,
    num_solpts: int,
) -> None:
    """Write the Rusanov common flux along ``direction`` into ``flux_itf``."""
    left, right = interface_slices(direction, num_solpts)
    sqrtG = (metric.sqrtG_itf_i_new, metric.sqrtG_itf_j_new, metric.sqrtG_itf_k_new)[direction]
    h_contra = (metric.h_contra_itf_i_new, metric.h_contra_itf_j_new, metric.h_contra_itf_k_new)[direction]

    u_l = velocity_itf[left]
    u_r = velocity_itf[right]

    if advection_only:
        # Tracer-only mode excludes acoustic wave speeds.
        eig_l = torch.abs(u_l)
        eig_r = torch.abs(u_r)
    else:
        # Use the advective and acoustic characteristic speeds.
        eig_l = torch.abs(u_l) + torch.sqrt(
            h_contra[direction, direction][left]
            * heat_capacity_ratio
            * pressure_itf[left]
            / variables_itf[idx_rho][left]
        )
        eig_r = torch.abs(u_r) + torch.sqrt(
            h_contra[direction, direction][right]
            * heat_capacity_ratio
            * pressure_itf[right]
            / variables_itf[idx_rho][right]
        )

    eig = torch.maximum(eig_l, eig_r)

    # Advective flux.
    flux_l = sqrtG[left] * u_l * variables_itf[left]
    flux_r = sqrtG[right] * u_r * variables_itf[right]

    # Pressure acts only on momentum rows.
    sqrtG_pressure_l = sqrtG[left] * pressure_itf[left]
    sqrtG_pressure_r = sqrtG[right] * pressure_itf[right]
    for i, momentum_row in enumerate(MOMENTUM_ROWS):
        flux_l[momentum_row] += sqrtG_pressure_l * h_contra[direction, i][left]
        flux_r[momentum_row] += sqrtG_pressure_r * h_contra[direction, i][right]

    # Store the same common flux on both traces.
    flux_itf[left] = 0.5 * (flux_l + flux_r - eig * sqrtG[left] * (variables_itf[right] - variables_itf[left]))
    flux_itf[right] = flux_itf[left]
    
    
def ausm_plus_up_3d(
    direction: int,
    velocity_itf: NDArray,
    variables_itf: NDArray,
    pressure_itf: NDArray,
    metric: Metric3DTopo,
    advection_only: bool,
    flux_itf: NDArray,
    num_solpts: int,
) -> None:
    """Low-Mach_Number AUSM+-up common flux."""

    if advection_only:
        rusanov_3d(direction, velocity_itf, variables_itf, pressure_itf, metric, True, flux_itf, num_solpts)
        return

    left, right = interface_slices(direction, num_solpts)

    sqrtG = (metric.sqrtG_itf_i_new, metric.sqrtG_itf_j_new, metric.sqrtG_itf_k_new)[direction]
    h_contra = (metric.h_contra_itf_i_new, metric.h_contra_itf_j_new, metric.h_contra_itf_k_new)[direction]

    beta = 0.125
    K_p = 0.25
    K_u = 0.75
    sigma = 1.0

    # Low-Mach cutoffs
    M_INF_P = 0.1
    M_INF_U = 1.0e-12

    rho_l = variables_itf[idx_rho][left]
    rho_r = variables_itf[idx_rho][right]

    u_l = velocity_itf[left]
    u_r = velocity_itf[right]

    p_l = pressure_itf[left]
    p_r = pressure_itf[right]

    a_l = torch.sqrt(h_contra[direction, direction][left] * heat_capacity_ratio * p_l / torch.clamp(rho_l, min=1.0e-12))
    a_r = torch.sqrt(h_contra[direction, direction][right] * heat_capacity_ratio * p_r / torch.clamp(rho_r, min=1.0e-12))

    a_half = 0.5 * (a_l + a_r)

    M_l = u_l / torch.clamp(a_l, min=1.0e-14)
    M_r = u_r / torch.clamp(a_r, min=1.0e-14)

    M_l = torch.nan_to_num(M_l, nan=0.0, posinf=0.0, neginf=0.0)
    M_r = torch.nan_to_num(M_r, nan=0.0, posinf=0.0, neginf=0.0)

    M_bar_sq = 0.5 * (M_l**2 + M_r**2)
    M_bar = torch.sqrt(torch.clamp(M_bar_sq, min=0.0))

    M_0_p = torch.minimum(torch.ones_like(M_bar), torch.maximum(M_bar, torch.full_like(M_bar, M_INF_P)))
    fa_p = M_0_p * (2.0 - M_0_p)

    M_0_u = torch.minimum(torch.ones_like(M_bar), torch.maximum(M_bar, torch.full_like(M_bar, M_INF_U)))
    fa_u = M_0_u * (2.0 - M_0_u)

    alpha = 0.1875 * (-4.0 + 5.0 * fa_p**2)

    M_l_plus = 0.25 * (M_l + 1.0) ** 2 * (1.0 + 4.0 * beta * (M_l - 1.0) ** 2)
    M_r_minus = -0.25 * (M_r - 1.0) ** 2 * (1.0 + 4.0 * beta * (M_r + 1.0) ** 2)

    # ============================================================
    # Physical sound speeds. Used ONLY in Mp normalization.
    # ============================================================

    c_l = torch.sqrt(heat_capacity_ratio * p_l / torch.clamp(rho_l, min=1.0e-12))
    c_r = torch.sqrt(heat_capacity_ratio * p_r / torch.clamp(rho_r, min=1.0e-12))

    c_half = 0.5 * (c_l + c_r)

    rho_half = 0.5 * (rho_l + rho_r)

    Mp = -(K_p / torch.clamp(fa_p, min=1.0e-12)) * torch.clamp(1.0 - sigma * M_bar_sq, min=0.0) * (p_r - p_l) / torch.clamp(rho_half * c_half**2, min=1.0e-12)

    M = M_l_plus + M_r_minus + Mp

    P_l_plus_coeff = 0.25 * (M_l + 1.0) ** 2 * (2.0 - M_l + 4.0 * alpha * M_l * (M_l - 1.0) ** 2)
    P_r_minus_coeff = 0.25 * (M_r - 1.0) ** 2 * (2.0 + M_r - 4.0 * alpha * M_r * (M_r + 1.0) ** 2)

    P_l_plus = P_l_plus_coeff * p_l
    P_r_minus = P_r_minus_coeff * p_r

    Pw = -K_u * P_l_plus_coeff * P_r_minus_coeff * (rho_l + rho_r) * fa_u * a_half * (u_r - u_l)

    P = P_l_plus + P_r_minus + Pw

    # ============================================================
    # Numerical flux
    # ============================================================

    adv_flux = sqrtG[left] * (torch.clamp(M, min=0.0) * a_l * variables_itf[left] + torch.clamp(M, max=0.0) * a_r * variables_itf[right])

    flux_itf[left] = adv_flux

    sqrtG_pressure = sqrtG[left] * P

    for i, momentum_row in enumerate(MOMENTUM_ROWS):
        flux_itf[left][momentum_row] += h_contra[direction, i][left] * sqrtG_pressure

    # Same interface flux on both sides.
    flux_itf[right] = flux_itf[left]