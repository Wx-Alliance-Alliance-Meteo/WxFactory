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
