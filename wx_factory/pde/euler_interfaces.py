"""Acoustic interface traces for the 5D Euler RHS.

Momentum uses pressure and mechanical-energy traces. The relative covariant
star velocity also supplies the curl interface lifting.
"""

from typing import NamedTuple

import torch
from torch import Tensor

from ..common.definitions import Rd, heat_capacity_ratio, p0


class EulerInterfaceGeometry(NamedTuple):
    """Static metric coefficients and indices for paired element-face traces.

    normal_scale is sqrt(G^{dd}); normal_row is G^{dj}/sqrt(G^{dd}).
    Contracting normal_row with v_j gives physical normal wind, whereas u^d
    is that wind multiplied by normal_scale.
    """

    left: tuple | None
    right: tuple | None
    inverse_spatial: Tensor
    geopotential: Tensor
    jacobian: Tensor
    normal_scale: Tensor
    normal_row: Tensor
    wall: Tensor | None


def pressure(rho_theta: Tensor) -> Tensor:
    """Return pressure from rho*theta using the ideal-gas EOS."""
    return p0 * torch.exp(heat_capacity_ratio * torch.log(rho_theta * (Rd / p0)))


def interface_slices(direction: int, num_solpts: int) -> tuple[tuple, tuple]:
    """Paired high/low traces in an element-face array with one halo element."""
    trailing = (slice(None),) * direction
    left = (Ellipsis, slice(None, -1)) + trailing + (slice(num_solpts**2, None),)
    right = (Ellipsis, slice(1, None)) + trailing + (slice(None, num_solpts**2),)
    return left, right


def principal_symbol(state: Tensor, inverse_spatial: Tensor, normal: Tensor) -> Tensor:
    """Return the frozen principal symbol for Q=(rho,v_i,rho*theta).

    The relative curl contributes to the principal part. ``normal`` is a
    covector with the same trailing point dimensions as state.
    """
    contravariant_velocity = torch.einsum("ij...,j...->i...", inverse_spatial, state[1:4])
    raised_normal = torch.einsum("ij...,j...->i...", inverse_spatial, normal)
    normal_advection_speed = (normal * contravariant_velocity).sum(dim=0)
    symbol = state.new_zeros((5, 5) + state.shape[1:])
    for i in range(5):
        symbol[i, i] = normal_advection_speed
    symbol[0, 1:4] = state[0] * raised_normal
    symbol[4, 1:4] = state[4] * raised_normal
    symbol[1:4, 4] = (heat_capacity_ratio * pressure(state[4]) / (state[0] * state[4])) * normal
    return symbol


def acoustic_interface(
    left: Tensor,
    right: Tensor,
    inverse_spatial: Tensor,
    direction: int,
    geopotential: Tensor,
    *,
    wall: Tensor | None = None,
    return_velocity: bool = False,
    normal_geometry: tuple[Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return common contravariant velocity u^d, pressure, and K+Phi_e.

    Use the arithmetic mean of rho*c as the acoustic impedance. Pressure is
    an independent numerical traction; it need not match the EOS evaluated
    on upwind density and temperature. The common velocity supplies both
    kinetic energy and relative-curl traces, with upwind tangential components.
    Supersonic same-sign states use upstream data. No positivity limiter is used.
    """
    pressure_left, pressure_right = pressure(left[4]), pressure(right[4])
    if normal_geometry is None:
        normal_scale = torch.sqrt(inverse_spatial[direction, direction])
        normal_row = inverse_spatial[direction] / normal_scale
    else:
        normal_scale, normal_row = normal_geometry
    normal_wind_left = (normal_row * left[1:4]).sum(dim=0)
    normal_wind_right = (normal_row * right[1:4]).sum(dim=0)
    sound_speed_left = torch.sqrt(heat_capacity_ratio * pressure_left / left[0])
    sound_speed_right = torch.sqrt(heat_capacity_ratio * pressure_right / right[0])
    impedance = (left[0] * sound_speed_left + right[0] * sound_speed_right) / 2
    common_normal_wind = (normal_wind_left + normal_wind_right) / 2 - (pressure_right - pressure_left) / (2 * impedance)
    common_pressure = (pressure_left + pressure_right) / 2 - impedance * (normal_wind_right - normal_wind_left) / 2

    supersonic_to_right = (normal_wind_left >= sound_speed_left) & (normal_wind_right >= sound_speed_right)
    supersonic_to_left = (normal_wind_left <= -sound_speed_left) & (normal_wind_right <= -sound_speed_right)
    common_normal_wind = torch.where(
        supersonic_to_right, normal_wind_left, torch.where(supersonic_to_left, normal_wind_right, common_normal_wind)
    )
    common_pressure = torch.where(
        supersonic_to_right, pressure_left, torch.where(supersonic_to_left, pressure_right, common_pressure)
    )
    if wall is not None:
        common_normal_wind = torch.where(wall, torch.zeros_like(common_normal_wind), common_normal_wind)

    upwind_covariant_velocity = torch.where(common_normal_wind[None] >= 0, left[1:4], right[1:4])
    upwind_normal_wind = (normal_row * upwind_covariant_velocity).sum(dim=0)
    common_covariant_velocity = upwind_covariant_velocity.clone()
    common_covariant_velocity[direction] += (common_normal_wind - upwind_normal_wind) / normal_scale
    common_kinetic_energy = (
        torch.einsum("i...,ij...,j...->...", common_covariant_velocity, inverse_spatial, common_covariant_velocity) / 2
    )
    result = (common_normal_wind * normal_scale, common_pressure, common_kinetic_energy + geopotential)
    return (*result, common_covariant_velocity) if return_velocity else result
