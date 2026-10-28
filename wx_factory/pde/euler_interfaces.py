"""Acoustic interface traces for the 5D Euler RHS.

Momentum uses pressure and mechanical-energy traces. The relative covariant
star velocity also supplies the curl interface lifting.
"""

from typing import NamedTuple

import torch
from torch import Tensor

from ..common.definitions import Rd, heat_capacity_ratio, p0
from ..common.tensor_math import spatial_matrix_vector


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


def physical_mach_number(covariant: Tensor, inverse_spatial: Tensor, sound_speed: Tensor) -> Tensor:
    """Return the coordinate-relative physical speed divided by sound speed."""
    contravariant = spatial_matrix_vector(inverse_spatial, covariant)
    mach_squared = (covariant * contravariant).sum(dim=0) / sound_speed.square()
    moving = mach_squared > 0
    # Keep the inactive square root finite so differentiation at rest gives zero.
    safe_squared = torch.where(moving, mach_squared, torch.ones_like(mach_squared))
    return torch.where(moving, torch.sqrt(safe_squared), torch.zeros_like(mach_squared))


def low_mach_scales(mach_left: Tensor, mach_right: Tensor, cutoff: float):
    """Return the interface Mach number and velocity-jump pressure scaling."""
    interface_mach = torch.where(mach_left >= mach_right, mach_left, mach_right)
    limited_mach = torch.clamp(interface_mach, min=cutoff, max=1.0)
    pressure_scale = limited_mach * (2 - limited_mach)
    return interface_mach, pressure_scale


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
    mach_cutoff: float = 0.3,
) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return common u^d, pressure, and K+Phi_e for low-Mach flow.

    The normal velocity comes from acoustic matching; tangential velocity is
    averaged. This common velocity supplies both energy and curl traces.
    Pressure is an independent traction and need not satisfy the EOS with
    the upwind scalars. Only the velocity-jump pressure correction is scaled.
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
    mach_left = physical_mach_number(left[1:4], inverse_spatial, sound_speed_left)
    mach_right = physical_mach_number(right[1:4], inverse_spatial, sound_speed_right)
    _, pressure_scale = low_mach_scales(mach_left, mach_right, mach_cutoff)
    common_normal_wind = (normal_wind_left + normal_wind_right) / 2 - (pressure_right - pressure_left) / (2 * impedance)
    common_pressure = (pressure_left + pressure_right) / 2 - (
        pressure_scale * impedance * (normal_wind_right - normal_wind_left) / 2
    )

    if wall is not None:
        common_normal_wind = torch.where(wall, torch.zeros_like(common_normal_wind), common_normal_wind)

    common_covariant_velocity = (left[1:4] + right[1:4]) / 2
    mean_normal_wind = (normal_row * common_covariant_velocity).sum(dim=0)
    common_covariant_velocity[direction] += (common_normal_wind - mean_normal_wind) / normal_scale
    common_kinetic_energy = (
        torch.einsum("i...,ij...,j...->...", common_covariant_velocity, inverse_spatial, common_covariant_velocity) / 2
    )
    result = (common_normal_wind * normal_scale, common_pressure, common_kinetic_energy + geopotential)
    return (*result, common_covariant_velocity) if return_velocity else result
