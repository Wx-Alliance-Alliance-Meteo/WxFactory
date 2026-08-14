import math

import torch
from torch import Tensor

from .cubed_sphere_2d import CubedSphere2D
from .cubed_sphere_3d import CubedSphere3D


def wind2contra_2d(u: float | Tensor, v: float | Tensor, geom: CubedSphere2D | CubedSphere3D):
    """Convert wind fields from the spherical basis (zonal, meridional) to panel-appropriate contrvariant winds, in two dimensions

    Parameters:
    ----------
    u : float | Tensor
       Input zonal winds, in meters per second
    v : float | Tensor
       Input meridional winds, in meters per second
    geom : CubedSphere
       Geometry object (CubedSphere), describing the grid configuration and globe paramters.
       Required parameters:
       earth_radius, coslat, lat_p, angle_p, X, Y, delta2

    Returns:
    -------
    (u1_contra, u2_contra) : tuple
       Tuple of contravariant winds"""

    # Convert winds coords to spherical basis

    if geom.nk > 1 and geom.deep:
        # In 3D code with the deep atmosphere, the conversion to λ and φ
        # uses the full radial height of the grid point:
        lambda_dot = u / ((geom.earth_radius + geom.coordVec_gnom[2, :, :, :]) * geom.coslat)
        phi_dot = v / (geom.earth_radius + geom.coordVec_gnom[2, :, :, :])
    else:
        # Otherwise, the conversion uses just the planetary radius, with no
        # correction for height above the surface
        lambda_dot = u / (geom.earth_radius * geom.coslat)
        phi_dot = v / geom.earth_radius

    if isinstance(geom, CubedSphere3D):
        X = geom.X_new
        Y = geom.Y_new
        delta2 = geom.delta2_new
    else:
        # CubedSphere2D
        X = geom.X
        Y = geom.Y
        delta2 = geom.delta2

    denom = torch.sqrt(
        (
            math.cos(geom.lat_p)
            + X * math.sin(geom.lat_p) * math.sin(geom.angle_p)
            - Y * math.sin(geom.lat_p) * math.cos(geom.angle_p)
        )
        ** 2
        + (X * math.cos(geom.angle_p) + Y * math.sin(geom.angle_p)) ** 2
    )

    dx1dlon = math.cos(geom.lat_p) * math.cos(geom.angle_p) + (
        X * Y * math.cos(geom.lat_p) * math.sin(geom.angle_p) - Y * math.sin(geom.lat_p)
    ) / (1.0 + X**2)
    dx2dlon = (X * Y * math.cos(geom.lat_p) * math.cos(geom.angle_p) + X * math.sin(geom.lat_p)) / (
        1.0 + Y**2
    ) + math.cos(geom.lat_p) * math.sin(geom.angle_p)

    dx1dlat = (
        -delta2 * ((math.cos(geom.lat_p) * math.sin(geom.angle_p) + X * math.sin(geom.lat_p)) / (1.0 + X**2)) / denom
    )
    dx2dlat = (
        delta2 * ((math.cos(geom.lat_p) * math.cos(geom.angle_p) - Y * math.sin(geom.lat_p)) / (1.0 + Y**2)) / denom
    )

    # transform to the reference element

    u1_contra = (dx1dlon * lambda_dot + dx1dlat * phi_dot) * 2.0 / geom.delta_x1
    u2_contra = (dx2dlon * lambda_dot + dx2dlat * phi_dot) * 2.0 / geom.delta_x2

    return u1_contra, u2_contra


def contra2wind_2d(u1: float | Tensor, u2: float | Tensor, geom: CubedSphere2D | CubedSphere3D):
    """Convert from reference element to "physical winds", in two dimensions

    Parameters:
    -----------
    u1 : float | Tensor
       Contravariant winds along first component (X)
    u2 : float | Tensor
       Contravariant winds along second component (Y)
    geom : CubedSphere
       Geometry object, containing:
          delta_x1, delta_x2, lat_p, angle_p, X, Y, coslat, earth_radius

    Returns:
    --------
    (u, v) : tuple
       Zonal/meridional winds, in m/s
    """

    u1_contra = u1 * geom.delta_x1 / 2.0
    u2_contra = u2 * geom.delta_x2 / 2.0

    if isinstance(geom, CubedSphere3D):
        X = geom.X_new
        Y = geom.Y_new
        delta2 = geom.delta2_new
    else:
        # CubedSphere2D
        X = geom.X
        Y = geom.Y
        delta2 = geom.delta2

    denom = (
        math.cos(geom.lat_p)
        + X * math.sin(geom.lat_p) * math.sin(geom.angle_p)
        - Y * math.sin(geom.lat_p) * math.cos(geom.angle_p)
    ) ** 2 + (X * math.cos(geom.angle_p) + Y * math.sin(geom.angle_p)) ** 2

    dlondx1 = (
        (
            math.cos(geom.lat_p) * math.cos(geom.angle_p)
            + X * Y * math.cos(geom.lat_p) * math.sin(geom.angle_p)
            - Y * math.sin(geom.lat_p)
        )
        * (1.0 + X**2)
        / denom
    )

    dlondx2 = (math.cos(geom.lat_p) * math.sin(geom.angle_p) + X * math.sin(geom.lat_p)) * (1.0 + Y**2) / denom

    denom[:, :] = torch.sqrt(
        (
            math.cos(geom.lat_p)
            + X * math.sin(geom.lat_p) * math.sin(geom.angle_p)
            - Y * math.sin(geom.lat_p) * math.cos(geom.angle_p)
        )
        ** 2
        + (X * math.cos(geom.angle_p) + Y * math.sin(geom.angle_p)) ** 2
    )

    dlatdx1 = -(
        (
            X * Y * math.cos(geom.lat_p) * math.cos(geom.angle_p)
            + X * math.sin(geom.lat_p)
            + (1.0 + Y**2) * math.cos(geom.lat_p) * math.sin(geom.angle_p)
        )
        * (1.0 + X**2)
    ) / (delta2 * denom)

    dlatdx2 = (
        (
            (1.0 + X**2) * math.cos(geom.lat_p) * math.cos(geom.angle_p)
            + X * Y * math.cos(geom.lat_p) * math.sin(geom.angle_p)
            - Y * math.sin(geom.lat_p)
        )
        * (1.0 + Y**2)
    ) / (delta2 * denom)

    if geom.nk > 1 and geom.deep:
        # If we are in a 3D geometry with the deep atmosphere, the conversion from
        # contravariant → spherical → zonal/meridional winds uses the full radial distance
        # at the last step
        u = (
            (dlondx1 * u1_contra + dlondx2 * u2_contra)
            * geom.coslat
            * (geom.earth_radius + geom.coordVec_gnom[2, :, :, :])
        )
        v = (dlatdx1 * u1_contra + dlatdx2 * u2_contra) * (geom.earth_radius + geom.coordVec_gnom[2, :, :, :])
    else:
        # Otherwise, the conversion is based on the spherical radius only, with no height correction
        u = (dlondx1 * u1_contra + dlondx2 * u2_contra) * geom.coslat * geom.earth_radius
        v = (dlatdx1 * u1_contra + dlatdx2 * u2_contra) * geom.earth_radius

    return u, v
