import math

import torch


def cart2sph(x, y, z):
    """Transform Cartesian to spherical coordinates.
    az,elev,r = cart2sph(X,Y,Z) transforms corresponding elements of
    data stored in Cartesian coordinates X,Y,Z to spherical
    coordinates (azimuth, elevation, and radius).  The arrays
    X,Y, and Z must be the same size (or any of them can be scalar).
    az and elev are returned in radians.

    az is the counterclockwise angle in the xy plane measured from the
    positive x axis.  elev is the elevation angle from the xy plane.
    """

    hypotxy = torch.hypot(x, y)
    r = torch.hypot(hypotxy, z)
    elev = torch.arctan2(z, hypotxy)
    az = torch.arctan2(y, x)

    # Map to the interval [0, 2 pi]
    az[az < 0.0] += 2.0 * math.pi

    return az, elev, r
