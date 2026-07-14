import numpy
import math


def array_module(*arrays):
    """The array library that owns these arrays (numpy, cupy or torch).

    numpy's ufuncs already dispatch to cupy, and to torch tensors that sit on the host, but they
    cannot touch a tensor that lives on a GPU. Picking the module from the input keeps these
    helpers working on every backend.
    """
    for array in arrays:
        module = type(array).__module__.split(".")[0]
        if module in ("torch", "cupy"):
            return __import__(module)
    return numpy


def sph2cart(az, elev, radius):
    """Transform spherical to Cartesian coordinates.
    [X,Y,Z] = sph2cart(TH,PHI,radius) transforms corresponding elements of
    data stored in spherical coordinates (azimuth TH, elevation PHI,
    radius radius) to Cartesian coordinates X,Y,Z.  The arrays TH, PHI, and
    radius must be the same size (or any of them can be scalar).  TH and
    PHI must be in radians.

    TH is the counterclockwise angle in the xy plane measured from the
    positive x axis.  PHI is the elevation angle from the xy plane.
    """

    xp = array_module(az, elev, radius)

    z = radius * xp.sin(elev)
    rcoselev = radius * xp.cos(elev)
    x = rcoselev * xp.cos(az)
    y = rcoselev * xp.sin(az)

    return x, y, z


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

    xp = array_module(x, y, z)

    hypotxy = xp.hypot(x, y)
    r = xp.hypot(hypotxy, z)
    elev = xp.arctan2(z, hypotxy)
    az = xp.arctan2(y, x)

    # Map to the interval [0, 2 pi]
    az[az < 0.0] += 2.0 * math.pi

    return az, elev, r
