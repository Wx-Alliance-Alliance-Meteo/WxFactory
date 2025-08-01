import math

import numpy

from common.configuration import Configuration
from common.definitions import cpd, gravity, p0, Rd
from geometry import CubedSphere3D, DFROperators, Metric3DTopo, wind2contra_2d, wind2contra_3d

# =======================================================================
#
#  Module for setting up initial conditions for the dynamical core tests:
#
#  11 - Deformational Advection Test
#  12 - Hadley Cell Advection Test
#  13 - Orography Advection Test
#  20 - Impact of orography on a steady-state at rest
#  21 and 22 - Non-Hydrostatic Mountain Waves Over A Schaer-Type Mountain without and with vertical wind shear
#  31 - Non-Hydrostatic Gravity Waves
#
# =======================================================================

# ==========================================================================================
# TEST CASE 11 - PURE ADVECTION - 3D DEFORMATIONAL FLOW
# ==========================================================================================

# The 3D deformational flow test is based on the deformational flow test of Nair and Lauritzen (JCP 2010),
# with a prescribed vertical wind velocity which makes the test truly 3D. An unscaled planet (with scale parameter
# X = 1) is selected.


def dcmip_T11_update_winds(geom, metric, mtrx, param, time=float(0)):
    """
    Test 11 - Deformational Advection

    The 3D deformational flow test is based on the deformational flow test of Nair and Lauritzen (JCP 2010),
    with a prescribed vertical wind velocity which makes the test truly 3D. An unscaled planet
    (with scale parameter X = 1) is selected.

    The velocities are time dependent and therefore must be updated in the dynamical core.
    """

    tau = 12.0 * 86400.0  # period of motion 12 days
    u0 = (2.0 * math.pi * geom.earth_radius) / tau  # 2 pi a / 12 days
    k0 = ((10.0 * geom.earth_radius) / tau,)  # Velocity Magnitude
    omega0 = (23000.0 * math.pi) / tau  # Velocity Magnitude
    T0 = 300.0  # temperature
    H = Rd * T0 / gravity  # scale height

    p = p0 * numpy.exp(-geom.height / H)
    ptop = p0 * math.exp(-12000.0 / H)

    lonp = geom.lon - 2.0 * math.pi * time / tau

    # Shape function
    bs = 0.2
    s = (
        1.0
        + math.exp((ptop - p0) / (bs * ptop))
        - numpy.exp((p - p0) / (bs * ptop))
        - numpy.exp((ptop - p) / (bs * ptop))
    )

    # Zonal Velocity

    ud = (
        (omega0 * geom.earth_radius)
        / (bs * ptop)
        * numpy.cos(lonp)
        * (numpy.cos(geom.lat) ** 2.0)
        * math.cos(2.0 * math.pi * time / tau)
        * (-numpy.exp((p - p0) / (bs * ptop)) + numpy.exp((ptop - p) / (bs * ptop)))
    )

    u = (
        k0 * numpy.sin(lonp) * numpy.sin(lonp) * numpy.sin(2.0 * geom.lat) * math.cos(math.pi * time / tau)
        + u0 * numpy.cos(geom.lat)
        + ud
    )

    # Meridional Velocity

    v = k0 * numpy.sin(2.0 * lonp) * numpy.cos(geom.lat) * math.cos(math.pi * time / tau)

    # Vertical Velocity

    w = (
        -((Rd * T0) / (gravity * p))
        * omega0
        * numpy.sin(lonp)
        * numpy.cos(geom.lat)
        * math.cos(2.0 * math.pi * time / tau)
        * s
    )

    # Contravariant components

    u1_contra, u2_contra = wind2contra_2d(u, v, geom)

    return u1_contra, u2_contra, w


# ==========================================================================================
# TEST CASE 12 - PURE ADVECTION - 3D HADLEY-LIKE FLOW
# ==========================================================================================


def dcmip_T12_update_winds(geom, metric, mtrx, param, time=float(0)):
    """
    Test 12 - 3D Hadley-like flow
    The velocities are time dependent and therefore must be updated in the dynamical core.
    """
    tau = 86400.0  # period of motion 1 day (in s)
    u0 = 40.0  # Zonal velocity magnitude (m/s)
    w0 = 0.15  # Vertical velocity magnitude (m/s), changed in v5
    T0 = 300.0  # temperature (K)
    H = Rd * T0 / gravity  # scale height
    K = 5.0  # number of Hadley-like cells

    # Height and pressure are aligned (p = p0 exp(-z/H))
    p = p0 * numpy.exp(-geom.height / H)

    # -----------------------------------------------------------------------
    #    TEMPERATURE IS CONSTANT 300 K
    # -----------------------------------------------------------------------

    t = T0

    # -----------------------------------------------------------------------
    #    RHO (density)
    # -----------------------------------------------------------------------

    rho = p / (Rd * t)
    rho0 = p0 / (Rd * t)

    # Zonal Velocity

    u = u0 * numpy.cos(geom.lat)

    # Meridional Velocity

    v = (
        -(rho0 / rho)
        * (geom.earth_radius * w0 * math.pi)
        / (K * param.ztop)
        * numpy.cos(geom.lat)
        * numpy.sin(K * geom.lat)
        * numpy.cos(math.pi * geom.height / param.ztop)
        * numpy.cos(math.pi * time / tau)
    )

    # Vertical Velocity - can be changed to vertical pressure velocity by
    # omega = -g*rho*w

    w = (
        (rho0 / rho)
        * (w0 / K)
        * (-2.0 * numpy.sin(K * geom.lat) * numpy.sin(geom.lat) + K * numpy.cos(geom.lat) * numpy.cos(K * geom.lat))
        * numpy.sin(math.pi * geom.height / param.ztop)
        * numpy.cos(math.pi * time / tau)
    )

    # Contravariant components

    u1_contra, u2_contra = wind2contra_2d(u, v, geom)

    return u1_contra, u2_contra, w


def dcmip_advection_deformation(geom, metric, mtrx, param):
    """
    Test 11 - Deformational Advection

    The 3D deformational flow test is based on the deformational flow test of Nair and Lauritzen (JCP 2010), with a prescribed vertical wind velocity which makes the test truly 3D. An unscaled planet (with scale parameter X = 1) is selected.
    """
    tau = 12.0 * 86400.0  # period of motion 12 days
    T0 = 300.0  # temperature
    H = Rd * T0 / gravity  # scale height
    RR = 1.0 / 2.0  # horizontal half width divided by 'a'
    ZZ = 1000.0  # vertical half width
    z0 = 5000.0  # center point in z
    lambda0 = 5.0 * math.pi / 6.0  # center point in longitudes
    lambda1 = 7.0 * math.pi / 6.0  # center point in longitudes
    phi0 = 0.0  # center point in latitudes
    phi1 = 0.0

    # -----------------------------------------------------------------------
    #    HEIGHT AND PRESSURE
    # -----------------------------------------------------------------------

    p = p0 * numpy.exp(-geom.height / H)

    # -----------------------------------------------------------------------
    #    WINDS
    # -----------------------------------------------------------------------

    u1_contra, u2_contra, w = dcmip_T11_update_winds(geom, metric, mtrx, param, time=0)

    # -----------------------------------------------------------------------
    #    TEMPERATURE IS CONSTANT 300 K
    # -----------------------------------------------------------------------

    t = T0

    # -----------------------------------------------------------------------
    #    RHO (density)
    # -----------------------------------------------------------------------

    rho = p / (Rd * t)

    # -----------------------------------------------------------------------
    #     Initialize theta (potential virtual temperature)
    # -----------------------------------------------------------------------

    tv = t
    theta = tv * (p0 / p) ** (Rd / cpd)

    # -----------------------------------------------------------------------
    #     initialize tracers
    # -----------------------------------------------------------------------

    # Tracer 1 - Cosine Bells

    # To calculate great circle distance
    sin_tmp = numpy.empty_like(p)
    cos_tmp = numpy.empty_like(p)
    sin_tmp2 = numpy.empty_like(p)
    cos_tmp2 = numpy.empty_like(p)

    sin_tmp[:, :, :] = numpy.sin(geom.lat) * math.sin(phi0)
    cos_tmp[:, :, :] = numpy.cos(geom.lat) * math.cos(phi0)
    sin_tmp2[:, :, :] = numpy.sin(geom.lat) * math.sin(phi1)
    cos_tmp2[:, :, :] = numpy.cos(geom.lat) * math.cos(phi1)

    # great circle distance without 'a'

    r = numpy.arccos(sin_tmp + cos_tmp * numpy.cos(geom.lon - lambda0))
    r2 = numpy.arccos(sin_tmp2 + cos_tmp2 * numpy.cos(geom.lon - lambda1))
    d1 = numpy.minimum(1.0, (r / RR) ** 2 + ((geom.height - z0) / ZZ) ** 2)
    d2 = numpy.minimum(1.0, (r2 / RR) ** 2 + ((geom.height - z0) / ZZ) ** 2)

    q1 = 0.5 * (1.0 + numpy.cos(math.pi * d1)) + 0.5 * (1.0 + numpy.cos(math.pi * d2))

    # Tracer 2 - Correlated Cosine Bells

    q2 = 0.9 - 0.8 * q1**2

    # Tracer 3 - Slotted Ellipse

    # Make the ellipse
    q3 = numpy.zeros_like(q1)
    nk, ni, nj = q3.shape
    for k in range(nk):
        for i in range(ni):
            for j in range(nj):
                # Make the ellipse
                if d1[k, i, j] <= RR:
                    q3[k, i, j] = 1.0
                elif d2[k, i, j] <= RR:
                    q3[k, i, j] = 1.0
                else:
                    q3[k, i, j] = 0.1

                # Put in the slot
                if geom.height[k, i, j] > z0 and abs(geom.lat[i, j]) < 0.125:
                    q3[k, i, j] = 0.1

    # Tracer 4: q4 is chosen so that, in combination with the other three tracer
    #           fields with weight (3/10), the sum is equal to one

    q4 = 1.0 - 0.3 * (q1 + q2 + q3)

    return rho, u1_contra, u2_contra, w, theta, q1, q2, q3, q4


def dcmip_advection_hadley(geom, metric, mtrx, param):
    """Test 12 - 3D Hadley-like flow"""
    tau = 86400.0  # period of motion 1 day (in s)
    T0 = 300.0  # temperature (K)
    H = Rd * T0 / gravity  # scale height
    z1 = 2000.0  # position of lower tracer bound (m), changed in v5
    z2 = 5000.0  # position of upper tracer bound (m), changed in v5
    z0 = 0.5 * (z1 + z2)  # midpoint (m)

    # -----------------------------------------------------------------------
    #    HEIGHT AND PRESSURE
    # -----------------------------------------------------------------------

    # Height and pressure are aligned (p = p0 exp(-z/H))
    p = p0 * numpy.exp(-geom.height / H)

    # -----------------------------------------------------------------------
    #    WINDS
    # -----------------------------------------------------------------------

    u1_contra, u2_contra, w = dcmip_T12_update_winds(geom, metric, mtrx, param, time=0)

    # -----------------------------------------------------------------------
    #    TEMPERATURE IS CONSTANT 300 K
    # -----------------------------------------------------------------------

    t = T0

    # -----------------------------------------------------------------------
    #    RHO (density)
    # -----------------------------------------------------------------------

    rho = p / (Rd * t)
    rho0 = p0 / (Rd * t)

    # -----------------------------------------------------------------------
    #     initialize TV (virtual temperature)
    # -----------------------------------------------------------------------
    tv = t
    theta = tv * (p0 / p) ** (Rd / cpd)

    # -----------------------------------------------------------------------
    #     initialize tracers
    # -----------------------------------------------------------------------

    # Tracer 1 - Layer

    q1 = numpy.zeros_like(p)
    nk, ni, nj = q1.shape
    for k in range(nk):
        for i in range(ni):
            for j in range(nj):
                if geom.height[k, i, j] < z2 and geom.height[k, i, j] > z1:
                    q1[k, i, j] = 0.5 * (1.0 + math.cos(2.0 * math.pi * (geom.height[k, i, j] - z0) / (z2 - z1)))

    return rho, u1_contra, u2_contra, w, theta, q1


# ============================================================================================
# TEST CASE 13 - HORIZONTAL ADVECTION OF THIN CLOUD-LIKE TRACERS IN THE PRESENCE OF OROGRAPHY
# ============================================================================================


def dcmip_mountain(geom: CubedSphere3D, metric, mtrx, param):

    lon_m = 3.0 * numpy.pi / 2.0
    # lon_m = 0.0
    lat_m = 0.0
    radius_m = 3.0 * numpy.pi / 4.0 * 0.5
    height_max = 2000.0
    oscillation_half_width = numpy.pi / 16.0

    def compute_distance_radian(lon, lat):
        """Compute the angular distance (in radians) of the given lon/lat coordinates from the center of the mountain"""
        return numpy.minimum(radius_m, numpy.sqrt((lon - lon_m) ** 2 + (lat - lat_m) ** 2))

    def compute_height_from_dist(dist):
        """Compute the height of the surface that corresponds to the given distance(s) from the mountain center.
        Based on the DCMIP case 1-3 description"""
        return (
            height_max
            / 2.0
            * (1.0 + numpy.cos(numpy.pi * dist / radius_m))
            * numpy.cos(numpy.pi * dist / oscillation_half_width)
        )

    # Distances from the mountain on all grid and interface points
    distance = compute_distance_radian(geom.lon[0, :, :], geom.lat[0, :, :])
    distance_itf_i = compute_distance_radian(geom.lon_itf_i, geom.lat_itf_i)
    distance_itf_j = compute_distance_radian(geom.lon_itf_j, geom.lat_itf_j)

    # Height at every grid and interface point
    h_surf = compute_height_from_dist(distance)

    num_interfaces_horiz = param.num_elements_horizontal + 1
    h_surf_itf_i = numpy.zeros((param.num_elements_horizontal + 2, param.num_solpts * param.num_elements_horizontal, 2))
    h_surf_itf_j = numpy.zeros((param.num_elements_horizontal + 2, 2, param.num_solpts * param.num_elements_horizontal))

    h_surf_itf_i[0:num_interfaces_horiz, :, 1] = compute_height_from_dist(distance_itf_i.T)
    h_surf_itf_i[1 : num_interfaces_horiz + 1, :, 0] = h_surf_itf_i[0:num_interfaces_horiz, :, 1]

    h_surf_itf_j[0:num_interfaces_horiz, 1, :] = compute_height_from_dist(distance_itf_j)
    h_surf_itf_j[1 : num_interfaces_horiz + 1, 0, :] = h_surf_itf_j[0:num_interfaces_horiz, 1, :]

    # Height derivative along x and y at every grid point
    _, ni, nj = geom.lon.shape
    dhdx1 = numpy.zeros((ni, nj))
    dhdx2 = numpy.zeros((ni, nj))

    offset = 1  # Offset due to the halo
    for elem in range(param.num_elements_horizontal):
        epais = elem * param.num_solpts + numpy.arange(param.num_solpts)

        # --- Direction x1
        dhdx1[:, epais] = h_surf[:, epais] @ mtrx.diff_solpt_tr + h_surf_itf_i[elem + offset, :, :] @ mtrx.correction_tr

        # --- Direction x2
        dhdx2[epais, :] = mtrx.diff_solpt @ h_surf[epais, :] + mtrx.correction @ h_surf_itf_j[elem + offset, :, :]

    return h_surf, h_surf_itf_i, h_surf_itf_j, dhdx1, dhdx2


# def dcmip_advection_orography(geom, metric, mtrx, param):
#   tau     = 12.0 * 86400.0             # period of motion 12 days (s)
#   u0      = 2.0*math.pi*a_ref/tau      # Velocity Magnitude (m/s)
#   T0      = 300.0                      # temperature (K)
#   H       = Rd * T0 / grav             # scale height (m)
#   alpha   = math.pi/6.0                # rotation angle (radians), 30 degrees
#   lambdam = 3.0*math.pi/2.0            # mountain longitude center point (radians)
#   phim    = 0.0                        # mountain latitude center point (radians)
#   h0      = 2000.0                     # peak height of the mountain range (m)
#   Rm      = 3.0*math.pi/4.0            # mountain radius (radians)
#   zetam   = math.pi/16.0               # mountain oscillation half-width (radians)
#   lambdap = math.pi/2.0                # cloud-like tracer longitude center point (radians)
#   phip    = 0.0                        # cloud-like tracer latitude center point (radians)
#   Rp      = mathpi/4.0                 # cloud-like tracer radius (radians)
#   zp1     = 3050.0                     # midpoint of first (lowermost) tracer (m)
#   zp2     = 5050.0                     # midpoint of second tracer (m)
#   zp3     = 8200.0                     # midpoint of third (topmost) tracer (m)
#   dzp1    = 1000.0                     # thickness of first (lowermost) tracer (m)
#   dzp2    = 1000.0                     # thickness of second tracer (m)
#   dzp3    = 400.0                      # thickness of third (topmost) tracer (m)
#   ztop    = 12000.0                     # model top (m)
#
#   return rho, u1_contra, u2_contra, w, theta, q1


# ==========================================================================================
# TEST CASE 2X - IMPACT OF OROGRAPHY ON A NON-ROTATING PLANET
# ==========================================================================================
# The tests in section 2-x examine the impact of 3D Schaer-like circular mountain profiles on an
# atmosphere at rest (2-0), and on flow fields with wind shear (2-1) and without vertical wind shear (2-2).
# A non-rotating planet is used for all configurations. Test 2-0 is conducted on an unscaled regular-size
# planet and primarily examines the accuracy of the pressure gradient calculation in a steady-state
# hydrostatically-balanced atmosphere at rest. This test is especially appealing for models with
# orography-following vertical coordinates. It increases the complexity of test 1-3, that investigated
# the impact of the same Schaer-type orographic profile on the accuracy of purely-horizontal passive
# tracer advection.
#
# Tests 2-1 and 2-2 increase the complexity even further since non-zero flow fields are now prescribed
# with and without vertical wind shear. In order to trigger non-hydrostatic responses the two tests are
# conducted on a reduced-size planet with reduction factor $X=500$ which makes the horizontal and
# vertical grid spacing comparable. This test clearly discriminates between non-hydrostatic and hydrostatic
# models since the expected response is in the non-hydrostatic regime. Therefore, the flow response is
# captured differently by hydrostatic models.


# =========================================================================
# Test 2-0:  Steady-State Atmosphere at Rest in the Presence of Orography
# =========================================================================


def dcmip_steady_state_mountain(geom: CubedSphere3D, metric, mtrx, param):
    T0 = 300.0  # temperature (K)
    gamma = 0.00650  # temperature lapse rate (K/m)
    lambdam = 3.0 * math.pi / 2.0  # mountain longitude center point (radians)
    phim = 0.0  # mountain latitude center point (radians)
    h0 = 2000.0  # peak height of the mountain range (m)
    Rm = 3.0 * math.pi / 4.0  # mountain radius (radians)
    zetam = math.pi / 16.0  # mountain oscillation half-width (radians)

    # -----------------------------------------------------------------------
    #    compute exponents
    # -----------------------------------------------------------------------
    exponent = 0.0
    if gamma != 0:
        exponent = gravity / (Rd * gamma)
        # exponent_rev = 1.0 / exponent # Unused

    # -----------------------------------------------------------------------
    #    Set topography
    # -----------------------------------------------------------------------
    zbot = numpy.zeros(geom.coordVec_latlon.shape[2:])
    zbot_itf_i = numpy.zeros(geom.coordVec_latlon_itf_i.shape[2:])
    zbot_itf_j = numpy.zeros(geom.coordVec_latlon_itf_j.shape[2:])

    for z, coord in zip(
        [zbot, zbot_itf_i, zbot_itf_j], [geom.coordVec_latlon, geom.coordVec_latlon_itf_i, geom.coordVec_latlon_itf_j]
    ):
        lat = coord[1, 0, :, :]
        lon = coord[0, 0, :, :]
        r = numpy.arccos(math.sin(phim) * numpy.sin(lat) + math.cos(phim) * numpy.cos(lat) * numpy.cos(lon - lambdam))
        z[r < Rm] = (
            (h0 / 2.0) * (1.0 + numpy.cos(math.pi * r[r < Rm] / Rm)) * numpy.cos(math.pi * r[r < Rm] / zetam) ** 2
        )  # mountain height

    # Update the geometry object with the new bottom topography
    geom.apply_topography(zbot, zbot_itf_i, zbot_itf_j)
    # And regenerate the metric to take this new topography into account
    metric.build_metric()

    # -----------------------------------------------------------------------
    #    PS (surface pressure)
    # -----------------------------------------------------------------------

    if gamma == 0.0:
        ps = p0 * numpy.exp(-gravity * zbot / (Rd * T0))
    else:
        ps = p0 * (1.0 - gamma / T0 * zbot) ** exponent

    # -----------------------------------------------------------------------
    #    PRESSURE
    # -----------------------------------------------------------------------

    if gamma != 0:
        p = p0 * (1.0 - gamma / T0 * geom.height) ** exponent
    else:
        p = p0 * numpy.exp(-gravity / Rd * geom.height / T0)

    # -----------------------------------------------------------------------
    #    THE VELOCITIES ARE ZERO (STATE AT REST)
    # -----------------------------------------------------------------------

    # Zonal Velocity

    u = 0.0

    # Meridional Velocity

    v = 0.0

    # Vertical Velocity

    w = 0.0

    u1_contra, u2_contra = wind2contra_2d(u, v, geom)

    # -----------------------------------------------------------------------
    #    TEMPERATURE WITH CONSTANT LAPSE RATE
    # -----------------------------------------------------------------------

    t = T0 - gamma * geom.height

    # -----------------------------------------------------------------------
    #    RHO (density)
    # -----------------------------------------------------------------------

    rho = p / (Rd * t)

    # -----------------------------------------------------------------------
    #     initialize Q, set to zero
    # -----------------------------------------------------------------------

    q = 0.0

    # -----------------------------------------------------------------------
    #     initialize TV (virtual temperature)
    # -----------------------------------------------------------------------

    tv = t

    theta = tv * (p0 / p) ** (Rd / cpd)

    return rho, u1_contra, u2_contra, w, theta


def dcmip_schar_waves(geom: CubedSphere3D, metric, mtrx: DFROperators, param: Configuration, shear=False):
    """
    Tests 2-1 and 2-2:  Non-hydrostatic Mountain Waves over a Schaer-type Mountain
    """
    xp = geom.device.xp

    T0 = 300.0  # temperature (K)
    lambdam = math.pi / 4.0  # mountain longitude center point (radians)
    phim = 0.0  # mountain latitude center point (radians)
    h0 = 250.0  # peak height of the mountain range (m)
    Dm = 5000.0  # mountain radius (meters)
    Dxi = 4000.0  # Mountain wavelength (meters)
    Ueq = 20.0  # Reference zonal wind velocity (equator)
    Peq = 100000.0  # Reference surface pressure (Pa)

    ratio = 1.0
    if param.enable_schar_mountain:
        lambdam = param.schar_mountain_longitude
        phim = param.schar_mountain_lattitude
        h0 = param.schar_mountain_height
        Dm = param.schar_mountain_radius
        Dxi = param.schar_mountain_length
        ratio = 0.0 if param.schar_mountain_step != 0 else 1.0


    if shear:
        Cs = 2.5e-4  # Wind shear rate (1/m), for shear case
    else:
        Cs = 0.0

    # -----------------------------------------------------------------------
    #    Set topography
    # -----------------------------------------------------------------------

    # Build topography based on lateral great-circle distance from the mountain center
    def build_topo_old(latlon):
        lat = latlon[1, 0, :, :]
        lon = latlon[0, 0, :, :]
        r = geom.earth_radius * xp.arccos(
            math.sin(phim) * xp.sin(lat) + math.cos(phim) * xp.cos(lat) * xp.cos(lon - lambdam)
        )
        z = xp.zeros(lat.shape)
        z[:, :] = h0 * xp.exp(-(r**2) / Dm**2) * xp.cos(xp.pi * r / Dxi) ** 2
        return z

    def build_topo(latlon):
        lat = latlon[1]
        lon = latlon[0]
        r = geom.earth_radius * xp.arccos(
            math.sin(phim) * xp.sin(lat) + math.cos(phim) * xp.cos(lat) * xp.cos(lon - lambdam)
        )

        return h0 * xp.exp(-(r**2) / Dm**2) * xp.cos(xp.pi * r / Dxi) ** 2

    zbot = build_topo_old(geom.coordVec_latlon)
    zbot_itf_i = build_topo_old(geom.coordVec_latlon_itf_i)
    zbot_itf_j = build_topo_old(geom.coordVec_latlon_itf_j)

    zbot_new = build_topo(geom.get_floor(geom.polar))
    zbot_itf_i_new = build_topo(geom.get_itf_i_floor(geom.polar_itf_i))
    zbot_itf_j_new = build_topo(geom.get_itf_j_floor(geom.polar_itf_j))
    zbot_itf_i_new[geom.floor_west_edge] = 0.0
    zbot_itf_i_new[geom.floor_east_edge] = 0.0
    zbot_itf_j_new[geom.floor_south_edge] = 0.0
    zbot_itf_j_new[geom.floor_north_edge] = 0.0

    diff = zbot_new - geom.to_new_floor(zbot)
    diffn = xp.linalg.norm(diff)

    diffi = zbot_itf_i_new - geom.to_new_itf_i_floor(zbot_itf_i)
    diffin = xp.linalg.norm(diffi)

    diffj = zbot_itf_j_new - geom.to_new_itf_j_floor(zbot_itf_j)
    diffjn = xp.linalg.norm(diffj)

    if diffn > 0.0 or diffin > 0.0 or diffjn > 0.0:
        raise ValueError

    # Update the geometry object with the new bottom topography
    geom.apply_topography(zbot * ratio, zbot_itf_i * ratio, zbot_itf_j * ratio, zbot_new * ratio, zbot_itf_i_new * ratio, zbot_itf_j_new * ratio)
    # And regenerate the metric to take this new topography into account
    metric.build_metric()

    ## Coordinate vectors in 3D

    # lat = geom.coordVec_latlon[1, :, :, :]  # Latitude as 3D field
    # z_3d = geom.coordVec_latlon[2, :, :, :]  # Retrieve all z-levels
    lat = geom.polar[1, ...]  # Latitude as 3D field
    z_3d = geom.polar[2, ...]  # Retrieve all z-levels

    ## Temperature in 3D
    if Ueq != 0:
        T = T0 * (1 - Cs * Ueq**2 / gravity * xp.sin(lat) ** 2)
    else:
        T = T0 * xp.ones_like(lat)

    ### NOTE: These equations are not in exact balance for the no-hill case.
    ### The DCMIP document assumes a shallow-atmosphere discretization,
    ### whereas we have a deep atmosphere.  This case still produces gravity
    ### waves that propagate, but that is overlaid on top of a background
    ### adjustment.

    ## Pressure (eqn 80)
    p = Peq * xp.exp(-(Ueq**2) / (2 * Rd * T0) * xp.sin(lat) ** 2 - gravity * z_3d / (Rd * T))

    # Zonal Velocity (eqn 82)

    u = Ueq * xp.cos(lat) * (2 * T0 / T * Cs * z_3d + T / T0) ** 0.5

    # Meridional Velocity

    v = xp.zeros_like(lat)

    # Vertical Velocity

    w = xp.zeros_like(lat)

    # u1_contra, u2_contra, u3_contra = wind2contra_3d(u, v, w, geom, metric)
    u1_contra, u2_contra, u3_contra = geom.wind2contra(u, v, w, metric)

    # -----------------------------------------------------------------------
    #    RHO (density)
    # -----------------------------------------------------------------------

    rho = p / (Rd * T)

    # -----------------------------------------------------------------------
    #     potential temperature
    # -----------------------------------------------------------------------

    theta = T * (p0 / p) ** (Rd / cpd)

    return rho, u1_contra, u2_contra, u3_contra, theta


def dcmip_schar_damping(
    forcing: numpy.ndarray,
    rho: numpy.ndarray,
    u1: numpy.ndarray,
    u2: numpy.ndarray,
    u3: numpy.ndarray,
    metric: Metric3DTopo,
    geom: CubedSphere3D,
    shear: bool,
    new_layout: bool,
):
    """Implements the required Rayleigh damping for DCMIP cases 2-1 and 2-2

    Parameters:
    -----------
    forcing : numpy.ndarray
       The RHS forcing variable as used by rhs_euler, which will be modified in-place to add
       the required Rayleigh damping.  This variable is in flux form (ρu1, ρu2, etc), so this
       function will calculate the required momentum fluxes.
    rho, u1, u2, u3 : numpy.ndarray
       Input variables at the current timestemp
    metric : Metric3DTopo
       3D metric, used to convert velocities between contravariant and geophysical winds
    geom : CubedSphere3D
       Geometry object, also used for velocity conversion
    shear : bool
       flag for whether the reference velocity field has vertical shear (case 2-2) or not (2-1)"""

    # Grab forcing index variables from 'definitions', since forcing is modified in-place
    from common.definitions import idx_rho_u1, idx_rho_u2, idx_rho_w

    xp = geom.device.xp

    # Case parameters
    T0 = 300.0  # temperature (K)
    Ueq = 20.0  # Reference zonal wind velocity (equator)
    Zh = 20000.0  # Threshold level for Rayleigh damping/sponge layer (m)
    tau0 = 25.0  # Time scale of Rayleigh damping (s)

    if shear:
        Cs = 2.5e-4  # Wind shear rate (1/m), for shear case
    else:
        Cs = 0.0

    # Get coordinates
    if new_layout:
        lat = geom.polar[1, ...]
        z_3d = geom.polar[2, ...]
    else:
        lat = geom.coordVec_latlon[1, :, :, :]  # Latitude as 3D field
        z_3d = geom.coordVec_latlon[2, :, :, :]  # Retrieve all z-levels

    # Build the damping mask (eqn 79), weighted by ρ and τ0^(-1)
    damping_weight = (
        rho / tau0 * xp.sin(xp.pi / 2 * (z_3d - Zh) / (geom.ztop - Zh)) ** 2
    )  # z > zh, defined everywhere at first
    # Reset to 0 below the threshold height
    damping_weight[z_3d <= Zh] = 0.0

    ## Temperature in 3D
    if Ueq != 0:
        Tref = T0 * (1 - Cs * Ueq**2 / gravity * xp.sin(lat) ** 2)
    else:
        Tref = T0

    # Get u, v, w reference velocities and convert to contravariant
    uref = Ueq * xp.cos(lat) * (2 * T0 / Tref * Cs * z_3d + Tref / T0) ** 0.5
    vref = 0.0
    wref = 0.0

    if new_layout:
        u1ref, u2ref, u3ref = geom.wind2contra(uref, vref, wref, metric)
    else:
        u1ref, u2ref, u3ref = wind2contra_3d(uref, vref, wref, geom, metric)

    # Increment velocity forcing (eqn 78).  Take note that this modification is in-place,
    # and the sign is positive because rhs_euler includes its own negative sign
    forcing[idx_rho_u1] += damping_weight * (u1 - u1ref)
    forcing[idx_rho_u2] += damping_weight * (u2 - u2ref)
    forcing[idx_rho_w] += damping_weight * (u3 - u3ref)


# ==========================================================================================
# TEST CASE 3 - GRAVITY WAVES
# ==========================================================================================


def dcmip_gravity_wave(geom: CubedSphere3D, metric: Metric3DTopo, mtrx: DFROperators, param: Configuration):
    """
    Test case 31 - gravity waves

    The non-hydrostatic gravity wave test examines the response of models to short time-scale wavemotion triggered
    by a localized perturbation. The formulation presented in this document is new, but is based on previous
    approaches by Skamarock et al. (JAS 1994), Tomita and Satoh (FDR 2004), and
    Jablonowski et al. (NCAR Tech Report 2008)
    """

    xp = geom.device.xp

    u0 = 20.0  # Reference Velocity
    Teq = 300.0  # Temperature at Equator
    Peq = 100000.0  # Reference PS at Equator
    lambdac = 2.0 * math.pi / 3.0  # Lon of Pert Center
    d = 5000.0  # Width for Pert
    phic = 0.0  # Lat of Pert Center
    delta_theta = 1.0  # Max Amplitude of Pert
    Lz = 20000.0  # Vertical Wavelength of Pert
    N = 0.01  # Brunt-Vaisala frequency
    N2 = N * N  # Brunt-Vaisala frequency Squared
    bigG = (gravity**2) / (N2 * cpd)

    kappa = Rd / cpd
    inv_kappa = cpd / Rd

    # -----------------------------------------------------------------------
    #    THE VELOCITIES
    # -----------------------------------------------------------------------

    # Zonal Velocity

    u = u0 * xp.cos(geom.lat_new)

    # Meridional Velocity

    v = xp.zeros_like(u)

    # Vertical Velocity = Vertical Pressure Velocity = 0

    w = xp.zeros_like(u)

    ## Set a trivial topography
    zbot = xp.zeros_like(geom.coordVec_latlon[0, 0])
    zbot_itf_i = xp.zeros_like(geom.coordVec_latlon_itf_i[0, 0])
    zbot_itf_j = xp.zeros_like(geom.coordVec_latlon_itf_j[0, 0])

    # Update the geometry object with the new bottom topography
    geom.apply_topography(zbot, zbot_itf_i, zbot_itf_j, None, None, None)
    # And regenerate the metric to take this new topography into account
    metric.build_metric()

    # u1_contra, u2_contra = wind2contra_2d(u, v, geom)
    u1_contra, u2_contra = geom.wind2contra_2d(u, v)

    # -----------------------------------------------------------------------
    #    SURFACE TEMPERATURE
    # -----------------------------------------------------------------------

    TS = bigG + (Teq - bigG) * xp.exp(
        -(u0 * N2 / (4.0 * gravity**2))
        * (u0 + 2.0 * geom.rotation_speed * geom.earth_radius)
        * (xp.cos(2.0 * geom.lat_new) - 1.0)
    )

    # -----------------------------------------------------------------------
    #    PS (surface pressure)
    # -----------------------------------------------------------------------

    ps = (
        Peq
        * xp.exp(
            (u0 / (4.0 * bigG * Rd))
            * (u0 + 2.0 * geom.rotation_speed * geom.earth_radius)
            * (xp.cos(2.0 * geom.lat_new) - 1.0)
        )
        * (TS / Teq) ** inv_kappa
    )

    # -----------------------------------------------------------------------
    #    HEIGHT AND PRESSURE AND MEAN TEMPERATURE
    # -----------------------------------------------------------------------

    p = ps * ((bigG / TS) * xp.exp(-N2 * geom.height_new / gravity) + 1.0 - (bigG / TS)) ** inv_kappa

    t_mean = bigG * (1.0 - xp.exp(N2 * geom.height_new / gravity)) + TS * xp.exp(N2 * geom.height_new / gravity)

    theta_base = t_mean * (p0 / p) ** kappa

    # -----------------------------------------------------------------------
    #    rho (density), unperturbed using the background temperature t_mean
    # -----------------------------------------------------------------------

    rho = p / (Rd * t_mean)

    # -----------------------------------------------------------------------
    #    POTENTIAL TEMPERATURE PERTURBATION,
    #    here: converted to temperature and added to the temperature field
    #    models with a prognostic potential temperature field can utilize
    #    the potential temperature perturbation theta_pert directly and add it
    #    to the background theta field (not included here)
    # -----------------------------------------------------------------------

    sin_tmp = xp.sin(geom.lat_new) * math.sin(phic)
    cos_tmp = xp.cos(geom.lat_new) * math.cos(phic)

    # great circle distance with 'a/X'

    r = geom.earth_radius * xp.arccos(sin_tmp + cos_tmp * xp.cos(geom.lon_new - lambdac))

    s = (d**2) / (d**2 + r**2)

    theta_pert = delta_theta * s * xp.sin(2.0 * math.pi * geom.height_new / Lz)
    #   theta_pert = 0. # for debuging

    theta = theta_base + theta_pert

    return rho, u1_contra, u2_contra, w, theta


# =========================================================================
# Test 77:  Acoustic Wave
# =========================================================================


def acoustic_wave(geom: CubedSphere3D, metric: Metric3DTopo):

    xp = geom.device.xp

    T0 = 300.0
    Δp = 100
    eta_v = 1
    re = 6371000
    rc = re / 3
    ztop = 10000

    # -----------------------------------------------------------------------
    #    TEMPERATURE IS CONSTANT 300 K
    # -----------------------------------------------------------------------

    t = T0

    # -----------------------------------------------------------------------
    #    THE VELOCITIES ARE ZERO (STATE AT REST)
    # -----------------------------------------------------------------------

    # Zonal Velocity

    u = 0.0

    # Meridional Velocity

    v = 0.0

    # Vertical Velocity

    w = 0.0

    # ## Set a trivial topography
    # zbot = numpy.zeros(geom.coordVec_latlon.shape[2:])
    # zbot_itf_i = numpy.zeros(geom.coordVec_latlon_itf_i.shape[2:])
    # zbot_itf_j = numpy.zeros(geom.coordVec_latlon_itf_j.shape[2:])
    # # Update the geometry object with the new bottom topography
    # geom.apply_topography(zbot, zbot_itf_i, zbot_itf_j)

    # And regenerate the metric to take this new topography into account
    metric.build_metric()

    # u1_contra, u2_contra = wind2contra_2d(u, v, geom)
    u1_contra, u2_contra = geom.wind2contra_2d(u, v)

    # -----------------------------------------------------------------------
    #    Pressure
    # -----------------------------------------------------------------------

    H = Rd * T0 / gravity
    p_mean = p0 * xp.exp(-geom.height_new / H)

    lat = geom.polar[1, ...]
    lon = geom.polar[0, ...]
    r = re * xp.arccos(xp.cos(lat) * xp.cos(lon))
    f = numpy.where(r > rc, 0.0, (Δp / 2) * (1 + numpy.cos((math.pi * r) / rc)))
    g = numpy.sin((eta_v * math.pi * r) / ztop)
    p_perturb = f * g
    pressure = p_mean + p_perturb
    # pdb.set_trace()

    # Full_u_p = MPI.COMM_WORLD.gather(p_mean, root=0)
    # if MPI.COMM_WORLD.rank == 0:
    #     numpy.save("p_mean2.npy", Full_u_p)

    # -----------------------------------------------------------------------
    #    RHO (density)
    # -----------------------------------------------------------------------

    rho = pressure / (Rd * t)

    # -----------------------------------------------------------------------
    #     initialize TV (virtual temperature)
    # -----------------------------------------------------------------------

    theta = t * (p0 / pressure) ** (Rd / cpd)

    return rho, u1_contra, u2_contra, w, theta
