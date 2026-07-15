import math

import numpy

from ..common.configuration import Configuration
from ..common.definitions import cpd, gravity, p0, Rd
from ..geometry import CubedSphere3D, DFROperators, Metric3DTopo, wind2contra_2d

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


def dcmip_prescribed_rho_theta(geom):
    """Prescribed density and rho*theta for the DCMIP advection tests (1-1 and 1-2).

    The atmosphere is isothermal at T0 = 300 K and hydrostatic, so the pressure, the density and the
    potential temperature are analytic and time independent. DCMIP requires that the dynamic updates
    of the density, temperature and pressure be disabled for these tests, so a step hook restores
    these fields after every step."""
    xp = geom.device.xp

    T0 = 300.0  # temperature
    H = Rd * T0 / gravity  # scale height

    p = p0 * xp.exp(-geom.height_new / H)
    rho = p / (Rd * T0)
    theta = T0 * (p0 / p) ** (Rd / cpd)

    return rho, rho * theta


def dcmip_T11_update_winds(geom, metric, mtrx, param, time=float(0)):
    """
    Test 11 - Deformational Advection

    The 3D deformational flow test is based on the deformational flow test of Nair and Lauritzen (JCP 2010),
    with a prescribed vertical wind velocity which makes the test truly 3D. An unscaled planet
    (with scale parameter X = 1) is selected.

    The velocities are time dependent and therefore must be updated in the dynamical core.
    """

    xp = geom.device.xp

    # Coordinates in the element-wise ("new") memory layout, matching the state vector.
    lon = geom.lon_new
    lat = geom.lat_new
    height = geom.height_new

    tau = 12.0 * 86400.0  # period of motion 12 days
    u0 = (2.0 * math.pi * geom.earth_radius) / tau  # 2 pi a / 12 days
    k0 = (10.0 * geom.earth_radius) / tau  # Velocity Magnitude
    omega0 = (23000.0 * math.pi) / tau  # Velocity Magnitude
    T0 = 300.0  # temperature
    H = Rd * T0 / gravity  # scale height

    p = p0 * xp.exp(-height / H)
    ptop = p0 * math.exp(-param.ztop / H)

    lonp = lon - 2.0 * math.pi * time / tau

    # Shape function
    bs = 0.2
    s = 1.0 + math.exp((ptop - p0) / (bs * ptop)) - xp.exp((p - p0) / (bs * ptop)) - xp.exp((ptop - p) / (bs * ptop))

    # Zonal Velocity

    ud = (
        (omega0 * geom.earth_radius)
        / (bs * ptop)
        * xp.cos(lonp)
        * (xp.cos(lat) ** 2.0)
        * math.cos(2.0 * math.pi * time / tau)
        * (-xp.exp((p - p0) / (bs * ptop)) + xp.exp((ptop - p) / (bs * ptop)))
    )

    u = k0 * xp.sin(lonp) * xp.sin(lonp) * xp.sin(2.0 * lat) * math.cos(math.pi * time / tau) + u0 * xp.cos(lat) + ud

    # Meridional Velocity

    v = k0 * xp.sin(2.0 * lonp) * xp.cos(lat) * math.cos(math.pi * time / tau)

    # Vertical Velocity

    w = -((Rd * T0) / (gravity * p)) * omega0 * xp.sin(lonp) * xp.cos(lat) * math.cos(2.0 * math.pi * time / tau) * s

    # The state vector holds the contravariant components of the wind in the cubed-sphere
    # coordinates, not the (zonal, meridional, vertical) components of the DCMIP document.
    u1_contra, u2_contra, u3_contra = geom.wind2contra(u, v, w, metric)

    return u1_contra, u2_contra, u3_contra


# ==========================================================================================
# TEST CASE 12 - PURE ADVECTION - 3D HADLEY-LIKE FLOW
# ==========================================================================================


def dcmip_T12_update_winds(geom, metric, mtrx, param, time=float(0)):
    """
    Test 12 - 3D Hadley-like flow
    The velocities are time dependent and therefore must be updated in the dynamical core.
    """
    xp = geom.device.xp

    # Coordinates in the element-wise ("new") memory layout, matching the state vector.
    lat = geom.lat_new
    height = geom.height_new

    tau = 86400.0  # period of motion 1 day (in s)
    u0 = 40.0  # Zonal velocity magnitude (m/s)
    w0 = 0.15  # Vertical velocity magnitude (m/s), changed in v5
    T0 = 300.0  # temperature (K)
    H = Rd * T0 / gravity  # scale height
    K = 5.0  # number of Hadley-like cells

    # Height and pressure are aligned (p = p0 exp(-z/H))
    p = p0 * xp.exp(-height / H)

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

    u = u0 * xp.cos(lat)

    # Meridional Velocity

    v = (
        -(rho0 / rho)
        * (geom.earth_radius * w0 * math.pi)
        / (K * param.ztop)
        * xp.cos(lat)
        * xp.sin(K * lat)
        * xp.cos(math.pi * height / param.ztop)
        * math.cos(math.pi * time / tau)
    )

    # Vertical Velocity - can be changed to vertical pressure velocity by
    # omega = -g*rho*w

    w = (
        (rho0 / rho)
        * (w0 / K)
        * (-2.0 * xp.sin(K * lat) * xp.sin(lat) + K * xp.cos(lat) * xp.cos(K * lat))
        * xp.sin(math.pi * height / param.ztop)
        * math.cos(math.pi * time / tau)
    )

    # The state vector holds the contravariant components of the wind in the cubed-sphere
    # coordinates, not the (zonal, meridional, vertical) components of the DCMIP document.
    u1_contra, u2_contra, u3_contra = geom.wind2contra(u, v, w, metric)

    return u1_contra, u2_contra, u3_contra


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

    xp = geom.device.xp

    # The surface is flat for this test (z_s = 0, so Phi_s = 0), but the metric is only
    # token-initialized by its constructor and must still be built explicitly.
    metric.build_metric()

    # Coordinates in the element-wise ("new") memory layout, matching the state vector.
    lon = geom.lon_new
    lat = geom.lat_new
    height = geom.height_new

    p = p0 * xp.exp(-height / H)

    # -----------------------------------------------------------------------
    #    WINDS
    # -----------------------------------------------------------------------

    u1_contra, u2_contra, u3_contra = dcmip_T11_update_winds(geom, metric, mtrx, param, time=0)

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

    # Tracer 1 - Cosine Bells (DCMIP eq. 28-30)

    # Great circle distance to each bell centre, normalized by the Earth radius 'a'
    r1 = xp.arccos(math.sin(phi0) * xp.sin(lat) + math.cos(phi0) * xp.cos(lat) * xp.cos(lon - lambda0))
    r2 = xp.arccos(math.sin(phi1) * xp.sin(lat) + math.cos(phi1) * xp.cos(lat) * xp.cos(lon - lambda1))

    d1 = xp.minimum(1.0, (r1 / RR) ** 2 + ((height - z0) / ZZ) ** 2)
    d2 = xp.minimum(1.0, (r2 / RR) ** 2 + ((height - z0) / ZZ) ** 2)

    q1 = 0.5 * (1.0 + xp.cos(math.pi * d1)) + 0.5 * (1.0 + xp.cos(math.pi * d2))

    # Tracer 2 - Correlated Cosine Bells (DCMIP eq. 31)

    q2 = 0.9 - 0.8 * q1**2

    # Tracer 3 - Slotted Ellipse (DCMIP eq. 32-33): 1 inside either ellipse, 0.1 elsewhere,
    # with a slot cut out above the tracer centre height near the equator.
    q3 = xp.where((d1 <= 0.5) | (d2 <= 0.5), 1.0, 0.1)
    q3 = xp.where((height > z0) & (xp.abs(lat) < 0.125), 0.1, q3)

    # Tracer 4: q4 is chosen so that, in combination with the other three tracer
    #           fields with weight (3/10), the sum is equal to one (DCMIP eq. 34)

    q4 = 1.0 - 0.3 * (q1 + q2 + q3)

    return rho, u1_contra, u2_contra, u3_contra, theta, q1, q2, q3, q4


def dcmip_advection_hadley(geom, metric, mtrx, param):
    """Test 12 - 3D Hadley-like flow"""
    xp = geom.device.xp

    metric.build_metric()

    height = geom.height_new

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
    p = p0 * xp.exp(-height / H)

    # -----------------------------------------------------------------------
    #    WINDS
    # -----------------------------------------------------------------------

    u1_contra, u2_contra, u3_contra = dcmip_T12_update_winds(geom, metric, mtrx, param, time=0)

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

    # Tracer 1 - Layer (DCMIP eq. 39): a cosine bell in the vertical, zero outside [z1, z2]

    q1 = xp.where(
        (height > z1) & (height < z2),
        0.5 * (1.0 + xp.cos(2.0 * math.pi * (height - z0) / (z2 - z1))),
        0.0,
    )

    return rho, u1_contra, u2_contra, u3_contra, theta, q1


# ============================================================================================
# TEST CASE 13 - HORIZONTAL ADVECTION OF THIN CLOUD-LIKE TRACERS IN THE PRESENCE OF OROGRAPHY
# ============================================================================================


def dcmip_advection_orography(geom: CubedSphere3D, metric, mtrx, param):
    """
    Test 13 - Horizontal advection of thin cloud-like tracers in the presence of orography

    Three thin cloud decks are carried once around the sphere, at constant height above mean sea
    level, over a Schar-like mountain with compact support. Because the vertical coordinate follows
    the terrain, a wind that is purely horizontal in physical space still crosses the coordinate
    surfaces: that "perceived" vertical velocity (section 1.3 and Appendix B of the document) is what
    the test is really about. Here it comes out of the metric on its own, since wind2contra converts
    the physical wind (u, v, w = 0) exactly.
    """
    xp = geom.device.xp

    tau = 12.0 * 86400.0  # period of motion 12 days (s)
    u0 = 2.0 * math.pi * geom.earth_radius / tau  # velocity magnitude (m/s)
    T0 = 300.0  # isothermal temperature (K)
    H = Rd * T0 / gravity  # scale height (m)
    alpha = math.pi / 6.0  # advection angle (radians), 30 degrees

    # Mountain (Table XI)
    lambdam = 3.0 * math.pi / 2.0  # mountain longitude center point (radians)
    phim = 0.0  # mountain latitude center point (radians)
    h0 = 2000.0  # peak height of the mountain range (m)
    Rm = 3.0 * math.pi / 4.0  # mountain radius (radians)
    zetam = math.pi / 16.0  # mountain oscillation half-width (radians)

    # Cloud-like tracers (Table XI)
    lambdap = math.pi / 2.0  # cloud longitude center point (radians)
    phip = 0.0  # cloud latitude center point (radians)
    Rp = math.pi / 4.0  # cloud radius (radians)
    zp = (3050.0, 5050.0, 8200.0)  # midpoint of each cloud deck (m)
    dzp = (1000.0, 1000.0, 400.0)  # thickness of each cloud deck (m)

    def surface_height(latlon, large_scale_only=False):
        """
        The Schar-like mountain of DCMIP eqs. 47 and 48, from a (lon, lat) pair of fields.

        The mountain is a smooth bell h*(rm) modulated by a short-wavelength ripple cos²(π rm/ζm).
        With cos² x = (1 + cos 2x)/2, it splits exactly into

            zs = h* cos²(π rm/ζm) = h*/2  +  (h*/2) cos(2π rm/ζm),

        a bell that carries the whole mountain height and a ripple of zero mean. The first term is
        the large-scale part h1 that SLEVE asks for (this is also the choice made by Schar et al.
        2002, eq. 27, for their own two-dimensional version of this mountain), the second is the
        small-scale part h2 that we want to decay quickly with height.
        """
        lon, lat = latlon[0], latlon[1]

        # Great circle distance from the centre of the mountain, in radians.
        rm = xp.arccos(math.sin(phim) * xp.sin(lat) + math.cos(phim) * xp.cos(lat) * xp.cos(lon - lambdam))

        bell = 0.5 * h0 * (1.0 + xp.cos(math.pi * rm / Rm))
        shape = 0.5 if large_scale_only else xp.cos(math.pi * rm / zetam) ** 2

        return xp.where(rm < Rm, bell * shape, 0.0)

    # ------------------------------------------------------------------------------------------
    #     Topography. The vertical coordinate is terrain following, so this has to be in place
    #     before the metric, the heights and therefore the tracers can be computed.
    # ------------------------------------------------------------------------------------------

    zbot_new = surface_height(geom.get_floor(geom.polar))
    zbot_itf_i_new = surface_height(geom.get_itf_i_floor(geom.polar_itf_i))
    zbot_itf_j_new = surface_height(geom.get_itf_j_floor(geom.polar_itf_j))

    zbot = surface_height(geom.coordVec_latlon[:, 0])
    zbot_itf_i = surface_height(geom.coordVec_latlon_itf_i[:, 0])
    zbot_itf_j = surface_height(geom.coordVec_latlon_itf_j[:, 0])

    large_new = surface_height(geom.get_floor(geom.polar), large_scale_only=True)
    large_itf_i_new = surface_height(geom.get_itf_i_floor(geom.polar_itf_i), large_scale_only=True)
    large_itf_j_new = surface_height(geom.get_itf_j_floor(geom.polar_itf_j), large_scale_only=True)

    large = surface_height(geom.coordVec_latlon[:, 0], large_scale_only=True)
    large_itf_i = surface_height(geom.coordVec_latlon_itf_i[:, 0], large_scale_only=True)
    large_itf_j = surface_height(geom.coordVec_latlon_itf_j[:, 0], large_scale_only=True)

    geom.apply_topography(
        zbot,
        zbot_itf_i,
        zbot_itf_j,
        zbot_new,
        zbot_itf_i_new,
        zbot_itf_j_new,
        large,
        large_itf_i,
        large_itf_j,
        large_new,
        large_itf_i_new,
        large_itf_j_new,
    )
    metric.build_metric()

    # Coordinates in the element-wise ("new") layout. The heights follow the terrain, so they are
    # only meaningful once the topography above has been applied.
    lon = geom.lon_new
    lat = geom.lat_new
    height = geom.height_new

    # ------------------------------------------------------------------------------------------
    #     Winds (eqs. 44-46). The flow is horizontal with respect to mean sea level, so the
    #     physical vertical velocity vanishes.
    # ------------------------------------------------------------------------------------------

    u = u0 * (xp.cos(lat) * math.cos(alpha) + xp.sin(lat) * xp.cos(lon) * math.sin(alpha))
    v = -u0 * xp.sin(lon) * math.sin(alpha)
    w = xp.zeros_like(u)

    u1_contra, u2_contra, u3_contra = geom.wind2contra(u, v, w, metric)

    # ------------------------------------------------------------------------------------------
    #     Isothermal atmosphere at rest with respect to the mass field
    # ------------------------------------------------------------------------------------------

    p = p0 * xp.exp(-height / H)
    rho = p / (Rd * T0)
    theta = T0 * (p0 / p) ** (Rd / cpd)

    # ------------------------------------------------------------------------------------------
    #     The three cloud decks (eqs. 49-53), initially away from the mountain
    # ------------------------------------------------------------------------------------------

    # Great circle distance from the centre of the cloud decks, in radians.
    rp = xp.arccos(math.sin(phip) * xp.sin(lat) + math.cos(phip) * xp.cos(lat) * xp.cos(lon - lambdap))
    inside = rp < Rp

    # The lower and medium decks are disk shaped (eq. 51) ...
    def disk(i):
        rz = xp.abs(height - zp[i])
        return xp.where(
            inside & (rz < 0.5 * dzp[i]),
            0.25 * (1.0 + xp.cos(2.0 * math.pi * rz / dzp[i])) * (1.0 + xp.cos(math.pi * rp / Rp)),
            0.0,
        )

    q1 = disk(0)
    q2 = disk(1)

    # ... and the upper one is box shaped (eq. 52)
    q3 = xp.where(inside & (xp.abs(height - zp[2]) < 0.5 * dzp[2]), 1.0, 0.0)

    # The total tracer field (eq. 53)
    q4 = q1 + q2 + q3

    return rho, u1_contra, u2_contra, u3_contra, theta, q1, q2, q3, q4


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

    metric.build_metric()

    T0 = 300.0  # temperature (K)
    Ueq = 20.0  # Reference zonal wind velocity (equator)
    Peq = 100000.0  # Reference surface pressure (Pa)

    if shear:
        Cs = 2.5e-4  # Wind shear rate (1/m), for shear case
    else:
        Cs = 0.0

    ## Coordinate vectors in 3D

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
    from ..common.definitions import idx_rho_u1, idx_rho_u2, idx_rho_w

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
    lat = geom.polar[1, ...]
    z_3d = geom.polar[2, ...]

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

    u1ref, u2ref, u3ref = geom.wind2contra(uref, vref, wref, metric)

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

    # -----------------------------------------------------------------------
    #    RHO (density)
    # -----------------------------------------------------------------------

    rho = pressure / (Rd * t)

    # -----------------------------------------------------------------------
    #     initialize TV (virtual temperature)
    # -----------------------------------------------------------------------

    theta = t * (p0 / pressure) ** (Rd / cpd)

    return rho, u1_contra, u2_contra, w, theta
