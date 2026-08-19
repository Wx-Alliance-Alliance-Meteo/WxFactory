import math

import torch
import xarray as xr
from torch import Tensor

from ..common.configuration import Configuration
from ..common.definitions import Rd, cpd, gravity, p0
from ..geometry import CubedSphere3D, DFROperators, Metric3DTopo
from ..output.input_manager import extract_available_levels
from .vertical_interpolation import vertical_interp

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
    T0 = 300.0  # temperature
    H = Rd * T0 / gravity  # scale height

    p = p0 * torch.exp(-geom.height_new / H)
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

    p = p0 * torch.exp(-height / H)
    ptop = p0 * math.exp(-param.ztop / H)

    lonp = lon - 2.0 * math.pi * time / tau

    # Shape function
    bs = 0.2
    s = (
        1.0
        + math.exp((ptop - p0) / (bs * ptop))
        - torch.exp((p - p0) / (bs * ptop))
        - torch.exp((ptop - p) / (bs * ptop))
    )

    # Zonal Velocity

    ud = (
        (omega0 * geom.earth_radius)
        / (bs * ptop)
        * torch.cos(lonp)
        * (torch.cos(lat) ** 2.0)
        * math.cos(2.0 * math.pi * time / tau)
        * (-torch.exp((p - p0) / (bs * ptop)) + torch.exp((ptop - p) / (bs * ptop)))
    )

    u = (
        k0 * torch.sin(lonp) * torch.sin(lonp) * torch.sin(2.0 * lat) * math.cos(math.pi * time / tau)
        + u0 * torch.cos(lat)
        + ud
    )

    # Meridional Velocity

    v = k0 * torch.sin(2.0 * lonp) * torch.cos(lat) * math.cos(math.pi * time / tau)

    # Vertical Velocity

    w = (
        -((Rd * T0) / (gravity * p))
        * omega0
        * torch.sin(lonp)
        * torch.cos(lat)
        * math.cos(2.0 * math.pi * time / tau)
        * s
    )

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
    p = p0 * torch.exp(-height / H)

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

    u = u0 * torch.cos(lat)

    # Meridional Velocity

    v = (
        -(rho0 / rho)
        * (geom.earth_radius * w0 * math.pi)
        / (K * param.ztop)
        * torch.cos(lat)
        * torch.sin(K * lat)
        * torch.cos(math.pi * height / param.ztop)
        * math.cos(math.pi * time / tau)
    )

    # Vertical Velocity - can be changed to vertical pressure velocity by
    # omega = -g*rho*w

    w = (
        (rho0 / rho)
        * (w0 / K)
        * (-2.0 * torch.sin(K * lat) * torch.sin(lat) + K * torch.cos(lat) * torch.cos(K * lat))
        * torch.sin(math.pi * height / param.ztop)
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

    # The surface is flat for this test (z_s = 0, so Phi_s = 0), but the metric is only
    # token-initialized by its constructor and must still be built explicitly.
    metric.build_metric()

    # Coordinates in the element-wise ("new") memory layout, matching the state vector.
    lon = geom.lon_new
    lat = geom.lat_new
    height = geom.height_new

    p = p0 * torch.exp(-height / H)

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
    r1 = torch.arccos(math.sin(phi0) * torch.sin(lat) + math.cos(phi0) * torch.cos(lat) * torch.cos(lon - lambda0))
    r2 = torch.arccos(math.sin(phi1) * torch.sin(lat) + math.cos(phi1) * torch.cos(lat) * torch.cos(lon - lambda1))

    d1 = torch.clamp((r1 / RR) ** 2 + ((height - z0) / ZZ) ** 2, max=1.0)
    d2 = torch.clamp((r2 / RR) ** 2 + ((height - z0) / ZZ) ** 2, max=1.0)

    q1 = 0.5 * (1.0 + torch.cos(math.pi * d1)) + 0.5 * (1.0 + torch.cos(math.pi * d2))

    # Tracer 2 - Correlated Cosine Bells (DCMIP eq. 31)

    q2 = 0.9 - 0.8 * q1**2

    # Tracer 3 - Slotted Ellipse (DCMIP eq. 32-33): 1 inside either ellipse, 0.1 elsewhere,
    # with a slot cut out above the tracer centre height near the equator.
    q3 = torch.where((d1 <= 0.5) | (d2 <= 0.5), 1.0, 0.1)
    q3 = torch.where((height > z0) & (torch.abs(lat) < 0.125), 0.1, q3)

    # Tracer 4: q4 is chosen so that, in combination with the other three tracer
    #           fields with weight (3/10), the sum is equal to one (DCMIP eq. 34)

    q4 = 1.0 - 0.3 * (q1 + q2 + q3)

    return rho, u1_contra, u2_contra, u3_contra, theta, q1, q2, q3, q4


def dcmip_advection_hadley(geom, metric, mtrx, param):
    """Test 12 - 3D Hadley-like flow"""
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
    p = p0 * torch.exp(-height / H)

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

    q1 = torch.where(
        (height > z1) & (height < z2),
        0.5 * (1.0 + torch.cos(2.0 * math.pi * (height - z0) / (z2 - z1))),
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
    tau = 12.0 * 86400.0  # period of motion 12 days (s)
    u0 = 2.0 * math.pi * geom.earth_radius / tau  # velocity magnitude (m/s)
    T0 = 300.0  # isothermal temperature (K)
    H = Rd * T0 / gravity  # scale height (m)
    alpha = math.pi / 6.0  # advection angle (radians), 30 degrees

    # Mountain (Table XI)
    lambdam = 3.0 * math.pi / 2.0  # mountain longitude center point (radians)
    phim = 0.0  # mountain latitude center point (radians)
    Rm = 3.0 * math.pi / 4.0  # mountain radius (radians)
    zetam = math.pi / 16.0  # mountain oscillation half-width (radians)

    if param.case_number == 201:
        # If z_s = h0 f(r), its physical radial slope is h0 f'(r) / a.
        # The maximum of |f'| for the DCMIP bell and ripple below is
        # 16.004700029593717. Choose h0 so max|dz_s/ds| = tan(70 degrees).
        maximum_normalized_slope = 16.004700029593717
        h0 = math.tan(math.radians(70.0)) * geom.earth_radius / maximum_normalized_slope
    else:
        h0 = 2000.0  # peak height of the DCMIP 2-0-0 mountain (m)

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
        rm = torch.arccos(math.sin(phim) * torch.sin(lat) + math.cos(phim) * torch.cos(lat) * torch.cos(lon - lambdam))

        bell = 0.5 * h0 * (1.0 + torch.cos(math.pi * rm / Rm))
        shape = 0.5 if large_scale_only else torch.cos(math.pi * rm / zetam) ** 2

        return torch.where(rm < Rm, bell * shape, 0.0)

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

    u = u0 * (torch.cos(lat) * math.cos(alpha) + torch.sin(lat) * torch.cos(lon) * math.sin(alpha))
    v = -u0 * torch.sin(lon) * math.sin(alpha)
    w = torch.zeros_like(u)

    u1_contra, u2_contra, u3_contra = geom.wind2contra(u, v, w, metric)

    # ------------------------------------------------------------------------------------------
    #     Isothermal atmosphere at rest with respect to the mass field
    # ------------------------------------------------------------------------------------------

    p = p0 * torch.exp(-height / H)
    rho = p / (Rd * T0)
    theta = T0 * (p0 / p) ** (Rd / cpd)

    # ------------------------------------------------------------------------------------------
    #     The three cloud decks (eqs. 49-53), initially away from the mountain
    # ------------------------------------------------------------------------------------------

    # Great circle distance from the centre of the cloud decks, in radians.
    rp = torch.arccos(math.sin(phip) * torch.sin(lat) + math.cos(phip) * torch.cos(lat) * torch.cos(lon - lambdap))
    inside = rp < Rp

    # The lower and medium decks are disk shaped (eq. 51) ...
    def disk(i):
        rz = torch.abs(height - zp[i])
        return torch.where(
            inside & (rz < 0.5 * dzp[i]),
            0.25 * (1.0 + torch.cos(2.0 * math.pi * rz / dzp[i])) * (1.0 + torch.cos(math.pi * rp / Rp)),
            0.0,
        )

    q1 = disk(0)
    q2 = disk(1)

    # ... and the upper one is box shaped (eq. 52)
    q3 = torch.where(inside & (torch.abs(height - zp[2]) < 0.5 * dzp[2]), 1.0, 0.0)

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
    """DCMIP-2012 test 2-0: hydrostatic atmosphere at rest.

    The same atmospheric state is used for the flat control and the terrain
    run.  With terrain enabled, geometric height follows the compact,
    moderately steep mountain from equations 63--64 of the specification.
    """
    T0 = 300.0  # surface temperature (K)
    gamma = 0.0065  # temperature lapse rate (K/m)
    lambdam = 3.0 * math.pi / 2.0  # mountain longitude center point (radians)
    phim = 0.0  # mountain latitude center point (radians)
    h0 = 2000.0  # peak height of the mountain range (m)
    Rm = 3.0 * math.pi / 4.0  # mountain radius (radians)
    zetam = math.pi / 16.0  # mountain oscillation half-width (radians)

    def surface_height(latlon, large_scale_only=False):
        lon, lat = latlon[0], latlon[1]
        cosine = math.sin(phim) * torch.sin(lat) + math.cos(phim) * torch.cos(lat) * torch.cos(lon - lambdam)
        # Clamp protects arccos from a one-ulp overshoot at the mountain centre.
        rm = torch.arccos(torch.clamp(cosine, -1.0, 1.0))
        bell = 0.5 * h0 * (1.0 + torch.cos(math.pi * rm / Rm))
        shape = 0.5 if large_scale_only else torch.cos(math.pi * rm / zetam) ** 2
        return torch.where(rm < Rm, bell * shape, 0.0)

    # 200 and 201 are mountain cases. 202 is the deliberately non-standard
    # flat control; legacy case number 20 retains its old meaning.
    if param.case_number in (20, 200, 201):
        zbot_new = surface_height(geom.get_floor(geom.polar))
        zbot_itf_i_new = surface_height(geom.get_itf_i_floor(geom.polar_itf_i))
        zbot_itf_j_new = surface_height(geom.get_itf_j_floor(geom.polar_itf_j))
        zbot = surface_height(geom.coordVec_latlon[:, 0])
        zbot_itf_i = surface_height(geom.coordVec_latlon_itf_i[:, 0])
        zbot_itf_j = surface_height(geom.coordVec_latlon_itf_j[:, 0])

        # Preserve the analytic bell/ripple split when SLEVE is selected.
        large_new = surface_height(geom.get_floor(geom.polar), True)
        large_itf_i_new = surface_height(geom.get_itf_i_floor(geom.polar_itf_i), True)
        large_itf_j_new = surface_height(geom.get_itf_j_floor(geom.polar_itf_j), True)
        large = surface_height(geom.coordVec_latlon[:, 0], True)
        large_itf_i = surface_height(geom.coordVec_latlon_itf_i[:, 0], True)
        large_itf_j = surface_height(geom.coordVec_latlon_itf_j[:, 0], True)

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
    else:  # case 202
        geom.apply_topography(None, None, None, None, None, None)
    metric.build_metric()

    height = geom.height_new
    exponent = gravity / (Rd * gamma)
    temperature = T0 - gamma * height
    pressure = p0 * (1.0 - gamma * height / T0) ** exponent
    rho = pressure / (Rd * temperature)
    theta = temperature * (p0 / pressure) ** (Rd / cpd)

    zero = torch.zeros_like(rho)
    return rho, zero, zero, zero, theta


def dcmip_schar_waves(geom: CubedSphere3D, metric, mtrx: DFROperators, param: Configuration, shear=False):
    """
    Tests 2-1 and 2-2:  Non-hydrostatic Mountain Waves over a Schaer-type Mountain
    """

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
        T = T0 * (1 - Cs * Ueq**2 / gravity * torch.sin(lat) ** 2)
    else:
        T = T0 * torch.ones_like(lat)

    ### NOTE: These equations are not in exact balance for the no-hill case.
    ### The DCMIP document assumes a shallow-atmosphere discretization,
    ### whereas we have a deep atmosphere.  This case still produces gravity
    ### waves that propagate, but that is overlaid on top of a background
    ### adjustment.

    ## Pressure (eqn 80)
    p = Peq * torch.exp(-(Ueq**2) / (2 * Rd * T0) * torch.sin(lat) ** 2 - gravity * z_3d / (Rd * T))

    # Zonal Velocity (eqn 82)

    u = Ueq * torch.cos(lat) * (2 * T0 / T * Cs * z_3d + T / T0) ** 0.5

    # Meridional Velocity

    v = torch.zeros_like(lat)

    # Vertical Velocity

    w = torch.zeros_like(lat)

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


def dcmip_schar_damping_coeffs(metric: Metric3DTopo, geom: CubedSphere3D, shear: bool):
    """Frozen (state-independent) coefficients of the DCMIP 2-1/2-2 Rayleigh sponge.

    The sponge forcing is ``rho * mask/tau0 * (u^i - u^i_ref)``; the mask (a function of height only)
    and the reference contravariant velocities depend on the geometry, not on the state. Returns
    ``(rate, u1ref, u2ref, u3ref)`` with ``rate = mask/tau0``, so the forcing is
    ``rho * rate * (u^i - u^i_ref)``. This lets both the forcing itself and its analytic Jacobian
    share one definition of the sponge."""
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

    # Build the damping mask (eqn 79), weighted by tau0^(-1); the rho weighting is applied by the caller
    rate = torch.sin(torch.pi / 2 * (z_3d - Zh) / (geom.ztop - Zh)) ** 2 / tau0  # z > zh, everywhere at first
    # Reset to 0 below the threshold height
    rate[z_3d <= Zh] = 0.0

    ## Temperature in 3D
    if Ueq != 0:
        Tref = T0 * (1 - Cs * Ueq**2 / gravity * torch.sin(lat) ** 2)
    else:
        Tref = T0

    # Get u, v, w reference velocities and convert to contravariant
    uref = Ueq * torch.cos(lat) * (2 * T0 / Tref * Cs * z_3d + Tref / T0) ** 0.5
    vref = 0.0
    wref = 0.0

    u1ref, u2ref, u3ref = geom.wind2contra(uref, vref, wref, metric)

    return rate, u1ref, u2ref, u3ref


def dcmip_schar_damping(
    forcing: Tensor,
    rho: Tensor,
    u1: Tensor,
    u2: Tensor,
    u3: Tensor,
    metric: Metric3DTopo,
    geom: CubedSphere3D,
    shear: bool,
):
    """Implements the required Rayleigh damping for DCMIP cases 2-1 and 2-2

    Parameters:
    -----------
    forcing : Tensor
       The RHS forcing variable as used by rhs_euler, which will be modified in-place to add
       the required Rayleigh damping.  This variable is in flux form (ρu1, ρu2, etc), so this
       function will calculate the required momentum fluxes.
    rho, u1, u2, u3 : Tensor
       Input variables at the current timestemp
    metric : Metric3DTopo
       3D metric, used to convert velocities between contravariant and geophysical winds
    geom : CubedSphere3D
       Geometry object, also used for velocity conversion
    shear : bool
       flag for whether the reference velocity field has vertical shear (case 2-2) or not (2-1)"""

    # Grab forcing index variables from 'definitions', since forcing is modified in-place
    from ..common.definitions import idx_rho_u1, idx_rho_u2, idx_rho_u3

    rate, u1ref, u2ref, u3ref = dcmip_schar_damping_coeffs(metric, geom, shear)
    damping_weight = rho * rate  # eqn 79, weighted by rho and tau0^(-1)

    # Increment velocity forcing (eqn 78).  Take note that this modification is in-place,
    # and the sign is positive because rhs_euler includes its own negative sign
    forcing[idx_rho_u1] += damping_weight * (u1 - u1ref)
    forcing[idx_rho_u2] += damping_weight * (u2 - u2ref)
    forcing[idx_rho_u3] += damping_weight * (u3 - u3ref)


# ==========================================================================================
# TEST CASE 3 - GRAVITY WAVES
# ==========================================================================================


def dcmip_gravity_wave(geom: CubedSphere3D, metric: Metric3DTopo, mtrx: DFROperators, param: Configuration):
    """DCMIP-2012 test 3-1: non-hydrostatic gravity wave.

    A localized potential-temperature perturbation is superposed on a
    hydrostatic, gradient-wind-balanced state on an X=125 non-rotating
    planet. Density deliberately uses the unperturbed temperature, as
    required by version 3 of the reference initial-condition routine.
    """

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

    # Test 3-1 has a flat lower boundary. Reapplying it explicitly keeps the
    # geometry and metric initialization order consistent with terrain cases.
    geom.apply_topography(None, None, None, None, None, None)
    metric.build_metric()

    u = u0 * torch.cos(geom.lat_new)
    v = torch.zeros_like(u)
    w = torch.zeros_like(u)
    u1_contra, u2_contra, u3_contra = geom.wind2contra(u, v, w, metric)

    # -----------------------------------------------------------------------
    #    SURFACE TEMPERATURE
    # -----------------------------------------------------------------------

    TS = bigG + (Teq - bigG) * torch.exp(
        -(u0 * N2 / (4.0 * gravity**2))
        * (u0 + 2.0 * geom.rotation_speed * geom.earth_radius)
        * (torch.cos(2.0 * geom.lat_new) - 1.0)
    )

    # -----------------------------------------------------------------------
    #    PS (surface pressure)
    # -----------------------------------------------------------------------

    ps = (
        Peq
        * torch.exp(
            (u0 / (4.0 * bigG * Rd))
            * (u0 + 2.0 * geom.rotation_speed * geom.earth_radius)
            * (torch.cos(2.0 * geom.lat_new) - 1.0)
        )
        * (TS / Teq) ** inv_kappa
    )

    # -----------------------------------------------------------------------
    #    HEIGHT AND PRESSURE AND MEAN TEMPERATURE
    # -----------------------------------------------------------------------

    p = ps * ((bigG / TS) * torch.exp(-N2 * geom.height_new / gravity) + 1.0 - (bigG / TS)) ** inv_kappa

    t_mean = bigG * (1.0 - torch.exp(N2 * geom.height_new / gravity)) + TS * torch.exp(N2 * geom.height_new / gravity)

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

    sin_tmp = torch.sin(geom.lat_new) * math.sin(phic)
    cos_tmp = torch.cos(geom.lat_new) * math.cos(phic)

    # great circle distance with 'a/X'

    r = geom.earth_radius * torch.arccos(sin_tmp + cos_tmp * torch.cos(geom.lon_new - lambdac))

    s = (d**2) / (d**2 + r**2)

    theta_pert = delta_theta * s * torch.sin(2.0 * math.pi * geom.height_new / Lz)

    theta = theta_base + theta_pert

    return rho, u1_contra, u2_contra, u3_contra, theta


# ==========================================================================================
# TEST CASE 41 - Dry Baroclinic Instability
# ==========================================================================================


def dcmip_baroclinic_instability(
    geom: CubedSphere3D,
    metric: Metric3DTopo,
    mtrx: DFROperators,
    param: Configuration,
):
    """
    DCMIP-2012 Test 4-1-x: Dry Baroclinic Instability.

    The analytic initial condition is originally specified in the
    pressure-based vertical coordinate

        eta = p / ps,

    with ps = p0 = 1000 hPa.

    Since this model uses height as the vertical coordinate, eta is
    obtained at each model point by solving

        -g z + Phi(lon, lat, eta) = 0

    with the Newton iteration prescribed in DCMIP Appendix F.5.

    The basic state consists of two midlatitude zonal jets in balanced
    hydrostatic/gradient-wind equilibrium. A localized 1 m/s zonal-wind
    perturbation triggers the baroclinic instability.
    """

    # ------------------------------------------------------------------
    # DCMIP constants
    # ------------------------------------------------------------------

    eta_tropo = 0.2
    eta0 = 0.252

    u0 = 35.0  # m/s
    up = 1.0  # m/s

    T0 = 288.0  # K
    delta_T = 4.8e5  # K
    gamma = 0.005  # K/m

    lambdac = math.pi / 9.0  # 20 degrees E
    phic = 2.0 * math.pi / 9.0  # 40 degrees N

    eta_sfc = 1.0

    # Use the model's physical/scaled constants.
    # This is important for the small-planet versions 4-1-1,4-1-2 and 4-1-3:
    #     a     = a_ref / X
    #     Omega = Omega_ref * X
    a = geom.earth_radius
    omega = geom.rotation_speed

    # p0 imported from common.definitions is assumed to be SI pressure
    # (100000 Pa = 1000 hPa).
    # Do NOT redefine p0 = 1000 here, since rho = p/(Rd*T) requires pressure in Pa.
    p_ref = p0

    exponent = Rd * gamma / gravity

    # ==================================================================
    # Horizontal-mean geopotential
    # DCMIP equations (125)-(127)
    # ==================================================================

    def horiz_mean_geopotential(eta):
        """
        Horizontal-mean geopotential Phi_bar(eta).
        """

        phi_mean = T0 * gravity / gamma * (1.0 - eta**exponent)

        # Eq. (127)
        delta_phi = (
            Rd
            * delta_T
            * (
                (torch.log(eta / eta_tropo) + 137.0 / 60.0) * eta_tropo**5
                - 5.0 * eta_tropo**4 * eta
                + 5.0 * eta_tropo**3 * eta**2
                - (10.0 / 3.0) * eta_tropo**2 * eta**3
                + (5.0 / 4.0) * eta_tropo * eta**4
                - (1.0 / 5.0) * eta**5
            )
        )

        # Eq. (125) below tropopause in pressure coordinate,
        # Eq. (126) above tropopause.
        return torch.where(
            eta < eta_tropo,
            phi_mean - delta_phi,
            phi_mean,
        )

    # ==================================================================
    # Full 3-D geopotential
    #
    # DCMIP equation (124)
    # ==================================================================

    def geopotential(lon, lat, eta):
        """
        Full geopotential Phi(lon, lat, eta).

        The DCMIP analytic expression has no explicit longitude
        dependence, but lon is kept in the argument list for
        consistency with the z -> eta inversion.
        """

        del lon  # longitude does not appear explicitly in Eq. (124)

        eta_v = (eta - eta0) * 0.5 * math.pi

        cos_eta = torch.cos(eta_v)

        jet_factor = u0 * cos_eta**1.5

        sin_lat = torch.sin(lat)
        cos_lat = torch.cos(lat)

        horizontal_1 = -2.0 * sin_lat**6 * (cos_lat**2 + 1.0 / 3.0) + 10.0 / 63.0

        horizontal_2 = (8.0 / 5.0) * cos_lat**3 * (sin_lat**2 + 2.0 / 3.0) - math.pi / 4.0

        phi_deviation = jet_factor * (horizontal_1 * jet_factor + horizontal_2 * a * omega)

        return horiz_mean_geopotential(eta) + phi_deviation

    # ==================================================================
    # Temperature
    # DCMIP equations (120)-(122)
    # ==================================================================

    def temperature(lon, lat, eta):
        """
        Return

            T_total(lon,lat,eta), T_mean(eta)

        where T_total is the temperature that must be used to
        initialize rho and theta.
        """

        del lon  # Eq. (120) has no explicit longitude dependence

        eta_v = (eta - eta0) * 0.5 * math.pi

        sin_eta = torch.sin(eta_v)
        cos_eta = torch.cos(eta_v)

        sin_lat = torch.sin(lat)
        cos_lat = torch.cos(lat)

        horizontal_1 = -2.0 * sin_lat**6 * (cos_lat**2 + 1.0 / 3.0) + 10.0 / 63.0

        horizontal_2 = (8.0 / 5.0) * cos_lat**3 * (sin_lat**2 + 2.0 / 3.0) - math.pi / 4.0

        # Eq. (120)
        factor = eta * math.pi * u0 / Rd

        t_deviation = (
            0.75 * factor * sin_eta * cos_eta**0.5 * (horizontal_1 * 2.0 * u0 * cos_eta**1.5 + horizontal_2 * a * omega)
        )

        # Eq. (121)
        t_lower = T0 * eta**exponent

        # Eq. (122)
        t_upper = T0 * eta**exponent + delta_T * (eta_tropo - eta) ** 5

        t_mean = torch.where(
            eta < eta_tropo,
            t_upper,
            t_lower,
        )

        t_total = t_mean + t_deviation

        return t_total, t_mean

    # ==================================================================
    # Surface geopotential / lower boundary
    # DCMIP equation (128)
    # ==================================================================
    # This test DOES NOT have a flat lower boundary. Phi_s is obtained by evaluating Eq. (124) at eta = 1:
    #     z_s = Phi_s / g
    # The surface geopotential is necessary to balance the non-zero zonal wind at the surface.
    # ==================================================================

    def surface_height(latlon):
        """
        Surface elevation z_s = Phi_s / g.
        """

        lon_s = latlon[0]
        lat_s = latlon[1]

        eta_surface = torch.ones_like(lat_s) * eta_sfc

        phi_surface = geopotential(
            lon_s,
            lat_s,
            eta_surface,
        )

        return phi_surface / gravity

    # ------------------------------------------------------------------
    # Evaluate the surface elevation on every geometry representation.
    # ------------------------------------------------------------------

    zbot_new = surface_height(geom.get_floor(geom.polar))

    zbot_itf_i_new = surface_height(geom.get_itf_i_floor(geom.polar_itf_i))

    zbot_itf_j_new = surface_height(geom.get_itf_j_floor(geom.polar_itf_j))

    zbot = surface_height(geom.coordVec_latlon[:, 0])

    zbot_itf_i = surface_height(geom.coordVec_latlon_itf_i[:, 0])

    zbot_itf_j = surface_height(geom.coordVec_latlon_itf_j[:, 0])

    # There is no separate small-scale topographic component here.
    # Treat the whole balanced surface geopotential as the large-scale
    # surface when SLEVE-type coordinates are used.

    geom.apply_topography(
        zbot,
        zbot_itf_i,
        zbot_itf_j,
        zbot_new,
        zbot_itf_i_new,
        zbot_itf_j_new,
    )

    # IMPORTANT: The metric and physical heights must be rebuilt BEFORE solving z -> eta.
    metric.build_metric()

    # ------------------------------------------------------------------
    # Coordinates in the element-wise layout used by the state vector
    # ------------------------------------------------------------------

    lon = geom.lon_new
    lat = geom.lat_new
    z = geom.height_new

    # ==================================================================
    # Convert height z -> eta
    # DCMIP Appendix F.5, equations (245)-(247)
    # ==================================================================

    def eta_from_z(lon, lat, z):
        """
        Vectorized Newton solution of

            F(eta) = -g*z + Phi(lon,lat,eta) = 0.

        DCMIP specifies eta_0 = 1e-7 for every grid point.
        """

        eta_val = torch.full_like(
            z,
            1.0e-7,
        )

        convergence = 1.0e-14
        max_iterations = 26

        converged = False

        for _ in range(max_iterations):

            phi = geopotential(
                lon,
                lat,
                eta_val,
            )

            temp, _ = temperature(
                lon,
                lat,
                eta_val,
            )

            # Eq. (246)
            f = -gravity * z + phi

            # Eq. (247)
            df = -(Rd / eta_val) * temp

            # Eq. (245)
            eta_new = eta_val - f / df

            error = torch.max(torch.abs(eta_new - eta_val))

            eta_val = eta_new

            if error.item() <= convergence:
                converged = True
                break

        if not converged:
            raise ValueError(
                "DCMIP 4-1: z -> eta Newton iteration did not converge. " f"Maximum |delta eta| = {error.item():.6e}"
            )

        return eta_val

    eta = eta_from_z(
        lon,
        lat,
        z,
    )

    # ==================================================================
    # Velocity
    # DCMIP equations (117)-(119)
    # ==================================================================

    sin_tmp = math.sin(phic) * torch.sin(lat)
    cos_tmp = math.cos(phic) * torch.cos(lat)

    # Great-circle angular distance.
    # DCMIP:
    #     r_phys = a * acos(...)
    #     R      = a / 10
    # Therefore:
    #     (r_phys/R)^2 = (10*r_angle)^2
    #
    acos_arg = sin_tmp + cos_tmp * torch.cos(lon - lambdac)

    acos_arg = torch.clamp(
        acos_arg,
        -1.0,
        1.0,
    )

    r_angle = torch.arccos(acos_arg)

    # Localized perturbation in Eq. (117)
    u_perturb = up * torch.exp(-((10.0 * r_angle) ** 2))

    # Basic zonal jet
    eta_v = (eta - eta0) * 0.5 * math.pi

    u_wind = u0 * torch.cos(eta_v) ** 1.5 * torch.sin(2.0 * lat) ** 2

    u = u_wind + u_perturb

    # Eq. (118)
    v = torch.zeros_like(u)
    w = torch.zeros_like(u)

    # Convert physical zonal/meridional/vertical winds to the
    # contravariant velocity components used by the state vector.
    u1_contra, u2_contra, u3_contra = geom.wind2contra(
        u,
        v,
        w,
        metric,
    )

    # ==================================================================
    # Thermodynamic state
    # ==================================================================

    # DCMIP:
    #     p = eta * p0
    # p_ref is in Pa, so p is also in Pa.
    p = p_ref * eta

    T, _ = temperature(
        lon,
        lat,
        eta,
    )

    # Eq. (123) / Appendix F.5 Eq. (249)
    rho = p / (Rd * T)

    # Appendix F.5 Eq. (250)
    theta = T * (p_ref / p) ** (Rd / cpd)

    return (
        rho,
        u1_contra,
        u2_contra,
        u3_contra,
        theta,
    )


# =========================================================================
# Test 77:  Acoustic Wave
# =========================================================================


def acoustic_wave(geom: CubedSphere3D, metric: Metric3DTopo):

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
    p_mean = p0 * torch.exp(-geom.height_new / H)

    lat = geom.polar[1, ...]
    lon = geom.polar[0, ...]
    r = re * torch.arccos(torch.cos(lat) * torch.cos(lon))
    f = torch.where(r > rc, 0.0, (Δp / 2) * (1 + torch.cos((math.pi * r) / rc)))
    g = torch.sin((eta_v * math.pi * r) / ztop)
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


def euler_from_era5(geom: CubedSphere3D, metric: Metric3DTopo, filename: str, time: str):

    metric.build_metric()

    ds = xr.open_zarr(filename, consolidated=True).isel(time=[0])
    feature_map = {str(f): i for i, f in enumerate(ds["features"].values)}
    levels = extract_available_levels(ds)
    levels.reverse()

    def get_feature_ids(name: str):
        "For a given feature base name, get the list of all level IDs for that feature"
        return [feature_map[f"{name}_h{l}"] for l in levels]

    # Get raw ERA5 data
    geo_era = ds["data"].isel(features=get_feature_ids("geopotential"))
    u_wind_era = ds["data"].isel(features=get_feature_ids("u_component_of_wind"))
    v_wind_era = ds["data"].isel(features=get_feature_ids("v_component_of_wind"))
    w_wind_era = ds["data"].isel(features=get_feature_ids("vertical_velocity"))
    temp_era = ds["data"].isel(features=get_feature_ids("temperature"))

    target_lon = geom.context.to_host(geom.lon * 180 / math.pi).reshape(-1)
    target_lat = geom.context.to_host(geom.lat * 180 / math.pi).reshape(-1)

    shape = (len(levels),) + geom.lon.shape

    def horizontal_interp(a: xr.DataArray):
        "Interpolate horizontally to the (block) cubed-sphere grid points."
        cs_lin = a.interp(longitude=("points", target_lon), latitude=("points", target_lat), method="linear").values
        return geom.context.tensor(cs_lin.reshape(shape))

    # Compute height from geopotential at every grid point
    geo_cs = horizontal_interp(geo_era)
    source_heights = geo_cs / gravity
    target_heights = geom.to_single_block(geom.polar[2])

    def era_to_cs(a: xr.DataArray):
        """Convert an ERA 5 field to the current cubed-sphere (CS) grid
        - Interpolate horizontally to the block-shape CS grid
        - Interpolate vertically to the new grid levels
        - Reshape the array into its final memory layout (by element)."""
        return geom._to_new(vertical_interp(horizontal_interp(a), source_heights, target_heights, interp_type="cubic"))

    u_wind_cs = era_to_cs(u_wind_era)
    v_wind_cs = era_to_cs(v_wind_era)
    w_wind_cs = era_to_cs(w_wind_era)
    temp_cs = era_to_cs(temp_era)

    # Compute state variables
    H = Rd * temp_cs / gravity  # scale height
    p = p0 * torch.exp(-geom.height_new / H)
    rho = p / (Rd * temp_cs)
    theta = temp_cs * (p0 / p) ** (Rd / cpd)
    u1, u2, u3 = geom.wind2contra(u_wind_cs, v_wind_cs, w_wind_cs, metric)

    return rho, u1, u2, u3, theta
