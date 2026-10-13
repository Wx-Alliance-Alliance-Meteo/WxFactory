import math

import torch

from ..common.configuration import Configuration
from ..common.definitions import Rd, cpd, gravity, p0
from ..geometry import CubedSphere3D, DFROperators, Metric5D


def dcmip_prescribed_rho_theta(geom):
    """The prescribed advection tests keep an isothermal, hydrostatic mass field fixed."""
    T0 = 300.0
    H = Rd * T0 / gravity

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

    lon = geom.lon_new
    lat = geom.lat_new
    height = geom.height_new

    tau = 12.0 * 86400.0
    u0 = (2.0 * math.pi * geom.earth_radius) / tau
    k0 = (10.0 * geom.earth_radius) / tau
    omega0 = (23000.0 * math.pi) / tau
    T0 = 300.0
    H = Rd * T0 / gravity

    p = p0 * torch.exp(-height / H)
    ptop = p0 * math.exp(-param.ztop / H)

    lonp = lon - 2.0 * math.pi * time / tau

    bs = 0.2
    s = (
        1.0
        + math.exp((ptop - p0) / (bs * ptop))
        - torch.exp((p - p0) / (bs * ptop))
        - torch.exp((ptop - p) / (bs * ptop))
    )


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


    v = k0 * torch.sin(2.0 * lonp) * torch.cos(lat) * math.cos(math.pi * time / tau)


    w = (
        -((Rd * T0) / (gravity * p))
        * omega0
        * torch.sin(lonp)
        * torch.cos(lat)
        * math.cos(2.0 * math.pi * time / tau)
        * s
    )

    u1_contra, u2_contra, u3_contra = geom.wind2contra(u, v, w, metric)

    return u1_contra, u2_contra, u3_contra


def dcmip_T12_update_winds(geom, metric, mtrx, param, time=float(0)):
    """
    Test 12 - 3D Hadley-like flow
    The velocities are time dependent and therefore must be updated in the dynamical core.
    """
    lat = geom.lat_new
    height = geom.height_new

    tau = 86400.0
    u0 = 40.0
    w0 = 0.15  # Vertical velocity magnitude (m/s), changed in v5
    T0 = 300.0
    H = Rd * T0 / gravity
    K = 5.0

    p = p0 * torch.exp(-height / H)


    t = T0


    rho = p / (Rd * t)
    rho0 = p0 / (Rd * t)


    u = u0 * torch.cos(lat)


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

    u1_contra, u2_contra, u3_contra = geom.wind2contra(u, v, w, metric)

    return u1_contra, u2_contra, u3_contra


def dcmip_advection_deformation(geom, metric, mtrx, param):
    """
    Test 11 - Deformational Advection

    The 3D deformational flow test is based on the deformational flow test of Nair and Lauritzen (JCP 2010), with a prescribed vertical wind velocity which makes the test truly 3D. An unscaled planet (with scale parameter X = 1) is selected.
    """
    T0 = 300.0
    H = Rd * T0 / gravity
    RR = 1.0 / 2.0  # horizontal half width divided by 'a'
    ZZ = 1000.0  # vertical half width
    z0 = 5000.0
    lambda0 = 5.0 * math.pi / 6.0
    lambda1 = 7.0 * math.pi / 6.0
    phi0 = 0.0
    phi1 = 0.0


    # The surface is flat for this test (z_s = 0, so Phi_s = 0), but the metric is only
    # token-initialized by its constructor and must still be built explicitly.
    metric.build_metric()

    lon = geom.lon_new
    lat = geom.lat_new
    height = geom.height_new

    p = p0 * torch.exp(-height / H)


    u1_contra, u2_contra, u3_contra = dcmip_T11_update_winds(geom, metric, mtrx, param, time=0)


    t = T0


    rho = p / (Rd * t)


    tv = t
    theta = tv * (p0 / p) ** (Rd / cpd)


    # Tracer 1 - Cosine Bells (DCMIP eq. 28-30)

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

    T0 = 300.0
    H = Rd * T0 / gravity
    z1 = 2000.0  # position of lower tracer bound (m), changed in v5
    z2 = 5000.0  # position of upper tracer bound (m), changed in v5
    z0 = 0.5 * (z1 + z2)


    p = p0 * torch.exp(-height / H)


    u1_contra, u2_contra, u3_contra = dcmip_T12_update_winds(geom, metric, mtrx, param, time=0)


    t = T0


    rho = p / (Rd * t)

    tv = t
    theta = tv * (p0 / p) ** (Rd / cpd)


    # Tracer 1 - Layer (DCMIP eq. 39): a cosine bell in the vertical, zero outside [z1, z2]

    q1 = torch.where(
        (height > z1) & (height < z2),
        0.5 * (1.0 + torch.cos(2.0 * math.pi * (height - z0) / (z2 - z1))),
        0.0,
    )

    return rho, u1_contra, u2_contra, u3_contra, theta, q1


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
    tau = 12.0 * 86400.0
    u0 = 2.0 * math.pi * geom.earth_radius / tau
    T0 = 300.0
    H = Rd * T0 / gravity
    alpha = math.pi / 6.0  # advection angle (radians), 30 degrees

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
        h0 = 2000.0

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

        rm = torch.arccos(math.sin(phim) * torch.sin(lat) + math.cos(phim) * torch.cos(lat) * torch.cos(lon - lambdam))

        bell = 0.5 * h0 * (1.0 + torch.cos(math.pi * rm / Rm))
        shape = 0.5 if large_scale_only else torch.cos(math.pi * rm / zetam) ** 2

        return torch.where(rm < Rm, bell * shape, 0.0)


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

    lon = geom.lon_new
    lat = geom.lat_new
    height = geom.height_new


    u = u0 * (torch.cos(lat) * math.cos(alpha) + torch.sin(lat) * torch.cos(lon) * math.sin(alpha))
    v = -u0 * torch.sin(lon) * math.sin(alpha)
    w = torch.zeros_like(u)

    u1_contra, u2_contra, u3_contra = geom.wind2contra(u, v, w, metric)


    p = p0 * torch.exp(-height / H)
    rho = p / (Rd * T0)
    theta = T0 * (p0 / p) ** (Rd / cpd)


    rp = torch.arccos(math.sin(phip) * torch.sin(lat) + math.cos(phip) * torch.cos(lat) * torch.cos(lon - lambdap))
    inside = rp < Rp

    def disk(i):
        rz = torch.abs(height - zp[i])
        return torch.where(
            inside & (rz < 0.5 * dzp[i]),
            0.25 * (1.0 + torch.cos(2.0 * math.pi * rz / dzp[i])) * (1.0 + torch.cos(math.pi * rp / Rp)),
            0.0,
        )

    q1 = disk(0)
    q2 = disk(1)

    q3 = torch.where(inside & (torch.abs(height - zp[2]) < 0.5 * dzp[2]), 1.0, 0.0)

    q4 = q1 + q2 + q3

    return rho, u1_contra, u2_contra, u3_contra, theta, q1, q2, q3, q4


def dcmip_steady_state_mountain(geom: CubedSphere3D, metric, mtrx, param):
    """DCMIP-2012 test 2-0: the flat control and mountain use the same hydrostatic profile."""
    T0 = 300.0
    gamma = 0.0065
    lambdam = 3.0 * math.pi / 2.0
    phim = 0.0
    h0 = 2000.0
    Rm = 3.0 * math.pi / 4.0
    zetam = math.pi / 16.0

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
    else:
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

    T0 = 300.0
    Ueq = 20.0
    Peq = 100000.0

    if shear:
        Cs = 2.5e-4  # Wind shear rate (1/m), for shear case
    else:
        Cs = 0.0


    lat = geom.polar[1, ...]
    z_3d = geom.polar[2, ...]

    if Ueq != 0:
        T = T0 * (1 - Cs * Ueq**2 / gravity * torch.sin(lat) ** 2)
    else:
        T = T0 * torch.ones_like(lat)

    # The prescribed shallow-atmosphere profile also adjusts in deep-atmosphere runs.

    p = Peq * torch.exp(-(Ueq**2) / (2 * Rd * T0) * torch.sin(lat) ** 2 - gravity * z_3d / (Rd * T))


    u = Ueq * torch.cos(lat) * (2 * T0 / T * Cs * z_3d + T / T0) ** 0.5


    v = torch.zeros_like(lat)


    w = torch.zeros_like(lat)

    u1_contra, u2_contra, u3_contra = geom.wind2contra(u, v, w, metric)


    rho = p / (Rd * T)


    theta = T * (p0 / p) ** (Rd / cpd)

    return rho, u1_contra, u2_contra, u3_contra, theta


def dcmip_schar_damping_coeffs(metric: Metric5D, geom: CubedSphere3D, shear: bool):
    """Geometry-dependent (rate, u1ref, u2ref, u3ref) are shared by damping and its Jacobian."""
    T0 = 300.0
    Ueq = 20.0
    Zh = 20000.0
    tau0 = 25.0

    if shear:
        Cs = 2.5e-4  # Wind shear rate (1/m), for shear case
    else:
        Cs = 0.0

    lat = geom.polar[1, ...]
    z_3d = geom.polar[2, ...]

    rate = torch.sin(torch.pi / 2 * (z_3d - Zh) / (geom.ztop - Zh)) ** 2 / tau0
    rate[z_3d <= Zh] = 0.0

    if Ueq != 0:
        Tref = T0 * (1 - Cs * Ueq**2 / gravity * torch.sin(lat) ** 2)
    else:
        Tref = T0

    uref = Ueq * torch.cos(lat) * (2 * T0 / Tref * Cs * z_3d + Tref / T0) ** 0.5
    vref = 0.0
    wref = 0.0

    u1ref, u2ref, u3ref = geom.wind2contra(uref, vref, wref, metric)

    return rate, u1ref, u2ref, u3ref


def dcmip_gravity_wave(geom: CubedSphere3D, metric: Metric5D, mtrx: DFROperators, param: Configuration):
    """DCMIP-2012 test 3-1: non-hydrostatic gravity wave.

    A localized potential-temperature perturbation is superposed on a
    hydrostatic, gradient-wind-balanced state on an X=125 non-rotating
    planet. Density deliberately uses the unperturbed temperature, as
    required by version 3 of the reference initial-condition routine.
    """

    u0 = 20.0
    Teq = 300.0
    Peq = 100000.0
    lambdac = 2.0 * math.pi / 3.0
    d = 5000.0
    phic = 0.0
    delta_theta = 1.0
    Lz = 20000.0
    N = 0.01  # Brunt-Vaisala frequency
    N2 = N * N
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


    TS = bigG + (Teq - bigG) * torch.exp(
        -(u0 * N2 / (4.0 * gravity**2))
        * (u0 + 2.0 * geom.rotation_speed * geom.earth_radius)
        * (torch.cos(2.0 * geom.lat_new) - 1.0)
    )


    ps = (
        Peq
        * torch.exp(
            (u0 / (4.0 * bigG * Rd))
            * (u0 + 2.0 * geom.rotation_speed * geom.earth_radius)
            * (torch.cos(2.0 * geom.lat_new) - 1.0)
        )
        * (TS / Teq) ** inv_kappa
    )


    p = ps * ((bigG / TS) * torch.exp(-N2 * geom.height_new / gravity) + 1.0 - (bigG / TS)) ** inv_kappa

    t_mean = bigG * (1.0 - torch.exp(N2 * geom.height_new / gravity)) + TS * torch.exp(N2 * geom.height_new / gravity)

    theta_base = t_mean * (p0 / p) ** kappa


    rho = p / (Rd * t_mean)


    sin_tmp = torch.sin(geom.lat_new) * math.sin(phic)
    cos_tmp = torch.cos(geom.lat_new) * math.cos(phic)

    # great circle distance with 'a/X'

    r = geom.earth_radius * torch.arccos(sin_tmp + cos_tmp * torch.cos(geom.lon_new - lambdac))

    s = (d**2) / (d**2 + r**2)

    theta_pert = delta_theta * s * torch.sin(2.0 * math.pi * geom.height_new / Lz)

    theta = theta_base + theta_pert

    return rho, u1_contra, u2_contra, u3_contra, theta


def dcmip_baroclinic_instability(
    geom: CubedSphere3D,
    metric: Metric5D,
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

    # Small-planet cases scale radius and rotation together; use the geometry constants.
    a = geom.earth_radius
    omega = geom.rotation_speed

    # The specification uses hPa; the EOS requires the imported SI pressure in Pa.
    p_ref = p0

    exponent = Rd * gamma / gravity


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

    # Surface geopotential balances the nonzero surface wind, so the boundary is not flat.

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
    if hasattr(metric, "height_map"):

        def radial_surface(radial):
            sin_lat = radial[2]
            cos_squared = (radial[0] ** 2 + radial[1] ** 2).clamp_min(1e-30)
            jet = u0 * math.cos((eta_sfc - eta0) * 0.5 * math.pi) ** 1.5
            horizontal_1 = -2 * sin_lat ** 6 * (cos_squared + 1 / 3) + 10 / 63
            horizontal_2 = 8 / 5 * cos_squared ** 1.5 * (sin_lat ** 2 + 2 / 3) - math.pi / 4
            mean = horiz_mean_geopotential(torch.full_like(sin_lat, eta_sfc))
            return (mean + jet * (horizontal_1 * jet + horizontal_2 * a * omega)) / gravity

        metric.height_map.radial_surface = radial_surface
    metric.build_metric()


    lon = geom.lon_new
    lat = geom.lat_new
    z = geom.height_new


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

    eta_v = (eta - eta0) * 0.5 * math.pi

    u_wind = u0 * torch.cos(eta_v) ** 1.5 * torch.sin(2.0 * lat) ** 2

    u = u_wind + u_perturb

    # Eq. (118)
    v = torch.zeros_like(u)
    w = torch.zeros_like(u)

    u1_contra, u2_contra, u3_contra = geom.wind2contra(
        u,
        v,
        w,
        metric,
    )


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


def acoustic_wave(geom: CubedSphere3D, metric: Metric5D):

    T0 = 300.0
    Δp = 100
    eta_v = 1
    re = 6371000
    rc = re / 3
    ztop = 10000


    t = T0


    u = 0.0


    v = 0.0


    w = 0.0

    metric.build_metric()

    u1_contra, u2_contra = geom.wind2contra_2d(u, v)


    H = Rd * T0 / gravity
    p_mean = p0 * torch.exp(-geom.height_new / H)

    lat = geom.polar[1, ...]
    lon = geom.polar[0, ...]
    r = re * torch.arccos(torch.cos(lat) * torch.cos(lon))
    f = torch.where(r > rc, 0.0, (Δp / 2) * (1 + torch.cos((math.pi * r) / rc)))
    g = torch.sin((eta_v * math.pi * r) / ztop)
    p_perturb = f * g
    pressure = p_mean + p_perturb


    rho = pressure / (Rd * t)


    theta = t * (p0 / pressure) ** (Rd / cpd)

    return rho, u1_contra, u2_contra, w, theta
