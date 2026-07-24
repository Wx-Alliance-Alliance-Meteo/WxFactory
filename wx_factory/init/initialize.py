import numpy
from numpy.typing import NDArray
import xarray as xr

from ..common.definitions import (
    idx_rho,
    idx_rho_u1,
    idx_rho_u2,
    idx_rho_u3,
    idx_rho_theta,
    idx_h,
    idx_u1,
    idx_u2,
    idx_hu1,
    idx_hu2,
    gravity,
    cpd,
    cvd,
    Rd,
    p0,
)
from ..common import Configuration
from ..geometry import CubedSphere3D, CubedSphere2D, DFROperators, Metric2D, Metric3DTopo

from .dcmip import (
    dcmip_advection_deformation,
    dcmip_advection_hadley,
    dcmip_advection_orography,
    dcmip_gravity_wave,
    dcmip_schar_waves,
    dcmip_steady_state_mountain,
    acoustic_wave,
)
from .shallow_water import (
    case_galewsky,
    case_matsuno,
    case_unsteady_zonal,
    circular_vortex,
    sw_from_ERA5,
    sw_from_file,
    williamson_case1,
    williamson_case2,
    williamson_case5,
    williamson_case6,
)


class Topo:
    def __init__(self, hsurf, dzdx1, dzdx2, hsurf_itf_i, hsurf_itf_j):
        self.hsurf = hsurf
        self.dzdx1 = dzdx1
        self.dzdx2 = dzdx2
        self.hsurf_itf_i = hsurf_itf_i
        self.hsurf_itf_j = hsurf_itf_j


def initialize_euler(geom: CubedSphere3D, metric: Metric3DTopo, mtrx: DFROperators, param: Configuration):

    # -------------------------------------------------------------------------|
    # case DCMIP 2012    | Pure advection                                     |
    #                    | ---------------------------------------------------|
    #                    | 11: 3D deformational flow                          |
    #                    | 12: 3D Hadley-like meridional circulation          |
    #                    | 13: 2D solid-body rotation of thin cloud-like      |
    #                    |     tracer in the presence of orography            |
    #                    | ---------------------------------------------------|
    #                    | 20: Steady-state at rest in presence of orography. |
    #                    | ---------------------------------------------------|
    #                    | Gravity waves, Non-rotating small-planet           |
    #                    | ---------------------------------------------------|
    #                    | 21: Mountain waves over a Schaer-type mountain     |
    #                    | 22: As 21 but with wind shear                      |
    #                    | 31: Gravity wave along the equator                 |
    #                    | ---------------------------------------------------|
    #                    | Rotating planet: Hydro. to non-hydro. scales (X)   |
    #                    | ---------------------------------------------------|
    #                    | 41X: Dry Baroclinic Instability Small Planet       |
    #                    | ---------------------------------------------------|
    #                    | 43 : Moist Baroclinic Instability Simple physics   |
    # --------------------|----------------------------------------------------|
    # case DCMIP 2016    | 161: Baroclinic wave with Toy Terminal Chemistry   |
    #                    | 162: Tropical cyclone                              |
    #                    | 163: Supercell (Small Planet)                      |
    # --------------------|----------------------------------------------------|
    # DCMIP_2012: https://www.earthsystemcog.org/projects/dcmip-2012/         |
    # DCMIP_2016: https://www.earthsystemcog.org/projects/dcmip-2016/         |
    # -------------------------------------------------------------------------|

    xp = geom.device.xp

    num_equations = 5

    if param.case_number == 11:
        num_equations = 9
        rho, u1_contra, u2_contra, w, potential_temperature, q1, q2, q3, q4 = dcmip_advection_deformation(
            geom, metric, mtrx, param
        )
    elif param.case_number == 12:
        num_equations = 6
        rho, u1_contra, u2_contra, w, potential_temperature, q1 = dcmip_advection_hadley(geom, metric, mtrx, param)
    elif param.case_number == 13:
        num_equations = 9
        rho, u1_contra, u2_contra, w, potential_temperature, q1, q2, q3, q4 = dcmip_advection_orography(
            geom, metric, mtrx, param
        )
    elif param.case_number == 20:
        rho, u1_contra, u2_contra, w, potential_temperature = dcmip_steady_state_mountain(geom, metric, mtrx, param)
    elif param.case_number == 21:
        rho, u1_contra, u2_contra, w, potential_temperature = dcmip_schar_waves(geom, metric, mtrx, param, False)
    elif param.case_number == 22:
        rho, u1_contra, u2_contra, w, potential_temperature = dcmip_schar_waves(geom, metric, mtrx, param, True)
    elif param.case_number == 31:
        rho, u1_contra, u2_contra, w, potential_temperature = dcmip_gravity_wave(geom, metric, mtrx, param)
    elif param.case_number == 77:
        rho, u1_contra, u2_contra, w, potential_temperature = acoustic_wave(geom, metric)
    else:
        raise ValueError(f"Unknown case number {param.case_number}")

    Q = xp.zeros((num_equations,) + rho.shape, dtype=rho.dtype)

    Q[idx_rho, ...] = rho
    Q[idx_rho_u1, ...] = rho * u1_contra
    Q[idx_rho_u2, ...] = rho * u2_contra
    Q[idx_rho_u3, ...] = rho * w
    Q[idx_rho_theta, ...] = rho * potential_temperature

    if param.case_number in (11, 12, 13):
        Q[5, ...] = rho * q1
    if param.case_number in (11, 13):
        Q[6, ...] = rho * q2
        Q[7, ...] = rho * q3
        Q[8, ...] = rho * q4

    return Q, None


def extract_available_levels(ds):
    features = list(ds["features"].values)
    feature_set = set(str(f) for f in features)

    levels = []

    for f in features:
        name = str(f)

        if name.startswith("geopotential_h"):
            level = name.split("_h")[-1]

            geo = f"geopotential_h{level}"
            u = f"u_component_of_wind_h{level}"
            v = f"v_component_of_wind_h{level}"

            if geo in feature_set and u in feature_set and v in feature_set:
                levels.append(int(level))

    return sorted(set(levels))


def initialize_sw(geom: CubedSphere2D, metric: Metric2D, mtrx: DFROperators, param: Configuration):

    xp = geom.device.xp
    dtype = xp.float64
    dataset = None

    # ni, nj = geom.lon.shape
    num_equations = 3

    base_shape = geom.lon.shape
    itf_i_shape = geom.lon_itf_i.shape
    itf_j_shape = geom.lon_itf_j.shape
    Q_shape = (num_equations,)

    hsurf = xp.zeros(base_shape, dtype=dtype)
    dzdx1 = xp.zeros(base_shape, dtype=dtype)
    dzdx2 = xp.zeros(base_shape, dtype=dtype)
    hsurf_itf_i = xp.zeros(itf_i_shape, dtype=dtype)
    hsurf_itf_j = xp.zeros(itf_j_shape, dtype=dtype)

    # --- Shallow water
    #   0 : deformation flow (passive advection only)
    #   1 : cosine hill (passive advection only)
    #   2 : zonal flow (shallow water)
    #   5 : zonal flow over an isolated mountain (shallow water)
    #   6 : Rossby-Haurvitz waves (shallow water)
    #   8 : Unstable jet (shallow water)
    if param.case_number == -2:
        ds = xr.open_zarr(param.initial_condition, consolidated=True)
        time_start = str(param.time_start)
        time_end = str(param.time_end)

        if time_start and time_end:
            dataset = ds.sel(time=slice(time_start, time_end))
        else:
            dataset = ds

        features = list(dataset["features"].values)
        feature_map = {str(f): i for i, f in enumerate(features)}

        levels = extract_available_levels(ds)
        NZ = len(levels)
        # For output_manager
        geom.z_levels = levels

        Q_shape = (num_equations, NZ)

        u1_contra, u2_contra, fluid_height = sw_from_ERA5(geom, dataset, 0, levels, feature_map)

    elif param.case_number == -1:
        u1_contra, u2_contra, fluid_height, hsurf, dzdx1, dzdx2, hsurf_itf_i, hsurf_itf_j = sw_from_file(
            geom, mtrx, param
        )

    elif param.case_number == 0:
        u1_contra, u2_contra, fluid_height = circular_vortex(geom, metric, param)

    elif param.case_number == 1:
        u1_contra, u2_contra, fluid_height = williamson_case1(geom, metric, param)

    elif param.case_number == 2:
        u1_contra, u2_contra, fluid_height = williamson_case2(geom, metric, param)

    elif param.case_number == 5:
        u1_contra, u2_contra, fluid_height, hsurf, dzdx1, dzdx2, hsurf_itf_i, hsurf_itf_j = williamson_case5(
            geom, metric, mtrx, param
        )

    elif param.case_number == 6:
        u1_contra, u2_contra, fluid_height = williamson_case6(geom, metric, param)

    elif param.case_number == 8:
        u1_contra, u2_contra, fluid_height = case_galewsky(geom, metric, param)

    elif param.case_number == 9:
        u1_contra, u2_contra, fluid_height = case_matsuno(geom, metric, param)

    elif param.case_number == 10:
        u1_contra, u2_contra, fluid_height, hsurf, dzdx1, dzdx2, hsurf_itf_i, hsurf_itf_j = case_unsteady_zonal(
            geom, metric, mtrx, param
        )

    else:
        raise ValueError(f"Unknown case number {param.case_number} for Shallow Water equations")

    Q = xp.zeros(Q_shape + base_shape, dtype=dtype)
    Q[idx_h, ...] = fluid_height

    if param.case_number in [0, 1]:
        # advection only
        Q[idx_u1, ...] = u1_contra
        Q[idx_u2, ...] = u2_contra
    else:
        Q[idx_hu1, ...] = fluid_height * u1_contra
        Q[idx_hu2, ...] = fluid_height * u2_contra

    topo = None
    if param.case_number in [-1, -2, 5, 10]:
        topo = Topo(hsurf, dzdx1, dzdx2, hsurf_itf_i, hsurf_itf_j)

    return Q, topo, dataset


def initialize_cartesian3d(geom, param: Configuration) -> NDArray[numpy.float64]:
    """Initialize a problem on a flat 3D cartesian slab, from the same case numbers as the 2D grid.

    The 2D cartesian cases live in the (x, z) plane; here they are extruded uniformly in y (the flow
    stays y-invariant, u2 = 0), so a 2D bubble becomes a 3D ridge. Every case's potential-temperature
    perturbation, base stratification and background wind are reused verbatim -- only the coordinates
    are the flat slab's physical X1 (x) / X3 (z), and the state carries the 5th (y-momentum) variable.
    """
    xp = geom.device.xp
    x1, x3 = geom.X1, geom.X3  # physical x and z, (ne3, ne2, ne1, ns**3)

    uu = xp.zeros_like(x1)
    ww = xp.zeros_like(x1)
    θ = xp.ones_like(x1)
    if param.case_number != 0:
        θ *= param.bubble_theta

    if param.case_number == 0:
        # Stratified background with a uniform wind (the step-mountain flow). The mountain topography
        # itself is a terrain-following-metric concern (update_topo) and is not applied here yet.
        xc = (geom.X1.min() + geom.X1.max()) / 2.0  # noqa: F841 (kept for parity with the 2D setup)
    elif param.case_number == 1:
        # Pill
        xc, zc, pert = 500.0, 260.0, 0.5
        r = (x1 - xc) ** 2 + (x3 - zc) ** 2
        θ = xp.where(r < param.bubble_rad**2, θ + pert, θ)
    elif param.case_number == 2:
        # Gaussian bubble
        A, a, s, x0, z0 = 0.5, 50, 100, 500, 260
        r = xp.sqrt((x1 - x0) ** 2 + (x3 - z0) ** 2)
        θ = xp.where(r <= a, θ + A, θ + A * xp.exp(-(((r - a) / s) ** 2)))
    elif param.case_number == 3:
        # Colliding bubbles: warm then cold
        for A, a, s, x0, z0 in ((0.5, 150, 50, 500, 300), (-0.15, 0, 50, 560, 640)):
            r = xp.sqrt((x1 - x0) ** 2 + (x3 - z0) ** 2)
            θ = xp.where(r <= a, θ + A, θ + A * xp.exp(-(((r - a) / s) ** 2)))
    elif param.case_number == 4:
        # Density current (cold anomaly)
        xc, zc, xr, zr = 0.0, 3000.0, 4000.0, 2000.0
        r = xp.sqrt(((x1 - xc) / xr) ** 2 + ((x3 - zc) / zr) ** 2)
        θ = θ + xp.where(r <= 1.0, -15.0 * (1.0 + xp.cos(xp.pi * r)) / 2.0, 0.0)

    if param.case_number == 0:
        N_star, t0 = 0.01, 288.0
        a00 = N_star**2 / gravity
        capc1 = gravity**2 / (N_star**2 * cpd * t0)
        exner = 1.0 - capc1 * (1.0 - xp.exp(-a00 * x3))
        θ = t0 * xp.exp(a00 * x3)
        uu = uu + 10.0
    else:
        exner = 1.0 - gravity / (cpd * θ) * x3

    ρ = p0 / (Rd * θ) * exner ** (cvd / Rd)

    Q = xp.zeros((5,) + geom.grid_shape_3d_new, dtype=x1.dtype)
    Q[idx_rho] = ρ
    Q[idx_rho_u1] = ρ * uu
    Q[idx_rho_u2] = 0.0  # y-invariant extrusion
    Q[idx_rho_u3] = ρ * ww
    Q[idx_rho_theta] = ρ * θ
    return Q
