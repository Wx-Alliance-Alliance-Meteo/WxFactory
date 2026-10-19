import numpy
import torch
import xarray as xr
from numpy.typing import NDArray

from ..common import Configuration
from ..common.definitions import (
    Rd,
    cpd,
    cvd,
    gravity,
    heat_capacity_ratio,
    idx_h,
    idx_hu1,
    idx_hu2,
    idx_rho,
    idx_rho_theta,
    idx_u1,
    idx_u2,
    p0,
)
from ..geometry import CubedSphere2D, CubedSphere3D, DFROperators, Metric2D, Metric5D
from ..output.input_manager import extract_available_levels
from .dcmip import (
    acoustic_wave,
    dcmip_advection_deformation,
    dcmip_advection_hadley,
    dcmip_advection_orography,
    dcmip_baroclinic_instability,
    dcmip_gravity_wave,
    dcmip_schar_waves,
    dcmip_steady_state_mountain,
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


def initialize_euler(geom: CubedSphere3D, metric: Metric5D, mtrx: DFROperators, param: Configuration):

    if param.case_number < 0 and param.initial_conditions != "":
        from .euler import initialize_era5

        initializer = initialize_era5
        rho, u1_contra, u2_contra, w, potential_temperature = initializer(
            geom, metric, param.initial_conditions, param.topography_file, param.time_start
        )
    elif param.case_number == 11:
        rho, u1_contra, u2_contra, w, potential_temperature, q1, q2, q3, q4 = dcmip_advection_deformation(
            geom, metric, mtrx, param
        )
    elif param.case_number == 12:
        rho, u1_contra, u2_contra, w, potential_temperature, q1 = dcmip_advection_hadley(geom, metric, mtrx, param)
    elif param.case_number == 13:
        rho, u1_contra, u2_contra, w, potential_temperature, q1, q2, q3, q4 = dcmip_advection_orography(
            geom, metric, mtrx, param
        )
    elif param.case_number in (20, 200, 201, 202):
        rho, u1_contra, u2_contra, w, potential_temperature = dcmip_steady_state_mountain(geom, metric, mtrx, param)
    elif param.case_number == 21:
        rho, u1_contra, u2_contra, w, potential_temperature = dcmip_schar_waves(geom, metric, mtrx, param, False)
    elif param.case_number == 22:
        rho, u1_contra, u2_contra, w, potential_temperature = dcmip_schar_waves(geom, metric, mtrx, param, True)
    elif param.case_number == 31:
        rho, u1_contra, u2_contra, w, potential_temperature = dcmip_gravity_wave(geom, metric, mtrx, param)
    elif param.case_number == 41:
        rho, u1_contra, u2_contra, w, potential_temperature = dcmip_baroclinic_instability(geom, metric, mtrx, param)
    elif param.case_number == 77:
        rho, u1_contra, u2_contra, w, potential_temperature = acoustic_wave(geom, metric)
    else:
        raise ValueError(f"Unknown case number {param.case_number}")

    from .euler import pack_euler_state

    state = pack_euler_state(rho, (u1_contra, u2_contra, w), potential_temperature, metric, param)
    if param.case_number in (11, 12, 13):
        tracers = (q1,) if param.case_number == 12 else (q1, q2, q3, q4)
        state = torch.cat((state, torch.stack([rho * tracer for tracer in tracers])))
    return state, None


def initialize_sw(geom: CubedSphere2D, metric: Metric2D, mtrx: DFROperators, param: Configuration):

    dtype = torch.float64
    dataset = None

    num_equations = 3

    base_shape = geom.lon.shape
    itf_i_shape = geom.lon_itf_i.shape
    itf_j_shape = geom.lon_itf_j.shape
    Q_shape = (num_equations,)

    hsurf = torch.zeros(base_shape, dtype=dtype)
    dzdx1 = torch.zeros(base_shape, dtype=dtype)
    dzdx2 = torch.zeros(base_shape, dtype=dtype)
    hsurf_itf_i = torch.zeros(itf_i_shape, dtype=dtype)
    hsurf_itf_j = torch.zeros(itf_j_shape, dtype=dtype)

    if param.case_number == -2:
        ds = xr.open_zarr(param.initial_conditions, consolidated=True)
        time_start = str(param.time_start)
        time_end = str(param.time_end)

        if time_start and time_end:
            dataset = ds.sel(time=slice(time_start, time_end))
        else:
            dataset = ds.isel(time=[0])

        if len(dataset.time) > 1 and dataset["time"].values[-1] == numpy.datetime64(
            str(param.time_end).replace("t", "T")
        ):
            dataset = dataset.isel(time=slice(None, -1))

        features = list(dataset["features"].values)
        feature_map = {str(f): i for i, f in enumerate(features)}

        levels = extract_available_levels(ds)
        if len(levels) > 1:
            NZ = len(levels) - 1
        else:
            NZ = 1
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

    Q = torch.zeros(Q_shape + base_shape, dtype=dtype)
    Q[idx_h, ...] = fluid_height

    if param.case_number in [0, 1]:
        Q[idx_u1, ...] = u1_contra
        Q[idx_u2, ...] = u2_contra
    else:
        Q[idx_hu1, ...] = fluid_height * u1_contra
        Q[idx_hu2, ...] = fluid_height * u2_contra

    Q = Q.astype(geom.dtype)

    topo = None
    if param.case_number in [-1, -2, 5, 10]:
        topo = Topo(
            hsurf.astype(geom.dtype),
            dzdx1.astype(geom.dtype),
            dzdx2.astype(geom.dtype),
            hsurf_itf_i.astype(geom.dtype),
            hsurf_itf_j.astype(geom.dtype),
        )

    return Q, topo, dataset


def initialize_cartesian3d(geom, metric, param: Configuration) -> NDArray[numpy.float64]:
    """Initialize a problem on a flat 3D cartesian slab, from the same case numbers as the 2D grid.

    The 2D cartesian cases live in the (x, z) plane; here they are extruded uniformly in y (the flow
    stays y-invariant, u2 = 0), so a 2D bubble becomes a 3D ridge. Every case's potential-temperature
    perturbation, base stratification and background wind are reused verbatim -- only the coordinates
    are the flat slab's physical X1 (x) / X3 (z), and the state carries the 5th (y-momentum) variable.
    """
    x1, x3 = geom.X1, geom.X3  # physical x and z, (ne3, ne2, ne1, ns**3)

    uu = torch.zeros_like(x1)
    ww = torch.zeros_like(x1)
    θ = torch.ones_like(x1)
    if param.case_number != 0:
        θ *= param.bubble_theta

    if param.case_number == 0:
        # Stratified background with a uniform wind (the step-mountain flow). The mountain topography
        # itself is a terrain-following-metric concern (update_topo) and is not applied here yet.
        xc = (geom.X1.min() + geom.X1.max()) / 2.0
    elif param.case_number == 1:
        xc, zc, pert = 500.0, 260.0, 0.5
        r = (x1 - xc) ** 2 + (x3 - zc) ** 2
        θ = torch.where(r < param.bubble_rad**2, θ + pert, θ)
    elif param.case_number == 2:
        A, a, s, x0, z0 = 0.5, 50, 100, 500, 260
        r = torch.sqrt((x1 - x0) ** 2 + (x3 - z0) ** 2)
        θ = torch.where(r <= a, θ + A, θ + A * torch.exp(-(((r - a) / s) ** 2)))
    elif param.case_number == 3:
        for A, a, s, x0, z0 in ((0.5, 150, 50, 500, 300), (-0.15, 0, 50, 560, 640)):
            r = torch.sqrt((x1 - x0) ** 2 + (x3 - z0) ** 2)
            θ = torch.where(r <= a, θ + A, θ + A * torch.exp(-(((r - a) / s) ** 2)))
    elif param.case_number == 4:
        xc, zc, xr, zr = 0.0, 3000.0, 4000.0, 2000.0
        r = torch.sqrt(((x1 - xc) / xr) ** 2 + ((x3 - zc) / zr) ** 2)
        exner = 1.0 - gravity * x3 / (cpd * param.bubble_theta)
        temperature_perturbation = torch.where(r <= 1.0, -15.0 * (1.0 + torch.cos(torch.pi * r)) / 2.0, 0.0)
        # Straka prescribes a temperature anomaly at unchanged background pressure.
        θ = θ + temperature_perturbation / exner
    elif param.case_number == 666:
        T0 = 300.0
        H = Rd * T0 / gravity
        t = T0
        pressure = p0 * torch.exp(-geom.X3 / H)
        ρ = pressure / (Rd * t)
        θ = t * (p0 / pressure) ** (Rd / cpd)
    elif param.case_number == 651:
        max_mach = param.gresho_mach
        if not numpy.isfinite(max_mach) or max_mach <= 0:
            raise ValueError("gresho_mach must be finite and positive")
        x = x1 - 0.5
        z = x3 - 0.5
        radius = torch.sqrt(x**2 + z**2)
        angle = torch.atan2(z, x)
        inner = radius < 0.2
        annulus = (radius >= 0.2) & (radius < 0.4)
        tangential_velocity = torch.zeros_like(radius)
        tangential_velocity[inner] = 5 * radius[inner]
        tangential_velocity[annulus] = 2 - 5 * radius[annulus]
        uu = -tangential_velocity * torch.sin(angle)
        ww = tangential_velocity * torch.cos(angle)
        ρ = torch.ones_like(radius)

        # At r = 0.2 the speed is one, fixing the maximum Mach number.
        base_pressure = 1 / (heat_capacity_ratio * max_mach**2) - 0.5
        if base_pressure <= 0:
            raise ValueError("gresho_mach is too large to give positive pressure throughout the vortex")
        pressure = torch.full_like(radius, base_pressure + 4 * numpy.log(2) - 2)
        pressure[inner] = base_pressure + 12.5 * radius[inner] ** 2
        r = radius[annulus]
        pressure[annulus] = base_pressure + 4 * torch.log(5 * r) + 4 - 20 * r + 12.5 * r**2
        θ = p0 / (Rd * ρ) * (pressure / p0) ** (1 / heat_capacity_ratio)

    if param.case_number == 0:
        N_star, t0 = 0.01, 288.0
        a00 = N_star**2 / gravity
        capc1 = gravity**2 / (N_star**2 * cpd * t0)
        exner = 1.0 - capc1 * (1.0 - torch.exp(-a00 * x3))
        θ = t0 * torch.exp(a00 * x3)
        uu = uu + 10.0
    elif param.case_number != 4:
        exner = 1.0 - gravity / (cpd * θ) * x3

    if param.case_number != 666 and param.case_number != 651:
        ρ = p0 / (Rd * θ) * exner ** (cvd / Rd)

    Q = torch.zeros((5,) + geom.grid_shape_3d_new, dtype=x1.dtype)
    Q[idx_rho] = ρ
    physical_momentum = torch.stack((ρ * uu, torch.zeros_like(ρ), ρ * ww))
    Q[1:4] = metric.volume.covariant_from_physical(physical_momentum / ρ)
    Q[idx_rho_theta] = ρ * θ

    return Q
