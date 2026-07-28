import math

import numpy
import torch

from ..common import Configuration
from ..common.definitions import day_in_secs, gravity
from ..geometry import CubedSphere2D, DFROperators
from ..init.matsuno import eval_field
from ..output.input_manager import InputManager


def eval_u_prime(lat):
    u_max = 80.0
    phi0 = math.pi / 7.0
    phi1 = math.pi / 2.0 - phi0

    if lat < phi0 or lat > phi1:
        return 0.0

    e_n = math.exp(-4.0 / ((phi1 - phi0) ** 2))

    u_p = math.exp(1.0 / ((lat - phi0) * (lat - phi1)))

    return u_max / e_n * u_p


def solid_body_rotation(geom: CubedSphere2D, metric, param):
    if param.case_number == 5:
        u0 = 20.0
    else:
        u0 = 2.0 * math.pi * geom.earth_radius / (12.0 * day_in_secs)

    u = u0 * geom.coslat
    v = 0.0
    u1, u2 = geom.wind2contra(u, v)

    return u1, u2


def circular_vortex(geom, metric, param):
    if geom.device.comm.rank == 0:
        print("--------------------------------------------------------------")
        print("CASE 0 (Tracer): Circular vortex, Nair and Machenhauer,2002   ")
        print("--------------------------------------------------------------")

    # Deformational Flow (Nair and Machenhauer, 2002)
    lon_center = math.pi - 0.8
    lat_center = math.pi / 4.8

    h, Omega = height_vortex(geom, metric, param, 0)

    u = (
        geom.earth_radius
        * Omega
        * (math.sin(lat_center) * geom.coslat - math.cos(lat_center) * torch.cos(geom.lon - lon_center) * geom.sinlat)
    )
    v = geom.earth_radius * Omega * math.cos(lat_center) * torch.sin(geom.lon - lon_center)
    u1, u2 = geom.wind2contra(u, v)

    return u1, u2, h


def height_vortex(geom, metric, param, step):

    step_time = step * param.dt

    lon_center = math.pi - 0.8
    lat_center = math.pi / 4.8

    V0 = 2.0 * math.pi / (12.0 * day_in_secs) * geom.earth_radius
    rho_0 = 3.0
    gamma = 5.0

    lonR = torch.atan2(
        geom.coslat * torch.sin(geom.lon - lon_center),
        geom.coslat * math.sin(lat_center) * torch.cos(geom.lon - lon_center) - math.cos(lat_center) * geom.sinlat,
    )

    lonR = torch.where(lonR < 0.0, lonR + 2.0 * math.pi, lonR)

    latR = torch.asin(
        geom.sinlat * math.sin(lat_center) + geom.coslat * math.cos(lat_center) * torch.cos(geom.lon - lon_center)
    )

    rho = rho_0 * torch.cos(latR)

    Vt = V0 * (3.0 / 2.0 * math.sqrt(3.0)) * (1.0 / torch.cosh(rho)) ** 2 * torch.tanh(rho)

    Omega = torch.zeros_like(geom.lat)

    mask = torch.abs(rho) > 1e-9
    Omega[mask] = Vt[mask] / (geom.earth_radius * rho[mask])

    h = 1.0 - torch.tanh((rho / gamma) * torch.sin(lonR - Omega * step_time))

    return h, Omega


def sw_from_ERA5(geom: CubedSphere2D, ds, t, levels, feature_map):
    idx_geo_all = [feature_map[f"geopotential_h{z}"] for z in levels]
    idx_u_all = [feature_map[f"u_component_of_wind_h{z}"] for z in levels]
    idx_v_all = [feature_map[f"v_component_of_wind_h{z}"] for z in levels]

    geopotential = ds["data"].isel(time=t, features=idx_geo_all)
    u = ds["data"].isel(time=t, features=idx_u_all)
    v = ds["data"].isel(time=t, features=idx_v_all)

    target_lon = geom.device.to_host((geom.lon * 180 / math.pi) % 360)
    target_lat = geom.device.to_host(geom.lat * 180 / math.pi)

    # Flatten for interpolation
    lon_flat = target_lon.reshape(-1)
    lat_flat = target_lat.reshape(-1)

    # Interpolate ERA5 → geom
    geop_interp = geopotential.interp(
        longitude=("points", lon_flat), latitude=("points", lat_flat), method="linear"
    ).values

    u_interp = u.interp(longitude=("points", lon_flat), latitude=("points", lat_flat), method="linear").values

    v_interp = v.interp(longitude=("points", lon_flat), latitude=("points", lat_flat), method="linear").values

    # Reshape back to cubed-sphere
    shape = (len(levels),) + geom.lon.shape
    geop_interp = geop_interp.reshape(shape)
    u_interp = u_interp.reshape(shape)
    v_interp = v_interp.reshape(shape)

    geop_interp = torch.asarray(geop_interp)
    u_interp = torch.asarray(u_interp)
    v_interp = torch.asarray(v_interp)

    g = 9.80616
    fluid_height = geop_interp / g

    u1_contra = torch.zeros_like(u_interp)
    u2_contra = torch.zeros_like(v_interp)

    u1_contra, u2_contra = geom.wind2contra(u_interp, v_interp)

    return u1_contra, u2_contra, fluid_height


def sw_from_file(geom: CubedSphere2D, operators: DFROperators, config: Configuration):
    h_surface = InputManager.read_mountain(config.topography_file, geom)
    h, u, v = InputManager.read_fields(config.initial_conditions_file, ["GZ", "UU", "VV"], geom)
    h[...] *= 10 / gravity

    num_solpts = geom.num_solpts
    num_elem = geom.num_elements_horizontal
    h_surface_itf_i = torch.zeros((num_elem, num_elem + 2, 2 * num_solpts), dtype=h_surface.dtype)
    h_surface_itf_j = torch.zeros((num_elem + 2, num_elem, 2 * num_solpts), dtype=h_surface.dtype)

    # Easier shape to work with
    h_split = h_surface.reshape(h_surface.shape[:-1] + (num_solpts, num_solpts))
    print(f"itf shapes: {h_split.shape}, {h_surface_itf_i.shape}, {h_surface_itf_j.shape}", flush=True)

    h_south = h_split[..., 0, :]
    h_north = h_split[..., -1, :]
    h_west = h_split[..., :, 0]
    h_east = h_split[..., :, -1]

    # Start transferring borders
    transfer = geom.process_topology.start_exchange_scalars(
        h_south[..., 0, :, :],
        h_north[..., -1, :, :],
        h_west[..., 0, :],
        h_east[..., -1, :],
        boundary_shape=(num_elem, num_solpts),
    )

    # Compute upper boundary (north, east)
    # We just take the average
    h_surface_itf_i[..., :, 1:-2, num_solpts:] = (h_west[..., :, 1:, :] + h_east[..., :, :-1, :]) * 0.5
    h_surface_itf_j[..., 1:-2, :, num_solpts:] = (h_south[..., 1:, :, :] + h_north[..., :-1, :, :]) * 0.5

    # Receive borders from neighbors
    h_s, h_n, h_w, h_e = transfer.wait()

    # Tile borders
    h_surface_itf_i[..., :, 0, num_solpts:] = (h_w + h_west[..., 0, :]) * 0.5
    h_surface_itf_i[..., :, -2, num_solpts:] = (h_east[..., -1, :] + h_e) * 0.5
    h_surface_itf_j[..., 0, :, num_solpts:] = (h_s + h_south[..., 0, :, :]) * 0.5
    h_surface_itf_j[..., -2, :, num_solpts:] = (h_north[..., -1, :, :] + h_n) * 0.5

    # Copy to lower boundary (south, west)
    h_surface_itf_i[..., :, 1:, :num_solpts] = h_surface_itf_i[..., :, :-1, num_solpts:]
    h_surface_itf_j[..., 1:, :, :num_solpts] = h_surface_itf_j[..., :-1, :, num_solpts:]

    # Compute mountain gradient
    dzdx1 = h_surface @ operators.derivative_x + geom.middle_itf_i(h_surface_itf_i) @ operators.correction_WE
    dzdx2 = h_surface @ operators.derivative_y + geom.middle_itf_j(h_surface_itf_j) @ operators.correction_SN

    u1, u2 = geom.wind2contra(u, v)

    return u1, u2, h - h_surface, h_surface, dzdx1, dzdx2, h_surface_itf_i, h_surface_itf_j


def williamson_case1(geom, metric, param):
    if geom.device.comm.rank == 0:
        print(
            "---------------------------------------------------------------\n"
            "WILLIAMSON CASE 1 (Tracer): Cosine Bell, Williamson et al.,1992\n"
            "---------------------------------------------------------------"
        )

    u1, u2 = solid_body_rotation(geom, metric, param)

    h = height_case1(geom, metric, param, 0)

    return u1, u2, h


def height_case1(geom: CubedSphere2D, metric, param, step):
    # Initialize gaussian bell
    step_time = step * param.dt

    ubar = 2.0 * math.pi / (12.0 * day_in_secs)

    lon_center = (3.0 * math.pi / 2.0) + ubar * step_time
    if lon_center > 2.0 * math.pi:
        lon_center -= 2.0 * math.pi

    lat_center = 0.0

    h0 = 1000.0

    radius = 1.0 / 3.0

    dist = torch.acos(
        math.sin(lat_center) * geom.sinlat + math.cos(lat_center) * geom.coslat * torch.cos(geom.lon - lon_center)
    )

    return 0.5 * h0 * (1.0 + torch.cos(math.pi * dist / radius)) * (dist <= radius)


def williamson_case2(geom, metric, param):
    if geom.device.comm.rank == 0:
        print("--------------------------------------------")
        print("WILLIAMSON CASE 2, Williamson et al. (1992) ")
        print("Steady state nonlinear geostrophic flow     ")
        print("--------------------------------------------")

    u1, u2 = solid_body_rotation(geom, metric, param)

    # Global Steady State Nonlinear Zonal Geostrophic Flow
    h = height_case2(geom, metric, param)
    return u1, u2, h


def height_case2(geom, metric, param):
    gh0 = 29400.0
    u0 = 2.0 * math.pi * geom.earth_radius / (12.0 * day_in_secs)

    h = (gh0 - (geom.earth_radius * geom.rotation_speed * u0 + (0.5 * u0**2)) * geom.sinlat**2) / gravity
    return h


def williamson_case5(geom: CubedSphere2D, metric, mtrx: DFROperators, param):
    if geom.device.comm.rank == 0:
        print(
            "--------------------------------------------\n"
            "WILLIAMSON CASE 5, Williamson et al. (1992) \n"
            "Zonal Flow over an isolated mountain        \n"
            "--------------------------------------------"
        )

    u0 = 20.0  # Max wind (m/s)
    h0 = 5960.0  # Mean height (m)

    u1, u2 = solid_body_rotation(geom, metric, param)

    h_star = (
        gravity * h0 - (geom.earth_radius * geom.rotation_speed * u0 + 0.5 * u0**2) * (geom.sinlat) ** 2
    ) / gravity

    # Isolated mountain
    hs0 = 2000.0
    rr = math.pi / 9.0

    # Mountain location
    lon_mountain = 3.0 * math.pi / 2.0
    lat_mountain = math.pi / 6.0

    r = torch.sqrt(torch.clamp((geom.lon - lon_mountain) ** 2 + (geom.lat - lat_mountain) ** 2, max=rr**2))

    r_itf_i = torch.sqrt(
        torch.clamp(
            (geom.lon_itf_i - lon_mountain) ** 2 + (geom.lat_itf_i - lat_mountain) ** 2,
            max=rr**2,
        )
    )
    r_itf_j = torch.sqrt(
        torch.clamp(
            (geom.lon_itf_j - lon_mountain) ** 2 + (geom.lat_itf_j - lat_mountain) ** 2,
            max=rr**2,
        )
    )

    r_itf_i[geom.west_edge] = 0.0
    r_itf_i[geom.east_edge] = 0.0
    r_itf_j[geom.south_edge] = 0.0
    r_itf_j[geom.north_edge] = 0.0

    hsurf = hs0 * (1 - r / rr)

    hsurf_itf_i = hs0 * (1.0 - r_itf_i / rr)
    hsurf_itf_j = hs0 * (1.0 - r_itf_j / rr)

    hsurf_itf_i[geom.west_edge] = 0.0
    hsurf_itf_i[geom.east_edge] = 0.0
    hsurf_itf_j[geom.south_edge] = 0.0
    hsurf_itf_j[geom.north_edge] = 0.0

    dzdx1 = hsurf @ mtrx.derivative_x + geom.middle_itf_i(hsurf_itf_i) @ mtrx.correction_WE
    dzdx2 = hsurf @ mtrx.derivative_y + geom.middle_itf_j(hsurf_itf_j) @ mtrx.correction_SN

    h = h_star - hsurf

    return u1, u2, h, hsurf, dzdx1, dzdx2, hsurf_itf_i, hsurf_itf_j


def williamson_case6(geom: CubedSphere2D, metric, param):
    if geom.device.comm.rank == 0:
        print(
            "--------------------------------------------\n"
            "WILLIAMSON CASE 6, Williamson et al. (1992) \n"
            "Rossby-Haurwitz wave                        \n"
            "--------------------------------------------"
        )

    # Rossby-Haurwitz wave

    R = 4

    omega = 7.848e-6
    K = omega
    h0 = 8000.0

    A = omega / 2.0 * (2.0 * geom.rotation_speed + omega) * geom.coslat**2 + (K**2) / 4.0 * geom.coslat ** (
        2 * R
    ) * ((R + 1) * geom.coslat**2 + (2.0 * R**2 - R - 2.0) - 2.0 * (R**2) * geom.coslat ** (-2))

    B = (
        2.0
        * (geom.rotation_speed + omega)
        * K
        / ((R + 1) * (R + 2))
        * geom.coslat**R
        * ((R**2 + 2 * R + 2) - (R + 1) ** 2 * geom.coslat**2)
    )

    C = (K**2) / 4.0 * geom.coslat ** (2 * R) * ((R + 1) * (geom.coslat**2) - (R + 2.0))

    h = (
        h0
        + (
            geom.earth_radius**2 * A
            + geom.earth_radius**2 * B * torch.cos(R * geom.lon)
            + geom.earth_radius**2 * C * torch.cos(2.0 * R * geom.lon)
        )
        / gravity
    )

    u = geom.earth_radius * omega * geom.coslat + geom.earth_radius * K * geom.coslat ** (R - 1) * (
        R * geom.sinlat**2 - geom.coslat**2
    ) * torch.cos(R * geom.lon)
    v = -geom.earth_radius * K * R * geom.coslat ** (R - 1) * geom.sinlat * torch.sin(R * geom.lon)

    u1, u2 = geom.wind2contra(u, v)

    return u1, u2, h


def case_galewsky(geom, metric, param):
    if geom.device.comm.rank == 0:
        print("--------------------------------------------")
        print("CASE 8, Galewsky et al. (2004)              ")
        print("Barotropic wave                             ")
        print("--------------------------------------------")

    h0 = 10158.18617045463179
    h_hat = 120.0
    phi2 = math.pi / 4.0
    alpha = 1.0 / 3.0
    beta = 1.0 / 15.0

    # This initialization contains a scalar numerical quadrature at every grid point.  Evaluate it
    # once on host copies, independently of the geometry's element/solution-point layout, then move
    # the completed fields back to the configured device.
    lat = geom.device.to_host(geom.lat)
    lon = geom.device.to_host(geom.lon)
    u = numpy.zeros_like(lat)
    v = numpy.zeros_like(lat)
    h = numpy.zeros_like(lat)

    for idx in numpy.ndindex(lat.shape):

        # Calculate height field via numerical integration
        nIntervals = int((lat[idx] + 0.5 * math.pi) / (1.0e-2))

        nIntervals = max(nIntervals, 1)

        latX = numpy.zeros(nIntervals + 1)

        for k in range(nIntervals + 1):
            latX[k] = -0.5 * math.pi + ((lat[idx] + 0.5 * math.pi) / nIntervals) * k

        h_integrand = 0.0

        for k in range(nIntervals):
            for m in range(-1, 2, 2):
                dXeval = 0.5 * (latX[k + 1] + latX[k]) + m * math.sqrt(1.0 / 3.0) * 0.5 * (latX[k + 1] - latX[k])

                dU = eval_u_prime(dXeval)

                h_integrand += (
                    2.0 * geom.earth_radius * geom.rotation_speed * math.sin(dXeval) + dU * math.tan(dXeval)
                ) * dU

        h_integrand *= 0.5 * (latX[1] - latX[0])

        h[idx] = h0 - h_integrand / gravity

        # Add perturbation
        h[idx] += (
            h_hat
            * math.cos(lat[idx])
            * math.exp(-((lon[idx] / alpha) ** 2))
            * math.exp(-(((phi2 - lat[idx]) / beta) ** 2))
        )

        # Evaluate the velocity field
        u_p = eval_u_prime(lat[idx])

        if abs(math.cos(lon[idx])) < 1.0e-13:
            u[idx] = u_p
        else:
            u[idx] = (v[idx] * math.sin(lat[idx]) * math.sin(lon[idx]) + u_p * math.cos(lon[idx])) / math.cos(lon[idx])

    dtype = geom.lon.dtype
    u = torch.asarray(u, dtype=dtype)
    v = torch.asarray(v, dtype=dtype)
    h = torch.asarray(h, dtype=dtype)
    u1, u2 = geom.wind2contra(u, v)

    return u1, u2, h


def case_matsuno(geom, metric, param):
    wave_type = {"rossby": "Rossby", "eig": "EIG", "wig": "WIG"}[param.matsuno_wave_type.lower()]
    if geom.device.comm.rank == 0:
        print("--------------------------------------------")
        print("CASE 9, Shamir et al.,2019,GMD,12,2181-2193 ")

        if wave_type == "Rossby":
            print("The Matsuno baroclinic wave (Rosby)         ")
        elif wave_type == "EIG":
            print("The Matsuno baroclinic wave (EIG)           ")
            print("--------------------------------------------")
        elif wave_type == "WIG":
            print("The Matsuno baroclinic wave (WIG)           ")
        print("--------------------------------------------")

    lat = geom.device.to_host(geom.lat)
    lon = geom.device.to_host(geom.lon)
    u = numpy.zeros_like(lat)
    v = numpy.zeros_like(lat)
    h = numpy.zeros_like(lat)

    for idx in numpy.ndindex(lat.shape):
        h[idx] = (
            eval_field(
                lat[idx],
                lon[idx],
                0.0,
                amp=param.matsuno_amp,
                field="phi",
                wave_type=wave_type,
            )
            / gravity
        )
        u[idx] = eval_field(lat[idx], lon[idx], 0.0, amp=param.matsuno_amp, field="u", wave_type=wave_type)
        v[idx] = eval_field(lat[idx], lon[idx], 0.0, amp=param.matsuno_amp, field="v", wave_type=wave_type)

    dtype = geom.lon.dtype
    u = torch.asarray(u, dtype=dtype)
    v = torch.asarray(v, dtype=dtype)
    h = torch.asarray(h, dtype=dtype)
    u1, u2 = geom.wind2contra(u, v)

    return u1, u2, h


def case_unsteady_zonal(geom, metric, mtrx, param):
    if geom.device.comm.rank == 0:
        print("--------------------------------------------")
        print("CASE 10, Läuter et al. (2005)               ")
        print("Zonal balanced time dependent flow          ")
        print("--------------------------------------------")

    u0 = 2.0 * math.pi * geom.earth_radius / (12.0 * 24.0 * 3600.0)

    # Note, units of k1 and k2 are gpm, m^2/s^2
    k2 = 10.0

    u = u0 * torch.cos(geom.lat)
    v = torch.zeros_like(geom.lat)

    # Geopotential heights
    h = height_unsteady_zonal(geom, metric, param)

    hs = 0.5 * (geom.earth_radius * geom.rotation_speed * torch.sin(geom.lat)) ** 2 + k2
    hsurf = hs / gravity

    hsurf_itf_i = (0.5 * (geom.earth_radius * geom.rotation_speed * torch.sin(geom.lat_itf_i)) ** 2 + k2) / gravity
    hsurf_itf_j = (0.5 * (geom.earth_radius * geom.rotation_speed * torch.sin(geom.lat_itf_j)) ** 2 + k2) / gravity
    dzdx1 = hsurf @ mtrx.derivative_x + geom.middle_itf_i(hsurf_itf_i) @ mtrx.correction_WE
    dzdx2 = hsurf @ mtrx.derivative_y + geom.middle_itf_j(hsurf_itf_j) @ mtrx.correction_SN

    u1, u2 = geom.wind2contra(u, v)
    return u1, u2, h, hsurf, dzdx1, dzdx2, hsurf_itf_i, hsurf_itf_j


def height_unsteady_zonal(geom, metric, param):

    u0 = 2.0 * math.pi * geom.earth_radius / (12.0 * 24.0 * 3600.0)

    # Note, units of k1 and k2 are gpm, m^2/s^2
    k1 = 133681.0
    k2 = 10.0

    sinlat = torch.sin(geom.lat)

    # Geopotential heights
    h = (
        -0.5 * (u0 * sinlat + geom.earth_radius * geom.rotation_speed * sinlat) ** 2
        + 0.5 * (geom.earth_radius * geom.rotation_speed * sinlat) ** 2
        + k1
    )

    hs = 0.5 * (geom.earth_radius * geom.rotation_speed * sinlat) ** 2 + k2

    # Revert to height, in metres
    # Note, need h as depth rather than height
    h = (h - hs) / gravity
    return h
