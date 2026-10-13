"""State packing and hydrostatic profiles for the 5D Euler path."""

import torch

from ..common.definitions import Rd, cpd, gravity, p0


def deep_rest_pressure(height, radius, temperature_surface=300.0, lapse_rate=0.0065):
    """Return hydrostatic pressure for T=T0-L*z and g(z)=g0*(a/(a+z))^2.

    The integral of d(log p)/dz=-g(z)/(Rd*T(z)) excludes centrifugal acceleration.
    This height-only profile is not in rest balance with the full rotating-frame
    potential. The DCMIP profile instead uses constant gravity.
    """
    temperature_offset = temperature_surface + lapse_rate * radius
    hydrostatic_integral = gravity * (
        radius**2
        * lapse_rate
        / temperature_offset**2
        * (torch.log1p(height / radius) - torch.log1p(-lapse_rate * height / temperature_surface))
        + radius * height / (temperature_offset * (radius + height))
    )
    return p0 * torch.exp(-hydrostatic_integral / Rd)


def pack_euler_state(rho, contravariant, theta, metric, config):
    """Pack density, relative covariant velocity, and rho*theta."""
    if metric.geometry.deep and config.case_number in (20, 200, 201, 202):
        height = metric.volume.height
        temperature = 300.0 - 0.0065 * height
        pressure = deep_rest_pressure(height, metric.geometry.earth_radius)
        rho = pressure / (Rd * temperature)
        theta = temperature * (p0 / pressure) ** (Rd / cpd)
    state = rho.new_empty((5,) + rho.shape)
    state[0] = rho
    velocity = torch.stack(
        [
            torch.broadcast_to(torch.as_tensor(component, dtype=rho.dtype, device=rho.device), rho.shape)
            for component in contravariant
        ]
    )
    state[1:4] = metric.volume.lower_velocity(velocity)
    state[4] = rho * theta
    return state


def initialize_era5(geometry, metric, filename, topography_filename, time):
    """Interpolate ERA5 physical fields after constructing the terrain-following grid."""
    import math

    import xarray as xr

    from ..geometry.terrain_mapping import SampledSphereSurface
    from ..output.input_manager import InputManager, extract_available_levels
    from .vertical_interpolation import vertical_interp

    surface = InputManager.read_mountain(topography_filename, geometry)
    height_map = metric.height_map
    height_map.tabulated_surface = SampledSphereSurface(geometry, surface, metric.frame)
    surface_volume = height_map.tabulated_surface(geometry.radians)
    surface = geometry.get_floor(surface_volume[None])[0]
    surface_i = height_map.tabulated_surface(torch.stack((geometry.radians_itf_i[0], geometry.radians_itf_i[1])))
    surface_j = height_map.tabulated_surface(torch.stack((geometry.radians_itf_j[0], geometry.radians_itf_j[1])))
    surface_i = geometry.get_itf_i_floor(surface_i[None])[0]
    surface_j = geometry.get_itf_j_floor(surface_j[None])[0]
    geometry.apply_topography(
        geometry.to_old_floor(surface),
        geometry.to_old_itf_i_floor(surface_i),
        geometry.to_old_itf_j_floor(surface_j),
        surface,
        surface_i,
        surface_j,
    )
    metric.build_metric()
    with xr.open_zarr(filename, consolidated=True) as dataset:
        levels = sorted(extract_available_levels(dataset), reverse=True)
        data = dataset["data"].sel(time=time)
        if data.sizes.get("time", 1) > 1:
            data = data.isel(time=0)
        longitude = geometry.context.to_host(geometry.lon * 180 / math.pi).flatten()
        origin = float(data.longitude.min())
        longitude = (longitude - origin) % 360 + origin
        if float(data.longitude.max()) - origin < 360:
            end = data.isel(longitude=[0]).assign_coords(longitude=[origin + 360])
            data = xr.concat((data, end), dim="longitude")
        latitude = geometry.context.to_host(geometry.lat * 180 / math.pi).flatten()
        shape = (len(levels),) + geometry.lon.shape

        def horizontal(name):
            fields = data.sel(features=[f"{name}_h{level}" for level in levels])
            values = fields.interp(longitude=("points", longitude), latitude=("points", latitude)).values
            return geometry.context.tensor(values.reshape(shape))

        source_heights = horizontal("geopotential") / gravity
        target_heights = geometry.to_single_block(metric.volume.height)

        def interpolate(name):
            block = vertical_interp(horizontal(name), source_heights, target_heights, interp_type="cubic")
            return geometry._to_new(block)

        eastward = interpolate("u_component_of_wind")
        northward = interpolate("v_component_of_wind")
        omega = interpolate("vertical_velocity")
        temperature = interpolate("temperature")
        source_pressure = torch.tensor(levels, dtype=source_heights.dtype, device=source_heights.device) * 100
        log_pressure = vertical_interp(
            torch.log(source_pressure)[:, None, None].expand_as(source_heights),
            source_heights,
            target_heights,
            interp_type="linear",
        )
        pressure = geometry._to_new(torch.exp(log_pressure))
        density = pressure / (Rd * temperature)
        theta = temperature * (p0 / pressure) ** (Rd / cpd)
        vertical = -omega / (density * gravity)
        contravariant = geometry.wind2contra(eastward, northward, vertical, metric)
    return density, *contravariant, theta
