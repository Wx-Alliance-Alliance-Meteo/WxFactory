#!/usr/bin/env python3
"""Publication-quality plots and diagnostics for DCMIP-2012 test 1-3.

This test uses terrain-following vertical coordinates.  The script therefore
keeps the native sloping levels and constant-height interpolation as separate
diagnostics, extracts the equator exactly, and masks terrain rather than
extrapolating tracer values through it.

Usage:
    python scripts/plot_dcmip13.py -o results --label gal_chen [--pdf] results/dcmip13.nc
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree

DAY = 86400.0
CLOUD_LEVELS = (3100.0, 5100.0, 8100.0)
SECTION_HEIGHTS = (2000.0, 10000.0)
SNAPSHOT_DAYS = (0.0, 6.0, 12.0)
TRACERS = ("q1", "q2", "q3", "q4")


def time_index(times, target):
    """Index of the output nearest a requested time."""
    return int(numpy.argmin(numpy.abs(times - target)))


def load(filename):
    """Read coordinates and only days 0, 6, and 12."""
    with netCDF4.Dataset(filename) as dataset:
        all_times = numpy.asarray(dataset.variables["time"][:])
        indices = [time_index(all_times, day * DAY) for day in SNAPSHOT_DAYS]
        data = {
            "time": all_times[indices],
            "lat2d": numpy.asarray(dataset.variables["lats"][:]),
            "lon2d": numpy.asarray(dataset.variables["lons"][:]),
            "elevation": numpy.asarray(dataset.variables["elev"][:]),
            "topography": numpy.asarray(dataset.variables["topo"][:]),
            "volume": numpy.asarray(dataset.variables["volume"][:]),
        }
        data["tracers"] = {
            name: numpy.asarray([dataset.variables[name][index] for index in indices]) for name in TRACERS
        }
    return data


def level_index(elevation, target):
    """Model level closest to a nominal flat-region height."""
    nominal_height = numpy.median(elevation, axis=(0, 2, 3))
    level = int(numpy.argmin(numpy.abs(nominal_height - target)))
    return level, float(nominal_height[level])


def integral(field, volume):
    """DCMIP global integral using exact model volume weights."""
    return float((field * volume).sum())


def error_norms(q, q_exact, volume):
    """Normalized l1, l2 and l-infinity error norms."""
    error = q - q_exact
    l1 = integral(numpy.abs(error), volume) / integral(numpy.abs(q_exact), volume)
    l2 = numpy.sqrt(integral(error**2, volume)) / numpy.sqrt(integral(q_exact**2, volume))
    linf = numpy.abs(error).max() / numpy.abs(q_exact).max()
    return l1, l2, linf


def unit_sphere(longitude, latitude):
    """Cartesian unit vectors for coordinates expressed in degrees."""
    lon = numpy.deg2rad(longitude)
    lat = numpy.deg2rad(latitude)
    cos_lat = numpy.cos(lat)
    return numpy.stack((cos_lat * numpy.cos(lon), cos_lat * numpy.sin(lon), numpy.sin(lat)), axis=-1)


def spherical_resampler(lat, lon, shape=(361, 721), neighbours=4):
    """Precompute seamless inverse-distance interpolation on the sphere."""
    plot_latitude = numpy.linspace(-90.0, 90.0, shape[0])
    plot_longitude = numpy.linspace(0.0, 360.0, shape[1])
    grid_lon, grid_lat = numpy.meshgrid(plot_longitude, plot_latitude)
    tree = cKDTree(unit_sphere(lon.ravel(), lat.ravel()))
    distance, index = tree.query(unit_sphere(grid_lon, grid_lat).reshape(-1, 3), k=neighbours)
    weight = 1.0 / numpy.maximum(distance, 1.0e-12) ** 2
    weight /= weight.sum(axis=1, keepdims=True)
    return plot_longitude, plot_latitude, index, weight


def apply_spherical_resampler(values, index, weight, shape):
    """Apply a precomputed spherical interpolation stencil."""
    return numpy.sum(values.ravel()[index] * weight, axis=1).reshape(shape)


def equator_stencil(lat, lon):
    """Bracketing stencils from equatorial cubed-sphere panels to latitude zero."""
    stencil = []
    equator_lon = []
    for panel in range(lat.shape[0]):
        for j in range(lat.shape[2]):
            latitude = lat[panel, :, j]
            crossings = numpy.flatnonzero(latitude[:-1] * latitude[1:] <= 0.0)
            if crossings.size == 0:
                continue
            i0 = int(crossings[numpy.argmin(numpy.abs(latitude[crossings]))])
            i1 = i0 + 1
            weight = float(-latitude[i0] / (latitude[i1] - latitude[i0]))
            lon0 = float(lon[panel, i0, j])
            delta = (float(lon[panel, i1, j]) - lon0 + 180.0) % 360.0 - 180.0
            stencil.append((panel, j, i0, i1, weight))
            equator_lon.append((lon0 + weight * delta) % 360.0)
    order = numpy.argsort(equator_lon)
    return numpy.asarray(equator_lon)[order], [stencil[index] for index in order]


def interpolate_equator(values, stencil):
    """Interpolate a ``(panel, [z,] x, y)`` array to latitude zero."""
    profiles = []
    for panel, j, i0, i1, weight in stencil:
        if values.ndim == 4:
            lower, upper = values[panel, :, i0, j], values[panel, :, i1, j]
        else:
            lower, upper = values[panel, i0, j], values[panel, i1, j]
        profiles.append(lower + weight * (upper - lower))
    return numpy.asarray(profiles)


def periodic_extend(longitude, values):
    """Add one column on either side of a periodic longitude array."""
    extended_lon = numpy.concatenate(([longitude[-1] - 360.0], longitude, [longitude[0] + 360.0]))
    extended_values = numpy.concatenate((values[-1:], values, values[:1]), axis=0)
    return extended_lon, extended_values


def constant_height_section(field, elevation, longitude, topography, heights, plot_longitude):
    """Interpolate terrain-following columns to constant geometric heights."""
    vertical = numpy.empty((longitude.size, heights.size))
    for column, (z_column, q_column) in enumerate(zip(elevation, field)):
        sample_height = numpy.clip(heights, z_column[0], z_column[-1])
        vertical[column] = PchipInterpolator(z_column, q_column)(sample_height)

    extended_lon, extended_values = periodic_extend(longitude, vertical)
    image = PchipInterpolator(extended_lon, extended_values, axis=0)(plot_longitude).T

    topo_lon, topo_values = periodic_extend(longitude, topography)
    surface = PchipInterpolator(topo_lon, topo_values)(plot_longitude)
    image = numpy.ma.masked_where(heights[:, None] < surface[None, :], image)
    return image, surface


def tracer_limits(data, name="q4"):
    """Initial-state colour range shared by every time panel."""
    initial = data["tracers"][name][0]
    return float(initial.min()), float(initial.max())


def save_figure(fig, path, pdf):
    """Write the requested figure formats."""
    fig.savefig(path, bbox_inches="tight")
    if pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return (path, path.with_suffix(".pdf")) if pdf else (path,)


def plot_topography(data, output_dir, label, pdf):
    """True Schär-type surface elevation."""
    longitude, latitude, index, weight = spherical_resampler(data["lat2d"], data["lon2d"])
    image = apply_spherical_resampler(
        data["topography"],
        index,
        weight,
        (latitude.size, longitude.size),
    )
    levels = numpy.linspace(0.0, float(data["topography"].max()), 22)

    with plt.rc_context({"font.family": "serif", "font.size": 9, "savefig.dpi": 300}):
        fig, ax = plt.subplots(figsize=(7.2, 2.9), constrained_layout=True)
        contour = ax.contourf(
            longitude,
            latitude,
            image,
            levels=levels,
            cmap="terrain",
            extend="max",
            antialiased=False,
        )
        colorbar = fig.colorbar(contour, ax=ax, pad=0.02)
        colorbar.set_label("Surface elevation (m)")
        ax.set(xlim=(0.0, 360.0), ylim=(-90.0, 90.0))
        ax.xaxis.set_major_locator(MultipleLocator(60.0))
        ax.yaxis.set_major_locator(MultipleLocator(30.0))
        ax.set_xlabel("Longitude (degrees east)")
        ax.set_ylabel("Latitude (degrees north)")
        ax.set_title(f"DCMIP 1-3 surface topography — {label}", fontsize=10)
        path = output_dir / f"dcmip13_{label}_topography.png"
        return save_figure(fig, path, pdf)


def plot_model_level_maps(data, output_dir, label, pdf):
    """q4 maps on the three requested terrain-following model levels."""
    plot_lon, plot_lat, index, weight = spherical_resampler(data["lat2d"], data["lon2d"])
    selected_levels = [level_index(data["elevation"], target) for target in CLOUD_LEVELS]
    vmin, vmax = tracer_limits(data)
    boundaries = numpy.linspace(vmin, vmax, 22)

    with plt.rc_context({"font.family": "serif", "font.size": 8, "savefig.dpi": 300}):
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.35), constrained_layout=True, sharex=True, sharey=True)
        contour = None
        for row, day in enumerate((6.0, 12.0)):
            snapshot = data["tracers"]["q4"][time_index(data["time"], day * DAY)]
            for column, (level, nominal_height) in enumerate(selected_levels):
                ax = axes[row, column]
                image = apply_spherical_resampler(
                    snapshot[:, level],
                    index,
                    weight,
                    (plot_lat.size, plot_lon.size),
                )
                contour = ax.contourf(
                    plot_lon,
                    plot_lat,
                    image,
                    levels=boundaries,
                    cmap="viridis",
                    extend="both",
                    antialiased=False,
                )
                if row == 0:
                    ax.set_title(f"Level {level + 1} ({nominal_height:.0f} m)", fontsize=8)
                if column == 0:
                    ax.set_ylabel(f"Day {day:g}\nLatitude (°N)")
                if row == 1:
                    ax.set_xlabel("Longitude (°E)")
                ax.set(xlim=(0.0, 360.0), ylim=(-90.0, 90.0))
                ax.xaxis.set_major_locator(MultipleLocator(120.0))
                ax.yaxis.set_major_locator(MultipleLocator(45.0))

        colorbar = fig.colorbar(contour, ax=axes, shrink=0.95, pad=0.015)
        colorbar.set_label(r"Total tracer $q_4$")
        fig.suptitle(f"DCMIP 1-3: $q_4$ on terrain-following model levels — {label}", fontsize=10)
        path = output_dir / f"dcmip13_{label}_latlon.png"
        return save_figure(fig, path, pdf)


def plot_native_sections(data, output_dir, label, pdf):
    """Exact-equator q4 sections on the native sloping model levels."""
    longitude, stencil = equator_stencil(data["lat2d"], data["lon2d"])
    elevation = interpolate_equator(data["elevation"], stencil)
    topography = interpolate_equator(data["topography"], stencil)
    extended_lon, extended_elevation = periodic_extend(longitude, elevation)
    plot_longitude = numpy.linspace(0.0, 360.0, 721)
    plot_elevation = PchipInterpolator(extended_lon, extended_elevation, axis=0)(plot_longitude)
    topo_lon, topo_values = periodic_extend(longitude, topography)
    surface = PchipInterpolator(topo_lon, topo_values)(plot_longitude)
    vmin, vmax = tracer_limits(data)

    with plt.rc_context({"font.family": "serif", "font.size": 9, "savefig.dpi": 300}):
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.75), constrained_layout=True, sharex=True, sharey=True)
        contour = None
        for ax, day in zip(axes, (6.0, 12.0)):
            snapshot = data["tracers"]["q4"][time_index(data["time"], day * DAY)]
            equator_field = interpolate_equator(snapshot, stencil)
            field_lon, field_values = periodic_extend(longitude, equator_field)
            plot_field = PchipInterpolator(field_lon, field_values, axis=0)(plot_longitude)
            x = numpy.broadcast_to(plot_longitude[:, None], plot_elevation.shape)
            contour = ax.contourf(
                x,
                plot_elevation / 1000.0,
                plot_field,
                levels=numpy.linspace(vmin, vmax, 22),
                cmap="viridis",
                extend="both",
                antialiased=False,
            )
            ax.fill_between(plot_longitude, 0.0, surface / 1000.0, color="0.25", linewidth=0.0)
            ax.text(
                0.012,
                0.92,
                f"Day {day:g}",
                transform=ax.transAxes,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.8},
            )
            ax.set_ylabel("Height (km)")
            ax.set(xlim=(0.0, 360.0), ylim=(0.0, 12.0))
            ax.yaxis.set_major_locator(MultipleLocator(2.0))

        axes[-1].set_xlabel("Longitude (degrees east)")
        axes[-1].xaxis.set_major_locator(MultipleLocator(60.0))
        colorbar = fig.colorbar(contour, ax=axes, shrink=0.95, pad=0.02)
        colorbar.set_label(r"Total tracer $q_4$")
        fig.suptitle(f"DCMIP 1-3: exact-equator $q_4$ on native sloping levels — {label}", fontsize=10)
        path = output_dir / f"dcmip13_{label}_lonlevel_q4.png"
        return save_figure(fig, path, pdf)


def plot_constant_height_sections(data, output_dir, label, pdf):
    """Exact-equator q4 sections interpolated to geometric height."""
    longitude, stencil = equator_stencil(data["lat2d"], data["lon2d"])
    elevation = interpolate_equator(data["elevation"], stencil)
    topography = interpolate_equator(data["topography"], stencil)
    heights = numpy.linspace(*SECTION_HEIGHTS, 321)
    plot_longitude = numpy.linspace(0.0, 360.0, 721)
    vmin, vmax = tracer_limits(data)

    images = []
    surfaces = []
    for day in (6.0, 12.0):
        snapshot = data["tracers"]["q4"][time_index(data["time"], day * DAY)]
        equator_field = interpolate_equator(snapshot, stencil)
        image, surface = constant_height_section(
            equator_field,
            elevation,
            longitude,
            topography,
            heights,
            plot_longitude,
        )
        images.append(image)
        surfaces.append(surface)

    with plt.rc_context({"font.family": "serif", "font.size": 9, "savefig.dpi": 300}):
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.75), constrained_layout=True, sharex=True, sharey=True)
        contour = None
        for ax, day, image, surface in zip(axes, (6.0, 12.0), images, surfaces):
            contour = ax.contourf(
                plot_longitude,
                heights / 1000.0,
                image,
                levels=numpy.linspace(vmin, vmax, 22),
                cmap="viridis",
                extend="both",
                antialiased=False,
            )
            ax.fill_between(plot_longitude, 0.0, surface / 1000.0, color="0.25", linewidth=0.0)
            ax.text(
                0.012,
                0.92,
                f"Day {day:g}",
                transform=ax.transAxes,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.8},
            )
            ax.set_ylabel("Height (km)")
            ax.set_ylim(SECTION_HEIGHTS[0] / 1000.0, SECTION_HEIGHTS[1] / 1000.0)
            ax.yaxis.set_major_locator(MultipleLocator(2.0))

        axes[-1].set_xlabel("Longitude (degrees east)")
        axes[-1].xaxis.set_major_locator(MultipleLocator(60.0))
        colorbar = fig.colorbar(contour, ax=axes, shrink=0.95, pad=0.02)
        colorbar.set_label(r"Total tracer $q_4$")
        fig.suptitle(f"DCMIP 1-3: exact-equator $q_4$ on constant-height levels — {label}", fontsize=10)
        path = output_dir / f"dcmip13_{label}_lonheight_q4.png"
        return save_figure(fig, path, pdf)


def report_error_norms(data, label):
    """Error norms at day 12, where the exact solution is the initial state."""
    final_index = time_index(data["time"], 12.0 * DAY)
    print(f"\nNormalized error norms at day 12 ({label}, exact solution = initial state)")
    print(f"  {'tracer':>6}  {'l1':>12}  {'l2':>12}  {'l_inf':>12}")
    for name in TRACERS:
        field = data["tracers"][name]
        l1, l2, linf = error_norms(field[final_index], field[0], data["volume"])
        print(f"  {name:>6}  {l1:12.4e}  {l2:12.4e}  {linf:12.4e}")


def main():
    parser = argparse.ArgumentParser(description="Publication-quality plots for DCMIP-2012 test 1-3.")
    parser.add_argument("netcdf_file", help="NetCDF output file produced by WxFactory")
    parser.add_argument("-o", "--output-dir", default="results", help="where to write the figures")
    parser.add_argument("--label", default=None, help="tag used in figure names and titles")
    parser.add_argument("--pdf", action="store_true", help="also write PDF figures")
    args = parser.parse_args()

    source = Path(args.netcdf_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or source.stem
    data = load(source)
    days = data["time"] / DAY
    print(f"Read {source}; selected diagnostic days: {', '.join(f'{day:g}' for day in days)}")

    figure_pairs = [
        plot_topography(data, output_dir, label, args.pdf),
        plot_model_level_maps(data, output_dir, label, args.pdf),
        plot_native_sections(data, output_dir, label, args.pdf),
        plot_constant_height_sections(data, output_dir, label, args.pdf),
    ]
    written = [path for pair in figure_pairs for path in pair]
    report_error_norms(data, label)

    print("\nFigures written:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
