#!/usr/bin/env python3
"""Diagnostics and publication-quality plots for DCMIP-2012 test 2-0.

The test-case document asks for day-6 native-level equatorial sections,
day-3/day-6 maps at the lowest and approximately 500-hPa model levels, the
500-hPa temperature field, and the time series of global-mean kinetic energy.

Usage:
    python scripts/plot_dcmip20.py -o results --label mountain [--pdf] results/dcmip20_mountain_rosexp.nc
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree

RD = 287.05
DAY = 86400.0
MAP_DAYS = (3.0, 6.0)
SECTION_DAY = 6.0
P500 = 50000.0


def nearest_indices(values, targets):
    return [int(numpy.argmin(numpy.abs(values - target))) for target in targets]


def equator_stencil(lat, lon):
    """Return exact-equator interpolation stencils, ordered by longitude."""
    stencils = []
    longitudes = []
    for panel in range(lat.shape[0]):
        for j in range(lat.shape[2]):
            column = lat[panel, :, j]
            crossing = numpy.flatnonzero(column[:-1] * column[1:] <= 0.0)
            if crossing.size == 0:
                continue
            i0 = int(crossing[numpy.argmin(numpy.abs(column[crossing]))])
            weight = float(-column[i0] / (column[i0 + 1] - column[i0]))
            lon0 = float(lon[panel, i0, j])
            delta = (float(lon[panel, i0 + 1, j]) - lon0 + 180.0) % 360.0 - 180.0
            longitudes.append((lon0 + weight * delta) % 360.0)
            stencils.append((panel, i0, i0 + 1, j, weight))
    order = numpy.argsort(longitudes)
    return numpy.asarray(longitudes)[order], [stencils[index] for index in order]


def at_equator(field, stencils):
    profiles = []
    for panel, i0, i1, j, weight in stencils:
        if field.ndim == 4:
            lower = field[panel, :, i0, j]
            upper = field[panel, :, i1, j]
        else:
            lower = field[panel, i0, j]
            upper = field[panel, i1, j]
        profiles.append(lower * (1.0 - weight) + upper * weight)
    return numpy.asarray(profiles)


def unit_vectors(lon, lat):
    lon = numpy.deg2rad(lon)
    lat = numpy.deg2rad(lat)
    coslat = numpy.cos(lat)
    return numpy.column_stack((coslat * numpy.cos(lon), coslat * numpy.sin(lon), numpy.sin(lat)))


class GlobalInterpolator:
    """Smooth scattered cubed-sphere data onto a regular spherical grid."""

    def __init__(self, lon, lat):
        self.longitude = numpy.linspace(0.0, 360.0, 721)
        self.latitude = numpy.linspace(-90.0, 90.0, 361)
        grid_lon, grid_lat = numpy.meshgrid(self.longitude, self.latitude)
        source = unit_vectors(lon.ravel(), lat.ravel())
        target = unit_vectors(grid_lon.ravel(), grid_lat.ravel())
        distances, self.indices = cKDTree(source).query(target, k=4)
        self.weights = 1.0 / numpy.maximum(distances, 1.0e-12) ** 2
        self.weights /= self.weights.sum(axis=1, keepdims=True)
        self.shape = grid_lon.shape

    def __call__(self, field):
        values = field.ravel()[self.indices]
        return numpy.sum(self.weights * values, axis=1).reshape(self.shape)


def interpolate_pressure_level(field, pressure, target=P500):
    result = numpy.empty(pressure.shape[0:1] + pressure.shape[2:])
    for panel in range(pressure.shape[0]):
        for i in range(pressure.shape[2]):
            for j in range(pressure.shape[3]):
                pcol = pressure[panel, :, i, j]
                qcol = field[panel, :, i, j]
                order = numpy.argsort(pcol)
                result[panel, i, j] = PchipInterpolator(pcol[order], qcol[order])(target)
    return result


def symmetric_limit(fields):
    maximum = max(float(numpy.nanmax(numpy.abs(field))) for field in fields)
    if maximum == 0.0:
        return 1.0
    scale = 10.0 ** numpy.floor(numpy.log10(maximum))
    return numpy.ceil(2.0 * maximum / scale) * 0.5 * scale


def save_figure(fig, path, pdf):
    """Write the requested figure formats."""
    fig.savefig(path, dpi=300)
    if pdf:
        fig.savefig(path.with_suffix(".pdf"), dpi=300)
    plt.close(fig)


def plot_native_sections(dataset, index, lon, stencils, output, label, pdf):
    fields = {
        "U": (r"$u$ (m s$^{-1}$)", "dcmip20_u_equator"),
        "W": (r"$w$ (m s$^{-1}$)", "dcmip20_w_equator"),
    }
    elevation = at_equator(numpy.asarray(dataset.variables["elev"][:]), stencils).T
    topography = at_equator(numpy.asarray(dataset.variables["topo"][:]), stencils)

    # Close the periodic seam explicitly; otherwise contouring can leave a
    # narrow blank strip between the first and last cubed-sphere panels.
    longitude = numpy.concatenate((lon, [lon[0] + 360.0]))
    elevation = numpy.concatenate((elevation, elevation[:, :1]), axis=1)
    topography = numpy.concatenate((topography, topography[:1]))

    for name, (colour_label, filename) in fields.items():
        section = at_equator(numpy.asarray(dataset.variables[name][index]), stencils).T
        section = numpy.concatenate((section, section[:, :1]), axis=1)
        limit = symmetric_limit([section])
        levels = numpy.linspace(-limit, limit, 22)
        fig, ax = plt.subplots(figsize=(10.0, 4.6), constrained_layout=True)
        image = ax.contourf(
            longitude[None, :] + numpy.zeros_like(elevation),
            elevation / 1000.0,
            section,
            levels=levels,
            cmap="RdBu_r",
            extend="both",
        )

        # These are the actual terrain-following solution levels. Drawing every
        # curve makes the compression and bending over the 70-degree mountain
        # directly visible without vertically remapping the field.
        for level in elevation:
            ax.plot(longitude, level / 1000.0, color="black", alpha=0.28, linewidth=0.35)
        ax.fill_between(longitude, 0.0, topography / 1000.0, color="0.35", zorder=5)

        ax.set(
            xlim=(0, 360),
            ylim=(0, 12),
            xlabel="Longitude (degrees east)",
            ylabel="Geometric height (km)",
        )
        ax.xaxis.set_major_locator(MultipleLocator(45))
        ax.grid(alpha=0.18, linewidth=0.5)
        fig.colorbar(image, ax=ax, label=colour_label, pad=0.02)
        ax.set_title(f"DCMIP 2-0 at the equator, day 6 — {label}")
        save_figure(fig, output / f"{filename}_{label}.png", pdf)


def plot_maps(dataset, indices, times, interpolator, output, label, pdf):
    for name, symbol in (("U", "u"), ("V", "v"), ("W", "w")):
        fields = []
        captions = []
        for index, time in zip(indices, times):
            pressure = numpy.asarray(dataset.variables["P"][index])
            level500 = int(numpy.argmin(numpy.abs(numpy.mean(pressure, axis=(0, 2, 3)) - P500)))
            data = numpy.asarray(dataset.variables[name][index])
            fields.extend((interpolator(data[:, 0]), interpolator(data[:, level500])))
            captions.extend(
                (f"day {time / DAY:g}, lowest level", f"day {time / DAY:g}, level {level500 + 1} (~500 hPa)")
            )

        limit = symmetric_limit(fields)
        levels = numpy.linspace(-limit, limit, 22)
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.2), sharex=True, sharey=True, constrained_layout=True)
        for ax, field, caption in zip(axes.flat, fields, captions):
            image = ax.contourf(
                interpolator.longitude, interpolator.latitude, field, levels=levels, cmap="RdBu_r", extend="both"
            )
            ax.set_title(caption)
            ax.set_xlim(0, 360)
            ax.set_ylim(-90, 90)
            ax.xaxis.set_major_locator(MultipleLocator(90))
            ax.yaxis.set_major_locator(MultipleLocator(30))
            ax.grid(alpha=0.18, linewidth=0.5)
        axes[-1, 0].set_xlabel("Longitude (degrees east)")
        axes[-1, 1].set_xlabel("Longitude (degrees east)")
        axes[0, 0].set_ylabel("Latitude (degrees north)")
        axes[1, 0].set_ylabel("Latitude (degrees north)")
        fig.colorbar(image, ax=axes, label=rf"${symbol}$ (m s$^{{-1}}$)", shrink=0.9)
        fig.suptitle(f"DCMIP 2-0 wind — {label}")
        save_figure(fig, output / f"dcmip20_{name.lower()}_maps_{label}.png", pdf)

    temperatures = []
    for index in indices:
        pressure = numpy.asarray(dataset.variables["P"][index])
        density = numpy.asarray(dataset.variables["rho"][index])
        temperatures.append(interpolator(interpolate_pressure_level(pressure / (RD * density), pressure)))
    levels = numpy.linspace(min(map(numpy.nanmin, temperatures)), max(map(numpy.nanmax, temperatures)), 21)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.2), sharex=True, sharey=True, constrained_layout=True)
    for ax, field, time in zip(axes, temperatures, times):
        image = ax.contourf(interpolator.longitude, interpolator.latitude, field, levels=levels, cmap="viridis")
        ax.set_title(f"day {time / DAY:g}")
        ax.set(xlim=(0, 360), ylim=(-90, 90), xlabel="Longitude (degrees east)")
        ax.xaxis.set_major_locator(MultipleLocator(90))
        ax.yaxis.set_major_locator(MultipleLocator(30))
        ax.grid(alpha=0.18, linewidth=0.5)
    axes[0].set_ylabel("Latitude (degrees north)")
    fig.colorbar(image, ax=axes, label=r"$T_{500}$ (K)", shrink=0.9)
    fig.suptitle(f"DCMIP 2-0 temperature at 500 hPa — {label}")
    save_figure(fig, output / f"dcmip20_t500_{label}.png", pdf)


def kinetic_energy(dataset, output, label, time_scale, pdf):
    times = numpy.asarray(dataset.variables["time"][:])
    volume = numpy.asarray(dataset.variables["volume"][:])
    denominator = numpy.sum(volume)
    values = numpy.empty(times.size)
    for index in range(times.size):
        u = numpy.asarray(dataset.variables["U"][index])
        v = numpy.asarray(dataset.variables["V"][index])
        w = numpy.asarray(dataset.variables["W"][index])
        values[index] = numpy.sum(0.5 * (u * u + v * v + w * w) * volume) / denominator

    with (output / f"dcmip20_kinetic_energy_{label}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("model_time_s", "scaled_time_s", "global_mean_kinetic_energy_m2_s-2"))
        writer.writerows(zip(times, times * time_scale, values))

    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    ax.plot(times * time_scale / DAY, values, color="#174a7e", linewidth=1.8)
    ax.set(
        xlabel="Time (days)",
        ylabel=r"Global-mean kinetic energy (m$^2$ s$^{-2}$)",
        title=f"DCMIP 2-0 pressure-gradient error — {label}",
        xlim=(0, 6),
    )
    ax.grid(alpha=0.25, linewidth=0.6)
    save_figure(fig, output / f"dcmip20_kinetic_energy_{label}.png", pdf)
    print(f"{label}: max global-mean kinetic energy = {numpy.max(values):.8e} m2 s-2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("."))
    parser.add_argument("--label", default=None, help="tag used in figure names and titles")
    parser.add_argument("--pdf", action="store_true", help="also write PDF figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    label = args.label or args.input.stem

    with netCDF4.Dataset(args.input) as dataset:
        times = numpy.asarray(dataset.variables["time"][:])
        case_number = int(getattr(dataset, "case_number", 0))
        time_scale = 500.0 if case_number == 201 else 1.0
        map_targets = tuple(day * DAY / time_scale for day in MAP_DAYS)
        section_target = SECTION_DAY * DAY / time_scale
        map_indices = nearest_indices(times, map_targets)
        section_index = nearest_indices(times, (section_target,))[0]
        lon = numpy.asarray(dataset.variables["lons"][:])
        lat = numpy.asarray(dataset.variables["lats"][:])
        equator_lon, stencils = equator_stencil(lat, lon)
        interpolator = GlobalInterpolator(lon, lat)

        plot_native_sections(dataset, section_index, equator_lon, stencils, args.output, label, args.pdf)
        plot_maps(dataset, map_indices, times[map_indices] * time_scale, interpolator, args.output, label, args.pdf)
        kinetic_energy(dataset, args.output, label, time_scale, args.pdf)


if __name__ == "__main__":
    main()
