#!/usr/bin/env python3
"""Publication-quality diagnostics for DCMIP-2012 test 3-1.

The specification requests a longitude-height section of potential-
temperature departure from the analytic base state and a line at z=5.5 km,
both along the equator at t=3000 s. Vertical velocity is included as a useful
non-hydrostatic companion diagnostic.

Usage:
    python scripts/plot_dcmip31.py -o results --label dcmip31 [--pdf] results/dcmip31_rosexp.nc
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy
from matplotlib.ticker import MultipleLocator

GRAVITY = 9.80616
RD = 287.05
CPD = 1005.46
P0 = 100000.0
TEQ = 300.0
U0 = 20.0
N = 0.01
PERTURBATION_LONGITUDE = 120.0
TARGET_TIME = 3000.0
TARGET_HEIGHT = 5500.0
REFERENCE_EARTH_RADIUS = 6371220.0
PLANET_FACTOR = 125.0


def equator_stencil(lat, lon):
    """Build exact-equator interpolation stencils ordered by longitude."""
    stencils = []
    longitudes = []
    for panel in range(lat.shape[0]):
        for j in range(lat.shape[2]):
            column = lat[panel, :, j]
            crossings = numpy.flatnonzero(column[:-1] * column[1:] <= 0.0)
            if crossings.size == 0:
                continue
            i0 = int(crossings[numpy.argmin(numpy.abs(column[crossings]))])
            weight = float(-column[i0] / (column[i0 + 1] - column[i0]))
            lon0 = float(lon[panel, i0, j])
            delta = (float(lon[panel, i0 + 1, j]) - lon0 + 180.0) % 360.0 - 180.0
            longitudes.append((lon0 + weight * delta) % 360.0)
            stencils.append((panel, i0, i0 + 1, j, weight))
    order = numpy.argsort(longitudes)
    return numpy.asarray(longitudes)[order], [stencils[index] for index in order]


def at_equator(field, stencils):
    """Interpolate a ``(panel, [z,] x, y)`` field to latitude zero."""
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


def analytic_base_theta(latitude, height):
    """Equation 97 of the DCMIP-2012 specification."""
    kappa = RD / CPD
    big_g = GRAVITY**2 / (N**2 * CPD)
    surface_temperature = big_g + (TEQ - big_g) * numpy.exp(
        -(U0 * N**2 / (4.0 * GRAVITY**2)) * U0 * (numpy.cos(2.0 * latitude) - 1.0)
    )
    surface_pressure = (
        P0
        * numpy.exp(U0**2 / (4.0 * big_g * RD) * (numpy.cos(2.0 * latitude) - 1.0))
        * (surface_temperature / TEQ) ** (1.0 / kappa)
    )
    return surface_temperature * (P0 / surface_pressure) ** kappa * numpy.exp(N**2 * height / GRAVITY)


def symmetric_limit(field):
    maximum = float(numpy.nanmax(numpy.abs(field)))
    if maximum == 0.0:
        return 1.0
    scale = 10.0 ** numpy.floor(numpy.log10(maximum))
    return numpy.ceil(2.0 * maximum / scale) * 0.5 * scale


def wavefront_longitudes(time):
    """Equation 112 phase-speed estimates at the equator."""
    intrinsic_speed = N * 20000.0 / (2.0 * math.pi)
    radius = REFERENCE_EARTH_RADIUS / PLANET_FACTOR
    speeds = (U0 - intrinsic_speed, U0 + intrinsic_speed)
    return tuple((PERTURBATION_LONGITUDE + math.degrees(speed * time / radius)) % 360.0 for speed in speeds)


def save_figure(fig, output, stem, pdf):
    fig.savefig(output / f"{stem}.png", dpi=300)
    if pdf:
        fig.savefig(output / f"{stem}.pdf", dpi=300)
    plt.close(fig)


def close_periodic(field):
    return numpy.concatenate((field, field[..., :1]), axis=-1)


def plot_initial_condition(longitude, elevation, theta_prime, output, label, pdf):
    """Plot the prescribed perturbation before the dynamical adjustment."""
    mean_heights = numpy.mean(elevation[:, :-1], axis=1)
    level = int(numpy.argmin(numpy.abs(mean_heights - TARGET_HEIGHT)))
    maximum = float(numpy.nanmax(theta_prime))
    levels = numpy.linspace(0.0, maximum if maximum > 0.0 else 1.0, 21)
    lon_mesh = longitude[None, :] + numpy.zeros_like(elevation)

    fig, axes = plt.subplots(
        2, 1, figsize=(10.0, 6.4), gridspec_kw={"height_ratios": (2.2, 1.0)}, constrained_layout=True
    )
    image = axes[0].contourf(lon_mesh, elevation / 1000.0, theta_prime, levels=levels, cmap="YlOrRd", extend="max")
    fig.colorbar(image, ax=axes[0], label=r"$\theta-\overline{\theta}$ (K)", pad=0.015)
    axes[0].axvline(PERTURBATION_LONGITUDE, color="black", linestyle="--", linewidth=0.8)
    axes[0].set(ylabel="Height (km)", ylim=(0, 10), xlim=(0, 360))
    axes[0].yaxis.set_major_locator(MultipleLocator(1))
    axes[0].grid(alpha=0.18, linewidth=0.5)

    axes[1].plot(longitude, theta_prime[level], color="#b33b1e", linewidth=1.8)
    axes[1].axvline(PERTURBATION_LONGITUDE, color="black", linestyle="--", linewidth=0.8)
    axes[1].set(
        xlim=(0, 360),
        xlabel="Longitude (degrees east)",
        ylabel=r"$\theta-\overline{\theta}$ (K)",
        title=f"z={mean_heights[level] / 1000.0:.1f} km",
    )
    axes[1].xaxis.set_major_locator(MultipleLocator(45))
    axes[1].grid(alpha=0.22, linewidth=0.5)
    fig.suptitle(f"DCMIP 3-1 initial potential-temperature perturbation — {label}")
    save_figure(fig, output, f"dcmip31_initial_condition_{label}", pdf)


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
        index = int(numpy.argmin(numpy.abs(times - TARGET_TIME)))
        if abs(float(times[index]) - TARGET_TIME) > 51.0:
            raise ValueError(f"No output near 3000 s; closest snapshot is {times[index]:g} s")

        latitude = numpy.asarray(dataset.variables["lats"][:])
        longitude = numpy.asarray(dataset.variables["lons"][:])
        equator_lon, stencils = equator_stencil(latitude, longitude)
        elevation = at_equator(numpy.asarray(dataset.variables["elev"][:]), stencils).T
        w = at_equator(numpy.asarray(dataset.variables["W"][index]), stencils).T

        latitude_radians = numpy.deg2rad(latitude)
        latitude_3d = numpy.broadcast_to(latitude_radians[:, None, :, :], dataset.variables["theta"][index].shape)
        base = analytic_base_theta(latitude_3d, numpy.asarray(dataset.variables["elev"][:]))
        theta_prime = at_equator(numpy.asarray(dataset.variables["theta"][index]) - base, stencils).T
        initial_theta_prime = at_equator(numpy.asarray(dataset.variables["theta"][0]) - base, stencils).T

    # Explicitly close the periodic longitude seam.
    longitude_closed = numpy.concatenate((equator_lon, [equator_lon[0] + 360.0]))
    elevation = close_periodic(elevation)
    theta_prime = close_periodic(theta_prime)
    initial_theta_prime = close_periodic(initial_theta_prime)
    w = close_periodic(w)
    lon_mesh = longitude_closed[None, :] + numpy.zeros_like(elevation)

    plot_initial_condition(longitude_closed, elevation, initial_theta_prime, args.output, label, args.pdf)

    angular_distance = numpy.abs(
        (numpy.deg2rad(longitude_closed - PERTURBATION_LONGITUDE) + math.pi) % (2 * math.pi) - math.pi
    )
    distance = REFERENCE_EARTH_RADIUS / PLANET_FACTOR * angular_distance
    horizontal_shape = 5000.0**2 / (5000.0**2 + distance**2)
    expected_initial = horizontal_shape[None, :] * numpy.sin(2.0 * math.pi * elevation / 20000.0)
    initial_error = float(numpy.max(numpy.abs(initial_theta_prime - expected_initial)))

    fig, axes = plt.subplots(2, 1, figsize=(10.2, 7.0), sharex=True, constrained_layout=True)
    for ax, field, colour_label in (
        (axes[0], theta_prime, r"$\theta-\overline{\theta}$ (K)"),
        (axes[1], w, r"$w$ (m s$^{-1}$)"),
    ):
        limit = symmetric_limit(field)
        image = ax.contourf(
            lon_mesh, elevation / 1000.0, field, levels=numpy.linspace(-limit, limit, 22), cmap="RdBu_r", extend="both"
        )
        fig.colorbar(image, ax=ax, label=colour_label, pad=0.015)
        ax.set(ylabel="Height (km)", ylim=(0, 10))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.grid(alpha=0.18, linewidth=0.5)
    axes[1].set(xlabel="Longitude (degrees east)", xlim=(0, 360))
    axes[1].xaxis.set_major_locator(MultipleLocator(45))
    fig.suptitle(f"DCMIP 3-1 gravity wave at t={times[index]:g} s — {label}")
    save_figure(fig, args.output, f"dcmip31_sections_{label}", args.pdf)

    mean_heights = numpy.mean(elevation[:, :-1], axis=1)
    level = int(numpy.argmin(numpy.abs(mean_heights - TARGET_HEIGHT)))
    line = theta_prime[level]
    west, east = wavefront_longitudes(float(times[index]))
    fig, ax = plt.subplots(figsize=(9.0, 4.0), constrained_layout=True)
    ax.plot(longitude_closed, line, color="#174a7e", linewidth=1.8)
    ax.axhline(0.0, color="0.25", linewidth=0.7)
    ax.axvline(west, color="#c44e52", linestyle="--", linewidth=1.0, label="phase-speed estimates")
    ax.axvline(east, color="#c44e52", linestyle="--", linewidth=1.0)
    ax.set(
        xlim=(0, 360),
        xlabel="Longitude (degrees east)",
        ylabel=r"$\theta-\overline{\theta}$ (K)",
        title=f"DCMIP 3-1 at z={mean_heights[level] / 1000.0:.1f} km, t={times[index]:g} s — {label}",
    )
    ax.xaxis.set_major_locator(MultipleLocator(45))
    ax.grid(alpha=0.22, linewidth=0.5)
    ax.legend(frameon=False)
    save_figure(fig, args.output, f"dcmip31_theta_5p5km_{label}", args.pdf)

    print(
        f"initial theta'=[{numpy.min(initial_theta_prime):.6g}, {numpy.max(initial_theta_prime):.6g}] K "
        f"(max analytic error {initial_error:.3e} K); "
        f"snapshot={times[index]:g} s, level={mean_heights[level]:.1f} m, "
        f"theta'=[{numpy.min(theta_prime):.6g}, {numpy.max(theta_prime):.6g}] K, "
        f"predicted fronts=({west:.2f}, {east:.2f}) degrees"
    )


if __name__ == "__main__":
    main()
