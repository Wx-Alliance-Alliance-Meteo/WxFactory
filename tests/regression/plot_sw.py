#!/usr/bin/env python3
"""Generic shallow-water cubed-sphere plotter for the regression suite.

The eight SW cases (case1/2/5/6, vortex, galewsky, matsuno, unsteady_zonal)
have no dedicated plot script, so this provides a small, deterministic one that
follows the ``plot_dcmip*.py`` convention:

* CLI: ``plot_sw.py <netcdf> -o <dir> --label <tag> [--pdf]``
* fixed colour limits (the initial-state range, so limits are stable across
  runs and overshoots stay visible),
* fixed dpi (300),
* deterministic file names ``sw_{case}_{var}_{t}.png``.

Figures, at three fixed times (t=0, t_mid, t_end):

* ``sw_{case}_h_{t}.png``      — fluid height ``h`` on a regular lat-lon map
  (cubed-sphere resampled with inverse-distance weighting on the unit sphere).
* ``sw_{case}_speed_{t}.png``  — wind speed ``sqrt(U^2+V^2)`` lat-lon map, only
  when ``U``/``V`` are present (i.e. ``case_number >= 2``).

The lat-lon resampler is a self-contained copy of the pattern used in
``scripts/plot_dcmip11.py`` so that this script has no cross-script imports.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy
from matplotlib.ticker import MultipleLocator
from scipy.spatial import cKDTree

DPI = 300
# Regular lat-lon grid used for the resampled maps.
LATLON_SHAPE = (181, 361)  # (lat, lon)


# --------------------------------------------------------------------------- #
# Cubed-sphere -> lat-lon resampling (self-contained copy of the DCMIP pattern)
# --------------------------------------------------------------------------- #


def unit_sphere(longitude, latitude):
    """Cartesian unit vectors for coordinates expressed in degrees."""
    lon = numpy.deg2rad(longitude)
    lat = numpy.deg2rad(latitude)
    cos_lat = numpy.cos(lat)
    return numpy.stack((cos_lat * numpy.cos(lon), cos_lat * numpy.sin(lon), numpy.sin(lat)), axis=-1)


def spherical_resampler(lat, lon, shape=LATLON_SHAPE, neighbours=4):
    """Map cubed-sphere (lat, lon) points onto a regular lat-lon grid.

    Uses unit-sphere distances to avoid longitude seams and polar
    singularities. Returns the plot axes plus the KDTree index/weight arrays
    used by :func:`apply_spherical_resampler`.
    """
    plot_latitudes = numpy.linspace(-90.0, 90.0, shape[0])
    plot_longitudes = numpy.linspace(0.0, 360.0, shape[1])
    grid_lon, grid_lat = numpy.meshgrid(plot_longitudes, plot_latitudes)

    tree = cKDTree(unit_sphere(lon.ravel(), lat.ravel()))
    distance, index = tree.query(unit_sphere(grid_lon, grid_lat).reshape(-1, 3), k=neighbours)
    weight = 1.0 / numpy.maximum(distance, 1.0e-12) ** 2
    weight /= weight.sum(axis=1, keepdims=True)
    return plot_longitudes, plot_latitudes, index, weight


def apply_spherical_resampler(values, index, weight, shape):
    return numpy.sum(values.ravel()[index] * weight, axis=1).reshape(shape)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load(filename):
    """Read the SW fields and coordinates needed for the plots."""
    with netCDF4.Dataset(filename) as ds:
        time = numpy.asarray(ds.variables["time"][:])
        lat = numpy.asarray(ds.variables["lats"][:])
        lon = numpy.asarray(ds.variables["lons"][:])
        h = numpy.asarray(ds.variables["h"][:])
        has_wind = "U" in ds.variables and "V" in ds.variables
        if has_wind:
            u = numpy.asarray(ds.variables["U"][:])
            v = numpy.asarray(ds.variables["V"][:])
        else:
            u = v = None

    return {"time": time, "lat": lat, "lon": lon, "h": h, "u": u, "v": v, "has_wind": has_wind}


def time_indices(time):
    """Indices for the three fixed times: first, middle, last."""
    n = time.size
    if n == 1:
        return [0]
    return [0, n // 2, n - 1]


def time_labels(indices):
    """Stable, human-readable labels for the selected time indices."""
    labels = []
    for i, idx in enumerate(indices):
        if i == 0:
            labels.append("t0")
        elif i == len(indices) - 1:
            labels.append("tlast")
        else:
            labels.append("tmid")
    return labels


def color_range(data, name):
    """Fix colour limits to the initial range so overshoots remain visible."""
    field = data[name]
    initial = field[0]
    return float(initial.min()), float(initial.max())


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def plot_h_latlon(data, it, tlabel, label, outdir, pdf):
    """Fluid height ``h`` on a regular lat-lon map at one time."""
    longitudes, latitudes, index, weight = spherical_resampler(data["lat"], data["lon"])
    image = apply_spherical_resampler(data["h"][it], index, weight, (latitudes.size, longitudes.size))
    vmin, vmax = color_range(data, "h")

    with plt.rc_context({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.7, "savefig.dpi": DPI}):
        fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
        contour = ax.contourf(
            longitudes,
            latitudes,
            image,
            levels=numpy.linspace(vmin, vmax, 22),
            cmap="viridis",
            extend="both",
            antialiased=False,
        )
        colorbar = fig.colorbar(contour, ax=ax, shrink=0.9, pad=0.02)
        colorbar.set_label("h (m)")
        ax.set_title(f"SW {label}: fluid height h ({tlabel})", fontsize=10)
        ax.set_xlim(0.0, 360.0)
        ax.set_ylim(-90.0, 90.0)
        ax.xaxis.set_major_locator(MultipleLocator(60.0))
        ax.yaxis.set_major_locator(MultipleLocator(30.0))
        ax.grid(color="white", lw=0.35, alpha=0.35)
        ax.set_xlabel("Longitude (degrees east)")
        ax.set_ylabel("Latitude (degrees north)")

        path = outdir / f"sw_{label}_h_{tlabel}.png"
        fig.savefig(path, bbox_inches="tight")
        if pdf:
            fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
    return (path, path.with_suffix(".pdf")) if pdf else (path,)


def plot_speed_latlon(data, it, tlabel, label, outdir, pdf):
    """Wind speed ``sqrt(U^2+V^2)`` on a regular lat-lon map at one time."""
    longitudes, latitudes, index, weight = spherical_resampler(data["lat"], data["lon"])
    speed = numpy.sqrt(data["u"][it] ** 2 + data["v"][it] ** 2)
    image = apply_spherical_resampler(speed, index, weight, (latitudes.size, longitudes.size))
    # Stable, case-fixed range: the initial wind-speed range.
    speed0 = numpy.sqrt(data["u"][0] ** 2 + data["v"][0] ** 2)
    vmin, vmax = 0.0, float(speed0.max())

    with plt.rc_context({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.7, "savefig.dpi": DPI}):
        fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
        contour = ax.contourf(
            longitudes,
            latitudes,
            image,
            levels=numpy.linspace(vmin, vmax, 22),
            cmap="magma",
            extend="both",
            antialiased=False,
        )
        colorbar = fig.colorbar(contour, ax=ax, shrink=0.9, pad=0.02)
        colorbar.set_label("|u| (m/s)")
        ax.set_title(f"SW {label}: wind speed ({tlabel})", fontsize=10)
        ax.set_xlim(0.0, 360.0)
        ax.set_ylim(-90.0, 90.0)
        ax.xaxis.set_major_locator(MultipleLocator(60.0))
        ax.yaxis.set_major_locator(MultipleLocator(30.0))
        ax.grid(color="white", lw=0.35, alpha=0.35)
        ax.set_xlabel("Longitude (degrees east)")
        ax.set_ylabel("Latitude (degrees north)")

        path = outdir / f"sw_{label}_speed_{tlabel}.png"
        fig.savefig(path, bbox_inches="tight")
        if pdf:
            fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
    return (path, path.with_suffix(".pdf")) if pdf else (path,)


def plot_time(data, it, tlabel, label, outdir, pdf):
    """Plot all figures for one selected time. Returns the written paths."""
    paths = []
    paths += plot_h_latlon(data, it, tlabel, label, outdir, pdf)
    if data["has_wind"]:
        paths += plot_speed_latlon(data, it, tlabel, label, outdir, pdf)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Generic shallow-water cubed-sphere plots.")
    parser.add_argument("netcdf_file", help="NetCDF output file produced by WxFactory")
    parser.add_argument("-o", "--output-dir", default="results", help="where to write the figures")
    parser.add_argument("--label", default=None, help="tag used in figure names and titles")
    parser.add_argument("--pdf", action="store_true", help="also write PDF figures")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    source = Path(args.netcdf_file)
    label = args.label or source.stem
    data = load(source)

    indices = time_indices(data["time"])
    labels = time_labels(indices)
    print(
        f"Read {args.netcdf_file}; {data['time'].size} timesteps, "
        f"wind={'yes' if data['has_wind'] else 'no'}; plotting times {labels}"
    )

    for it, tlabel in zip(indices, labels):
        for path in plot_time(data, it, tlabel, label, outdir, args.pdf):
            print(f"  wrote {path}")


if __name__ == "__main__":
    main()
