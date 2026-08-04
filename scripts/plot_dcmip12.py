#!/usr/bin/env python3
"""Publication-quality plots and diagnostics for DCMIP-2012 test 1-2.

The DCMIP document requests latitude-height sections of ``q1`` at longitude
180 degrees after 12 and 24 hours, plus normalized error norms at 24 hours.
Cubed-sphere nodes do not lie on that meridian, so the section is sampled in
Cartesian unit-sphere coordinates rather than by combining a longitude band.

Usage:
    python scripts/plot_dcmip12.py -o results --label dcmip12 [--pdf] results/dcmip12.nc
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

HOUR = 3600.0
SECTION_LONGITUDE = 180.0
SNAPSHOT_HOURS = (0.0, 12.0, 24.0)


def time_index(times, target):
    """Index of the output nearest a requested time."""
    return int(numpy.argmin(numpy.abs(times - target)))


def load(filename):
    """Read coordinates and only the snapshots needed for plots and norms."""
    with netCDF4.Dataset(filename) as dataset:
        all_times = numpy.asarray(dataset.variables["time"][:])
        indices = [time_index(all_times, hour * HOUR) for hour in SNAPSHOT_HOURS]
        return {
            "time": all_times[indices],
            "lat2d": numpy.asarray(dataset.variables["lats"][:]),
            "lon2d": numpy.asarray(dataset.variables["lons"][:]),
            "elevation": numpy.asarray(dataset.variables["elev"][:]),
            "volume": numpy.asarray(dataset.variables["volume"][:]),
            "q1": numpy.asarray([dataset.variables["q1"][index] for index in indices]),
        }


def integral(field, volume):
    """DCMIP global integral using the model's exact volume weights."""
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


def meridian_stencil(lat, lon, longitude, latitudes, neighbours=4):
    """Spherical inverse-distance stencil for one exact meridian."""
    tree = cKDTree(unit_sphere(lon.ravel(), lat.ravel()))
    target_lon = numpy.full_like(latitudes, longitude)
    distance, index = tree.query(unit_sphere(target_lon, latitudes), k=neighbours)
    weight = 1.0 / numpy.maximum(distance, 1.0e-12) ** 2
    weight /= weight.sum(axis=1, keepdims=True)
    return index, weight


def sample_meridian(values, index, weight):
    """Apply a horizontal meridian stencil to every vertical level."""
    num_levels = values.shape[1]
    sampled = numpy.empty((index.shape[0], num_levels))
    for level in range(num_levels):
        horizontal = values[:, level].ravel()
        sampled[:, level] = numpy.sum(horizontal[index] * weight, axis=1)
    return sampled


def regular_section(field, elevations, native_latitudes, heights, plot_latitudes):
    """Interpolate meridional profiles vertically and then across latitude."""
    vertical = numpy.empty((native_latitudes.size, heights.size))
    for column, (z_column, q_column) in enumerate(zip(elevations, field)):
        sample_height = numpy.clip(heights, z_column[0], z_column[-1])
        vertical[column] = PchipInterpolator(z_column, q_column)(sample_height)
    return PchipInterpolator(native_latitudes, vertical, axis=0)(plot_latitudes).T


def plot_latitude_height(data, output_dir, label, pdf):
    """Render the DCMIP-recommended 12 h and 24 h latitude-height sections."""
    native_latitudes = numpy.linspace(-90.0, 90.0, 361)
    plot_latitudes = numpy.linspace(-90.0, 90.0, 721)
    heights = numpy.linspace(0.0, 12_000.0, 301)
    index, weight = meridian_stencil(
        data["lat2d"],
        data["lon2d"],
        SECTION_LONGITUDE,
        native_latitudes,
    )

    elevations = sample_meridian(data["elevation"], index, weight)
    initial_min = float(data["q1"][0].min())
    initial_max = float(data["q1"][0].max())
    levels = numpy.linspace(initial_min, initial_max, 22)

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.linewidth": 0.7,
            "savefig.dpi": 300,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True, sharex=True, sharey=True)
        contour = None
        for ax, hour in zip(axes, (12.0, 24.0)):
            snapshot = data["q1"][time_index(data["time"], hour * HOUR)]
            meridian = sample_meridian(snapshot, index, weight)
            image = regular_section(meridian, elevations, native_latitudes, heights, plot_latitudes)
            contour = ax.contourf(
                plot_latitudes,
                heights / 1000.0,
                image,
                levels=levels,
                cmap="viridis",
                extend="both",
                antialiased=False,
            )
            ax.text(
                0.03,
                0.95,
                rf"$t={hour:g}$ h",
                transform=ax.transAxes,
                ha="left",
                va="top",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.8},
            )
            ax.set_xlim(-90.0, 90.0)
            ax.set_ylim(0.0, 12.0)
            ax.xaxis.set_major_locator(MultipleLocator(30.0))
            ax.yaxis.set_major_locator(MultipleLocator(2.0))
            ax.set_xlabel("Latitude (degrees north)")
            ax.tick_params(length=3.0, width=0.7)

        axes[0].set_ylabel("Height (km)")
        colorbar = fig.colorbar(contour, ax=axes, shrink=0.95, pad=0.02)
        colorbar.set_label(r"Tracer $q_1$")
        fig.suptitle(rf"DCMIP 1-2: $q_1$ at $\lambda={SECTION_LONGITUDE:g}^\circ$ — {label}", fontsize=10)

        path = output_dir / f"dcmip12_{label}_latheight.png"
        fig.savefig(path, bbox_inches="tight")
        if pdf:
            fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
    return (path, path.with_suffix(".pdf")) if pdf else (path,)


def report_error_norms(data):
    """Report error at 24 h, when the exact solution is the initial state."""
    final = data["q1"][time_index(data["time"], 24.0 * HOUR)]
    l1, l2, linf = error_norms(final, data["q1"][0], data["volume"])

    print("\nNormalized error norms for q1 at t = 24 h (exact solution = initial state)")
    print(f"  {'l1':>12}  {'l2':>12}  {'l_inf':>12}")
    print(f"  {l1:12.4e}  {l2:12.4e}  {linf:12.4e}")


def main():
    parser = argparse.ArgumentParser(description="Publication-quality plots for DCMIP-2012 test 1-2.")
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
    hours = data["time"] / HOUR
    print(f"Read {args.netcdf_file}; selected diagnostic hours: {', '.join(f'{hour:g}' for hour in hours)}")

    written = plot_latitude_height(data, output_dir, label, args.pdf)
    report_error_norms(data)

    print("\nFigures written:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
