#!/usr/bin/env python3
"""Plots and diagnostics for DCMIP-2012 test 1-2 (3D Hadley-like meridional circulation).

Takes the NetCDF file written by WxFactory and produces what the DCMIP-2012 Test Case Document
(v1.7, section 1.2) asks for:

  * latitude-height cross sections of the tracer q1 at lambda = 180 degrees, at t = 12 h and
    t = 24 h (the document's suggested analysis), plus the initial state for reference;
  * the normalized l1, l2 and l_inf error norms for q1 at t = 1 day, measured against the initial
    condition -- which is also the exact solution, since the flow reverses over one period.

Unlike test 1-1 this case carries a single tracer, and there are no mixing or correlation
diagnostics to compute.

Usage:
    python3 scripts/plot_dcmip12.py results/dcmip12.nc [-o output_dir]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy

HOUR = 3600.0

# The document suggests a cross section along this meridian.
SECTION_LONGITUDE = 180.0


def load(filename):
    """Read the fields we need, on the (panel, z, x, y) grid."""
    with netCDF4.Dataset(filename) as ds:
        time = numpy.asarray(ds.variables["time"][:])
        lat = numpy.asarray(ds.variables["lats"][:])
        lon = numpy.asarray(ds.variables["lons"][:])
        elev = numpy.asarray(ds.variables["elev"][:])
        volume = numpy.asarray(ds.variables["volume"][:])
        q1 = numpy.asarray(ds.variables["q1"][:])

    # lats/lons are (panel, x, y): broadcast them over the vertical so they match the tracer.
    num_z = elev.shape[1]
    lat = numpy.repeat(lat[:, numpy.newaxis], num_z, axis=1)
    lon = numpy.repeat(lon[:, numpy.newaxis], num_z, axis=1)

    return {"time": time, "lat": lat, "lon": lon, "elev": elev, "volume": volume, "q1": q1}


def time_index(time, target):
    """Index of the output closest to `target` seconds."""
    return int(numpy.argmin(numpy.abs(time - target)))


def integral(field, volume):
    """The DCMIP global integral I[x] = sum_j x_j V_j (the volume weights are exact)."""
    return float((field * volume).sum())


def error_norms(q, q_exact, volume):
    """Normalized l1, l2 and l_inf error norms."""
    err = q - q_exact
    l1 = integral(numpy.abs(err), volume) / integral(numpy.abs(q_exact), volume)
    l2 = numpy.sqrt(integral(err**2, volume)) / numpy.sqrt(integral(q_exact**2, volume))
    linf = numpy.abs(err).max() / numpy.abs(q_exact).max()
    return l1, l2, linf


def meridian_columns(data, longitude, tolerance=2.0):
    """The (panel, x, y) columns lying near a meridian, with their vertical profiles.

    The cubed sphere has no column exactly on a given meridian, so widen the tolerance until the band
    holds enough columns. The model levels are unevenly spaced Gauss-Legendre points, so the
    interpolation to regular heights has to be done one column at a time, up a monotonic line;
    triangulating the raw points in the (latitude, height) plane instead stripes the picture.
    """
    lon2d = data["lon"][:, 0]  # (panel, x, y)
    lat2d = data["lat"][:, 0]

    # Angular distance to the meridian, taking the 0/360 wrap into account.
    distance = numpy.abs((lon2d - longitude + 180.0) % 360.0 - 180.0)
    columns = numpy.sort(distance.ravel())
    minimum_points = max(3, int(0.01 * columns.size))
    tolerance = max(tolerance, columns[minimum_points])

    mask = distance < tolerance  # (panel, x, y)
    z_cols = numpy.moveaxis(data["elev"], 1, -1)[mask]  # (ncol, nz)
    lat_cols = lat2d[mask]
    return lat_cols, z_cols, mask, tolerance


def plot_lat_height(data, hours, outdir):
    """Latitude-height cross sections of q1 along the section meridian, one panel per time."""
    lat_cols, z_cols, mask, tolerance = meridian_columns(data, SECTION_LONGITUDE)

    # Bound the axes by the data. The outermost grid points fall short of the poles and of the
    # ground, and extending the axes past them would only leave empty strips.
    latmin, latmax = float(lat_cols.min()), float(lat_cols.max())
    zbot, ztop = float(z_cols.min()), float(z_cols.max())

    latitudes = numpy.linspace(latmin, latmax, 181)
    heights = numpy.linspace(zbot, ztop, 181)
    order = numpy.argsort(lat_cols)
    lat_sorted = lat_cols[order]

    # A colour scale fixed on the initial state makes the times comparable, and lets the
    # over- and undershoots of an unlimited scheme show up as saturation.
    initial = data["q1"][0]
    vmin, vmax = float(initial.min()), float(initial.max())

    fig, axes = plt.subplots(1, len(hours), figsize=(5.2 * len(hours), 4.0), constrained_layout=True, sharey=True)
    axes = numpy.atleast_1d(axes)

    for ax, hour in zip(axes, hours):
        it = time_index(data["time"], hour * HOUR)
        q_cols = numpy.moveaxis(data["q1"][it], 1, -1)[mask]  # (panel, z, x, y) -> (ncol, nz)

        # First up each monotonic column onto the regular heights, then across latitude.
        on_heights = numpy.empty((len(lat_cols), heights.size))
        for c in range(len(lat_cols)):
            on_heights[c] = numpy.interp(heights, z_cols[c], q_cols[c])

        image = numpy.empty((heights.size, latitudes.size))
        for r in range(heights.size):
            image[r] = numpy.interp(latitudes, lat_sorted, on_heights[order, r])

        rendered = ax.imshow(
            image,
            origin="lower",
            extent=[latmin, latmax, zbot, ztop],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="bilinear",
        )
        fig.colorbar(rendered, ax=ax, shrink=0.85)
        ax.set_title(f"t = {hour:g} h   [{image.min():.3f}, {image.max():.3f}]")
        ax.set_xlabel("latitude (deg)")

    axes[0].set_ylabel("height (m)")
    fig.suptitle(
        f"DCMIP 1-2: tracer q1 along lambda = {SECTION_LONGITUDE:g} deg "
        f"(|lon - {SECTION_LONGITUDE:g}| < {tolerance:.1f} deg)"
    )
    path = os.path.join(outdir, "dcmip12_latheight.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def report_error_norms(data):
    """Error norms at the end of the period, where the exact solution is the initial state."""
    it = time_index(data["time"], data["time"].max())
    hours = data["time"][it] / HOUR

    l1, l2, linf = error_norms(data["q1"][it], data["q1"][0], data["volume"])

    print(f"\nNormalized error norms for q1 at t = {hours:g} h (exact solution = initial state)")
    print(f"  {'l1':>12}  {'l2':>12}  {'l_inf':>12}")
    print(f"  {l1:12.4e}  {l2:12.4e}  {linf:12.4e}")


def main():
    parser = argparse.ArgumentParser(description="Plots and diagnostics for DCMIP-2012 test 1-2.")
    parser.add_argument("netcdf_file", help="NetCDF output file produced by WxFactory")
    parser.add_argument("-o", "--output-dir", default="results", help="where to write the figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load(args.netcdf_file)

    hours = data["time"] / HOUR
    print(f"Read {args.netcdf_file}: {len(hours)} output times, from {hours[0]:g} h to {hours[-1]:g} h")

    # Hour 0 is the initial condition, and (the flow being periodic) the exact solution at hour 24.
    written = [plot_lat_height(data, (0, 12, 24), args.output_dir)]

    report_error_norms(data)

    print("\nFigures written:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
