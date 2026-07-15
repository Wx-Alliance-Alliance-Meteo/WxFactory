#!/usr/bin/env python3
"""Plots and diagnostics for DCMIP-2012 test 1-3 (thin cloud-like tracers over orography).

Takes the NetCDF file written by WxFactory and produces what the DCMIP-2012 Test Case Document
(v1.7, section 1.3) asks for:

  * latitude-longitude cross sections of q4 at the model levels closest to z = 3100, 5100 and
    8100 m, at t = 6 and 12 days;
  * longitude-height cross sections of q4 along the equator at t = 6 and 12 days, interpolated to
    constant height levels between 2000 and 10000 m (the model levels themselves follow the
    terrain, so a plot on model levels would fold the mountain into the picture);
  * the normalized l1, l2 and l_inf error norms for q1 ... q4 at t = 12 days, measured against the
    initial condition -- which is also the exact solution, since the flow carries the clouds exactly
    once around the sphere in 12 days;
  * the initial state, for reference.

Usage:
    python3 scripts/plot_dcmip13.py results/dcmip13.nc [-o output_dir] [--label gal_chen]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy
from scipy.interpolate import griddata

DAY = 86400.0

# The document asks for the model levels closest to these heights (Table XI puts the cloud decks at
# 3050, 5050 and 8200 m).
CLOUD_LEVELS = (3100.0, 5100.0, 8100.0)

# Vertical extent of the longitude-height sections.
SECTION_HEIGHTS = (2000.0, 10000.0)

TRACERS = ("q1", "q2", "q3", "q4")


def load(filename):
    """Read the tracers and the grid, on the (panel, z, x, y) layout."""
    with netCDF4.Dataset(filename) as ds:
        time = numpy.asarray(ds.variables["time"][:])
        lat = numpy.asarray(ds.variables["lats"][:])
        lon = numpy.asarray(ds.variables["lons"][:])
        elev = numpy.asarray(ds.variables["elev"][:])
        volume = numpy.asarray(ds.variables["volume"][:])
        tracers = {name: numpy.asarray(ds.variables[name][:]) for name in TRACERS}

    # Keep the horizontal coordinates on their native (panel, x, y) grid for the column-wise vertical
    # interpolation, and also broadcast them over the vertical for the whole-volume diagnostics.
    num_z = elev.shape[1]
    lat3 = numpy.repeat(lat[:, numpy.newaxis], num_z, axis=1)
    lon3 = numpy.repeat(lon[:, numpy.newaxis], num_z, axis=1)

    data = {
        "time": time,
        "lat": lat3,
        "lon": lon3,
        "lat2d": lat,
        "lon2d": lon,
        "elev": elev,
        "volume": volume,
    }
    data.update(tracers)
    return data


def time_index(time, target):
    """Index of the output closest to `target` seconds."""
    return int(numpy.argmin(numpy.abs(time - target)))


def level_index(elev, target):
    """The model level closest to a given height.

    The levels follow the terrain, so their height varies from column to column. The median over the
    sphere is dominated by the flat regions, where the levels sit at their nominal height, which is
    what the document means by "the model level closest to z".
    """
    heights = numpy.median(elev.reshape(elev.shape[0], elev.shape[1], -1), axis=(0, 2))
    k = int(numpy.argmin(numpy.abs(heights - target)))
    return k, float(heights[k])


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


def scatter_to_grid(x, y, values, x_range, y_range, shape, method="linear"):
    """Interpolate the scattered cubed-sphere points onto a regular grid.

    Contouring the raw points triangulates them in the plotting plane, which on a cubed sphere leaves
    slivers along the panel edges and wedges near the poles. Interpolating first, and letting imshow
    smooth, avoids all of that. Linear interpolation is safe for fields with sharp edges (the box
    cloud), but leaves visible triangle facets on a smooth, finely oscillating field near the poles;
    "cubic" removes them and is appropriate wherever the field is smooth.
    """
    grid_x, grid_y = numpy.meshgrid(
        numpy.linspace(x_range[0], x_range[1], shape[1]),
        numpy.linspace(y_range[0], y_range[1], shape[0]),
    )
    return griddata((x, y), values, (grid_x, grid_y), method=method)


def latlon_image(lon, lat, values, shape=(361, 721), method="linear"):
    """A lat-lon image, with the points duplicated across the 0/360 seam.

    Without the duplication the interpolation has nothing to work with on either side of the seam,
    and leaves a blank stripe down the middle of the map.
    """
    lon = numpy.concatenate([lon - 360.0, lon, lon + 360.0])
    lat = numpy.concatenate([lat, lat, lat])
    values = numpy.concatenate([values, values, values])
    image = scatter_to_grid(lon, lat, values, (0.0, 360.0), (-90.0, 90.0), shape, method=method)

    # The cubed sphere's outermost points fall a little short of the poles, so the top and bottom
    # rows of the target grid lie outside the data and come back as NaN, which imshow would draw as
    # blank strips. The tracer is zero there, so fill them with zero.
    return numpy.nan_to_num(image, nan=0.0)


def equator_band(lat, tolerance=2.0):
    """The grid points lying near the equator, widening the band until it holds enough points."""
    distance = numpy.abs(lat)
    rows = numpy.sort(distance[:, 0].ravel())
    minimum_points = max(3, int(0.01 * rows.size))
    tolerance = max(tolerance, rows[minimum_points])
    return distance < tolerance, tolerance


def plot_latlon(data, days, outdir, label):
    """Lat-lon cross sections of q4 at the three cloud levels, one row per day."""
    q4 = data["q4"]
    vmin, vmax = float(q4[0].min()), float(q4[0].max())

    levels = [level_index(data["elev"], target) for target in CLOUD_LEVELS]

    fig, axes = plt.subplots(
        len(days), len(levels), figsize=(5.6 * len(levels), 3.0 * len(days)), constrained_layout=True
    )
    axes = numpy.atleast_2d(axes)

    for row, day in enumerate(days):
        it = time_index(data["time"], day * DAY)
        for col, (k, height) in enumerate(levels):
            ax = axes[row, col]
            q = q4[it][:, k].ravel()
            image = latlon_image(data["lon"][:, k].ravel(), data["lat"][:, k].ravel(), q)

            rendered = ax.imshow(
                image,
                origin="lower",
                extent=[0.0, 360.0, -90.0, 90.0],
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
                interpolation="bilinear",
            )
            fig.colorbar(rendered, ax=ax, shrink=0.85)
            ax.set_title(f"day {day:g},  z = {height:.0f} m   [{q.min():.3f}, {q.max():.3f}]")
            if row == len(days) - 1:
                ax.set_xlabel("longitude (deg)")
            if col == 0:
                ax.set_ylabel("latitude (deg)")

    fig.suptitle(f"DCMIP 1-3 ({label}): total tracer q4 on the model levels nearest the cloud decks")
    path = os.path.join(outdir, f"dcmip13_{label}_latlon.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def equator_columns(data, tolerance=2.0):
    """The (panel, x, y) columns whose foot lies near the equator.

    Returns the column longitudes and, for each column, the full vertical profile of heights. The
    model levels follow the terrain and are unevenly spaced (Gauss-Legendre points, clustered near
    the element boundaries), so the interpolation to constant heights has to be done one column at a
    time, up a monotonic line. Triangulating the raw points in the (longitude, height) plane instead
    is what stripes the picture.
    """
    lat2d, lon2d = data["lat2d"], data["lon2d"]
    distance = numpy.abs(lat2d)
    columns = numpy.sort(distance.ravel())
    minimum_points = max(3, int(0.01 * columns.size))
    tolerance = max(tolerance, columns[minimum_points])

    mask = distance < tolerance  # (panel, x, y)
    # elev is (panel, z, x, y); pick the near-equator columns and put height on the last axis.
    z_cols = numpy.moveaxis(data["elev"], 1, -1)[mask]  # (ncol, nz)
    lon_cols = lon2d[mask]  # (ncol,)
    return lon_cols, z_cols, mask, tolerance


def plot_lon_height(data, days, outdir, label, field="q4"):
    """Longitude-height cross sections along the equator, interpolated to constant heights."""
    lon_cols, z_cols, mask, tolerance = equator_columns(data)
    zbot, ztop = SECTION_HEIGHTS

    heights = numpy.linspace(zbot, ztop, 161)
    longitudes = numpy.linspace(0.0, 360.0, 361)
    order = numpy.argsort(lon_cols)
    lon_sorted = lon_cols[order]

    initial = data[field][0]
    vmin, vmax = float(initial.min()), float(initial.max())

    fig, axes = plt.subplots(len(days), 1, figsize=(11.0, 3.2 * len(days)), constrained_layout=True, sharex=True)
    axes = numpy.atleast_1d(axes)

    for ax, day in zip(axes, days):
        it = time_index(data["time"], day * DAY)
        q_cols = numpy.moveaxis(data[field][it], 1, -1)[mask]  # (panel, z, x, y) -> (ncol, nz)

        # First interpolate each column onto the regular heights (exact, up a monotonic line),
        # then interpolate across longitude, wrapping around the 0/360 seam.
        on_heights = numpy.empty((len(lon_cols), heights.size))
        for c in range(len(lon_cols)):
            on_heights[c] = numpy.interp(heights, z_cols[c], q_cols[c], left=0.0, right=0.0)

        image = numpy.empty((heights.size, longitudes.size))
        for r in range(heights.size):
            row = on_heights[order, r]
            image[r] = numpy.interp(longitudes, lon_sorted, row, period=360.0)

        rendered = ax.imshow(
            image,
            origin="lower",
            extent=[0.0, 360.0, zbot, ztop],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="bilinear",
        )
        fig.colorbar(rendered, ax=ax, shrink=0.9)
        ax.set_title(f"day {day:g}   [{image.min():.3f}, {image.max():.3f}]")
        ax.set_ylabel("height (m)")

    axes[-1].set_xlabel("longitude (deg)")
    fig.suptitle(
        f"DCMIP 1-3 ({label}): {field} along the equator (|lat| < {tolerance:.1f} deg), on height levels"
    )
    path = os.path.join(outdir, f"dcmip13_{label}_lonheight_{field}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_topography(data, outdir, label):
    """The mountain, as the model sees it: the height of the lowest model level."""
    zs = data["elev"][:, 0].ravel()
    # The topography is smooth, so cubic interpolation is safe and removes the triangle facets that
    # linear interpolation leaves on the finely oscillating mountain near the poles. Cubic can
    # overshoot right at the convex-hull edge (the outermost latitudes), so clip back to the data
    # range to keep those pixels clean.
    image = latlon_image(data["lon"][:, 0].ravel(), data["lat"][:, 0].ravel(), zs, method="cubic")
    image = numpy.clip(image, float(zs.min()), float(zs.max()))

    fig, ax = plt.subplots(figsize=(11.0, 4.4), constrained_layout=True)
    rendered = ax.imshow(
        image,
        origin="lower",
        extent=[0.0, 360.0, -90.0, 90.0],
        cmap="terrain",
        aspect="auto",
        interpolation="bilinear",
    )
    fig.colorbar(rendered, ax=ax, shrink=0.9, label="surface height (m)")
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    ax.set_title(f"DCMIP 1-3 ({label}): surface elevation, max = {zs.max():.0f} m")

    path = os.path.join(outdir, f"dcmip13_{label}_topography.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def report_error_norms(data, label):
    """Error norms at day 12, where the exact solution is the initial state."""
    it = time_index(data["time"], data["time"].max())
    days = data["time"][it] / DAY

    print(f"\nNormalized error norms at t = {days:g} days ({label}, exact solution = initial state)")
    print(f"  {'tracer':>6}  {'l1':>12}  {'l2':>12}  {'l_inf':>12}")
    for name in TRACERS:
        l1, l2, linf = error_norms(data[name][it], data[name][0], data["volume"])
        print(f"  {name:>6}  {l1:12.4e}  {l2:12.4e}  {linf:12.4e}")


def main():
    parser = argparse.ArgumentParser(description="Plots and diagnostics for DCMIP-2012 test 1-3.")
    parser.add_argument("netcdf_file", help="NetCDF output file produced by WxFactory")
    parser.add_argument("-o", "--output-dir", default="results", help="where to write the figures")
    parser.add_argument(
        "--label",
        default=None,
        help="tag for the figure names and titles (defaults to the name of the NetCDF file)",
    )
    args = parser.parse_args()

    label = args.label or os.path.splitext(os.path.basename(args.netcdf_file))[0]

    os.makedirs(args.output_dir, exist_ok=True)
    data = load(args.netcdf_file)

    days = data["time"] / DAY
    print(f"Read {args.netcdf_file}: {len(days)} output times, from day {days[0]:g} to day {days[-1]:g}")

    written = [
        plot_topography(data, args.output_dir, label),
        plot_latlon(data, (0, 6, 12), args.output_dir, label),
        plot_lon_height(data, (0, 6, 12), args.output_dir, label),
    ]

    report_error_norms(data, label)

    print("\nFigures written:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
