#!/usr/bin/env python3
"""Plots for DCMIP-2012 test 2-1 (non-hydrostatic mountain waves over a Schar-type mountain).

Takes the NetCDF file written by WxFactory and produces what the DCMIP-2012 Test Case Document
(v1.7, section 2.X / 2.1) suggests for analysis:

  * longitude-height cross sections of the temperature perturbation T'(lambda, z) = T - Teq along
    the equator, at t = 2400 s, 3600 s and 7200 s (the document's suggested snapshots), plus the
    initial state for reference. The 2400 s and 3600 s frames catch the wave before it has wrapped
    around the small planet; the 7200 s frame is after it has interfered with itself and the sponge;
  * the same sections for the vertical velocity w, which the document also recommends analysing;
  * the surface elevation (the Schar-type mountain), for reference.

The model writes pressure P and density rho, so the temperature is recovered from the ideal gas law
T = P / (Rd rho). For test 2-1 the background is isothermal at Teq = 300 K, so T' is zero at t = 0
and is created entirely by the flow over the mountain.

Usage:
    python3 scripts/plot_dcmip21.py results/dcmip21.nc [-o output_dir] [--label sleve]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy

# Gas constant for dry air and the reference (equatorial) temperature of the balanced state.
RD = 287.0
TEQ = 300.0

# The document's suggested analysis snapshots (seconds, unscaled Earth time).
SNAPSHOTS = (0.0, 2400.0, 3600.0, 7200.0)

# Longitude of the mountain centre (lambda_c = pi / 4), marked on the sections for orientation.
MOUNTAIN_LON = 45.0


def load(filename):
    """Read pressure, density, vertical velocity and the grid, on the (panel, z, x, y) layout."""
    with netCDF4.Dataset(filename) as ds:
        time = numpy.asarray(ds.variables["time"][:])
        lat = numpy.asarray(ds.variables["lats"][:])
        lon = numpy.asarray(ds.variables["lons"][:])
        elev = numpy.asarray(ds.variables["elev"][:])
        topo = numpy.asarray(ds.variables["topo"][:])
        pressure = numpy.asarray(ds.variables["P"][:])
        rho = numpy.asarray(ds.variables["rho"][:])
        w = numpy.asarray(ds.variables["W"][:])

    # Temperature from the ideal gas law, then its perturbation from the isothermal background.
    temperature = pressure / (RD * rho)
    tprime = temperature - TEQ

    return {
        "time": time,
        "lat2d": lat,
        "lon2d": lon,
        "elev": elev,
        "topo": topo,
        "tprime": tprime,
        "w": w,
    }


def time_index(time, target):
    """Index of the output closest to `target` seconds."""
    return int(numpy.argmin(numpy.abs(time - target)))


def equator_columns(data, tolerance=2.0):
    """The (panel, x, y) columns whose foot lies near the equator.

    Returns the column longitudes and, for each column, the full vertical profile of heights. The
    model levels follow the terrain and are unevenly spaced (Gauss-Legendre points, clustered near
    the element boundaries), so the interpolation to constant heights has to be done one column at a
    time, up a monotonic line. Triangulating the raw points in the (longitude, height) plane instead
    stripes the picture.
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


def section_image(field_it, mask, z_cols, lon_cols, heights, longitudes, order, lon_sorted):
    """Interpolate one snapshot onto the regular (longitude, height) grid, column by column."""
    q_cols = numpy.moveaxis(field_it, 1, -1)[mask]  # (panel, z, x, y) -> (panel, x, y, z), mask -> (ncol, nz)

    # First up each monotonic column onto the regular heights, then across longitude (0/360 wrap).
    # Outside the column's height range there is no data; hold the nearest value so the terrain foot
    # and the model top do not leave blank bands.
    on_heights = numpy.empty((len(lon_cols), heights.size))
    for c in range(len(lon_cols)):
        on_heights[c] = numpy.interp(heights, z_cols[c], q_cols[c])

    image = numpy.empty((heights.size, longitudes.size))
    for r in range(heights.size):
        image[r] = numpy.interp(longitudes, lon_sorted, on_heights[order, r], period=360.0)
    return image


def plot_sections(data, outdir, label, field, cmap, title, filename, symmetric=True):
    """Longitude-height equatorial sections of `field` at the document's snapshots."""
    lon_cols, z_cols, mask, tolerance = equator_columns(data)
    zbot, ztop = 0.0, float(numpy.max(data["elev"]))

    heights = numpy.linspace(zbot, ztop, 241)
    longitudes = numpy.linspace(0.0, 360.0, 481)
    order = numpy.argsort(lon_cols)
    lon_sorted = lon_cols[order]

    # Build every image first so a single colour scale, set from the evolved field, spans them all.
    # The initial perturbation is essentially zero, so it must not drive the scale.
    images = []
    times = []
    for target in SNAPSHOTS:
        it = time_index(data["time"], target)
        images.append(section_image(data[field][it], mask, z_cols, lon_cols, heights, longitudes, order, lon_sorted))
        times.append(float(data["time"][it]))

    evolved = numpy.concatenate([img.ravel() for img, t in zip(images, times) if t > 0.0])
    scale = float(numpy.percentile(numpy.abs(evolved), 99.9)) or 1.0
    if symmetric:
        vmin, vmax = -scale, scale
    else:
        vmin, vmax = 0.0, scale

    fig, axes = plt.subplots(len(images), 1, figsize=(11.0, 2.7 * len(images)), constrained_layout=True, sharex=True)
    axes = numpy.atleast_1d(axes)

    for ax, image, t in zip(axes, images, times):
        rendered = ax.imshow(
            image,
            origin="lower",
            extent=[0.0, 360.0, zbot / 1000.0, ztop / 1000.0],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="bilinear",
        )
        fig.colorbar(rendered, ax=ax, shrink=0.9)
        ax.axvline(MOUNTAIN_LON, color="k", lw=0.6, ls=":", alpha=0.5)
        ax.set_title(f"t = {t:.0f} s   [{image.min():.2f}, {image.max():.2f}]")
        ax.set_ylabel("height (km)")

    axes[-1].set_xlabel("longitude (deg)")
    fig.suptitle(f"DCMIP 2-1 ({label}): {title} along the equator (|lat| < {tolerance:.1f} deg)")
    path = os.path.join(outdir, filename.format(label=label))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_topography(data, outdir, label):
    """The Schar-type surface elevation along the equator.

    A full lat-lon map is dominated by the flat background (the Schar mountain is only 250 m tall and
    compact), so a zoom on the equatorial profile around the mountain centre is more informative. The
    true surface height is used (the `topo` field), not the first model level, which sits about 170 m
    up because the lowest Gauss-Legendre node is offset from the terrain floor.
    """
    _, _, mask, tolerance = equator_columns(data)
    lon_cols = data["lon2d"][mask]
    zs = data["topo"][mask]
    order = numpy.argsort(lon_cols)

    fig, ax = plt.subplots(figsize=(9.0, 3.4), constrained_layout=True)
    ax.plot(lon_cols[order], zs[order], color="saddlebrown")
    ax.fill_between(lon_cols[order], 0.0, zs[order], color="saddlebrown", alpha=0.3)
    ax.axvline(MOUNTAIN_LON, color="k", lw=0.6, ls=":", alpha=0.5)
    ax.set_xlim(MOUNTAIN_LON - 20.0, MOUNTAIN_LON + 20.0)
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("surface height (m)")
    ax.set_title(f"DCMIP 2-1 ({label}): Schar-type mountain at the equator, max = {zs.max():.0f} m")

    path = os.path.join(outdir, f"dcmip21_{label}_topography.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Plots for DCMIP-2012 test 2-1.")
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

    times = data["time"]
    print(f"Read {args.netcdf_file}: {len(times)} output times, from {times[0]:g} s to {times[-1]:g} s")
    print(f"  T' range over the run: [{data['tprime'].min():.3f}, {data['tprime'].max():.3f}] K")
    print(f"  w  range over the run: [{data['w'].min():.3f}, {data['w'].max():.3f}] m/s")

    written = [
        plot_topography(data, args.output_dir, label),
        plot_sections(
            data,
            args.output_dir,
            label,
            field="tprime",
            cmap="RdBu_r",
            title="temperature perturbation T' = T - Teq (K)",
            filename="dcmip21_{label}_tprime.png",
        ),
        plot_sections(
            data,
            args.output_dir,
            label,
            field="w",
            cmap="RdBu_r",
            title="vertical velocity w (m/s)",
            filename="dcmip21_{label}_w.png",
        ),
    ]

    print("\nFigures written:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
