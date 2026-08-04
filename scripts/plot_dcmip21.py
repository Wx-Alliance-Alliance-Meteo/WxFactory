#!/usr/bin/env python3
"""Publication-quality plots for DCMIP-2012 test 2-1.

The DCMIP document recommends longitude-height sections at the equator at
2400, 3600, and 7200 s.  Cubed-sphere nodes do not lie exactly on the
equator, so this script brackets latitude zero on each equatorial panel and
interpolates the model fields to it.  Treating a latitude band as a single
line produces visible stripes and must be avoided.

Both constant-height sections and counterparts on the native SLEVE levels
are written.  The former are the primary inter-model comparison; the latter
show the terrain-following coordinate deformation without vertical remapping.

Usage:
    python scripts/plot_dcmip21.py -o results --label dcmip21 [--pdf] results/dcmip21_rosexp.nc
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

# Keep diagnostics consistent with wx_factory/common/definitions.py.  Using
# 287.0 instead leaves a false ~0.052 K perturbation in the initial state.
RD = 287.05
TEQ = 300.0
SNAPSHOTS = (2400.0, 3600.0, 7200.0)
MOUNTAIN_LON = 45.0


def time_indices(times, targets):
    """Return the nearest output index for every requested time."""
    return [int(numpy.argmin(numpy.abs(times - target))) for target in targets]


def equator_stencil(lat, lon):
    """Build interpolation stencils from cubed-sphere nodes to latitude zero.

    Each returned tuple contains a panel, longitude index, the two latitude
    indices bracketing the equator, and the interpolation weight of the upper
    point.  Polar panels are excluded naturally because they do not bracket
    zero.
    """
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

            # Interpolate longitude on the circle so the 0/360 seam cannot
            # produce a spurious value near 180 degrees.
            lon0 = float(lon[panel, i0, j])
            delta = (float(lon[panel, i1, j]) - lon0 + 180.0) % 360.0 - 180.0
            longitude = (lon0 + weight * delta) % 360.0

            stencil.append((panel, j, i0, i1, weight))
            equator_lon.append(longitude)

    order = numpy.argsort(equator_lon)
    return numpy.asarray(equator_lon)[order], [stencil[i] for i in order]


def interpolate_equator(values, stencil):
    """Interpolate a ``(panel, [z,] x, y)`` array to the exact equator."""
    profiles = []
    has_vertical_axis = values.ndim == 4
    for panel, j, i0, i1, weight in stencil:
        if has_vertical_axis:
            lower = values[panel, :, i0, j]
            upper = values[panel, :, i1, j]
        else:
            lower = values[panel, i0, j]
            upper = values[panel, i1, j]
        profiles.append(lower + weight * (upper - lower))
    return numpy.asarray(profiles)


def regular_section(field, elevations, longitudes, topography, heights, plot_longitudes):
    """Interpolate terrain-following profiles to a regular longitude-height grid."""
    vertical = numpy.empty((longitudes.size, heights.size))
    for column, (z_column, q_column) in enumerate(zip(elevations, field)):
        interpolator = PchipInterpolator(z_column, q_column, extrapolate=False)
        sample_heights = numpy.clip(heights, z_column[0], z_column[-1])
        vertical[column] = interpolator(sample_heights)

    # Periodic, shape-preserving interpolation across the four equatorial
    # panels.  Adding one point on either side makes the 0/360 seam continuous.
    lon_extended = numpy.concatenate(([longitudes[-1] - 360.0], longitudes, [longitudes[0] + 360.0]))
    field_extended = numpy.concatenate((vertical[-1:], vertical, vertical[:1]), axis=0)
    image = PchipInterpolator(lon_extended, field_extended, axis=0)(plot_longitudes).T

    topo_extended = numpy.concatenate(([topography[-1]], topography, [topography[0]]))
    surface = PchipInterpolator(lon_extended, topo_extended)(plot_longitudes)
    image = numpy.ma.masked_where(heights[:, None] < surface[None, :], image)
    return image, surface


def load_sections(filename):
    """Read coordinates and only the three snapshots used in the figures."""
    with netCDF4.Dataset(filename) as dataset:
        times_all = numpy.asarray(dataset.variables["time"][:])
        indices = time_indices(times_all, SNAPSHOTS)
        times = times_all[indices]

        lat = numpy.asarray(dataset.variables["lats"][:])
        lon = numpy.asarray(dataset.variables["lons"][:])
        equator_lon, stencil = equator_stencil(lat, lon)

        elevations = interpolate_equator(numpy.asarray(dataset.variables["elev"][:]), stencil)
        topography = interpolate_equator(numpy.asarray(dataset.variables["topo"][:]), stencil)

        tprime = []
        vertical_velocity = []
        for index in indices:
            pressure = numpy.asarray(dataset.variables["P"][index])
            density = numpy.asarray(dataset.variables["rho"][index])
            w = numpy.asarray(dataset.variables["W"][index])
            tprime.append(interpolate_equator(pressure / (RD * density) - TEQ, stencil))
            vertical_velocity.append(interpolate_equator(w, stencil))

    return {
        "time": times,
        "longitude": equator_lon,
        "elevation": elevations,
        "topography": topography,
        "tprime": numpy.asarray(tprime),
        "w": numpy.asarray(vertical_velocity),
    }


def nice_symmetric_limit(images):
    """Choose a round shared limit that contains every plotted value."""
    values = numpy.concatenate([numpy.abs(numpy.ma.asarray(image).compressed()) for image in images])
    raw = float(numpy.max(values))
    if not numpy.isfinite(raw) or raw == 0.0:
        return 1.0
    exponent = 10.0 ** numpy.floor(numpy.log10(raw))
    return numpy.ceil(raw / exponent * 2.0) / 2.0 * exponent


def plot_sections(data, output_dir, label, field, symbol, units, filename, pdf):
    """Render the DCMIP sections after interpolation to geometric height."""
    heights = numpy.linspace(0.0, 30_000.0, 301)
    plot_longitudes = numpy.linspace(0.0, 360.0, 721)
    images = []
    surfaces = []
    for snapshot in data[field]:
        image, surface = regular_section(
            snapshot,
            data["elevation"],
            data["longitude"],
            data["topography"],
            heights,
            plot_longitudes,
        )
        images.append(image)
        surfaces.append(surface)

    limit = nice_symmetric_limit(data[field])
    # An even number of boundaries leaves zero in the middle of a neutral
    # colour band.  If zero itself is a boundary, roundoff-level sign changes
    # in the quiet upper atmosphere appear as distracting horizontal stripes.
    levels = numpy.linspace(-limit, limit, 22)

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.linewidth": 0.7,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.dpi": 300,
        }
    ):
        fig, axes = plt.subplots(
            len(images),
            1,
            figsize=(7.2, 6.7),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )

        contour = None
        for ax, image, surface, time in zip(axes, images, surfaces, data["time"]):
            contour = ax.contourf(
                plot_longitudes,
                heights / 1000.0,
                image,
                levels=levels,
                cmap="RdBu_r",
                extend="both",
                antialiased=False,
            )
            ax.fill_between(
                plot_longitudes,
                0.0,
                surface / 1000.0,
                color="0.25",
                linewidth=0.0,
                zorder=5,
            )
            ax.axvline(MOUNTAIN_LON, color="0.15", lw=0.55, ls=(0, (2, 2)), alpha=0.8)
            ax.text(
                0.012,
                0.93,
                f"$t={time:.0f}$ s",
                transform=ax.transAxes,
                ha="left",
                va="top",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.8},
            )
            ax.set_xlim(0.0, 360.0)
            ax.set_ylim(0.0, 30.0)
            ax.set_ylabel("Height (km)")
            ax.xaxis.set_major_locator(MultipleLocator(45.0))
            ax.yaxis.set_major_locator(MultipleLocator(5.0))
            ax.tick_params(length=3.0, width=0.7)

        axes[-1].set_xlabel("Longitude (degrees east)")
        colorbar = fig.colorbar(contour, ax=axes, location="right", shrink=0.96, pad=0.02)
        colorbar.set_label(f"{symbol} ({units})")
        colorbar.set_ticks(numpy.linspace(-limit, limit, 9))
        fig.suptitle(f"DCMIP 2-1: {symbol} at the equator, constant-height levels — {label}", fontsize=10)

        path = output_dir / filename.format(label=label)
        fig.savefig(path, bbox_inches="tight")
        if pdf:
            fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
    return (path, path.with_suffix(".pdf")) if pdf else (path,)


def periodic_interpolate(longitude, values, plot_longitude):
    """Periodically interpolate column-major values across longitude."""
    extended_lon = numpy.concatenate(([longitude[-1] - 360.0], longitude, [longitude[0] + 360.0]))
    extended_values = numpy.concatenate((values[-1:], values, values[:1]), axis=0)
    return PchipInterpolator(extended_lon, extended_values, axis=0)(plot_longitude)


def plot_native_sections(data, output_dir, label, field, symbol, units, filename, pdf):
    """Render exact-equator sections on the native terrain-following levels."""
    plot_longitude = numpy.linspace(0.0, 360.0, 721)
    plot_elevation = periodic_interpolate(data["longitude"], data["elevation"], plot_longitude)
    surface = periodic_interpolate(data["longitude"], data["topography"], plot_longitude)
    limit = nice_symmetric_limit(data[field])
    levels = numpy.linspace(-limit, limit, 22)
    x = numpy.broadcast_to(plot_longitude[:, None], plot_elevation.shape)

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.linewidth": 0.7,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.dpi": 300,
        }
    ):
        fig, axes = plt.subplots(
            len(data["time"]),
            1,
            figsize=(7.2, 6.7),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        contour = None
        for ax, snapshot, time in zip(axes, data[field], data["time"]):
            plot_field = periodic_interpolate(data["longitude"], snapshot, plot_longitude)
            contour = ax.contourf(
                x,
                plot_elevation / 1000.0,
                plot_field,
                levels=levels,
                cmap="RdBu_r",
                extend="both",
                antialiased=False,
            )
            ax.fill_between(plot_longitude, 0.0, surface / 1000.0, color="0.25", linewidth=0.0, zorder=5)
            ax.axvline(MOUNTAIN_LON, color="0.15", lw=0.55, ls=(0, (2, 2)), alpha=0.8)
            ax.text(
                0.012,
                0.93,
                f"$t={time:.0f}$ s",
                transform=ax.transAxes,
                ha="left",
                va="top",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.8},
            )
            ax.set(xlim=(0.0, 360.0), ylim=(0.0, 30.0))
            ax.set_ylabel("Height (km)")
            ax.xaxis.set_major_locator(MultipleLocator(45.0))
            ax.yaxis.set_major_locator(MultipleLocator(5.0))
            ax.tick_params(length=3.0, width=0.7)

        axes[-1].set_xlabel("Longitude (degrees east)")
        colorbar = fig.colorbar(contour, ax=axes, location="right", shrink=0.96, pad=0.02)
        colorbar.set_label(f"{symbol} ({units})")
        colorbar.set_ticks(numpy.linspace(-limit, limit, 9))
        fig.suptitle(f"DCMIP 2-1: {symbol} at the equator, native SLEVE levels — {label}", fontsize=10)

        path = output_dir / filename.format(label=label)
        fig.savefig(path, bbox_inches="tight")
        if pdf:
            fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
    return (path, path.with_suffix(".pdf")) if pdf else (path,)


def plot_topography(data, output_dir, label, pdf):
    """Plot the exact-equator Schär mountain profile."""
    longitude = data["longitude"]
    topography = data["topography"]
    keep = (longitude >= MOUNTAIN_LON - 20.0) & (longitude <= MOUNTAIN_LON + 20.0)

    dense_lon = numpy.linspace(MOUNTAIN_LON - 20.0, MOUNTAIN_LON + 20.0, 801)
    dense_topo = PchipInterpolator(longitude[keep], topography[keep])(dense_lon)

    with plt.rc_context({"font.family": "serif", "font.size": 9, "savefig.dpi": 300}):
        fig, ax = plt.subplots(figsize=(7.2, 2.5), constrained_layout=True)
        ax.plot(dense_lon, dense_topo, color="0.15", lw=1.0)
        ax.fill_between(dense_lon, 0.0, dense_topo, color="0.65", linewidth=0.0)
        ax.axvline(MOUNTAIN_LON, color="0.15", lw=0.55, ls=(0, (2, 2)))
        ax.set(xlim=(25.0, 65.0), ylim=(0.0, None), xlabel="Longitude (degrees east)", ylabel="Height (m)")
        ax.xaxis.set_major_locator(MultipleLocator(5.0))
        ax.set_title(f"DCMIP 2-1: Schär mountain at the equator — {label}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

        path = output_dir / f"dcmip21_{label}_topography.png"
        fig.savefig(path, bbox_inches="tight")
        if pdf:
            fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
    return (path, path.with_suffix(".pdf")) if pdf else (path,)


def main():
    parser = argparse.ArgumentParser(description="Publication-quality plots for DCMIP-2012 test 2-1.")
    parser.add_argument("netcdf_file", help="NetCDF output file produced by WxFactory")
    parser.add_argument("-o", "--output-dir", default="results", help="where to write the figures")
    parser.add_argument("--label", default=None, help="tag used in figure names and titles")
    parser.add_argument("--pdf", action="store_true", help="also write PDF figures")
    args = parser.parse_args()

    source = Path(args.netcdf_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or source.stem

    data = load_sections(source)
    print(f"Read {source}; selected output times: {', '.join(f'{time:g}' for time in data['time'])} s")

    figure_pairs = [
        plot_topography(data, output_dir, label, args.pdf),
        plot_sections(
            data,
            output_dir,
            label,
            field="tprime",
            symbol=r"Temperature perturbation $T^\prime$",
            units="K",
            filename="dcmip21_{label}_tprime.png",
            pdf=args.pdf,
        ),
        plot_native_sections(
            data,
            output_dir,
            label,
            field="tprime",
            symbol=r"Temperature perturbation $T^\prime$",
            units="K",
            filename="dcmip21_{label}_tprime_native.png",
            pdf=args.pdf,
        ),
        plot_sections(
            data,
            output_dir,
            label,
            field="w",
            symbol=r"Vertical velocity $w$",
            units=r"m s$^{-1}$",
            filename="dcmip21_{label}_w.png",
            pdf=args.pdf,
        ),
        plot_native_sections(
            data,
            output_dir,
            label,
            field="w",
            symbol=r"Vertical velocity $w$",
            units=r"m s$^{-1}$",
            filename="dcmip21_{label}_w_native.png",
            pdf=args.pdf,
        ),
    ]
    written = [path for pair in figure_pairs for path in pair]

    print("Figures written:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
