#!/usr/bin/env python3
"""Plots and diagnostics for DCMIP-2012 test 1-1 (3D deformational flow).

Takes the NetCDF file written by WxFactory and produces the figures and numbers that the DCMIP-2012
Test Case Document (v1.7, section 1.1 and Appendix A) asks for:

  * lat-lon cross sections of q1..q4 at 4900 m, at day 6 (maximum deformation) and day 12 (the flow
    has reversed, so the exact solution is the initial condition again);
  * longitude-height cross sections along the equator at day 12;
  * the normalized l1, l2 and l_inf error norms at day 12, measured against the initial state;
  * the mixing diagnostics l_r (real mixing), l_u (range-preserving unmixing) and l_o (overshooting)
    at day 6, computed from the (q1, q2) pairs on the five levels near 4500-5300 m;
  * the q1-q2 correlation scatter plots on those same five levels at day 6.

Usage:
    python3 scripts/plot_dcmip11.py results/dcmip11.nc [-o output_dir]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import netCDF4
import numpy
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree

# The initial tracer distribution is bounded by these values, and q2 follows q1 along the curve
# psi(chi) = 0.9 - 0.8 chi^2 (DCMIP eq. 173 and the definition of q2).
CHI_MIN, CHI_MAX = 0.0, 1.0
XI_MIN, XI_MAX = 0.1, 0.9

# DCMIP restricts the mixing diagnostics and the correlation plots to these five levels.
MIXING_LEVELS = (4500.0, 4700.0, 4900.0, 5100.0, 5300.0)

DAY = 86400.0


def psi(chi):
    """The initial correlation curve between q1 and q2 (DCMIP eq. 167)."""
    return 0.9 - 0.8 * chi**2


def load(filename):
    """Read coordinates and only the snapshots needed by the DCMIP diagnostics."""
    with netCDF4.Dataset(filename) as ds:
        all_times = numpy.asarray(ds.variables["time"][:])
        indices = [time_index(all_times, day * DAY) for day in (0.0, 6.0, 12.0)]
        time = all_times[indices]
        lat = numpy.asarray(ds.variables["lats"][:])
        lon = numpy.asarray(ds.variables["lons"][:])
        elev = numpy.asarray(ds.variables["elev"][:])
        volume = numpy.asarray(ds.variables["volume"][:])
        tracers = {
            name: numpy.asarray([ds.variables[name][index] for index in indices])
            for name in ("q1", "q2", "q3", "q4")
        }

    return {
        "time": time,
        "lat2d": lat,
        "lon2d": lon,
        "elev": elev,
        "volume": volume,
        "tracers": tracers,
    }


def time_index(time, target):
    """Index of the output closest to `target` seconds."""
    return int(numpy.argmin(numpy.abs(time - target)))


def level_index(elev, target):
    """Index of the model level whose mean height is closest to `target` metres."""
    mean_height = elev.mean(axis=(0, 2, 3))
    return int(numpy.argmin(numpy.abs(mean_height - target)))


def integral(field, volume):
    """The DCMIP global integral I[x] = sum_j x_j V_j (the volume weights are exact)."""
    return float((field * volume).sum())


def error_norms(q, q_exact, volume):
    """Normalized l1, l2 and l_inf error norms (DCMIP section on error measures)."""
    err = q - q_exact
    l1 = integral(numpy.abs(err), volume) / integral(numpy.abs(q_exact), volume)
    l2 = numpy.sqrt(integral(err**2, volume)) / numpy.sqrt(integral(q_exact**2, volume))
    linf = numpy.abs(err).max() / numpy.abs(q_exact).max()
    return l1, l2, linf


def distance_to_curve(chi, xi):
    """Normalized distance from each (chi, xi) pair to the initial correlation curve.

    Follows DCMIP eqs. 169-172: the closest point on the parabola is found analytically as the root
    of a cubic. The discriminant is negative over much of the range, so the cube root is taken in
    the complex plane and only the real part is kept.
    """
    chi = numpy.asarray(chi, dtype=float)
    xi = numpy.asarray(xi, dtype=float)

    chi_c = chi.astype(numpy.complex128)
    xi_c = xi.astype(numpy.complex128)

    # Minimizing the normalized distance to psi leads to the cubic chi^3 + p*chi - chi_k/2 = 0,
    # with p = (5/8)(2 xi_k - 1); Cardano's root is c - p/(3c). The factor 6 under the radical is
    # missing from eq. 169 as printed, but it is what makes the root satisfy the cubic.
    c = (432.0 * chi_c + 6.0 * numpy.sqrt(750.0 * (2.0 * xi_c - 1.0) ** 3 + 5184.0 * chi_c**2)) ** (1.0 / 3.0) / 12.0

    # c only vanishes at (chi, xi) = (0, 1/2), where the cubic gives chi = 0 directly.
    degenerate = numpy.abs(c) < 1e-14
    c = numpy.where(degenerate, 1.0, c)

    # The cubic has three real roots whenever the point lies inside the evolute of the parabola, and
    # the principal cube root is not always the closest one. Take every root and keep the best.
    def normalized_distance(chi_curve):
        chi_curve = numpy.clip(chi_curve, CHI_MIN, CHI_MAX)
        return numpy.sqrt(
            ((chi - chi_curve) / (CHI_MAX - CHI_MIN)) ** 2 + ((xi - psi(chi_curve)) / (XI_MAX - XI_MIN)) ** 2
        )

    best = None
    for unit_root in (1.0, numpy.exp(2j * numpy.pi / 3.0), numpy.exp(-2j * numpy.pi / 3.0)):
        ck = c * unit_root
        root = numpy.where(degenerate, 0.0, ck + (5.0 / 24.0 - (5.0 / 12.0) * xi_c) / ck)
        distance = normalized_distance(numpy.real(root))
        best = distance if best is None else numpy.minimum(best, distance)

    return best


def classify(chi, xi):
    """Split the (chi, xi) pairs into the three DCMIP mixing regions (Figure 1).

    Mixing two parcels that sit on the (convex) initial curve produces a point on the straight line
    between them, so everything physically reachable by mixing lies in the lens between the curve
    and its chord: that is region A. Pairs that stay inside the bounds but leave that lens are
    unmixed in a range-preserving way (region B). Pairs that leave the bounds have overshot.
    """
    in_range = (chi >= CHI_MIN) & (chi <= CHI_MAX) & (xi >= XI_MIN) & (xi <= XI_MAX)
    chord = 0.9 - 0.8 * chi  # the straight line joining the two ends of the curve

    real_mixing = in_range & (xi >= chord) & (xi <= psi(chi))
    unmixing = in_range & ~real_mixing
    overshooting = ~in_range

    return real_mixing, unmixing, overshooting


def mixing_diagnostics(q1, q2, weight):
    """l_r, l_u and l_o (DCMIP eqs. 174-176), as area-weighted mean distances to the curve."""
    d = distance_to_curve(q1, q2)
    real_mixing, unmixing, overshooting = classify(q1, q2)

    total = weight.sum()
    return {
        "lr": float((d * weight * real_mixing).sum() / total),
        "lu": float((d * weight * unmixing).sum() / total),
        "lo": float((d * weight * overshooting).sum() / total),
    }


def unit_sphere(longitude, latitude):
    """Cartesian unit vectors for coordinates expressed in degrees."""
    lon = numpy.deg2rad(longitude)
    lat = numpy.deg2rad(latitude)
    cos_lat = numpy.cos(lat)
    return numpy.stack((cos_lat * numpy.cos(lon), cos_lat * numpy.sin(lon), numpy.sin(lat)), axis=-1)


def spherical_resampler(lat, lon, shape=(361, 721), neighbours=4):
    """Precompute seamless inverse-distance interpolation on the sphere.

    Planar triangulation of longitude and latitude treats 0/360 as distant and
    degenerates near the poles.  Neighbour lookup in Cartesian unit-sphere
    coordinates has neither singularity.
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
    """Apply a precomputed spherical interpolation stencil."""
    return numpy.sum(values.ravel()[index] * weight, axis=1).reshape(shape)


def color_range(data, name):
    """Colour limits taken from the initial state, so every time can be compared against it.

    Keeping the scale fixed is also what makes the over- and undershoots visible: they saturate.
    """
    initial = data["tracers"][name][0]
    return float(initial.min()), float(initial.max())


def tracer_label(name):
    """Typeset tracer names consistently."""
    return rf"$q_{name[1:]}$"


def plot_lat_lon(data, day, level, outdir):
    """Lat-lon cross section of every tracer at one day and one level."""
    it = time_index(data["time"], day * DAY)
    iz = level_index(data["elev"], level)
    height = data["elev"].mean(axis=(0, 2, 3))[iz]
    longitudes, latitudes, index, weight = spherical_resampler(data["lat2d"], data["lon2d"])

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.linewidth": 0.7,
            "savefig.dpi": 300,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), constrained_layout=True, sharex=True, sharey=True)
        for ax, (name, field) in zip(axes.ravel(), data["tracers"].items()):
            q = field[it, :, iz]
            image = apply_spherical_resampler(q, index, weight, (latitudes.size, longitudes.size))
            vmin, vmax = color_range(data, name)
            boundaries = numpy.linspace(vmin, vmax, 22)

            contour = ax.contourf(
                longitudes,
                latitudes,
                image,
                levels=boundaries,
                cmap="viridis",
                extend="both",
                antialiased=False,
            )
            colorbar = fig.colorbar(contour, ax=ax, shrink=0.9, pad=0.02)
            colorbar.set_label(tracer_label(name))
            ax.set_title(tracer_label(name), fontsize=10)
            ax.set_xlim(0.0, 360.0)
            ax.set_ylim(-90.0, 90.0)
            ax.xaxis.set_major_locator(MultipleLocator(60.0))
            ax.yaxis.set_major_locator(MultipleLocator(30.0))
            ax.grid(color="white", lw=0.35, alpha=0.35)
            ax.set_xlabel("Longitude (degrees east)")
            ax.set_ylabel("Latitude (degrees north)")

        fig.suptitle(f"DCMIP 1-1 tracers at day {day:g}, $z={height:.0f}$ m", fontsize=10)
        path = outdir / f"dcmip11_latlon_day{day:g}.png"
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
    return path, path.with_suffix(".pdf")


def equator_stencil(lat, lon):
    """Interpolation stencils from the four equatorial panels to latitude zero."""
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


def periodic_section(field, elevations, longitude, heights, plot_longitude):
    """Shape-preserving interpolation onto a regular longitude-height grid."""
    vertical = numpy.empty((longitude.size, heights.size))
    for column, (z_column, q_column) in enumerate(zip(elevations, field)):
        sample_height = numpy.clip(heights, z_column[0], z_column[-1])
        vertical[column] = PchipInterpolator(z_column, q_column)(sample_height)

    extended_lon = numpy.concatenate(([longitude[-1] - 360.0], longitude, [longitude[0] + 360.0]))
    extended_field = numpy.concatenate((vertical[-1:], vertical, vertical[:1]), axis=0)
    return PchipInterpolator(extended_lon, extended_field, axis=0)(plot_longitude).T


def plot_lon_height(data, day, outdir):
    """Longitude-height section interpolated to the exact equator."""
    it = time_index(data["time"], day * DAY)
    longitude, stencil = equator_stencil(data["lat2d"], data["lon2d"])
    elevations = interpolate_equator(data["elev"], stencil)
    heights = numpy.linspace(0.0, 12_000.0, 301)
    plot_longitude = numpy.linspace(0.0, 360.0, 721)

    with plt.rc_context({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.7, "savefig.dpi": 300}):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), constrained_layout=True, sharex=True, sharey=True)
        for ax, (name, field) in zip(axes.ravel(), data["tracers"].items()):
            equator_field = interpolate_equator(field[it], stencil)
            image = periodic_section(equator_field, elevations, longitude, heights, plot_longitude)
            vmin, vmax = color_range(data, name)
            contour = ax.contourf(
                plot_longitude,
                heights / 1000.0,
                image,
                levels=numpy.linspace(vmin, vmax, 22),
                cmap="viridis",
                extend="both",
                antialiased=False,
            )
            colorbar = fig.colorbar(contour, ax=ax, shrink=0.9, pad=0.02)
            colorbar.set_label(tracer_label(name))
            ax.set_title(tracer_label(name), fontsize=10)
            ax.set_xlim(0.0, 360.0)
            ax.set_ylim(0.0, 12.0)
            ax.xaxis.set_major_locator(MultipleLocator(60.0))
            ax.yaxis.set_major_locator(MultipleLocator(2.0))
            ax.set_xlabel("Longitude (degrees east)")
            ax.set_ylabel("Height (km)")

        fig.suptitle(f"DCMIP 1-1 tracers at the equator, day {day:g}", fontsize=10)
        path = outdir / f"dcmip11_lonheight_day{day:g}.png"
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
    return path, path.with_suffix(".pdf")


def plot_correlation(data, day, outdir):
    """q1 against q2 on the five DCMIP levels, with the initial correlation curve for reference."""
    it = time_index(data["time"], day * DAY)
    mean_height = data["elev"].mean(axis=(0, 2, 3))

    # Coarse vertical grids map several of the DCMIP target levels onto the same model level; only
    # plot each model level once.
    levels = sorted({level_index(data["elev"], target) for target in MIXING_LEVELS})

    with plt.rc_context({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.7, "savefig.dpi": 300}):
        fig, axes = plt.subplots(
            1,
            len(levels),
            figsize=(7.2, 2.25),
            constrained_layout=True,
            sharex=True,
            sharey=True,
        )
        axes = numpy.atleast_1d(axes)
        curve = numpy.linspace(CHI_MIN, CHI_MAX, 300)
        density = None

        for ax, iz in zip(axes, levels):
            q1 = data["tracers"]["q1"][it, :, iz].ravel()
            q2 = data["tracers"]["q2"][it, :, iz].ravel()
            density = ax.hexbin(
                q1,
                q2,
                gridsize=65,
                extent=(-0.08, 1.08, 0.02, 0.98),
                bins="log",
                mincnt=1,
                cmap="Blues",
                linewidths=0.0,
                rasterized=True,
            )
            ax.plot(curve, psi(curve), color="0.1", lw=1.0, label="Initial curve", zorder=3)
            ax.plot(
                [CHI_MIN, CHI_MAX],
                [psi(CHI_MIN), psi(CHI_MAX)],
                color="0.25",
                ls=(0, (3, 2)),
                lw=0.8,
                label="Chord",
                zorder=3,
            )
            ax.set_xlabel(r"$q_1$")
            ax.set_title(f"{mean_height[iz]:.0f} m", fontsize=9)
            ax.set_xlim(-0.08, 1.08)
            ax.set_ylim(0.02, 0.98)
            ax.tick_params(length=2.5)

        axes[0].set_ylabel(r"$q_2$")
        axes[0].legend(loc="lower left", fontsize=7, frameon=False)
        colorbar = fig.colorbar(density, ax=axes, shrink=0.9, pad=0.01)
        colorbar.set_label("Sample density")
        fig.suptitle(rf"DCMIP 1-1: $q_1$–$q_2$ correlation at day {day:g}", fontsize=10)
        path = outdir / f"dcmip11_correlation_day{day:g}.png"
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
    return path, path.with_suffix(".pdf")


def report_error_norms(data, day):
    """Error norms against the initial state, which is the exact solution after a full period."""
    it = time_index(data["time"], day * DAY)
    volume = data["volume"]

    print(f"\nNormalized error norms at day {day:g} (exact solution = initial state)")
    print(f"  {'tracer':>8}  {'l1':>12}  {'l2':>12}  {'l_inf':>12}")
    for name in ("q1", "q3", "q4"):
        field = data["tracers"][name]
        l1, l2, linf = error_norms(field[it], field[0], volume)
        print(f"  {name:>8}  {l1:12.4e}  {l2:12.4e}  {linf:12.4e}")


def report_mixing(data, day):
    """Mixing diagnostics over the five DCMIP levels."""
    it = time_index(data["time"], day * DAY)
    mean_height = data["elev"].mean(axis=(0, 2, 3))

    levels = sorted({level_index(data["elev"], target) for target in MIXING_LEVELS})
    # Index the vertical axis the same way everywhere, so the weights line up with the tracers.
    q1 = data["tracers"]["q1"][it][:, levels].ravel()
    q2 = data["tracers"]["q2"][it][:, levels].ravel()
    weight = data["volume"][:, levels].ravel()

    diagnostics = mixing_diagnostics(q1, q2, weight)

    heights = ", ".join(f"{mean_height[i]:.0f}" for i in levels)
    print(f"\nMixing diagnostics at day {day:g}, on the levels at {heights} m")
    print(f"  real mixing              l_r = {diagnostics['lr']:.4e}")
    print(f"  range-preserving unmixing l_u = {diagnostics['lu']:.4e}")
    print(f"  overshooting             l_o = {diagnostics['lo']:.4e}")


def main():
    parser = argparse.ArgumentParser(description="Plots and diagnostics for DCMIP-2012 test 1-1.")
    parser.add_argument("netcdf_file", help="NetCDF output file produced by WxFactory")
    parser.add_argument("-o", "--output-dir", default="results", help="where to write the figures")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = load(args.netcdf_file)

    days = data["time"] / DAY
    print(f"Read {args.netcdf_file}; selected diagnostic days: {', '.join(f'{day:g}' for day in days)}")

    figure_pairs = []
    for day in (6, 12):
        figure_pairs.append(plot_lat_lon(data, day, 4900.0, outdir))
    figure_pairs.append(plot_lon_height(data, 12, outdir))
    figure_pairs.append(plot_correlation(data, 6, outdir))
    written = [path for pair in figure_pairs for path in pair]

    report_error_norms(data, 12)
    report_mixing(data, 6)

    print("\nFigures written:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
