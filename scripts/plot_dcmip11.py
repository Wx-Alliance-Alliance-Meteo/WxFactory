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
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy
from scipy.interpolate import griddata

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
    """Read the fields we need. Arrays come back flattened over the (panel, z, x, y) grid."""
    with netCDF4.Dataset(filename) as ds:
        time = numpy.asarray(ds.variables["time"][:])
        # lats/lons are (panel, x, y); broadcast them over the vertical to match the tracers.
        lat = numpy.asarray(ds.variables["lats"][:])
        lon = numpy.asarray(ds.variables["lons"][:])
        elev = numpy.asarray(ds.variables["elev"][:])
        volume = numpy.asarray(ds.variables["volume"][:])
        tracers = {name: numpy.asarray(ds.variables[name][:]) for name in ("q1", "q2", "q3", "q4")}

    num_z = elev.shape[1]
    lat = numpy.repeat(lat[:, numpy.newaxis], num_z, axis=1)
    lon = numpy.repeat(lon[:, numpy.newaxis], num_z, axis=1)

    return {
        "time": time,
        # The writer already stores these in degrees (units "degrees_north" / "degrees_east").
        "lat": lat,
        "lon": lon,
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


def scatter_to_grid(x, y, values, x_range, y_range, shape, periodic_x=None):
    """Interpolate the scattered cubed-sphere points onto a regular grid.

    Contouring the raw points directly (tricontourf) triangulates them in the plotting plane, which
    on a cubed sphere produces visible artifacts: slivers along the panel edges, wedges near the
    poles, and a seam where longitude wraps. Interpolating onto a regular grid first, and letting
    imshow do the smoothing, removes all of them. `periodic_x` repeats the points either side of the
    domain so the seam at 0/360 degrees closes.
    """
    grid_x, grid_y = numpy.meshgrid(
        numpy.linspace(x_range[0], x_range[1], shape[1]),
        numpy.linspace(y_range[0], y_range[1], shape[0]),
    )

    if periodic_x is not None:
        x = numpy.concatenate([x - periodic_x, x, x + periodic_x])
        y = numpy.tile(y, 3)
        values = numpy.tile(values, 3)

    return griddata((x, y), values, (grid_x, grid_y), method="linear")


def color_range(data, name):
    """Colour limits taken from the initial state, so every time can be compared against it.

    Keeping the scale fixed is also what makes the over- and undershoots visible: they saturate.
    """
    initial = data["tracers"][name][0]
    return float(initial.min()), float(initial.max())


def plot_lat_lon(data, day, level, outdir):
    """Lat-lon cross section of every tracer at one day and one level."""
    it = time_index(data["time"], day * DAY)
    iz = level_index(data["elev"], level)
    height = data["elev"].mean(axis=(0, 2, 3))[iz]

    lon = data["lon"][:, iz].ravel()
    lat = data["lat"][:, iz].ravel()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for ax, (name, field) in zip(axes.ravel(), data["tracers"].items()):
        q = field[it, :, iz].ravel()
        vmin, vmax = color_range(data, name)

        image = scatter_to_grid(lon, lat, q, (0, 360), (-90, 90), (181, 361), periodic_x=360.0)
        rendered = ax.imshow(
            image,
            origin="lower",
            extent=[0, 360, -90, 90],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="bilinear",
        )
        fig.colorbar(rendered, ax=ax, shrink=0.85)
        ax.set_title(f"{name}   [{q.min():.3f}, {q.max():.3f}]")
        ax.set_xlabel("longitude (deg)")
        ax.set_ylabel("latitude (deg)")

    fig.suptitle(f"DCMIP 1-1: tracers at day {day:g}, z = {height:.0f} m")
    path = os.path.join(outdir, f"dcmip11_latlon_day{day:g}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def equator_band(lat, lat_tol):
    """A band of grid points around the equator.

    The cubed sphere has no row exactly on the equator, and how close it gets depends on the
    resolution, so widen the requested tolerance until the band holds enough points to contour.
    """
    abs_lat = numpy.abs(lat)
    columns = numpy.sort(abs_lat[:, 0].ravel())  # the horizontal grid, one level is enough
    minimum_points = max(3, int(0.01 * columns.size))
    tol = max(lat_tol, columns[minimum_points])

    return abs_lat < tol, tol


def plot_lon_height(data, day, outdir, lat_tol=2.0):
    """Longitude-height cross section along the equator."""
    it = time_index(data["time"], day * DAY)

    near_equator, lat_tol = equator_band(data["lat"], lat_tol)
    lon = data["lon"][near_equator]
    z = data["elev"][near_equator]

    # The first and last levels sit half a cell away from the ground and the model top, so bound the
    # axis by the data itself: extending it further would only leave an empty strip.
    zbot, ztop = float(z.min()), float(z.max())

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for ax, (name, field) in zip(axes.ravel(), data["tracers"].items()):
        q = field[it][near_equator]
        vmin, vmax = color_range(data, name)

        image = scatter_to_grid(lon, z, q, (0, 360), (zbot, ztop), (181, 361), periodic_x=360.0)
        rendered = ax.imshow(
            image,
            origin="lower",
            extent=[0, 360, zbot, ztop],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="bilinear",
        )
        fig.colorbar(rendered, ax=ax, shrink=0.85)
        ax.set_title(f"{name}   [{q.min():.3f}, {q.max():.3f}]")
        ax.set_xlabel("longitude (deg)")
        ax.set_ylabel("height (m)")

    fig.suptitle(f"DCMIP 1-1: tracers along the equator (|lat| < {lat_tol:g} deg) at day {day:g}")
    path = os.path.join(outdir, f"dcmip11_lonheight_day{day:g}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_correlation(data, day, outdir):
    """q1 against q2 on the five DCMIP levels, with the initial correlation curve for reference."""
    it = time_index(data["time"], day * DAY)
    mean_height = data["elev"].mean(axis=(0, 2, 3))

    # Coarse vertical grids map several of the DCMIP target levels onto the same model level; only
    # plot each model level once.
    levels = sorted({level_index(data["elev"], target) for target in MIXING_LEVELS})

    fig, axes = plt.subplots(1, len(levels), figsize=(4 * len(levels), 4.2), constrained_layout=True, sharey=True)
    axes = numpy.atleast_1d(axes)
    curve = numpy.linspace(CHI_MIN, CHI_MAX, 200)

    for ax, iz in zip(axes, levels):
        q1 = data["tracers"]["q1"][it, :, iz].ravel()
        q2 = data["tracers"]["q2"][it, :, iz].ravel()

        ax.plot(curve, psi(curve), "k-", lw=2, label="initial curve", zorder=3)
        ax.plot([CHI_MIN, CHI_MAX], [psi(CHI_MIN), psi(CHI_MAX)], "k--", lw=1, label="chord", zorder=3)
        ax.scatter(q1, q2, s=1, alpha=0.3, color="tab:blue")
        ax.set_xlabel("q1")
        ax.set_title(f"z = {mean_height[iz]:.0f} m")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(0.0, 1.0)

    axes[0].set_ylabel("q2")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"DCMIP 1-1: q1-q2 correlation at day {day:g}")
    path = os.path.join(outdir, f"dcmip11_correlation_day{day:g}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


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

    os.makedirs(args.output_dir, exist_ok=True)
    data = load(args.netcdf_file)

    days = data["time"] / DAY
    print(f"Read {args.netcdf_file}: {len(days)} output times, from day {days[0]:g} to day {days[-1]:g}")

    written = []
    # Day 0 is the initial condition, and (since the flow reverses) also the exact solution at day 12.
    for day in (0, 6, 12):
        written.append(plot_lat_lon(data, day, 4900.0, args.output_dir))
    for day in (0, 12):
        written.append(plot_lon_height(data, day, args.output_dir))
    written.append(plot_correlation(data, 0, args.output_dir))
    written.append(plot_correlation(data, 6, args.output_dir))

    report_error_norms(data, 12)
    report_mixing(data, 6)

    print("\nFigures written:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
