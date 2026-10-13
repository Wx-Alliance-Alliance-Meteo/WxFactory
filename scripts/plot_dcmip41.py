#!/usr/bin/env python3
"""DCMIP 4-1 maps, tracer comparisons, and kinetic-energy spectrum."""

import argparse
from pathlib import Path

import matplotlib
import netCDF4
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator

INPUT_FILE = "results/dcmip41.nc"

TARGET_PRESSURE = 85000.0
OUTPUT_PREFIX = "dcmip41"

RD = 287.05
CPD = 1005.46
P0 = 100000.0
KAPPA = RD / CPD
EARTH_RADIUS = 6371220.0

TITLE_SIZE = 12
LABEL_SIZE = 11
TICK_SIZE = 9
CBAR_LABEL_SIZE = 10
CBAR_TICK_SIZE = 9
DPI = 180


def temperature_from_theta(theta, pressure):
    return theta * (pressure / P0) ** KAPPA


def select_time(raw_time, target_seconds):
    raw_time = np.asarray(raw_time, dtype=np.float64)
    if raw_time.ndim != 1:
        raise ValueError(f"time must be one-dimensional; got {raw_time.shape}")
    if raw_time.size == 0:
        raise ValueError("No output times were found.")
    if not np.all(np.isfinite(raw_time)):
        raise ValueError("Non-finite values found in time.")

    elapsed_seconds = raw_time - raw_time[0]
    if elapsed_seconds.size > 1:
        dt = np.diff(elapsed_seconds)
        if np.any(dt < 0.0):
            raise ValueError("NetCDF time coordinate is not monotonically increasing.")
    first_seconds = float(elapsed_seconds[0])
    last_seconds = float(elapsed_seconds[-1])
    if target_seconds < first_seconds or target_seconds > last_seconds:
        raise ValueError(
            "\nRequested time is outside available output.\n"
            "----------------------------------------------\n"
            f"Requested time       : {target_seconds:.6f} s\n"
            f"First available time : {first_seconds:.6f} s\n"
            f"Last available time  : {last_seconds:.6f} s"
        )

    difference = np.abs(elapsed_seconds - target_seconds)
    itime = int(np.argmin(difference))
    actual_seconds = float(elapsed_seconds[itime])
    difference_seconds = float(difference[itime])

    if elapsed_seconds.size > 1:
        output_dt = np.diff(elapsed_seconds)
        output_dt = output_dt[output_dt > 0.0]
        if output_dt.size > 0:
            typical_output_interval = float(np.median(output_dt))
        else:
            typical_output_interval = np.nan
    else:
        typical_output_interval = np.nan

    return (itime, elapsed_seconds, actual_seconds, difference_seconds, typical_output_interval)


def interpolate_to_pressure(field, pressure, target_pressure):
    if field.shape != pressure.shape:
        raise ValueError("field and pressure must have identical shapes.")

    npanel, _nz, ny, nx = field.shape
    result = np.full((npanel, ny, nx), np.nan, dtype=np.float64)
    for panel in range(npanel):
        for j in range(ny):
            for i in range(nx):
                pcol = pressure[panel, :, j, i]
                fcol = field[panel, :, j, i]
                valid = np.isfinite(pcol) & np.isfinite(fcol)
                if np.count_nonzero(valid) < 2:
                    continue
                p = np.asarray(pcol[valid], dtype=np.float64)
                f = np.asarray(fcol[valid], dtype=np.float64)
                order = np.argsort(p)
                p = p[order]
                f = f[order]
                p_unique, index = np.unique(p, return_index=True)
                f_unique = f[index]
                if p_unique.size < 2:
                    continue
                if target_pressure < p_unique[0] or target_pressure > p_unique[-1]:
                    continue
                result[panel, j, i] = np.interp(target_pressure, p_unique, f_unique)
    return result


def unwrap_longitude(lon):
    lon_rad = np.deg2rad(lon)
    lon_rad = np.unwrap(lon_rad, axis=-1)
    lon_rad = np.unwrap(lon_rad, axis=-2)
    return lon_rad


def relative_vorticity_panel(u, v, lat, lon, earth_radius):
    npanel, _ny, _nx = u.shape

    zeta = np.full_like(u, np.nan, dtype=np.float64)
    for panel in range(npanel):
        phi = np.deg2rad(lat[panel])
        lam = unwrap_longitude(lon[panel])
        up = np.asarray(u[panel], dtype=np.float64)
        vp = np.asarray(v[panel], dtype=np.float64)
        cosphi = np.cos(phi)
        dv_dj, dv_di = np.gradient(vp)
        ucos = up * cosphi
        ducos_dj, ducos_di = np.gradient(ucos)
        dlam_dj, dlam_di = np.gradient(lam)
        dphi_dj, dphi_di = np.gradient(phi)

        determinant = dlam_di * dphi_dj - dlam_dj * dphi_di

        determinant = np.where(np.abs(determinant) < 1.0e-14, np.nan, determinant)
        dv_dlambda = (dv_di * dphi_dj - dv_dj * dphi_di) / determinant
        ducos_dphi = (dlam_di * ducos_dj - dlam_dj * ducos_di) / determinant

        denominator = earth_radius * cosphi

        denominator = np.where(np.abs(cosphi) < 1.0e-8, np.nan, denominator)
        zeta[panel] = (dv_dlambda - ducos_dphi) / denominator
    return zeta


def make_levels(field, vmin=None, vmax=None, nlevels=20, symmetric=False):
    finite = np.asarray(field)[np.isfinite(field)]
    if finite.size == 0:
        raise ValueError("No finite values available.")
    if symmetric:
        if vmin is None and vmax is None:
            vmax_abs = float(np.nanmax(np.abs(finite)))
        else:
            candidates = []
            if vmin is not None:
                candidates.append(abs(vmin))
            if vmax is not None:
                candidates.append(abs(vmax))
            vmax_abs = max(candidates)
        if vmax_abs == 0.0:
            vmax_abs = 1.0
        return np.linspace(-vmax_abs, vmax_abs, nlevels + 1)
    if vmin is None:
        vmin = float(np.nanmin(finite))
    if vmax is None:
        vmax = float(np.nanmax(finite))
    if vmin == vmax:
        delta = abs(vmin) * 0.01 if vmin != 0.0 else 1.0
        vmin -= delta
        vmax += delta
    return np.linspace(vmin, vmax, nlevels + 1)


def make_colorbar_ticks(vmin, vmax, nbins=5):
    locator = MaxNLocator(nbins=nbins, min_n_ticks=3)
    ticks = locator.tick_values(vmin, vmax)
    tolerance = abs(vmax - vmin) * 1.0e-10
    ticks = ticks[(ticks >= vmin - tolerance) & (ticks <= vmax + tolerance)]
    return ticks


def add_horizontal_colorbar(fig, ax, image, levels, label, nbins, tick_format):
    ticks = make_colorbar_ticks(levels[0], levels[-1], nbins=nbins)
    cbar = fig.colorbar(
        image, ax=ax, orientation="horizontal", pad=0.16, fraction=0.055, shrink=0.94, aspect=28, ticks=ticks
    )
    cbar.ax.xaxis.set_major_formatter(FormatStrFormatter(tick_format))
    cbar.set_label(label, fontsize=CBAR_LABEL_SIZE, labelpad=5)
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE, length=3, pad=2)
    return cbar


def plot_cubesphere_field(ax, lon, lat, field, levels, cmap, draw_contours=False, line_levels=None):
    image = None
    for panel in range(field.shape[0]):
        lon_panel = np.asarray(lon[panel], dtype=np.float64).copy()
        lat_panel = np.asarray(lat[panel], dtype=np.float64).copy()
        field_panel = np.asarray(field[panel], dtype=np.float64).copy()
        lon_panel = ((lon_panel + 180.0) % 360.0) - 180.0
        seam = np.zeros_like(lon_panel, dtype=bool)
        diff_x = np.abs(np.diff(lon_panel, axis=1))
        diff_y = np.abs(np.diff(lon_panel, axis=0))
        seam[:, :-1] |= diff_x > 180.0
        seam[:, 1:] |= diff_x > 180.0
        seam[:-1, :] |= diff_y > 180.0
        seam[1:, :] |= diff_y > 180.0
        masked = np.ma.masked_where(seam | ~np.isfinite(field_panel), field_panel)
        image = ax.contourf(lon_panel, lat_panel, masked, levels=levels, cmap=cmap, extend="both", antialiased=True)
        if draw_contours and line_levels is not None:
            ax.contour(lon_panel, lat_panel, masked, levels=line_levels, colors="white", linewidths=0.45, alpha=0.70)
    return image


def configure_axis(ax, show_ylabel=True):
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xlabel("Longitude (°)", fontsize=LABEL_SIZE)
    if show_ylabel:
        ax.set_ylabel("Latitude (°)", fontsize=LABEL_SIZE)
    else:
        ax.set_ylabel("")
    ax.xaxis.set_major_locator(MultipleLocator(60))
    ax.yaxis.set_major_locator(MultipleLocator(30))
    ax.tick_params(axis="both", labelsize=TICK_SIZE, direction="out", length=3)
    ax.grid(linewidth=0.4, alpha=0.20)


def read_field(dataset, name, index):
    return np.asarray(np.ma.filled(dataset.variables[name][index], np.nan), dtype=np.float64)


def surface_pressure(dataset, pressure, elevation, index):
    if "PS" in dataset.variables:
        return read_field(dataset, "PS", index)
    if "topo" not in dataset.variables:
        raise ValueError("Surface pressure requires PS or the surface elevation topo.")
    surface = read_field(dataset, "topo", slice(None))
    # Continue log(p) from the lowest two nodes to the actual boundary.
    slope = (np.log(pressure[:, 1]) - np.log(pressure[:, 0])) / (elevation[:, 1] - elevation[:, 0])
    return np.exp(np.log(pressure[:, 0]) + slope * (surface - elevation[:, 0]))


def interpolate_to_isentrope(field, theta, elevation, target=315.0):
    result = np.full((field.shape[0],) + field.shape[2:], np.nan)
    for panel, j, i in np.ndindex(result.shape):
        order = np.argsort(elevation[panel, :, j, i])
        levels = theta[panel, order, j, i]
        values = field[panel, order, j, i]
        crossing = np.flatnonzero((levels[:-1] <= target) & (levels[1:] > target))
        if crossing.size:
            # Choose the lowest upward crossing when the isentrope folds.
            k = crossing[0]
            weight = (target - levels[k]) / (levels[k + 1] - levels[k])
            result[panel, j, i] = (1 - weight) * values[k] + weight * values[k + 1]
    return result


def horizontal_derivatives(field, lat, lon):
    phi = np.deg2rad(lat)[:, None]
    lam = np.stack([unwrap_longitude(panel) for panel in lon])[:, None]
    dlam_dj, dlam_di = np.gradient(lam, axis=(2, 3))
    dphi_dj, dphi_di = np.gradient(phi, axis=(2, 3))
    df_dj, df_di = np.gradient(field, axis=(2, 3))
    determinant = dlam_di * dphi_dj - dlam_dj * dphi_di
    determinant = np.where(np.abs(determinant) > 1e-14, determinant, np.nan)
    return ((df_di * dphi_dj - df_dj * dphi_di) / determinant, (dlam_di * df_dj - dlam_dj * df_di) / determinant)


def dynamic_epv(u, v, w, theta, rho, elevation, lat, lon, radius, rotation):
    """Evaluate shallow-atmosphere Ertel PV using differences of saved physical fields."""
    height_lambda, height_phi = horizontal_derivatives(elevation, lat, lon)
    height_step = np.gradient(elevation, axis=1)

    def derivatives(field):
        vertical = np.gradient(field, axis=1) / height_step
        longitude, latitude = horizontal_derivatives(field, lat, lon)
        return longitude - vertical * height_lambda, latitude - vertical * height_phi, vertical

    _u_lambda, u_phi, u_z = derivatives(u)
    v_lambda, _v_phi, v_z = derivatives(v)
    w_lambda, w_phi, _w_z = derivatives(w)
    theta_lambda, theta_phi, theta_z = derivatives(theta)
    phi = np.deg2rad(lat)[:, None]
    cosine = np.where(np.abs(np.cos(phi)) > 1e-8, np.cos(phi), np.nan)
    east = w_phi / radius - v_z
    north = u_z - w_lambda / (radius * cosine) + 2 * rotation * cosine
    radial = (v_lambda - cosine * u_phi + u * np.sin(phi)) / (radius * cosine) + 2 * rotation * np.sin(phi)
    return (east * theta_lambda / (radius * cosine) + north * theta_phi / radius + radial * theta_z) / rho


def save_figure(fig, path, pdf):
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"Saved {path}")
    if pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        print(f"Saved {path.with_suffix('.pdf')}")
    plt.close(fig)


def plot_maps(lat, lon, panels, title, path, pdf, northern=False):
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.6), squeeze=False)
    fig.subplots_adjust(bottom=0.24, top=0.84, wspace=0.28)
    for ax, (name, data, units, symmetric) in zip(axes[0], panels):
        if northern:
            data = np.where(lat >= 0, data, np.nan)
        if not np.isfinite(data).any():
            ax.text(0.5, 0.5, "No valid values", ha="center", transform=ax.transAxes)
            configure_axis(ax)
        else:
            comparable = [
                np.where(lat >= 0, values, np.nan) if northern else values
                for _, values, other_units, other_symmetric in panels
                if other_units == units and other_symmetric == symmetric
            ]
            levels = make_levels(np.stack(comparable), symmetric=symmetric)
            image = plot_cubesphere_field(ax, lon, lat, data, levels, "RdBu_r" if symmetric else "viridis")
            configure_axis(ax)
            add_horizontal_colorbar(fig, ax, image, levels, units, 5, "%.3g")
        if northern:
            ax.set_ylim(0, 90)
        ax.set_title(name, fontsize=TITLE_SIZE)
    fig.suptitle(title, fontsize=TITLE_SIZE)
    save_figure(fig, path, pdf)


def kinetic_energy_spectrum(u, v, lat, lon, degree):
    from scipy.spatial import cKDTree
    from scipy.special import sph_harm_y

    mu, weights = np.polynomial.legendre.leggauss(2 * degree + 2)
    longitude = np.arange(4 * degree + 4) * (2 * np.pi / (4 * degree + 4))
    colatitude, longitude = np.meshgrid(np.arccos(mu), longitude, indexing="ij")

    def position(latitude, longitude):
        return np.stack(
            (np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude)), axis=-1
        ).reshape(-1, 3)

    valid = np.isfinite(u) & np.isfinite(v)
    if np.count_nonzero(valid) < 4:
        raise ValueError("The KE spectrum needs at least four valid wind samples at 700 hPa.")
    tree = cKDTree(position(np.deg2rad(lat[valid]), np.deg2rad(lon[valid])))
    distance, indices = tree.query(position(np.pi / 2 - colatitude, longitude), k=4)
    # Spherical interpolation avoids a discontinuity at the longitude seam.
    interpolation = 1 / np.maximum(distance, 1e-12) ** 2
    interpolation /= interpolation.sum(axis=1, keepdims=True)
    east = (u[valid][indices] * interpolation).sum(axis=1).reshape(colatitude.shape)
    north = (v[valid][indices] * interpolation).sum(axis=1).reshape(colatitude.shape)
    quadrature = weights[:, None] * (2 * np.pi / longitude.shape[1])
    spectrum = np.zeros(degree + 1)
    for n in range(1, degree + 1):
        for m in range(n + 1):
            _, gradient = sph_harm_y(n, m, colatitude, longitude, diff_n=1)
            meridional = gradient[..., 0].conj() / np.sqrt(n * (n + 1))
            zonal = gradient[..., 1].conj() / (np.sin(colatitude) * np.sqrt(n * (n + 1)))
            divergent = np.sum((-north * meridional + east * zonal) * quadrature)
            rotational = np.sum((east * meridional + north * zonal) * quadrature)
            spectrum[n] += (1 if m == 0 else 2) * (abs(divergent) ** 2 + abs(rotational) ** 2) / (8 * np.pi)
    return spectrum


def main():
    parser = argparse.ArgumentParser(description="DCMIP 4-1 diagnostics from Section 4.1 of the test document.")
    parser.add_argument("netcdf_file", nargs="?", default=INPUT_FILE)
    parser.add_argument("-o", "--output-dir", default="results", help="where to write figures")
    parser.add_argument("--label", help="tag used in figure names and titles")
    parser.add_argument("--pdf", action="store_true", help="also write PDF figures")
    times = parser.add_mutually_exclusive_group()
    times.add_argument("--days", type=float, nargs="+", help="small-planet days for the primary and theta maps")
    times.add_argument("--times", type=float, nargs="+", help="elapsed seconds for all snapshot diagnostics")
    parser.add_argument("--planet-scaling", type=float, help="DCMIP X; otherwise inferred from earth_radius")
    parser.add_argument("--spectrum-degree", type=int, default=30, help="maximum spherical-harmonic degree")
    args = parser.parse_args()
    if args.spectrum_degree < 1:
        parser.error("--spectrum-degree must be positive")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    label = args.label or Path(args.netcdf_file).stem
    prefix = f"dcmip41_{label}"
    with netCDF4.Dataset(args.netcdf_file) as ds:
        radius = float(getattr(ds, "earth_radius", EARTH_RADIUS))
        scaling = args.planet_scaling if args.planet_scaling is not None else EARTH_RADIUS / radius
        if not np.isfinite(scaling) or scaling <= 0:
            parser.error("--planet-scaling must be positive and finite")
        rotation = float(getattr(ds, "rotation_speed", 7.29212e-5 * scaling))
        lat = read_field(ds, "lats", slice(None))
        lon = read_field(ds, "lons", slice(None))
        elevation = read_field(ds, "elev", slice(None))
        raw_time = read_field(ds, "time", slice(None))
        units = getattr(ds.variables["time"], "units", "seconds")
        if not units.lower().startswith("seconds"):
            raise ValueError(f"Expected a time coordinate in seconds, got {units!r}")
        primary_times = (
            args.times if args.times is not None else [day * 86400 / scaling for day in (args.days or [7, 9, 12, 15])]
        )
        epv_times = args.times if args.times is not None else [day * 86400 / scaling for day in [8, 12, 15]]
        spectrum_time = 30 * 86400 / scaling
        requested = sorted(set(primary_times + epv_times + [spectrum_time]))
        if "q1" not in ds.variables:
            print("Unavailable: theta tracer comparisons (q1 is absent).")
        if "q2" not in ds.variables:
            print("Unavailable: EPV tracer comparisons and scatter plots (q2 is absent).")
        if "PS" not in ds.variables:
            print("Surface pressure is extrapolated in log(p) from the lowest two nodes to topo.")
        for target in requested:
            try:
                index, _, actual, difference, _ = select_time(raw_time, target)
            except ValueError as error:
                print(f"Skipping {target:g} s: {error}")
                continue
            if difference > max(1e-6, 1e-10 * abs(target)):
                print(f"Skipping {target:g} s: no matching snapshot (nearest is {actual:g} s).")
                continue
            print(f"Reading {actual:g} s (small-planet day {actual * scaling / 86400:g})")
            stamp = f"{actual:.6f}".rstrip("0").rstrip(".").replace(".", "p") + "s"
            title = f"DCMIP 4-1 — {label}, day {actual * scaling / 86400:g} (t={actual:g} s)"
            pressure = read_field(ds, "P", index)
            theta = read_field(ds, "theta", index)
            u, v, w = [read_field(ds, name, index) for name in ("U", "V", "W")]
            if target in primary_times:
                t850 = interpolate_to_pressure(temperature_from_theta(theta, pressure), pressure, TARGET_PRESSURE)
                u850 = interpolate_to_pressure(u, pressure, TARGET_PRESSURE)
                v850 = interpolate_to_pressure(v, pressure, TARGET_PRESSURE)
                w850 = interpolate_to_pressure(w, pressure, TARGET_PRESSURE)
                zeta = relative_vorticity_panel(u850, v850, lat, lon, radius)
                panels = [
                    ("Surface pressure", surface_pressure(ds, pressure, elevation, index) / 100, "hPa", False),
                    ("850 hPa temperature", t850, "K", False),
                    ("850 hPa vertical velocity", w850, "m/s", True),
                    ("850 hPa relative vorticity", zeta * 1e5, r"$10^{-5}$ s$^{-1}$", True),
                ]
                plot_maps(lat, lon, panels, title, output / f"{prefix}_{stamp}.png", args.pdf)
                if "q1" in ds.variables:
                    tracer = read_field(ds, "q1", index)
                    theta850 = interpolate_to_pressure(theta, pressure, TARGET_PRESSURE)
                    tracer850 = interpolate_to_pressure(tracer, pressure, TARGET_PRESSURE)
                    panels = [
                        ("850 hPa dynamic theta", theta850, "K", False),
                        ("850 hPa tracer theta", tracer850, "K", False),
                        (
                            "Tracer minus 315 K on theta=315 K",
                            interpolate_to_isentrope(tracer, theta, elevation) - 315,
                            "K",
                            True,
                        ),
                    ]
                    plot_maps(lat, lon, panels, title, output / f"{prefix}_theta_{stamp}.png", args.pdf)
            if target in epv_times:
                rho = read_field(ds, "rho", index)
                epv = dynamic_epv(u, v, w, theta, rho, elevation, lat, lon, radius, rotation)
                epv315 = interpolate_to_isentrope(epv, theta, elevation)
                panels = [("Dynamic EPV on theta=315 K", epv315 * 1e6, "PVU", False)]
                if "q2" in ds.variables:
                    tracer315 = interpolate_to_isentrope(read_field(ds, "q2", index), theta, elevation)
                    panels.append(("Tracer EPV on theta=315 K", tracer315 * 1e6, "PVU", False))
                    valid = (lat >= 30) & np.isfinite(epv315) & np.isfinite(tracer315)
                    fig, ax = plt.subplots()
                    ax.scatter(epv315[valid] * 1e6, tracer315[valid] * 1e6, s=2, alpha=0.3)
                    ax.set(xlabel="Dynamic EPV (PVU)", ylabel="Tracer EPV (PVU)", title=title)
                    save_figure(fig, output / f"{prefix}_epv_scatter_{stamp}.png", args.pdf)
                plot_maps(lat, lon, panels, title, output / f"{prefix}_epv_{stamp}.png", args.pdf, northern=True)
            if target == spectrum_time:
                spectrum = kinetic_energy_spectrum(
                    interpolate_to_pressure(u, pressure, 70000),
                    interpolate_to_pressure(v, pressure, 70000),
                    lat,
                    lon,
                    args.spectrum_degree,
                )
                fig, ax = plt.subplots()
                degrees = np.arange(1, spectrum.size)
                ax.loglog(degrees, spectrum[1:])
                ax.set(
                    xlabel="Spherical-harmonic degree",
                    ylabel=r"KE per degree (m$^2$ s$^{-2}$)",
                    title=title + " — 700 hPa",
                )
                print(
                    "KE spectrum uses spherical interpolation to Gauss-Legendre quadrature; high degrees depend on resolution."
                )
                save_figure(fig, output / f"{prefix}_ke700_{stamp}.png", args.pdf)


if __name__ == "__main__":
    main()
