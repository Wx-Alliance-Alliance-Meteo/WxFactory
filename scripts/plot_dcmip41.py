#!/usr/bin/env python3

import numpy as np
import netCDF4
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator, FormatStrFormatter

# ============================================================
# User settings
# ============================================================

INPUT_FILE = "results/dcmip41.nc"

# Requested elapsed simulation time in seconds
TARGET_SECONDS = 200.0

# Pressure surface
TARGET_PRESSURE = 85000.0  # Pa

# Output
OUTPUT_PREFIX = "dcmip41"


# ============================================================
# Physical constants
# ============================================================

RD = 287.05
CPD = 1005.46
P0 = 100000.0

KAPPA = RD / CPD

EARTH_RADIUS = 6371220.0


# ============================================================
# Filled-contour settings
# ============================================================

# ------------------------------------------------------------
# Near-surface pressure
# ------------------------------------------------------------

PS_MIN = None
PS_MAX = None
PS_NLEVELS = 20

# ------------------------------------------------------------
# 850-hPa temperature
# ------------------------------------------------------------

T850_MIN = None
T850_MAX = None
T850_NLEVELS = 20

# ------------------------------------------------------------
# 850-hPa relative vorticity
# ------------------------------------------------------------

VORT_MIN = None
VORT_MAX = None
VORT_NLEVELS = 20


# ============================================================
# Contour-line settings
# ============================================================

DRAW_PS_CONTOURS = True
DRAW_T850_CONTOURS = True
DRAW_VORT_CONTOURS = False

PS_NLINE_LEVELS = 8
T850_NLINE_LEVELS = 8
VORT_NLINE_LEVELS = 8


# ============================================================
# Colorbar settings
# ============================================================

# Maximum approximate number of colorbar labels
PS_CBAR_TICKS = 5
T850_CBAR_TICKS = 5
VORT_CBAR_TICKS = 5

# Number formatting
PS_CBAR_FORMAT = "%.0f"
T850_CBAR_FORMAT = "%.0f"
VORT_CBAR_FORMAT = "%.2f"


# ============================================================
# Colormaps
# ============================================================

PS_CMAP = "viridis"
T850_CMAP = "RdYlBu_r"
VORT_CMAP = "RdBu_r"


# ============================================================
# Figure settings
# ============================================================

FIGSIZE = (16.0, 5.1)

TITLE_SIZE = 15
LABEL_SIZE = 14
TICK_SIZE = 11

CBAR_LABEL_SIZE = 13
CBAR_TICK_SIZE = 10

DPI = 300


# ============================================================
# Thermodynamics
# ============================================================


def temperature_from_theta(theta, pressure):
    return theta * (pressure / P0) ** KAPPA


# ============================================================
# Time selection
# ============================================================


def select_time(raw_time, target_seconds):

    raw_time = np.asarray(raw_time, dtype=np.float64)

    if raw_time.ndim != 1:
        raise ValueError(f"time must be one-dimensional; got {raw_time.shape}")

    if raw_time.size == 0:
        raise ValueError("No output times were found.")

    if not np.all(np.isfinite(raw_time)):
        raise ValueError("Non-finite values found in time.")

    # Numerical time values produced by this model are
    # elapsed simulation seconds.
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

    return (
        itime,
        elapsed_seconds,
        actual_seconds,
        difference_seconds,
        typical_output_interval,
    )


# ============================================================
# Vertical interpolation
# ============================================================


def interpolate_to_pressure(
    field,
    pressure,
    target_pressure,
):

    if field.shape != pressure.shape:

        raise ValueError("field and pressure must have identical shapes.")

    npanel, nz, ny, nx = field.shape

    result = np.full(
        (npanel, ny, nx),
        np.nan,
        dtype=np.float64,
    )

    for panel in range(npanel):

        for j in range(ny):

            for i in range(nx):

                pcol = pressure[
                    panel,
                    :,
                    j,
                    i,
                ]

                fcol = field[
                    panel,
                    :,
                    j,
                    i,
                ]

                valid = np.isfinite(pcol) & np.isfinite(fcol)

                if np.count_nonzero(valid) < 2:
                    continue

                p = np.asarray(
                    pcol[valid],
                    dtype=np.float64,
                )

                f = np.asarray(
                    fcol[valid],
                    dtype=np.float64,
                )

                order = np.argsort(p)

                p = p[order]
                f = f[order]

                p_unique, index = np.unique(
                    p,
                    return_index=True,
                )

                f_unique = f[index]

                if p_unique.size < 2:
                    continue

                if target_pressure < p_unique[0] or target_pressure > p_unique[-1]:
                    continue

                result[
                    panel,
                    j,
                    i,
                ] = np.interp(
                    target_pressure,
                    p_unique,
                    f_unique,
                )

    return result


# ============================================================
# Longitude handling
# ============================================================


def unwrap_longitude(lon):

    lon_rad = np.deg2rad(lon)

    lon_rad = np.unwrap(
        lon_rad,
        axis=-1,
    )

    lon_rad = np.unwrap(
        lon_rad,
        axis=-2,
    )

    return lon_rad


# ============================================================
# Relative vorticity
# ============================================================


def relative_vorticity_panel(
    u,
    v,
    lat,
    lon,
    earth_radius,
):

    npanel, ny, nx = u.shape

    zeta = np.full_like(
        u,
        np.nan,
        dtype=np.float64,
    )

    for panel in range(npanel):

        phi = np.deg2rad(lat[panel])

        lam = unwrap_longitude(lon[panel])

        up = np.asarray(
            u[panel],
            dtype=np.float64,
        )

        vp = np.asarray(
            v[panel],
            dtype=np.float64,
        )

        cosphi = np.cos(phi)

        # ----------------------------------------------------
        # Computational derivatives
        # ----------------------------------------------------

        dv_dj, dv_di = np.gradient(vp)

        ucos = up * cosphi

        ducos_dj, ducos_di = np.gradient(ucos)

        dlam_dj, dlam_di = np.gradient(lam)

        dphi_dj, dphi_di = np.gradient(phi)

        # ----------------------------------------------------
        # Coordinate transformation
        # ----------------------------------------------------

        determinant = dlam_di * dphi_dj - dlam_dj * dphi_di

        determinant = np.where(
            np.abs(determinant) < 1.0e-14,
            np.nan,
            determinant,
        )

        dv_dlambda = (dv_di * dphi_dj - dv_dj * dphi_di) / determinant

        ducos_dphi = (dlam_di * ducos_dj - dlam_dj * ducos_di) / determinant

        denominator = earth_radius * cosphi

        denominator = np.where(
            np.abs(cosphi) < 1.0e-8,
            np.nan,
            denominator,
        )

        zeta[panel] = (dv_dlambda - ducos_dphi) / denominator

    return zeta


# ============================================================
# Contour levels
# ============================================================


def make_levels(
    field,
    vmin=None,
    vmax=None,
    nlevels=20,
    symmetric=False,
):

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

        return np.linspace(
            -vmax_abs,
            vmax_abs,
            nlevels + 1,
        )

    if vmin is None:
        vmin = float(np.nanmin(finite))

    if vmax is None:
        vmax = float(np.nanmax(finite))

    if vmin == vmax:

        delta = abs(vmin) * 0.01 if vmin != 0.0 else 1.0

        vmin -= delta
        vmax += delta

    return np.linspace(
        vmin,
        vmax,
        nlevels + 1,
    )


def make_line_levels(
    levels,
    nlevels,
):

    return np.linspace(
        levels[0],
        levels[-1],
        nlevels,
    )


# ============================================================
# Clean colorbar ticks
# ============================================================


def make_colorbar_ticks(
    vmin,
    vmax,
    nbins=5,
):

    locator = MaxNLocator(
        nbins=nbins,
        min_n_ticks=3,
    )

    ticks = locator.tick_values(
        vmin,
        vmax,
    )

    tolerance = abs(vmax - vmin) * 1.0e-10

    ticks = ticks[(ticks >= vmin - tolerance) & (ticks <= vmax + tolerance)]

    return ticks


def add_horizontal_colorbar(
    fig,
    ax,
    image,
    levels,
    label,
    nbins,
    tick_format,
):

    ticks = make_colorbar_ticks(
        levels[0],
        levels[-1],
        nbins=nbins,
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
        orientation="horizontal",
        pad=0.16,
        fraction=0.055,
        shrink=0.94,
        aspect=28,
        ticks=ticks,
    )

    cbar.ax.xaxis.set_major_formatter(FormatStrFormatter(tick_format))

    cbar.set_label(
        label,
        fontsize=CBAR_LABEL_SIZE,
        labelpad=5,
    )

    cbar.ax.tick_params(
        labelsize=CBAR_TICK_SIZE,
        length=3,
        pad=2,
    )

    return cbar


# ============================================================
# Cubed-sphere plotting
# ============================================================


def plot_cubesphere_field(
    ax,
    lon,
    lat,
    field,
    levels,
    cmap,
    draw_contours=False,
    line_levels=None,
):

    image = None

    for panel in range(field.shape[0]):

        lon_panel = np.asarray(
            lon[panel],
            dtype=np.float64,
        ).copy()

        lat_panel = np.asarray(
            lat[panel],
            dtype=np.float64,
        ).copy()

        field_panel = np.asarray(
            field[panel],
            dtype=np.float64,
        ).copy()

        lon_panel = ((lon_panel + 180.0) % 360.0) - 180.0

        # ----------------------------------------------------
        # Mask longitude discontinuity
        # ----------------------------------------------------

        seam = np.zeros_like(
            lon_panel,
            dtype=bool,
        )

        diff_x = np.abs(
            np.diff(
                lon_panel,
                axis=1,
            )
        )

        diff_y = np.abs(
            np.diff(
                lon_panel,
                axis=0,
            )
        )

        seam[:, :-1] |= diff_x > 180.0

        seam[:, 1:] |= diff_x > 180.0

        seam[:-1, :] |= diff_y > 180.0

        seam[1:, :] |= diff_y > 180.0

        masked = np.ma.masked_where(
            seam | ~np.isfinite(field_panel),
            field_panel,
        )

        image = ax.contourf(
            lon_panel,
            lat_panel,
            masked,
            levels=levels,
            cmap=cmap,
            extend="both",
            antialiased=True,
        )

        if draw_contours and line_levels is not None:

            ax.contour(
                lon_panel,
                lat_panel,
                masked,
                levels=line_levels,
                colors="white",
                linewidths=0.45,
                alpha=0.70,
            )

    return image


# ============================================================
# Axis formatting
# ============================================================


def configure_axis(
    ax,
    show_ylabel=True,
):

    ax.set_xlim(
        -180.0,
        180.0,
    )

    ax.set_ylim(
        -90.0,
        90.0,
    )

    ax.set_xlabel(
        "Longitude (°)",
        fontsize=LABEL_SIZE,
    )

    if show_ylabel:

        ax.set_ylabel(
            "Latitude (°)",
            fontsize=LABEL_SIZE,
        )

    else:

        ax.set_ylabel("")

    ax.xaxis.set_major_locator(MultipleLocator(60))

    ax.yaxis.set_major_locator(MultipleLocator(30))

    ax.tick_params(
        axis="both",
        labelsize=TICK_SIZE,
        direction="out",
        length=3,
    )

    ax.grid(
        linewidth=0.4,
        alpha=0.20,
    )


# ============================================================
# Main
# ============================================================


def main():

    # ========================================================
    # Read data
    # ========================================================

    with netCDF4.Dataset(INPUT_FILE) as ds:

        time_var = ds.variables["time"]

        raw_time = np.asarray(
            time_var[:],
            dtype=np.float64,
        )

        (
            itime,
            elapsed_seconds,
            actual_seconds,
            difference_seconds,
            output_dt,
        ) = select_time(
            raw_time,
            TARGET_SECONDS,
        )

        lat = np.asarray(
            ds.variables["lats"][:],
            dtype=np.float64,
        )

        lon = np.asarray(
            ds.variables["lons"][:],
            dtype=np.float64,
        )

        elevation = np.asarray(
            ds.variables["elev"][:],
            dtype=np.float64,
        )

        pressure = np.asarray(
            ds.variables["P"][itime],
            dtype=np.float64,
        )

        theta = np.asarray(
            ds.variables["theta"][itime],
            dtype=np.float64,
        )

        u = np.asarray(
            ds.variables["U"][itime],
            dtype=np.float64,
        )

        v = np.asarray(
            ds.variables["V"][itime],
            dtype=np.float64,
        )

        time_units = getattr(
            time_var,
            "units",
            "",
        )

    # ========================================================
    # Time diagnostics
    # ========================================================

    print("")
    print("==============================================")
    print("DCMIP 4-1 BAROCLINIC INSTABILITY")
    print("==============================================")
    print(f"Input file           : {INPUT_FILE}")
    print(f"NetCDF time units    : {time_units}")
    print(f"Number of snapshots  : {raw_time.size}")
    print(f"Requested time       : {TARGET_SECONDS:.6f} s")
    print(f"Selected time        : {actual_seconds:.6f} s")
    print(f"Difference           : {difference_seconds:.6f} s")
    print(f"Time index           : {itime}")
    print(f"First available time : {elapsed_seconds[0]:.6f} s")
    print(f"Last available time  : {elapsed_seconds[-1]:.6f} s")

    if np.isfinite(output_dt):

        print(f"Typical output dt    : " f"{output_dt:.6f} s")

    if not np.isclose(
        actual_seconds,
        TARGET_SECONDS,
        rtol=0.0,
        atol=1.0e-10,
    ):

        print("")
        print("WARNING:")
        print("Requested time is not available exactly.")
        print("Using the nearest available snapshot.")

    # ========================================================
    # Temperature
    # ========================================================

    temperature = temperature_from_theta(
        theta,
        pressure,
    )

    # ========================================================
    # Lowest-level pressure
    # ========================================================

    lowest_index = np.argmin(
        elevation,
        axis=1,
    )

    ps = np.take_along_axis(
        pressure,
        lowest_index[:, None, :, :],
        axis=1,
    )[:, 0]

    ps_hpa = ps / 100.0

    # ========================================================
    # 850-hPa interpolation
    # ========================================================

    print("")
    print("Interpolating temperature to 850 hPa...")

    t850 = interpolate_to_pressure(
        temperature,
        pressure,
        TARGET_PRESSURE,
    )

    print("Interpolating U to 850 hPa...")

    u850 = interpolate_to_pressure(
        u,
        pressure,
        TARGET_PRESSURE,
    )

    print("Interpolating V to 850 hPa...")

    v850 = interpolate_to_pressure(
        v,
        pressure,
        TARGET_PRESSURE,
    )

    # ========================================================
    # Relative vorticity
    # ========================================================

    print("Computing 850-hPa relative vorticity...")

    zeta850 = relative_vorticity_panel(
        u850,
        v850,
        lat,
        lon,
        EARTH_RADIUS,
    )

    zeta850_plot = zeta850 * 1.0e5

    # ========================================================
    # Field diagnostics
    # ========================================================

    print("")
    print("Field ranges")
    print("----------------------------------------------")

    print(f"Near-surface pressure : " f"{np.nanmin(ps_hpa):.3f} to " f"{np.nanmax(ps_hpa):.3f} hPa")

    print(f"T850                  : " f"{np.nanmin(t850):.3f} to " f"{np.nanmax(t850):.3f} K")

    print(
        f"zeta850               : "
        f"{np.nanmin(zeta850_plot):.3f} to "
        f"{np.nanmax(zeta850_plot):.3f} "
        f"x10^-5 s^-1"
    )

    # ========================================================
    # Filled contour levels
    # ========================================================

    ps_levels = make_levels(
        ps_hpa,
        vmin=PS_MIN,
        vmax=PS_MAX,
        nlevels=PS_NLEVELS,
    )

    t850_levels = make_levels(
        t850,
        vmin=T850_MIN,
        vmax=T850_MAX,
        nlevels=T850_NLEVELS,
    )

    vort_levels = make_levels(
        zeta850_plot,
        vmin=VORT_MIN,
        vmax=VORT_MAX,
        nlevels=VORT_NLEVELS,
        symmetric=True,
    )

    # ========================================================
    # Contour-line levels
    # ========================================================

    ps_line_levels = make_line_levels(
        ps_levels,
        PS_NLINE_LEVELS,
    )

    t850_line_levels = make_line_levels(
        t850_levels,
        T850_NLINE_LEVELS,
    )

    vort_line_levels = make_line_levels(
        vort_levels,
        VORT_NLINE_LEVELS,
    )

    # ========================================================
    # Figure
    # ========================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=FIGSIZE,
    )

    # Extra room below panels for horizontal colorbars
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.90,
        bottom=0.24,
        wspace=0.24,
    )

    # ========================================================
    # Panel (a)
    # ========================================================

    image = plot_cubesphere_field(
        axes[0],
        lon,
        lat,
        ps_hpa,
        ps_levels,
        PS_CMAP,
        draw_contours=DRAW_PS_CONTOURS,
        line_levels=ps_line_levels,
    )

    configure_axis(
        axes[0],
        show_ylabel=True,
    )

    axes[0].set_title(
        "(a) Near-surface pressure",
        fontsize=TITLE_SIZE,
        pad=7,
    )

    add_horizontal_colorbar(
        fig,
        axes[0],
        image,
        ps_levels,
        "Pressure (hPa)",
        PS_CBAR_TICKS,
        PS_CBAR_FORMAT,
    )

    # ========================================================
    # Panel (b)
    # ========================================================

    image = plot_cubesphere_field(
        axes[1],
        lon,
        lat,
        t850,
        t850_levels,
        T850_CMAP,
        draw_contours=DRAW_T850_CONTOURS,
        line_levels=t850_line_levels,
    )

    configure_axis(
        axes[1],
        show_ylabel=True,
    )

    axes[1].set_title(
        "(b) 850-hPa temperature",
        fontsize=TITLE_SIZE,
        pad=7,
    )

    add_horizontal_colorbar(
        fig,
        axes[1],
        image,
        t850_levels,
        "Temperature (K)",
        T850_CBAR_TICKS,
        T850_CBAR_FORMAT,
    )

    # ========================================================
    # Panel (c)
    # ========================================================

    image = plot_cubesphere_field(
        axes[2],
        lon,
        lat,
        zeta850_plot,
        vort_levels,
        VORT_CMAP,
        draw_contours=DRAW_VORT_CONTOURS,
        line_levels=vort_line_levels,
    )

    configure_axis(
        axes[2],
        show_ylabel=True,
    )

    axes[2].set_title(
        "(c) 850-hPa relative vorticity",
        fontsize=TITLE_SIZE,
        pad=7,
    )

    add_horizontal_colorbar(
        fig,
        axes[2],
        image,
        vort_levels,
        r"Relative vorticity ($10^{-5}$ s$^{-1}$)",
        VORT_CBAR_TICKS,
        VORT_CBAR_FORMAT,
    )

    # ========================================================
    # Output filename
    # ========================================================

    if np.isclose(
        actual_seconds,
        round(actual_seconds),
    ):

        time_string = f"{int(round(actual_seconds)):06d}s"

    else:

        time_string = f"{actual_seconds:.3f}".replace(".", "p") + "s"

    png_file = f"{OUTPUT_PREFIX}_{time_string}.png"

    pdf_file = f"{OUTPUT_PREFIX}_{time_string}.pdf"

    fig.savefig(
        png_file,
        dpi=DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_file,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("")
    print("Saved")
    print("----------------------------------------------")
    print(f"  {png_file}")
    print(f"  {pdf_file}")


if __name__ == "__main__":
    main()
