"""Phase 4 — compare the current run against a baseline.

Two complementary, *report-only* checks (the run never fails on these):

* **Data L2** — for each time-varying field present in both the current and the
  baseline NetCDF, the relative L2 error over the intersection of available
  timesteps: ``||a - b||_2 / ||b||_2``. Computed incrementally (one timestep at
  a time) so that large files (e.g. dcmip21, ~4.5 GB) are not loaded whole.
* **SSIM** — for each figure, matched by filename against the baseline, the
  structural similarity (grayscale, data range 0–1). Flagged if below the
  threshold (default 0.98).

Outputs:

* ``<output_root>/report.json`` — machine-readable.
* ``<output_root>/report.html`` — a browsable per-config table with run status,
  walltime, per-variable L2, per-figure SSIM pass/flag, and side-by-side figure
  links against the baseline.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

import numpy

from .regression_common import (
    CATEGORY_LABELS,
    CATEGORY_ORDER,
    category_of,
    log,
    time_sort_key,
)
from .regression_discover import ConfigInfo

# Time-varying fields considered for the data-level L2, by equation set.
SW_VARS = ["h", "U", "V", "RV", "PV"]
EULER_VARS = ["U", "V", "W", "rho", "theta", "P", "q1", "q2", "q3", "q4"]

# Coordinate / metadata variables that are not model fields.
NON_FIELD_VARS = {
    "time",
    "lats",
    "lons",
    "elev",
    "topo",
    "volume",
    "nfaces",
    "Ydim",
    "Xdim",
    "Zdim",
}


# --------------------------------------------------------------------------- #
# Data-level L2
# --------------------------------------------------------------------------- #


def _field_vars(info: ConfigInfo, names_a: set, names_b: set) -> list[str]:
    """The model fields to compare: the equation-specific list, restricted to
    variables present in both files."""
    preferred = SW_VARS if info.equations == "shallow_water" else EULER_VARS
    common = (names_a & names_b) - NON_FIELD_VARS
    # Keep the preferred order, then append any other common fields.
    ordered = [v for v in preferred if v in common]
    ordered += sorted(v for v in common if v not in preferred)
    return ordered


def _match_times(ta: numpy.ndarray, tb: numpy.ndarray) -> list[tuple[int, int]]:
    """Pair up timesteps by nearest time value, within a tolerance.

    Returns a list of ``(ia, ib)`` index pairs. The tolerance is half the median
    timestep of the current run, so identical configs pair 1:1 while a slightly
    shorter baseline still pairs on the overlapping range.
    """
    if ta.size == 0 or tb.size == 0:
        return []
    dts = numpy.diff(ta)
    dts = dts[dts > 0]
    tol = 0.5 * float(numpy.median(dts)) if dts.size else 0.0
    pairs = []
    for ia, t in enumerate(ta):
        ib = int(numpy.argmin(numpy.abs(tb - t)))
        if abs(float(t) - float(tb[ib])) <= tol:
            pairs.append((ia, ib))
    return pairs


def rel_l2_variable(ds_a, ds_b, var: str) -> float | None:
    """Relative L2 ``||a-b||_2 / ||b||_2`` over the matched timesteps."""
    ta = numpy.asarray(ds_a.variables["time"][:])
    tb = numpy.asarray(ds_b.variables["time"][:])
    pairs = _match_times(ta, tb)
    if not pairs:
        return None

    sum_sq_diff = 0.0
    sum_sq_b = 0.0
    va = ds_a.variables[var]
    vb = ds_b.variables[var]
    for ia, ib in pairs:
        a = numpy.asarray(va[ia], dtype=numpy.float64)
        b = numpy.asarray(vb[ib], dtype=numpy.float64)
        if a.shape != b.shape:
            continue
        diff = a - b
        sum_sq_diff += float((diff * diff).sum())
        sum_sq_b += float((b * b).sum())

    if sum_sq_b <= 0.0:
        return None
    return float(numpy.sqrt(sum_sq_diff / sum_sq_b))


def data_l2(info: ConfigInfo, nc_a: Path, nc_b: Path) -> dict[str, float | None]:
    """Relative L2 for every comparable field. Returns ``{var: value}``."""
    import netCDF4

    result: dict[str, float | None] = {}
    with netCDF4.Dataset(nc_a) as ds_a, netCDF4.Dataset(nc_b) as ds_b:
        names_a = set(ds_a.variables)
        names_b = set(ds_b.variables)
        for var in _field_vars(info, names_a, names_b):
            try:
                result[var] = rel_l2_variable(ds_a, ds_b, var)
            except (KeyError, ValueError, TypeError) as exc:
                log(f"  [{info.stem}] L2 for '{var}' failed: {exc}")
                result[var] = None
    return result


# --------------------------------------------------------------------------- #
# Image SSIM
# --------------------------------------------------------------------------- #


def _load_gray(path: Path) -> numpy.ndarray | None:
    """Load an image as a float64 grayscale array in [0, 1]."""
    try:
        from PIL import Image
    except ImportError:
        return None
    with Image.open(path) as img:
        arr = numpy.asarray(img.convert("L"), dtype=numpy.float64)
    if arr.size == 0:
        return None
    return arr / 255.0


def ssim_pair(a: Path, b: Path) -> float | None:
    """Structural similarity of two images (None if not comparable)."""
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        log("  scikit-image not available; SSIM disabled")
        return None

    img_a = _load_gray(a)
    img_b = _load_gray(b)
    if img_a is None or img_b is None:
        return None
    if img_a.shape != img_b.shape:
        return None
    return float(structural_similarity(img_a, img_b, data_range=1.0))


def compare_figures(current_dir: Path | None, baseline_dir: Path | None, threshold: float) -> dict:
    """Match figures by filename and compute SSIM.

    Returns a dict with the per-figure SSIM values and the sets of figures that
    are missing on either side. Either directory may be ``None`` (no plots).
    """
    current = {p.name: p for p in sorted(current_dir.glob("*.png"))} if current_dir and current_dir.is_dir() else {}
    baseline = {p.name: p for p in sorted(baseline_dir.glob("*.png"))} if baseline_dir and baseline_dir.is_dir() else {}

    ssim: dict[str, float | None] = {}
    for name in sorted(set(current) & set(baseline)):
        ssim[name] = ssim_pair(current[name], baseline[name])

    return {
        "ssim": ssim,
        "missing_in_baseline": sorted(set(current) - set(baseline)),
        "missing_in_current": sorted(set(baseline) - set(current)),
        "threshold": threshold,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _plots_dir(run_dir: Path) -> Path:
    return run_dir / "plots"


def _nc_path(info: ConfigInfo) -> Path | None:
    if info.run_dir is None:
        return None
    return info.run_dir / f"{info.base_output_file}.nc"


def compare_all(
    infos: list[ConfigInfo],
    output_root: Path,
    baseline_root: Path | None,
    ssim_threshold: float,
    walltime_delta: dict | None = None,
) -> dict:
    """Compare every config against the baseline and write the reports.

    ``walltime_delta`` maps ``stem -> (current_walltime, baseline_walltime)``
    (seconds); when present, the report shows the per-config walltime change so
    performance regressions are visible at a glance.
    """
    walltime_delta = walltime_delta or {}
    report: dict = {
        "run_id": output_root.name,
        "baseline_id": baseline_root.name if baseline_root else None,
        "ssim_threshold": ssim_threshold,
        "configs": {},
    }

    for info in infos:
        cur_wt, base_wt = walltime_delta.get(info.stem, (info.walltime, None))
        entry: dict = {
            "category": category_of(info.grid_type, info.equations),
            "status": info.status,
            "walltime": info.walltime,
            "walltime_baseline": base_wt,
            "walltime_delta": (cur_wt - base_wt) if (cur_wt is not None and base_wt is not None) else None,
            "exit_code": info.exit_code,
            "data_l2": None,
            "ssim": None,
            "ssim_min": None,
            "figures_current": [],
            "figures_baseline": [],
            "figures_missing_in_baseline": [],
            "figures_missing_in_current": [],
        }

        cur_plots = _plots_dir(info.run_dir) if info.run_dir else None
        entry["figures_current"] = (
            sorted(p.name for p in cur_plots.glob("*.png")) if cur_plots and cur_plots.is_dir() else []
        )

        if info.status != "ok" or baseline_root is None:
            report["configs"][info.stem] = entry
            continue

        # --- data L2 ------------------------------------------------------- #
        nc_a = _nc_path(info)
        nc_b = baseline_root / info.stem / f"{info.base_output_file}.nc"
        if info.is_cubed_sphere and nc_a is not None and nc_a.is_file() and nc_b.is_file():
            log(f"  [{info.stem}] computing data L2 vs baseline ...")
            entry["data_l2"] = data_l2(info, nc_a, nc_b)
        elif info.is_cubed_sphere:
            entry["data_l2_note"] = "baseline NetCDF not found"

        # --- SSIM ---------------------------------------------------------- #
        base_plots = baseline_root / info.stem / "plots"
        fig = compare_figures(cur_plots, base_plots, ssim_threshold)
        entry["ssim"] = fig["ssim"]
        entry["figures_missing_in_baseline"] = fig["missing_in_baseline"]
        entry["figures_missing_in_current"] = fig["missing_in_current"]
        entry["figures_baseline"] = sorted(p.name for p in base_plots.glob("*.png")) if base_plots.is_dir() else []

        values = [v for v in fig["ssim"].values() if v is not None]
        entry["ssim_min"] = min(values) if values else None

        report["configs"][info.stem] = entry

    # --- write reports ----------------------------------------------------- #
    json_path = output_root / "report.json"
    json_path.write_text(json.dumps(report, indent=2))
    log(f"Wrote {json_path}")

    html_path = output_root / "report.html"
    html_path.write_text(render_html(report, output_root, baseline_root))
    log(f"Wrote {html_path}")

    return report


# --------------------------------------------------------------------------- #
# HTML rendering
# --------------------------------------------------------------------------- #


def _rel_link(from_dir: Path, target: Path) -> str:
    """A relative href from ``from_dir`` to ``target``.

    Uses ``os.path.relpath`` (not ``Path.relative_to``) so that targets in a
    *sibling* directory — e.g. the baseline run, which lives next to the current
    run — resolve to ``../<baseline>/...`` instead of an absolute path the
    browser cannot load.
    """
    return html.escape(os.path.relpath(target, from_dir))


def _fmt(value, spec=".3e") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return format(value, spec)
    return html.escape(str(value))


def _fmt_time(seconds: float | None) -> str:
    """Format a duration in seconds as M:SS (or H:MM:SS when >= 1 h)."""
    if seconds is None:
        return "—"
    total = round(abs(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def render_html(report: dict, output_root: Path, baseline_root: Path | None) -> str:
    """Render the per-config comparison table as a self-contained HTML page."""
    threshold = report["ssim_threshold"]
    parts: list[str] = []
    parts.append("<!DOCTYPE html>\n<html><head><meta charset='utf-8'>")
    parts.append("<title>WxFactory regression report</title>")
    parts.append(
        "<style>"
        "body{font-family:system-ui,sans-serif;margin:24px;color:#1a1a1a;}"
        "h1{font-size:20px;}"
        ".cat{font-size:18px;border-bottom:2px solid #ccc;padding-bottom:4px;margin-top:32px;}"
        "h3{font-size:14px;margin:16px 0 4px;color:#333;}"
        "table{border-collapse:collapse;width:100%;font-size:13px;}"
        "th,td{border:1px solid #ccc;padding:6px 8px;text-align:left;vertical-align:top;}"
        "th{background:#f0f0f0;}"
        ".ok{color:#0a7a2b;font-weight:600;} .failed{color:#c0392b;font-weight:600;}"
        ".flag{color:#c0392b;font-weight:600;} .pass{color:#0a7a2b;}"
        ".meta{color:#555;font-size:12px;}"
        "img{max-width:220px;border:1px solid #ddd;}"
        ".pair{display:flex;gap:12px;flex-wrap:wrap;}"
        ".pair div{font-size:11px;}"
        "</style></head><body>"
    )
    parts.append("<h1>WxFactory regression report</h1>")
    parts.append(
        f"<p class='meta'>run: <b>{html.escape(report['run_id'])}</b> &nbsp;|&nbsp; "
        f"baseline: <b>{html.escape(report['baseline_id'] or 'none (first run)')}</b> &nbsp;|&nbsp; "
        f"SSIM threshold: {threshold}</p>"
    )

    # Order configs by display category (DCMIP / shallow water / cartesian),
    # then by name within each category.
    ordered: list[tuple[str, str]] = []
    for category in CATEGORY_ORDER:
        for stem in sorted(s for s, e in report["configs"].items() if e.get("category", "dcmip") == category):
            ordered.append((category, stem))
    for stem in sorted(report["configs"]):  # defensive: any unexpected category
        if all(stem != s for _, s in ordered):
            ordered.append((report["configs"][stem].get("category", "dcmip"), stem))

    last_category = None
    for category, stem in ordered:
        if category != last_category:
            parts.append(f"<h2 class='cat'>{html.escape(CATEGORY_LABELS.get(category, category))}</h2>")
            last_category = category
        e = report["configs"][stem]
        status_cls = "ok" if e["status"] == "ok" else "failed"
        parts.append(f"<h3>{html.escape(stem)} <span class='{status_cls}'>[{e['status']}]</span></h3>")
        parts.append("<table>")
        parts.append(
            f"<tr><th>run</th><td>{e['status']} (exit {e['exit_code']})</td>"
            f"<th>walltime</th><td>{_fmt_time(e['walltime'])}</td></tr>"
        )
        # Walltime change vs the baseline run (performance tracking).
        if e.get("walltime_delta") is not None:
            delta = e["walltime_delta"]
            base = e.get("walltime_baseline") or 0
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "＝")
            cls = "flag" if delta > 0 else ("pass" if delta < 0 else "meta")
            sign = "+" if delta > 0 else ("−" if delta < 0 else "")
            parts.append(
                f"<tr><th>Δ walltime</th>"
                f"<td colspan='3' class='{cls}'>{arrow} {sign}{_fmt_time(delta)} "
                f"(baseline {_fmt_time(base)})</td></tr>"
            )

        # Data L2 row.
        if e.get("data_l2"):
            # Inline spans (not nested <td>) so the values stay inside the
            # single colspan'd cell rather than spilling into new columns.
            cells = " &nbsp; ".join(f"<b>{html.escape(v)}</b>: {_fmt(val)}" for v, val in e["data_l2"].items())
            parts.append(f"<tr><th>data L2</th><td colspan='3'>{cells}</td></tr>")
        elif e.get("data_l2_note"):
            parts.append(f"<tr><th>data L2</th><td colspan='3' class='meta'>{html.escape(e['data_l2_note'])}</td></tr>")

        # SSIM rows: one per figure, with side-by-side links. Ordered by
        # simulation time (t0 < tmid < tlast, day6 < day12) so the middle time
        # sorts between the first and the last.
        ssim = e.get("ssim") or {}
        if ssim:
            parts.append("<tr><th>figure</th><th>SSIM</th><th>verdict</th><th>current vs baseline</th></tr>")
            for name in sorted(ssim, key=time_sort_key):
                val = ssim[name]
                if val is None:
                    verdict, cls = "n/a", "meta"
                elif val >= threshold:
                    verdict, cls = "pass", "pass"
                else:
                    verdict, cls = "FLAG", "flag"
                cur_img = output_root / stem / "plots" / name
                base_img = (baseline_root / stem / "plots" / name) if baseline_root else None
                pair = f"<img src='{_rel_link(output_root, cur_img)}' alt='{html.escape(name)}'>"
                if base_img is not None and base_img.is_file():
                    pair += f" <img src='{_rel_link(output_root, base_img)}' alt='baseline {html.escape(name)}'>"
                parts.append(
                    f"<tr><td>{html.escape(name)}</td><td>{_fmt(val, '.5f')}</td>"
                    f"<td class='{cls}'>{verdict}</td><td>{pair}</td></tr>"
                )
            if e.get("figures_missing_in_baseline"):
                parts.append(
                    "<tr><td colspan='4' class='meta'>only in current: "
                    + html.escape(", ".join(e["figures_missing_in_baseline"]))
                    + "</td></tr>"
                )
            if e.get("figures_missing_in_current"):
                parts.append(
                    "<tr><td colspan='4' class='meta'>only in baseline: "
                    + html.escape(", ".join(e["figures_missing_in_current"]))
                    + "</td></tr>"
                )

        parts.append("</table>")

    parts.append("</body></html>")
    return "\n".join(parts)
