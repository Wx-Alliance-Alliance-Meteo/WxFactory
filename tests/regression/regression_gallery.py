"""Phase 5 — publish a browsable gallery under ``~/public_html``.

For a finished run this:

* copies ``report.html`` and every config's ``plots/*.png`` into
  ``<gallery-dir>/<run-id>/`` (a flat, web-servable tree),
* regenerates ``<gallery-dir>/index.html`` from the directory listing — all
  runs newest-first, each with per-config figure thumbnails and a link to its
  report,
* updates the ``<gallery-dir>/latest`` symlink to point at the newest run,
* prints the public URL hint.

Files are copied (not symlinked) so the tree is self-contained and works when
served over HTTP from ``~/public_html``.
"""

from __future__ import annotations

import html
import json
import os
import shutil
from pathlib import Path

from .regression_common import (
    CATEGORY_LABELS,
    CATEGORY_ORDER,
    figure_type,
    figure_type_label,
    log,
    time_sort_key,
)

# The gallery root. Overridable via WX_REGRESSION_GALLERY. The
# user symlinks ~/public_html/wx_factory/regression -> this dir to serve it.
DEFAULT_GALLERY_DIR = Path(
    os.environ.get(
        "WX_REGRESSION_GALLERY",
        str(Path.home() / "public_html" / "wx_factory" / "regression"),
    )
)


def _copy_plots(output_root: Path, dest: Path) -> int:
    """Copy report.html and every config's plots into ``dest``. Returns count."""
    count = 0
    report = output_root / "report.html"
    if report.is_file():
        shutil.copy2(report, dest / "report.html")
        count += 1
    manifest = output_root / "manifest.json"
    if manifest.is_file():
        shutil.copy2(manifest, dest / "manifest.json")
        count += 1

    for stem_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        plots = stem_dir / "plots"
        if not plots.is_dir():
            continue
        dest_plots = dest / stem_dir.name / "plots"
        dest_plots.mkdir(parents=True, exist_ok=True)
        for png in sorted(plots.glob("*.png")):
            shutil.copy2(png, dest_plots / png.name)
            count += 1
    return count


def _run_entries(gallery_dir: Path) -> list[Path]:
    """The run directories under the gallery, newest-first (by name).

    A run is listed if it has a ``report.html`` (finished) or an ``.in_progress``
    marker (registered by ``register_run`` but not yet aggregated). Symlinks
    (e.g. the user-managed ``baseline`` pointer) are skipped so they are not
    listed as their own run.
    """
    runs = [
        p
        for p in gallery_dir.iterdir()
        if p.is_dir() and not p.is_symlink() and ((p / "report.html").is_file() or (p / ".in_progress").is_file())
    ]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def _baseline_target(gallery_dir: Path) -> str | None:
    """The run name the ``baseline`` symlink points to (or ``None``)."""
    link = gallery_dir / "baseline"
    if link.is_symlink():
        return link.resolve().name
    return None


# Preferred display order for image-type rows within a config (unknown types
# sort last, alphabetically).
TYPE_ORDER = ["h", "speed", "latlon", "lonheight", "correlation", "cartesian", "figures"]


def _category_from_plots(png_names: list[str]) -> str:
    """Infer a config's display category from its plot filenames.

    The plot scripts use a deterministic prefix per grid/equation set:
    ``sw_`` (shallow water), ``cart_`` (cartesian), ``dcmipNN_`` (DCMIP Euler).
    """
    for name in png_names:
        if name.startswith("sw_"):
            return "shallow_water"
        if name.startswith("cart_"):
            return "cartesian"
        if name.startswith("dcmip"):
            return "dcmip"
    return "dcmip"


def _load_manifest_statuses(run_dir: Path) -> dict[str, dict]:
    """Read manifest.json from the gallery run dir and return {stem: {status, exit_code}}.

    Returns an empty dict if the manifest is missing or unreadable (e.g. an
    in-progress run that has not yet been aggregated).
    """
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        data = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    out: dict[str, dict] = {}
    for cfg in data.get("configs", []):
        stem = cfg.get("stem")
        if stem:
            out[stem] = {"status": cfg.get("status"), "exit_code": cfg.get("exit_code")}
    return out


def _config_thumbnails(run_dir: Path) -> list[tuple[str, str, dict[str, list[str]]]]:
    """Return ``[(stem, category, {type_key: [png names in time order]})]``.

    Figures are grouped by image type (one row per type in the index) and,
    within each type, ordered by simulation time so the middle time sorts
    between the first and the last. Every plot present is shown; the plot phase
    already skips failed runs, so a failed config simply has no figures.
    """
    out = []
    for stem_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        plots = stem_dir / "plots"
        if not plots.is_dir():
            continue
        pngs = sorted((p.name for p in plots.glob("*.png")), key=time_sort_key)
        if not pngs:
            continue
        by_type: dict[str, list[str]] = {}
        for name in pngs:
            by_type.setdefault(figure_type(name), []).append(name)
        out.append((stem_dir.name, _category_from_plots(pngs), by_type))
    return out


def render_index(gallery_dir: Path) -> str:
    """Build the gallery index.html from the directory listing.

    Runs are listed newest-first (symlinks skipped) as collapsible ``<details>``
    accordions; within a run, configs are grouped by category (DCMIP / shallow
    water / cartesian), each category also collapsible. The newest run (and its
    categories) is expanded by default. The run the ``baseline`` symlink points
    to is badged.
    """
    runs = _run_entries(gallery_dir)
    baseline_name = _baseline_target(gallery_dir)
    parts: list[str] = []
    parts.append("<!DOCTYPE html>\n<html><head><meta charset='utf-8'>")
    parts.append("<title>WxFactory regression gallery</title>")
    parts.append(
        "<style>"
        "body{font-family:system-ui,sans-serif;margin:24px;color:#1a1a1a;}"
        "h1{font-size:20px;}"
        "a{color:#1a56db;}"
        "details.run{border:1px solid #ddd;border-radius:8px;margin:16px 0;}"
        "details.run.baseline{border-color:#0a7a2b;}"
        "details.run>summary{cursor:pointer;padding:10px 16px;font-size:16px;font-weight:600;list-style:none;}"
        "details.run>summary::-webkit-details-marker{display:none;}"
        "details.run>summary::before{content:'▸';display:inline-block;margin-right:8px;transition:transform .1s;}"
        "details.run[open]>summary::before{transform:rotate(90deg);}"
        ".runbody{padding:0 16px 12px;}"
        ".reportlink{font-size:12px;font-weight:400;}"
        "details.cat{border-top:1px solid #eee;margin-top:10px;}"
        "details.cat>summary{cursor:pointer;padding:8px 0;font-size:14px;font-weight:600;color:#333;list-style:none;}"
        "details.cat>summary::-webkit-details-marker{display:none;}"
        "details.cat>summary::before{content:'▸';display:inline-block;margin-right:8px;transition:transform .1s;}"
        "details.cat[open]>summary::before{transform:rotate(90deg);}"
        ".catbody{padding:4px 0 8px 16px;}"
        ".meta{color:#555;font-size:12px;}"
        ".config{margin:10px 0;}"
        ".typerow{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px;align-items:flex-start;}"
        ".typerow .tlabel{flex:0 0 130px;font-size:12px;color:#555;padding-top:4px;}"
        "img{max-width:160px;border:1px solid #ccc;}"
        ".badge{background:#0a7a2b;color:#fff;border-radius:4px;padding:1px 6px;font-size:11px;}"
        ".badge.inprogress{background:#b45309;}"
        ".count{color:#555;font-size:12px;font-weight:400;margin-left:8px;}"
        ".inprogress-note{color:#b45309;font-size:13px;font-weight:600;margin:8px 0;}"
        ".failed-section{background:#fef2f2;border:1px solid #fecaca;border-radius:6px;padding:10px 14px;margin:10px 0;}"
        ".failed-section h3{color:#b91c1c;font-size:14px;margin:0 0 6px 0;}"
        ".failed-item{color:#991b1b;font-size:13px;margin:3px 0;}"
        ".badge.failed{background:#b91c1c;}"
        "</style></head><body>"
    )
    parts.append("<h1>WxFactory regression gallery</h1>")
    parts.append(
        f"<p class='meta'>{len(runs)} run(s), newest first. Click a run or a category to expand / collapse it.</p>"
    )

    for ri, run in enumerate(runs):
        is_baseline = run.name == baseline_name
        in_progress = (run / ".in_progress").is_file()
        cls = "run baseline" if is_baseline else "run"
        badge = " <span class='badge'>baseline</span>" if is_baseline else ""
        if in_progress:
            badge += " <span class='badge inprogress'>in progress</span>"

        # Group this run's configs by category (DCMIP / SW / cartesian).
        by_category: dict[str, list[tuple[str, dict[str, list[str]]]]] = {}
        for stem, category, by_type in _config_thumbnails(run):
            by_category.setdefault(category, []).append((stem, by_type))
        n_configs = sum(len(g) for g in by_category.values())
        n_figs = sum(len(names) for g in by_category.values() for _, by_type in g for names in by_type.values())

        # Load manifest statuses to detect failed configs (which have no plots).
        statuses = _load_manifest_statuses(run)
        plotted_stems = {stem for group in by_category.values() for stem, _ in group}
        failed_configs = [
            (stem, info)
            for stem, info in statuses.items()
            if info.get("status") == "failed" and stem not in plotted_stems
        ]
        n_failed = len(failed_configs)
        if n_failed:
            n_configs += n_failed  # include failed configs in the total count

        # The newest run (and its categories) is expanded by default.
        run_open = " open" if ri == 0 else ""
        if n_failed:
            badge += f" <span class='badge failed'>{n_failed} failed</span>"

        parts.append(f"<details class='{cls}'{run_open}")
        parts.append(
            f"<summary>{html.escape(run.name)}{badge}"
            f"<span class='count'>{n_configs} config(s), {n_figs} figure(s)</span></summary>"
        )
        parts.append("<div class='runbody'>")
        if in_progress:
            parts.append(
                "<p class='inprogress-note'>Run in progress &mdash; results will appear here as the "
                "compute jobs finish and the aggregation job publishes the report.</p>"
            )
        else:
            parts.append(
                f"<p class='meta reportlink'><a href='{html.escape(run.name)}/report.html'>full report &rarr;</a></p>"
            )
        if failed_configs:
            parts.append("<div class='failed-section'>")
            parts.append(f"<h3>&#9888; {n_failed} config(s) failed</h3>")
            for stem, info in sorted(failed_configs):
                exit_code = info.get("exit_code")
                detail = f" (exit code {exit_code})" if exit_code is not None else ""
                parts.append(f"<div class='failed-item'>{html.escape(stem)}{detail}</div>")
            parts.append("</div>")
        for category in CATEGORY_ORDER:
            group = by_category.get(category)
            if not group:
                continue
            cat_open = " open" if ri == 0 else ""
            parts.append(f"<details class='cat'{cat_open}>")
            parts.append(
                f"<summary>{html.escape(CATEGORY_LABELS[category])}"
                f"<span class='count'>{len(group)} config(s)</span></summary>"
            )
            parts.append("<div class='catbody'>")
            for stem, by_type in sorted(group):
                parts.append(f"<div class='config'><b>{html.escape(stem)}</b>")
                # One row per image type, in a stable order.
                for type_key in sorted(
                    by_type, key=lambda k: (TYPE_ORDER.index(k) if k in TYPE_ORDER else len(TYPE_ORDER), k)
                ):
                    names = by_type[type_key]
                    parts.append("<div class='typerow'>")
                    parts.append(f"<span class='tlabel'>{html.escape(figure_type_label(type_key))}</span>")
                    for name in names:
                        src = f"{run.name}/{stem}/plots/{name}"
                        parts.append(
                            f"<a href='{html.escape(src)}'><img src='{html.escape(src)}' alt='{html.escape(name)}'></a>"
                        )
                    parts.append("</div>")
                parts.append("</div>")
            parts.append("</div>")  # catbody
            parts.append("</details>")  # cat
        parts.append("</div>")  # runbody
        parts.append("</details>")  # run

    parts.append("</body></html>")
    return "\n".join(parts)


def publish(output_root: Path, gallery_dir: Path | None = None) -> Path:
    """Publish a run to the gallery and refresh the index. Returns the run dir."""
    gallery_dir = gallery_dir or DEFAULT_GALLERY_DIR
    gallery_dir.mkdir(parents=True, exist_ok=True)

    dest = gallery_dir / output_root.name
    dest.mkdir(parents=True, exist_ok=True)
    count = _copy_plots(output_root, dest)
    log(f"Published {count} file(s) to {dest}")

    # The run is finished: drop the in-progress marker (if present) BEFORE
    # regenerating the index, so the index no longer badges it as in progress.
    (dest / ".in_progress").unlink(missing_ok=True)

    # Refresh the index. (The 'baseline' symlink is user-managed and is left
    # untouched here; the index reads it to badge the baseline run.)
    index = gallery_dir / "index.html"
    index.write_text(render_index(gallery_dir))
    log(f"Regenerated {index}")

    log(f"Gallery: {gallery_dir}")
    log(f"Public URL hint: http://<host>/wx_factory/regression/{output_root.name}/report.html")
    return dest


def register_run(run_name: str, gallery_dir: Path | None = None) -> Path:
    """Register a run that has been launched but not yet aggregated.

    Creates ``<gallery-dir>/<run_name>/`` with a stub ``report.html`` and an
    ``.in_progress`` marker, then regenerates ``index.html`` so the run appears
    (empty, badged "in progress"). The aggregation job's ``publish()`` later
    overwrites the stub with the real report/figures and removes the marker.
    """
    gallery_dir = gallery_dir or DEFAULT_GALLERY_DIR
    dest = gallery_dir / run_name
    dest.mkdir(parents=True, exist_ok=True)

    (dest / ".in_progress").write_text("registered\n")
    stub = (
        "<!DOCTYPE html>\n<html><head><meta charset='utf-8'>"
        f"<title>WxFactory regression &mdash; {html.escape(run_name)}</title></head>"
        "<body style='font-family:system-ui,sans-serif;margin:24px;'>"
        f"<h1>Run {html.escape(run_name)} is in progress</h1>"
        "<p>The compute jobs have been submitted. This page will be replaced by the "
        "full report once the aggregation job runs.</p>"
        "<p><a href='../index.html'>&larr; back to the gallery index</a></p>"
        "</body></html>"
    )
    (dest / "report.html").write_text(stub)

    index = gallery_dir / "index.html"
    index.write_text(render_index(gallery_dir))
    log(f"Registered in-progress run {run_name} at {dest}")
    log(f"Regenerated {index}")
    return dest
