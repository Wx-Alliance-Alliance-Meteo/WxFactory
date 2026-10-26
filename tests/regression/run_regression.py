#!/usr/bin/env python3
"""WxFactory regression suite — main entry point.

Runs every runnable config in ``config/`` SEQUENTIALLY, produces outputs and
plots, compares against a previous run (SSIM + data L2, report-only) and
publishes a browsable gallery under ``~/public_html``.

Usable three ways (same script, same flags):

* standalone:      ``python tests/regression/run_regression.py --dry-run``
* from a job:      ``tests/regression/submit_regression.sh`` submits a set of
  ``regression_case.job`` jobs (one per config group) plus a
  ``regression_aggregate.job`` that writes the shared report / gallery,
* from CI:         the GitLab custom runner invokes the same script.

Phases (each independently skippable):

1. discover  — scan ``config/*.ini``, classify, auto-exclude unrunnable ones.
2. run       — sequential WxFactory runs (config copy + mpirun via run.sh).
3. plot      — dispatch to the DCMIP / SW / cartesian plotters.
4. compare   — SSIM + data L2 vs the baseline → report.json / report.html.
5. gallery   — publish to ``~/public_html`` + regenerate index.html.
6. summary   — per-config status, walltimes, disk usage.

Exit code: 0 when the completed phases are healthy; non-zero only if the run
phase executed and zero configs succeeded.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Allow running both as a script (``python tests/regression/run_regression.py``)
# and as a module (``python -m tests.regression.run_regression``).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tests.regression import (
        regression_common as common,
        regression_compare as compare,
        regression_discover as discover,
        regression_gallery as gallery,
        regression_plots as plots,
        regression_runner as runner,
    )
else:
    from . import (
        regression_common as common,
        regression_compare as compare,
        regression_discover as discover,
        regression_gallery as gallery,
        regression_plots as plots,
        regression_runner as runner,
    )

REGRESSION_ROOT = common.REPO_ROOT / "results" / "regression"


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WxFactory regression suite (run + plot + compare + gallery).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="config file-name globs to include (default: all *.ini in config/)",
    )
    parser.add_argument(
        "--list-runnable",
        action="store_true",
        help=(
            "print the file names of the runnable configs (one per line) and "
            "exit. Used by submit_regression.sh to compute the 'remainder' "
            "group (configs not covered by its hardcoded job groups)."
        ),
    )
    parser.add_argument(
        "--register-run",
        metavar="RUN_NAME",
        default=None,
        help=(
            "register an in-progress run in the gallery (create a stub "
            "report.html + .in_progress marker and refresh index.html), then "
            "exit. Used by submit_regression.sh so a freshly launched run "
            "appears in the gallery before the aggregation job fills it in."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="where to write this run (default: results/regression/<UTC>_<sha7>)",
    )
    parser.add_argument("--skip-run", action="store_true", help="do not run the configs")
    parser.add_argument("--skip-plot", action="store_true", help="do not (re)plot")
    parser.add_argument("--skip-compare", action="store_true", help="do not compare against a baseline")
    parser.add_argument("--skip-gallery", action="store_true", help="do not publish to the gallery")
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help=(
            "do not write manifest.json. Used by parallel category jobs that "
            "share one output root: each writes only its per-config outputs, and "
            "a later aggregation job writes the single manifest."
        ),
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="baseline run dir (default: most recent previous run under results/regression/)",
    )
    parser.add_argument("--ssim-threshold", type=float, default=0.98, help="SSIM pass threshold")
    parser.add_argument(
        "--gallery-dir",
        default=None,
        help="gallery root (default: ~/public_html/wx_factory_regression)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print discovered configs, exclusions and planned commands, then exit",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def find_baseline(output_root: Path) -> Path | None:
    """The baseline run to compare against.

    Driven by the user-managed ``baseline`` symlink in the gallery directory:
    it names the run (e.g. ``20261015T183228Z_4a2981b``) that new runs are
    compared against.

    * symlink -> a different, valid run: that run is the baseline.
    * symlink -> the current run (or a run with no manifest): the current run
      *is* the baseline, so there is nothing to compare against -> ``None``.
    * no symlink at all: fall back to the *oldest* timestamp-named run that has
      a manifest (a stable reference point). Only timestamp-named run dirs are
      considered, so ad-hoc dirs like ``smoke_gpu`` are never picked.

    The regression root is taken as the *parent of the resolved current run*
    (not ``REGRESSION_ROOT``) so the baseline is always a sibling of it on the
    same filesystem. This keeps the report's relative image links
    (``../<baseline>/...``) valid even when ``results`` is a symlink to scratch
    space (a cross-filesystem ``relpath`` would otherwise be broken).
    """
    regression_root = output_root.parent
    if not regression_root.is_dir():
        return None

    def _valid(run: Path) -> bool:
        return run.is_dir() and (run / "manifest.json").is_file() and run.resolve() != output_root.resolve()

    # 1. User-managed baseline symlink in the gallery (names the reference run).
    link = gallery.DEFAULT_GALLERY_DIR / "baseline"
    if link.is_symlink():
        candidate = regression_root / link.resolve().name
        if _valid(candidate):
            return candidate
        # The symlink points at the current run (or an invalid one): this run
        # is the baseline, so there is nothing to compare against.
        return None

    # 2. No symlink: fall back to the oldest timestamp-named run with a manifest.
    ts = re.compile(r"^\d{8}T\d{6}Z")
    runs = [p for p in regression_root.iterdir() if ts.match(p.name) and _valid(p)]
    if not runs:
        return None
    return min(runs, key=lambda p: p.name)


# --------------------------------------------------------------------------- #
# Per-config status + performance tracking
# --------------------------------------------------------------------------- #

PERFORMANCE_LOG = REGRESSION_ROOT / "performance.jsonl"


def _rehydrate_status(c) -> None:
    """Recover a config's status from its ``status.json`` (written by the job
    that ran it). Falls back to ``run.log`` presence, then ``skipped``."""
    status_file = c.run_dir / "status.json"
    if status_file.is_file():
        try:
            data = json.loads(status_file.read_text())
            c.status = data.get("status", "skipped")
            c.exit_code = data.get("exit_code")
            c.walltime = data.get("walltime")
            return
        except (OSError, ValueError):
            pass
    if (c.run_dir / "run.log").is_file():
        c.status = "ok"
    else:
        c.status = "skipped"


def _append_performance(run_id: str, infos) -> None:
    """Append one JSON line per config to the persistent performance log.

    ``performance.jsonl`` is an append-only history (one line per run+config)
    that lets performance changes be tracked across runs, independent of which
    job actually ran each config.
    """
    PERFORMANCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = common.utc_timestamp()
    sha = common.git_sha7()
    with PERFORMANCE_LOG.open("a") as fh:
        for c in infos:
            fh.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "timestamp": ts,
                        "git_sha": sha,
                        "stem": c.stem,
                        "status": c.status,
                        "walltime": c.walltime,
                    }
                )
                + "\n"
            )
    common.log(f"Performance log: {PERFORMANCE_LOG} ({len(infos)} config(s) appended)")


def _prev_walltime(baseline: Path | None, stem: str) -> float | None:
    """The walltime of ``stem`` in the baseline run (from its status.json)."""
    if baseline is None:
        return None
    status_file = baseline / stem / "status.json"
    if not status_file.is_file():
        return None
    try:
        return json.loads(status_file.read_text()).get("walltime")
    except (OSError, ValueError):
        return None


def dir_size(path: Path) -> int:
    """Total size in bytes of all regular files under ``path``."""
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024.0 or unit == "TB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} TB"


def write_manifest(output_root: Path, infos, excluded, args, baseline) -> None:
    """Write a self-describing manifest.json for this run."""
    manifest = {
        "run_id": output_root.name,
        "output_root": str(output_root),
        "git_sha": common.git_sha7(),
        "ssim_threshold": args.ssim_threshold,
        "baseline": str(baseline) if baseline else None,
        "phases": {
            "run": not args.skip_run,
            "plot": not args.skip_plot,
            "compare": not args.skip_compare,
            "gallery": not args.skip_gallery,
        },
        "configs": [
            {
                "stem": c.stem,
                "grid_type": c.grid_type,
                "equations": c.equations,
                "case_number": c.case_number,
                "ranks": c.ranks,
                "base_output_file": c.base_output_file,
                "command": " ".join(runner.build_command(c, Path("config.ini"))),
                "status": c.status,
                "exit_code": c.exit_code,
            }
            for c in infos
        ],
        "excluded": [{"stem": c.stem, "reason": c.exclude_reason} for c in excluded],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    common.log(f"Wrote {output_root / 'manifest.json'}")


# --------------------------------------------------------------------------- #
# Dry run
# --------------------------------------------------------------------------- #


def dry_run(infos, excluded, args, output_root, baseline) -> int:
    common.section("DRY RUN — no configs will be executed")
    discover.summarize(infos + excluded)
    common.log(f"\nPlanned output root: {output_root}")
    common.log(f"Planned baseline:    {baseline or '(auto: most recent previous run)'}")
    common.log(
        f"Phases: run={not args.skip_run} plot={not args.skip_plot} "
        f"compare={not args.skip_compare} gallery={not args.skip_gallery}"
    )
    common.log("\nPlanned commands (sequential):")
    for c in infos:
        cmd = " ".join(runner.build_command(c, Path(f"<output_root>/{c.stem}/config.ini")))
        common.log(f"  {c.stem:24s} {cmd}")
    return 0


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #


def print_summary(infos, output_root: Path, baseline) -> None:
    common.section("SUMMARY")
    ok = [c for c in infos if c.status == "ok"]
    failed = [c for c in infos if c.status == "failed"]
    skipped = [c for c in infos if c.status not in ("ok", "failed")]
    common.log(f"configs: {len(infos)} total | {len(ok)} ok | {len(failed)} failed | {len(skipped)} other")

    common.log(f"\n{'config':24s} {'status':8s} {'walltime':>10s}")
    for c in infos:
        wt = compare._fmt_time(c.walltime)
        common.log(f"{c.stem:24s} {c.status:8s} {wt:>10s}")

    run_bytes = dir_size(output_root)
    total_bytes = dir_size(REGRESSION_ROOT)
    common.log(f"\nthis run disk usage:   {human(run_bytes)}  ({output_root})")
    common.log(f"regression root total: {human(total_bytes)}  ({REGRESSION_ROOT})")
    if baseline:
        common.log(f"baseline:              {baseline}")
    common.log(f"report:                {output_root / 'report.html'}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Register an in-progress run in the gallery and exit (no discovery needed).
    if args.register_run:
        gallery.register_run(args.register_run, Path(args.gallery_dir).expanduser() if args.gallery_dir else None)
        return 0

    # --- Phase 1: discover (always) --------------------------------------- #
    all_infos = discover.discover(globs=args.configs)

    # Machine-readable listing (submit_regression.sh computes the remainder
    # group from it): file names only, no summary noise on stdout.
    if args.list_runnable:
        for c in all_infos:
            if c.runnable:
                print(c.path.name)
        return 0

    common.section("PHASE 1 — DISCOVER")
    discover.summarize(all_infos)
    infos, excluded = discover.split(all_infos)

    # Resolve the output root (needed for dry-run display and baseline).
    if args.output_root:
        output_root = Path(args.output_root).expanduser().resolve()
    else:
        output_root = REGRESSION_ROOT / common.run_id()

    # Resolve the baseline (for display / compare).
    if args.baseline:
        baseline = Path(args.baseline).expanduser().resolve()
    else:
        baseline = find_baseline(output_root)

    if args.dry_run:
        return dry_run(infos, excluded, args, output_root, baseline)

    if not infos:
        common.log("No runnable configs discovered; nothing to do.")
        return 1

    output_root.mkdir(parents=True, exist_ok=True)
    if args.no_manifest:
        common.log("Manifest write skipped (--no-manifest).")
    else:
        write_manifest(output_root, infos, excluded, args, baseline)

    # --- Phase 2: run ------------------------------------------------------ #
    if not args.skip_run:
        common.section(f"PHASE 2 — RUN ({len(infos)} configs, sequential)")
        runner.run_all(infos, output_root)
    else:
        common.section("PHASE 2 — RUN (skipped)")
        # Re-hydrate status from the per-config status.json (written by the
        # category job that actually ran it) so later phases and the summary are
        # meaningful when re-plotting / re-comparing. Falls back to run.log.
        for c in infos:
            c.run_dir = output_root / c.stem
            _rehydrate_status(c)

    # Record per-config walltime in the persistent performance log (one line per
    # run+config) so performance changes can be tracked across runs. Only the job
    # that actually ran the configs appends (the aggregation job skips the run
    # phase and would otherwise duplicate the entries).
    if not args.skip_run:
        _append_performance(output_root.name, infos)

    # --- Phase 3: plot ----------------------------------------------------- #
    if not args.skip_plot:
        common.section(f"PHASE 3 — PLOT ({len(infos)} configs)")
        plots.plot_all(infos)
    else:
        common.section("PHASE 3 — PLOT (skipped)")

    # --- Phase 4: compare -------------------------------------------------- #
    if not args.skip_compare:
        common.section("PHASE 4 — COMPARE")
        if baseline is None:
            common.log("No baseline available (first run); skipping comparison.")
        else:
            common.log(f"Baseline: {baseline}")
            walltime_delta = {c.stem: (c.walltime, _prev_walltime(baseline, c.stem)) for c in infos}
            compare.compare_all(infos, output_root, baseline, args.ssim_threshold, walltime_delta)
    else:
        common.section("PHASE 4 — COMPARE (skipped)")

    # --- Phase 5: gallery -------------------------------------------------- #
    if not args.skip_gallery:
        common.section("PHASE 5 — GALLERY")
        gallery_dir = Path(args.gallery_dir).expanduser() if args.gallery_dir else gallery.DEFAULT_GALLERY_DIR
        gallery.publish(output_root, gallery_dir)
    else:
        common.section("PHASE 5 — GALLERY (skipped)")

    # --- Phase 6: summary -------------------------------------------------- #
    print_summary(infos, output_root, baseline)

    # --- exit code --------------------------------------------------------- #
    if not args.skip_run:
        succeeded = sum(1 for c in infos if c.status == "ok")
        if succeeded == 0:
            common.log("ERROR: zero configs succeeded.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
