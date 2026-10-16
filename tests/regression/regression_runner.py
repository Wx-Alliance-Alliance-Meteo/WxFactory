"""Phase 2 — run each config sequentially.

Runs are strictly sequential: the configs share the same GPUs, so they must not
be launched concurrently. For each config we:

1. write a *copy* of the config into ``<output_root>/<stem>/config.ini`` with
   ``output_dir`` overridden to ``<output_root>/<stem>/`` (the repo config is
   never mutated — this is the pattern from ``run_both.sh``),
2. launch WxFactory:
   * cubed-sphere → ``./scripts/run.sh --bind-to none -n 6 ./WxFactory <copy>``
     (``run.sh`` re-exports the environment to the remote ranks, which is what
     makes the multi-node job work),
   * cartesian    → ``./WxFactory <copy>`` (single process, no mpirun),
3. capture stdout+stderr to ``<output_root>/<stem>/run.log`` and record the
   exit code and walltime.

A failed config is recorded and the batch continues with the next one.
"""

from __future__ import annotations

import json
from pathlib import Path

from .regression_common import (
    REPO_ROOT,
    RUN_SH,
    WxFACTORY,
    log,
    run_command,
    write_config_copy,
)
from .regression_discover import ConfigInfo


def build_command(info: ConfigInfo, config_copy: Path) -> list[str]:
    """Return the argv used to launch one config."""
    if info.is_cubed_sphere:
        return [
            str(RUN_SH),
            "--bind-to",
            "none",
            "-n",
            str(info.ranks),
            str(WxFACTORY),
            str(config_copy),
        ]
    # cartesian (or any single-process grid): run directly, no mpirun.
    return [str(WxFACTORY), str(config_copy)]


def run_one(info: ConfigInfo, output_root: Path) -> ConfigInfo:
    """Run a single config and update its status in place."""
    run_dir = output_root / info.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    info.run_dir = run_dir

    config_copy = write_config_copy(info.path, run_dir / "config.ini", str(run_dir))
    cmd = build_command(info, config_copy)
    log_file = run_dir / "run.log"

    log(f"  [{info.stem}] launching: {' '.join(cmd)}")
    exit_code, walltime = run_command(cmd, cwd=REPO_ROOT, log_file=log_file)

    info.exit_code = exit_code
    info.walltime = walltime
    info.status = "ok" if exit_code == 0 else "failed"
    log(f"  [{info.stem}] {info.status} (exit {exit_code}, {walltime / 60.0:.1f} min) — log: {log_file}")
    _write_status(info, run_dir)
    return info


def _write_status(info: ConfigInfo, run_dir: Path) -> None:
    """Persist a per-config status file.

    Written to ``<run_dir>/status.json`` so that a later *aggregation* job
    (which re-discovers the configs and re-hydrates ``run_dir``) can recover the
    exact status / exit code / walltime of each config, even though the config
    was actually run by a different (category) job.
    """
    status = {
        "stem": info.stem,
        "status": info.status,
        "exit_code": info.exit_code,
        "walltime": info.walltime,
    }
    (run_dir / "status.json").write_text(json.dumps(status, indent=2))


def run_all(infos: list[ConfigInfo], output_root: Path) -> list[ConfigInfo]:
    """Run every config sequentially and return the updated list."""
    output_root.mkdir(parents=True, exist_ok=True)
    for i, info in enumerate(infos, start=1):
        log(f"--- run {i}/{len(infos)}: {info.stem} ({info.grid_type}, {info.equations}, case {info.case_number}) ---")
        run_one(info, output_root)
    return infos
