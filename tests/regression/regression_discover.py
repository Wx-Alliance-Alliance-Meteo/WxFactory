"""Phase 1 — config discovery and runnability classification.

Every invocation re-scans ``config/*.ini`` (cheap, self-describing, adapts to
new or broken configs). Each config is classified by grid type, equations and
case number, and is either marked *runnable* or *excluded* with a reason.

Exclusion rule (per the plan, simplified at the user's request): a config that
references any external input data file is excluded. That is, any of the fields
``initial_condition`` / ``initial_conditions`` / ``grid_file`` /
``topography_file`` is set to a non-empty value. This catches the ERA5_*
configs (external dataset path), ``ERA5_euler.ini`` (literal ``[era5 file]``
placeholder) and ``rockies.ini`` (missing ``exp_cs.fst``) without any
hard-coded list.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path

from .regression_common import (
    CONFIG_DIR,
    RANKS_BY_GRID,
    data_file_refs,
    log,
    read_config,
)


@dataclass
class ConfigInfo:
    """A single discovered config and how it should be run."""

    stem: str
    path: Path
    grid_type: str
    equations: str
    case_number: str
    base_output_file: str
    output_dir: str
    output_format: str
    ranks: int
    runnable: bool = True
    exclude_reason: str = ""
    # Filled in by later phases:
    run_dir: Path | None = field(default=None)
    status: str = "pending"  # pending | ok | failed | skipped
    exit_code: int | None = None
    walltime: float | None = None

    @property
    def is_cubed_sphere(self) -> bool:
        return self.grid_type == "cubed_sphere"

    @property
    def is_cartesian(self) -> bool:
        return self.grid_type == "cartesian3d"


def _to_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def classify(path: Path) -> ConfigInfo:
    """Read one config file and build its :class:`ConfigInfo`."""
    cfg = read_config(path)
    grid_type = cfg.get("grid_type", "").strip()
    equations = cfg.get("equations", "").strip()
    case_number = cfg.get("case_number", "").strip()
    base_output_file = cfg.get("base_output_file", "out").strip() or "out"
    output_dir = cfg.get("output_dir", "results").strip() or "results"
    output_format = cfg.get("output_format", "netcdf").strip() or "netcdf"

    ranks = RANKS_BY_GRID.get(grid_type, 0)

    info = ConfigInfo(
        stem=path.stem,
        path=path,
        grid_type=grid_type,
        equations=equations,
        case_number=case_number,
        base_output_file=base_output_file,
        output_dir=output_dir,
        output_format=output_format,
        ranks=ranks,
    )

    # --- runnability ------------------------------------------------------- #
    refs = data_file_refs(cfg)
    if refs:
        detail = ", ".join(f"{f}={v}" for f, v in refs)
        info.runnable = False
        info.exclude_reason = f"requires input data file(s): {detail}"
        return info

    if grid_type not in RANKS_BY_GRID:
        info.runnable = False
        info.exclude_reason = f"unknown grid_type '{grid_type}'"
        return info

    return info


def discover(config_dir: Path | None = None, globs: list[str] | None = None) -> list[ConfigInfo]:
    """Discover and classify every config in ``config_dir``.

    ``globs`` (if given) restricts the scan to matching file names, e.g.
    ``["dcmip*", "case*"]``. Returns configs sorted by stem for deterministic
    ordering.
    """
    config_dir = config_dir or CONFIG_DIR
    if not config_dir.is_dir():
        log(f"WARNING: config directory not found: {config_dir}")
        return []

    patterns = globs or ["*.ini"]
    seen: dict[str, ConfigInfo] = {}
    for pattern in patterns:
        for path in sorted(config_dir.glob(pattern)):
            if not path.is_file():
                continue
            if path.stem in seen:
                continue
            try:
                seen[path.stem] = classify(path)
            except (configparser.Error, OSError) as exc:  # a malformed config must not kill discovery
                info = ConfigInfo(
                    stem=path.stem,
                    path=path,
                    grid_type="",
                    equations="",
                    case_number="",
                    base_output_file="out",
                    output_dir="results",
                    output_format="netcdf",
                    ranks=0,
                    runnable=False,
                    exclude_reason=f"failed to parse: {exc}",
                )
                seen[path.stem] = info

    return [seen[k] for k in sorted(seen)]


def split(infos: list[ConfigInfo]) -> tuple[list[ConfigInfo], list[ConfigInfo]]:
    """Split discovered configs into (runnable, excluded)."""
    runnable = [c for c in infos if c.runnable]
    excluded = [c for c in infos if not c.runnable]
    return runnable, excluded


def summarize(infos: list[ConfigInfo]) -> None:
    """Print a human-readable discovery summary."""
    runnable, excluded = split(infos)
    log(f"Discovered {len(infos)} config(s) in {CONFIG_DIR}")
    log(f"  runnable: {len(runnable)}   excluded: {len(excluded)}")

    by_grid: dict[str, int] = {}
    for c in runnable:
        by_grid[c.grid_type] = by_grid.get(c.grid_type, 0) + 1
    for grid, count in sorted(by_grid.items()):
        log(f"    {grid}: {count}")

    if excluded:
        log("  Excluded (with reasons):")
        for c in excluded:
            log(f"    - {c.stem}: {c.exclude_reason}")
