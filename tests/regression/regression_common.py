"""Shared helpers for the WxFactory regression suite.

This module holds the small, cross-cutting utilities used by the discovery,
runner, plotting, comparison and gallery phases:

* locating the repository root,
* reading / rewriting the ``output_dir`` of a config file (without mutating the
  repo copy),
* a few path / timestamp / git helpers,
* a thin command runner that captures stdout+stderr to a log file.

It intentionally has no dependency on the model itself so that it can be
imported cheaply from any phase (including ``--dry-run``).
"""

from __future__ import annotations

import configparser
import datetime
import os
import re
import subprocess
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

# tests/regression/regression_common.py -> parents[2] is the repository root
# (the directory that contains ``WxFactory``, ``config/`` and ``scripts/``).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
SCRIPTS_DIR = REPO_ROOT / "scripts"
WxFACTORY = REPO_ROOT / "WxFactory"
RUN_SH = SCRIPTS_DIR / "run.sh"

# Config fields that point at external input data files. A config that sets any
# of these to a non-empty value "requires input data files" and is therefore
# excluded from the regression run (see regression_discover).
DATA_FILE_FIELDS = ("initial_condition", "initial_conditions", "grid_file", "topography_file")

# Number of MPI ranks per grid type. Cubed-sphere needs one rank per face
# (multiple of 6, minimum 6); cartesian runs as a single process.
RANKS_BY_GRID = {"cubed_sphere": 6, "cartesian3d": 1}


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


def log(message: str) -> None:
    """Print a timestamped line to stdout (captured by the PBS job log)."""
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def section(title: str) -> None:
    """Print a phase header."""
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}", flush=True)


# --------------------------------------------------------------------------- #
# Config parsing / rewriting
# --------------------------------------------------------------------------- #


def read_config(path: Path) -> dict:
    """Parse an INI config into a flat ``{option: value}`` dict.

    Interpolation is disabled so that values containing ``%`` are read verbatim,
    and keys are lower-cased (matching how WxFactory's own ``ConfigParser``
    reads them). Only the raw string values are kept; no evaluation of
    expressions such as ``86400 / 900`` is performed here.
    """
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(path)
    flat: dict[str, str] = {}
    for section in parser.sections():
        for key, value in parser.items(section):
            flat[key] = value
    return flat


def data_file_refs(cfg: dict) -> list[tuple[str, str]]:
    """Return ``(field, value)`` for every data-file field that is set.

    An empty list means the config is self-contained (no external input data).
    """
    refs = []
    for field in DATA_FILE_FIELDS:
        value = cfg.get(field, "").strip()
        if value:
            refs.append((field, value))
    return refs


def override_output_dir(text: str, new_dir: str) -> str:
    """Return ``text`` with its ``output_dir`` set to ``new_dir``.

    This is a text-level edit so that comments, section order and any
    expression-valued options (e.g. ``output_freq = 86400 / 900``) are preserved
    exactly. If no ``output_dir`` line exists, one is inserted into the
    ``[Output_options]`` section (or a new section is appended as a fallback).
    """
    pattern = re.compile(r"(?im)^[ \t]*output_dir[ \t]*=.*$")
    if pattern.search(text):
        return pattern.sub(f"output_dir = {new_dir}", text, count=1)

    section_match = re.search(r"(?m)^[ \t]*\[Output_options\][ \t]*$", text)
    if section_match:
        insert_at = section_match.end()
        return text[:insert_at] + f"\noutput_dir = {new_dir}" + text[insert_at:]

    return text.rstrip() + f"\n\n[Output_options]\noutput_dir = {new_dir}\n"


def write_config_copy(source: Path, dest: Path, output_dir: str) -> Path:
    """Write a copy of ``source`` to ``dest`` with ``output_dir`` overridden.

    The repository config is never modified.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = source.read_text()
    dest.write_text(override_output_dir(text, output_dir))
    return dest


# --------------------------------------------------------------------------- #
# Timestamps / git
# --------------------------------------------------------------------------- #


def utc_timestamp() -> str:
    """A filesystem-safe UTC timestamp, e.g. ``20261015T153000Z``."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_sha7() -> str:
    """The short (7-char) git SHA of HEAD, or ``nogit`` outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        sha = out.stdout.strip()
        return sha if sha else "nogit"
    except Exception:
        return "nogit"


def run_id() -> str:
    """A unique, sortable identifier for a regression run."""
    return f"{utc_timestamp()}_{git_sha7()}"


# --------------------------------------------------------------------------- #
# Figure naming / ordering / grouping
#
# Shallow-water and cartesian figures carry a *type* token and a *time* token
# as the last two underscore-separated parts of the filename:
#
#   SW:         sw_{stem}_{h|speed}_{t0|tmid|tlast}.png
#   cartesian:  cart_{stem}_{t0|tmid|tlast}.png   (single image type)
#
# DCMIP figures use a heterogeneous naming scheme (several plot scripts,
# different token layouts), so they are NOT split by type or sorted by time —
# they are shown as a single "figures" row in as-generated (alphabetical)
# order.
#
# These helpers let the report and the gallery order figures by simulation time
# (so the middle time sorts between the first and the last) and group them by
# image type, without renaming the files on disk.
# --------------------------------------------------------------------------- #

# Friendly labels for each image type (one row per type in the gallery).
FIGURE_TYPE_LABELS = {
    "h": "h",
    "speed": "wind speed",
    "latlon": "lat-lon",
    "lonheight": "lon-height",
    "correlation": "correlation",
    "cartesian": "time series",
    "figures": "figures",
}

# Display categories and their order in the report / gallery.
CATEGORY_LABELS = {
    "dcmip": "DCMIP (Euler)",
    "shallow_water": "Shallow water",
    "cartesian": "Cartesian",
}
CATEGORY_ORDER = ["dcmip", "shallow_water", "cartesian"]


def category_of(grid_type: str, equations: str) -> str:
    """The display category for a config: ``dcmip``, ``shallow_water`` or
    ``cartesian`` (used to group configs in the report and gallery)."""
    if grid_type == "cartesian3d":
        return "cartesian"
    if equations == "shallow_water":
        return "shallow_water"
    return "dcmip"


def figure_type(filename: str) -> str:
    """The image *type* key of a figure filename.

    * cartesian figures (``cart_*``) have a single type: ``cartesian``.
    * DCMIP figures (``dcmip*``) are not split by type: they all map to
      ``figures`` (shown as one row, in as-generated order).
    * shallow-water figures (``sw_*``) carry the type as the second-to-last
      underscore-separated token (the last is the time token): ``h`` or
      ``speed``.
    """
    name = Path(filename).name
    if name.startswith("cart_"):
        return "cartesian"
    if name.startswith("dcmip"):
        return "figures"
    parts = Path(filename).stem.split("_")
    if len(parts) >= 2:
        return parts[-2]
    return "other"


def figure_type_label(type_key: str) -> str:
    """A friendly label for an image-type key (falls back to the key itself)."""
    return FIGURE_TYPE_LABELS.get(type_key, type_key)


def time_sort_key(filename: str) -> tuple:
    """A sort key that orders figure filenames by simulation time.

    Only shallow-water / cartesian figures carry a ``t0`` / ``tmid`` / ``tlast``
    time token, and this orders them so the middle time sorts between the first
    and the last. DCMIP figures (and anything without a recognised time token)
    sort by their raw name, i.e. as generated (alphabetical).
    """
    token = Path(filename).stem.rsplit("_", 1)[-1]
    order = {"t0": 0, "tmid": 1, "tlast": 2}
    if token in order:
        return (0, order[token], "")
    return (1, 0, Path(filename).name)


# --------------------------------------------------------------------------- #
# Command execution
# --------------------------------------------------------------------------- #


def run_command(
    cmd: list[str],
    cwd: Path,
    log_file: Path,
    env: dict | None = None,
) -> tuple[int, float]:
    """Run ``cmd``, streaming stdout+stderr to ``log_file``.

    Returns ``(exit_code, walltime_seconds)``. The process is run with the
    current environment (optionally extended by ``env``) so that the activated
    venv / module environment is inherited.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    full_env = dict(os.environ)
    if env:
        full_env.update(env)

    start = time.time()
    with open(log_file, "w") as handle:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=full_env,
        )
    walltime = time.time() - start
    return proc.returncode, walltime
