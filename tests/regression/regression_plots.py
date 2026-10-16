"""Phase 3 — plot dispatch.

Turns each run's output into a set of figures with *stable* names (label =
config stem, fixed times) so that the comparison phase can match them against
the baseline by filename.

Dispatch (keyed by grid type / equations / config stem):

* cartesian  → no post-run plotting; the model already wrote
  ``euler_cartesian_{case}_{step:08d}.png`` into the run dir. We copy the
  first / middle / last of them into the plots dir under stable names.
* shallow_water → the new generic ``tests/regression/plot_sw.py``.
* euler (DCMIP) → the existing ``scripts/plot_dcmip*.py`` scripts, chosen by
  the config-stem prefix.

All figures land in ``<output_root>/<stem>/plots/``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from .regression_common import SCRIPTS_DIR, log, run_command
from .regression_discover import ConfigInfo

# DCMIP euler plot scripts, chosen by config-stem prefix (most specific first).
DCMIP_PLOT_SCRIPTS = [
    ("dcmip11", "plot_dcmip11.py"),
    ("dcmip12", "plot_dcmip12.py"),
    ("dcmip13", "plot_dcmip13.py"),
    ("dcmip20", "plot_dcmip20.py"),
    ("dcmip21", "plot_dcmip21.py"),
    ("dcmip31", "plot_dcmip31.py"),
    ("dcmip41", "plot_dcmip41.py"),
]

PLOT_SW = Path(__file__).resolve().parent / "plot_sw.py"


def netcdf_path(info: ConfigInfo) -> Path:
    """The NetCDF output file for a cubed-sphere run."""
    return info.run_dir / f"{info.base_output_file}.nc"


def plots_dir(info: ConfigInfo) -> Path:
    return info.run_dir / "plots"


def _pick_times(paths: list[Path]) -> list[Path]:
    """Return the first, middle and last of a sorted list of paths."""
    if not paths:
        return []
    if len(paths) == 1:
        return paths
    return [paths[0], paths[len(paths) // 2], paths[-1]]


def _collect_cartesian(info: ConfigInfo) -> list[Path]:
    """Copy the first/mid/last in-run PNGs into the plots dir with stable names."""
    out = plots_dir(info)
    out.mkdir(parents=True, exist_ok=True)
    pngs = sorted(info.run_dir.glob("euler_cartesian_*.png"))
    if not pngs:
        log(f"  [{info.stem}] WARNING: no euler_cartesian_*.png found in {info.run_dir}")
        return []

    written = []
    for src, tlabel in zip(_pick_times(pngs), ("t0", "tmid", "tlast")):
        dest = out / f"cart_{info.stem}_{tlabel}.png"
        shutil.copy2(src, dest)
        written.append(dest)
    for p in written:
        log(f"  [{info.stem}] collected {p.name}")
    return written


def _run_plot_script(info: ConfigInfo, script: Path, nc: Path) -> list[Path]:
    """Invoke a plot script (plot_dcmip*.py or plot_sw.py) for one config."""
    out = plots_dir(info)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script),
        str(nc),
        "-o",
        str(out),
        "--label",
        info.stem,
    ]
    log(f"  [{info.stem}] plotting: {' '.join(cmd)}")
    exit_code, _ = run_command(cmd, cwd=script.parent, log_file=out / "plot.log")
    if exit_code != 0:
        log(f"  [{info.stem}] WARNING: plot script exited {exit_code} (see {out / 'plot.log'})")
    return sorted(out.glob("*.png"))


def plot_one(info: ConfigInfo) -> list[Path]:
    """Produce the figures for one config. Returns the list of PNG paths."""
    if info.is_cartesian:
        return _collect_cartesian(info)

    nc = netcdf_path(info)
    if not nc.is_file():
        log(f"  [{info.stem}] WARNING: no NetCDF output at {nc}; skipping plots")
        return []

    if info.equations == "shallow_water":
        return _run_plot_script(info, PLOT_SW, nc)

    if info.equations == "euler":
        for prefix, script_name in DCMIP_PLOT_SCRIPTS:
            if info.stem.startswith(prefix):
                return _run_plot_script(info, SCRIPTS_DIR / script_name, nc)
        log(f"  [{info.stem}] WARNING: no DCMIP plot script for stem '{info.stem}'")
        return []

    log(f"  [{info.stem}] WARNING: unknown equations '{info.equations}'; no plots")
    return []


def plot_all(infos: list[ConfigInfo]) -> dict[str, list[Path]]:
    """Plot every config. Returns ``{stem: [png paths]}``.

    Configs that did not run successfully (``status != "ok"``) are skipped.
    """
    results: dict[str, list[Path]] = {}
    for info in infos:
        if info.status != "ok":
            log(f"--- plot: {info.stem} --- (skipped, status={info.status})")
            results[info.stem] = []
            continue
        log(f"--- plot: {info.stem} ---")
        results[info.stem] = plot_one(info)
    return results
