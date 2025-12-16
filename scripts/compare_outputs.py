#!/usr/bin/env python3

import os
import sys
import argparse
from types import SimpleNamespace


root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
src_dir = os.path.join(root_dir, "wx_factory")
sys.path.append(root_dir)
sys.path.append(src_dir)

import numpy as np
import netCDF4 as nc
from numpy.typing import NDArray

from common.layout_conversion import sv_to_netcdf
from scipy.interpolate import RegularGridInterpolator

from typing import Dict, Optional, Tuple, Iterable, List

INTERP_METHODS = ["linear", "cubic", "quintic"]
interp_results: Dict[str, Dict[str, float]] = {}
interp_errors: Dict[str, str] = {}

# Equivalent to finite elements with x points
def get_interpolator(dest: NDArray, interp_method: str) -> Optional[RegularGridInterpolator]:

    nz, ny, nx = dest.shape[0], dest.shape[1], dest.shape[2]
    z, y, x = np.linspace(0, 1, nz), np.linspace(0, 1, ny), np.linspace(0, 1, nx)

    try:
        interp = RegularGridInterpolator((z, y, x), dest, method=interp_method)
        return interp
    except Exception as e:
        return none

    return interp


def evaluate_without_interpolation(
    grid1: NDArray,
    grid2: NDArray,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:

    results: Dict[str, Dict[str, float]] = {}
    errors: Dict[str, str] = {}

    try:
        err_abs, err_rel = get_error(grid1, grid2)
        sp_err_abs, sp_err_rel = spectral_error(grid1, grid2)
        rmse_val, nrmse_val = get_error_rmse(grid1, grid2)

        results["direct"] = {
            "rmse": float(rmse_val),
            "nrmse": float(nrmse_val),
            "sp_err_abs": float(sp_err_abs),
            "sp_err_rel": float(sp_err_rel),
            "err_abs": float(err_abs),
            "err_rel": float(err_rel),
        }
    except Exception as e:
        errors["direct"] = f"Failed direct metrics: {e}"

    return results, errors


def interpolate_and_metrics(source: NDArray, target: NDArray) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    
    results: Dict[str, Dict[str, float]] = {}
    errors: Dict[str, str] = {}

    for method in INTERP_METHODS:
        print(f"Method: {method}")
        interpolator = None
        try:
            interpolator = get_interpolator(source, method)
            if interpolator is None:
                raise RuntimeError(f"Interpolator failed for method '{method}'")
        except Exception as e:
            errors[method] = f"Failed to create interpolator ({method}): {e}"
            continue
        
        try:
            projected = project(target, interpolator)
        except Exception as e:
            errors[method] = f"Failed during projection ({method}): {e}"
            continue

        try:
            err_abs, err_rel = get_error(target, projected)
            sp_err_abs, sp_err_rel = spectral_error(target, projected)
            rmse_val, nrmse_val = get_error_rmse(target, projected)

            results[method] = {
                "rmse": float(rmse_val),
                "nrmse": float(nrmse_val),
                "sp_err_abs": float(sp_err_abs),
                "sp_err_rel": float(sp_err_rel),
                "err_abs": float(err_abs),
                "err_rel": float(err_rel),
            }
        except Exception as e:
            errors[method] = f"Failed computing metrics ({method}): {e}"
            continue

    return results, errors


def build_report_text(
    variable_label: str,
    panel_index: int | None,
    results: Dict[str, Dict[str, float]],
    errors: Dict[str, str],
) -> str:
    lines = list(iter_report_lines(variable_label, results, errors))
    return "\n".join(lines)


def iter_report_lines(
    variable_label: str,
    results_by_method: Dict[str, List[Dict[str, float]]],
    errors_by_method: Dict[str, List[str]],
) -> Iterable[str]:
    """
    Yield report lines aggregated (mean/max) across panels, separated per method.
    - results_by_method: method -> list of per-panel metric dicts
    - errors_by_method:  method -> list of error messages (strings)
    """

    def safe_mean(vals: List[float]) -> float:
        return float(np.mean(vals)) if len(vals) > 0 else float("nan")

    def safe_max(vals: List[float]) -> float:
        return float(np.max(vals)) if len(vals) > 0 else float("nan")

    # Header
    yield "-------------"
    yield f"Report for {variable_label}"

    # Method order: 'direct' first (if present), then interpolation methods
    method_order: List[str] = []
    if "direct" in results_by_method or "direct" in errors_by_method:
        method_order.append("direct")
    method_order.extend(INTERP_METHODS)

    for method in method_order:
        panels = results_by_method.get(method, [])

        nrmse_vals  = [rp["nrmse"]     for rp in panels if "nrmse"     in rp]
        sp_rel_vals = [rp["sp_err_rel"]for rp in panels if "sp_err_rel" in rp]
        err_rel_vals= [rp["err_rel"]   for rp in panels if "err_rel"    in rp]
        rmse_vals   = [rp["rmse"]      for rp in panels if "rmse"       in rp]
        sp_abs_vals = [rp["sp_err_abs"]for rp in panels if "sp_err_abs" in rp]
        err_abs_vals= [rp["err_abs"]   for rp in panels if "err_abs"    in rp]

        yield ""
        yield ("Method: direct (same-grid)" if method == "direct" else f"Method: {method}")

        if len(panels) == 0:
            yield "  No metrics (all panels failed or none applicable)."
        else:
            yield f"  NRMSE (mean across panels): {safe_mean(nrmse_vals):.6g}"
            yield f"  NRMSE (max  across panels): {safe_max(nrmse_vals):.6g}"
            yield f"  Spectral relative error (mean): {safe_mean(sp_rel_vals):.6g}"
            yield f"  Spectral relative error (max):  {safe_max(sp_rel_vals):.6g}"
            yield f"  Relative L2 error (mean): {safe_mean(err_rel_vals):.6g}"
            yield f"  Relative L2 error (max):  {safe_max(err_rel_vals):.6g}"

        for msg in errors_by_method.get(method, []):
            yield f"  Note: {msg}"





def project(data: NDArray, interpolator: RegularGridInterpolator):
    nz, ny, nx = data.shape

    z, y, x = np.linspace(0, 1, nz), np.linspace(0, 1, ny), np.linspace(0, 1, nx)
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

    interp_points = np.column_stack((Z.ravel(), Y.ravel(), X.ravel()))

    interp_vals = interpolator(interp_points)
    data_proj = interp_vals.reshape(nz, ny, nx)

    return data_proj


# L2 error
# Note that relative error scales badly with grid size
def get_error(grid1: NDArray, grid2: NDArray):
    err = np.abs(grid1 - grid2)
    err_l2 = np.linalg.norm(err)
    err_l2_rel = err_l2 / np.linalg.norm(grid1)

    return err_l2, err_l2_rel


# Root mean square error normalized with range normalization
def get_error_rmse(grid1: NDArray, grid2: NDArray):
    rmse = np.sqrt(np.mean((grid1 - grid2) ** 2))

    range = np.max(grid1) - np.min(grid1)
    nrmse = rmse / range
    return rmse, nrmse


# Choose variable, time and panel for the grid size
def process_netcdf(data1: NDArray, data2: NDArray, variable: str, panel: int, time_index: int):
    data1_var = data1[variable]
    data2_var = data2[variable]

    data1_ready = data1_var[time_index, panel, ...]
    data2_ready = data2_var[time_index, panel, ...]
    return data1_ready, data2_ready


# Choose variable and panel, in case of state vector modified to netcdf format
def process_reshaped_sv(data1: NDArray, data2: NDArray, variable_index: int, panel: int):
    data1_var = data1[panel, variable_index, ...]
    data2_var = data2[panel, variable_index, ...]
    return data1_var, data2_var


# Nrmse of spectral error
# Finds the error in frequency distribution
# In theory, it is impervious to shifts
def spectral_error(grid1: NDArray, grid2: NDArray):
    fft1 = np.abs(np.fft.fftn(grid1))
    fft2 = np.abs(np.fft.fftn(grid2))

    err = np.sqrt(np.mean((fft1 - fft2) ** 2))
    rms_base = np.sqrt(np.mean(fft1**2))
    err_rel = err / (rms_base)
    return err, err_rel


def main(args):
    
    
    all_reports_text: list[str] = []

    # Load data
    data1 = None
    data2 = None
    if args.input_type == "netcdf":
        data1 = nc.Dataset(args.data_file_1, "r")
        data2 = nc.Dataset(args.data_file_2, "r")
    elif args.input_type == "sv":
        # state vector
        # shape (panels, variables, elevs, verticals, horizontal, num_points * num_points)
        # variables: rho, u1_contra, u2_contra, w, potential_temperature -> defined in initialize.py according to config
        data1 = sv_to_netcdf(args.data_file_1)
        data2 = sv_to_netcdf(args.data_file_2)
    else:
        raise ValueError(f"Unsupported input type '{args.input_type}'")

    # Variable loop
    vars = args.vars

    # Iterate through each data variable such as rho, P, theta
    for var_index in range(len(vars)):


        results_by_method: Dict[str, List[Dict[str, float]]] = {
            "direct": [],
            "linear": [],
            "cubic": [],
            "quintic": [],
        }
        errors_by_method: Dict[str, List[str]] = {
            "direct": [],
            "linear": [],
            "cubic": [],
            "quintic": [],
        }


        variable_label = (
            str(vars[var_index]) if isinstance(vars[var_index], str) else f"var_idx_{var_index}"
        )

        # Iterate through each panels
        for p in range(0, 5):

            if args.input_type == "netcdf":
                data1_var, data2_var = process_netcdf(data1, data2, vars[var_index], p, args.time_index)
            elif args.input_type == "sv":
                data1_var, data2_var = process_reshaped_sv(data1, data2, var_index, p)

            # Interpolating one of the grids since different sizes
            if data1_var.shape != data2_var.shape:

                src_is_finer = np.prod(data1_var.shape) > np.prod(data2_var.shape)
                src = data1_var if src_is_finer else data2_var
                tgt = data2_var if src_is_finer else data1_var

                # Run interpolation and metrics with different interpolation methods
                method_results, method_errors = interpolate_and_metrics(src, tgt)

                for method in INTERP_METHODS:
                    if method in method_results:
                        results_by_method[method].append(method_results[method])
                    if method in method_errors:
                        errors_by_method[method].append(f"Panel {p}: {errors[method]}")


            # Same grid, we take l2 error and l2 spectral error
            else:
                results, errors = evaluate_without_interpolation(data1_var, data2_var)
                if "direct" in results:
                    results_by_method["direct"].append(results["direct"])
                if "direct" in errors:
                    errors_by_method["direct"].append(f"Panel {p}: {errors['direct']}")
        


        # Report stream
        report_text = build_report_text(variable_label, var_index, results_by_method, errors_by_method)
        for line in report_text.splitlines():
            print(line)
        all_reports_text.append(report_text)

    if args.save_path:
        try:
            with open(args.save_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(all_reports_text))
            print(f"\nSaved report to: {args.save_path}")
        except Exception as e:
            print(f"\nFailed to save report to {args.save_path}: {e}")
            
    return all_reports_text

    # Small wrapper to run from python
def run (
    data_file_1: str,
    data_file_2: str,
    *,
    input_type: str = "netcdf",
    time_index: int = -1,
    save_path: str = "",
    vars: list[str] = None,
) -> str:
    args = SimpleNamespace(
        data_file_1=data_file_1,
        data_file_2=data_file_2,
        input_type=input_type,
        time_index=time_index,
        save_path=save_path,
        vars=vars or ["P", "rho", "theta"],
    )
    return main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Prints error and spectral error between two datas.
    """
    )
    parser.add_argument("data_file_1", type=str, help="Path to the first output file netcdf")
    parser.add_argument("data_file_2", type=str, help="Path to the second output file netcdf")

    # Optionals
    parser.add_argument("--input_type", default="netcdf", type=str, help="netcdf or sv")
    parser.add_argument("--time_index", default=-1, type=int, help="Time step to compare. Defaults at last.")
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument(
        "--vars",
        nargs="+",
        type=str,
        default=["P", "rho", "theta"],
        help="""List of variables to estimate the error with. e.g. --vars P rho theta
        Options: P, rho, theta, U, V, W""",
    )

    # Run
    main(parser.parse_args())