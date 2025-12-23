#!/usr/bin/env python3

import os
import sys
import argparse
from dataclasses import dataclass

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
from types import SimpleNamespace
from typing import List


from output.state import load_state


SUPPORTED_TYPES = [".npy", ".nc"]
# INTERP_METHODS = ["linear", "cubic", "quintic"]
INTERP_METHODS = ["cubic"]

interp_results: Dict[str, Dict[str, float]] = {}
interp_errors: Dict[str, str] = {}

@dataclass
class MetricRecord:
    variable: str
    panel: int
    method: str # direct, linear, cubic, quintic, todo - lagrange
    rmse: float
    nrmse: float
    err_l2: float
    err_l2_rel: float
    sp_err_abs: float
    sp_err_rel: float  

# TODO: replace with know lagrange polynomials
def get_interpolator(dest: NDArray, interp_method: str) -> Optional[RegularGridInterpolator]:
    if (dest.ndim == 3):
        nz, ny, nx = dest.shape[0], dest.shape[1], dest.shape[2]
        z, y, x = np.linspace(0, 1, nz), np.linspace(0, 1, ny), np.linspace(0, 1, nx)
        try:
            interp = RegularGridInterpolator((z, y, x), dest, method=interp_method)
            return interp
        except Exception as e:
            return none
    elif (dest.ndim == 4):
        nz, ny, nx, nw = dest.shape
        z, y, x, w = np.linspace(0, 1, nz), np.linspace(0, 1, ny), np.linspace(0, 1, nx), np.linspace(0, 1, nw)
        try:
            interp = RegularGridInterpolator((z, y, x, w), dest, method=interp_method)
            return interp
        except Exception as e:
            print("Failed interpolation")
            return none
    else:
        return ValueError(f"Interpolation with {dest.ndim} dims not supported" )

    return interp


def evaluate_without_interpolation(
    grid1: NDArray,
    grid2: NDArray,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:

    results: Dict[str, Dict[str, float]] = {}
    errors: Dict[str, str] = {}

    err_abs, err_rel = get_error(grid1, grid2)
    sp_err_abs, sp_err_rel = spectral_error(grid1, grid2)
    rmse_val, nrmse_val = get_error_rmse(grid1, grid2)

    return {
        "direct": {
            "rmse": float(rmse_val),
            "nrmse": float(nrmse_val),
            "err_l2": float(err_abs),
            "err_l2_rel": float(err_rel),
            "sp_err_abs": float(sp_err_abs),
            "sp_err_rel": float(sp_err_rel),
        }
    }

def interpolate_and_metrics(source: NDArray, target: NDArray) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    
    results: Dict[str, Dict[str, float]] = {}
    errors: Dict[str, str] = {}

    for method in INTERP_METHODS:
        interpolator = None
        try:
            interpolator = get_interpolator(source, method)
            if interpolator is None:
                raise RuntimeError(f"Interpolator failed for method '{method}'")
        except Exception as e:
            errors[method] = f"Failed to create interpolator ({method}): {e}"
            print("interp failed")
            continue
        
        try:
            projected = project(target, interpolator)
        except Exception as e:
            errors[method] = f"Failed during projection ({method}): {e}"
            print("projected failed")
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
    - errors_by_method: method -> list of error messages (strings)
    """

    def safe_mean(vals: List[float]) -> float:
        return float(np.mean(vals)) if len(vals) > 0 else float("nan")

    def safe_max(vals: List[float]) -> float:
        return float(np.max(vals)) if len(vals) > 0 else float("nan")

    yield "-------------"
    yield f"Report for {variable_label}"

    method_order: List[str] = []
    if "direct" in results_by_method or "direct" in errors_by_method:
        method_order.append("direct")
    method_order.extend(INTERP_METHODS)

    for method in method_order:
        panels = results_by_method.get(method, [])

        nrmse_vals  = [rp["nrmse"] for rp in panels if "nrmse" in rp]
        sp_rel_vals = [rp["sp_err_rel"] for rp in panels if "sp_err_rel" in rp]
        err_rel_vals= [rp["err_rel"] for rp in panels if "err_rel" in rp]
        rmse_vals   = [rp["rmse"] for rp in panels if "rmse" in rp]
        sp_abs_vals = [rp["sp_err_abs"]for rp in panels if "sp_err_abs" in rp]
        err_abs_vals= [rp["err_abs"] for rp in panels if "err_abs" in rp]

        yield ""
        yield ("Method: direct (same-grid)" if method == "direct" else f"Method: {method}")

        if len(panels) == 0:
            yield "  No metrics (all panels failed or none applicable)."
        else:
            yield f" NRMSE (mean across panels): {safe_mean(nrmse_vals):.6g}"
            yield f" NRMSE (max  across panels): {safe_max(nrmse_vals):.6g}"
            yield f" Spectral relative error (mean): {safe_mean(sp_rel_vals):.6g}"
            yield f" Spectral relative error (max): {safe_max(sp_rel_vals):.6g}"
            yield f" Relative L2 error (mean): {safe_mean(err_rel_vals):.6g}"
            yield f" Relative L2 error (max):  {safe_max(err_rel_vals):.6g}"

        for msg in errors_by_method.get(method, []):
            yield f"  Note: {msg}"
            
def project(data: NDArray, interpolator: RegularGridInterpolator):
    
    if (data.ndim == 3):
        nz, ny, nx = data.shape

        z, y, x = np.linspace(0, 1, nz), np.linspace(0, 1, ny), np.linspace(0, 1, nx)
        Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

        interp_points = np.column_stack((Z.ravel(), Y.ravel(), X.ravel()))

        interp_vals = interpolator(interp_points)
        data_proj = interp_vals.reshape(nz, ny, nx)
    elif (data.ndim == 4):
        
        nz, ny, nx, nw = data.shape

        z = np.linspace(0, 1, nz)
        y = np.linspace(0, 1, ny)
        x = np.linspace(0, 1, nx)
        w = np.linspace(0, 1, nw)

        Z, Y, X, W = np.meshgrid(z, y, x, w, indexing="ij")

        interp_points = np.column_stack(
            (Z.ravel(), Y.ravel(), X.ravel(), W.ravel())
        )

        interp_vals = interpolator(interp_points)
        data_proj = interp_vals.reshape(nz, ny, nx, nw)
    else:
        raise ValueError("Unsupported number dims")

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

# Choose variable and panel for 
def process_netcdf(data1: NDArray, data2: NDArray, variable: str, panel: int, time_index: int):
    data1_var = data1[variable]
    data2_var = data2[variable]

    data1_ready = data1_var[time_index, panel, ...]
    data2_ready = data2_var[time_index, panel, ...]
    return data1_ready, data2_ready


# Choose variable and panel for state vector
def process_sv(data1: NDArray, data2: NDArray, variable_index: int, panel: int):
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
    
    all_metric_records: list[MetricRecord] = []

    # Load data
    data1 = None
    data2 = None

    data_path_1 = args.data_file_1
    data_path_2 = args.data_file_2
    extension_1 = os.path.splitext(data_path_1)[1]
    extension_2 = os.path.splitext(data_path_2)[1]
    
    if extension_1 not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported data type '{extension_1}'")
    elif extension_2 not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported data type '{extension_2}'")
    elif extension_1 != extension_2:
        raise ValueError(f"Incompatible data type '{extension_1}' and '{extension_2}'")
    
    if extension_1 == ".nc":
        data1 = nc.Dataset(args.data_file_1, "r")
        data2 = nc.Dataset(args.data_file_2, "r")
    elif extension_1 == ".npy":
        data1, config1 = load_state(args.data_file_1)
        data2, config2 = load_state(args.data_file_2)

    else:
        raise ValueError(f"Unsupported input type '{extension_1}'")

    # Variable loop
    vars = args.vars

    # Iterate through each data variable such as rho, P, theta
    for var_index in range(len(vars)):

        variable_label = (
            str(vars[var_index]) if isinstance(vars[var_index], str) else f"var_idx_{var_index}"
        )

        # Iterate through each panels
        for p in range(0, 5):

            if extension_1 == ".nc":
                data1_var, data2_var = process_netcdf(data1, data2, vars[var_index], p, args.time_index)
            elif extension_1 == ".npy":
                data1_var, data2_var = process_sv(data1, data2, var_index, p)

            # Interpolating one of the grids since different sizes
            if data1_var.shape != data2_var.shape:

                src_is_finer = np.prod(data1_var.shape) > np.prod(data2_var.shape)
                src = data1_var if src_is_finer else data2_var
                tgt = data2_var if src_is_finer else data1_var

                # Run interpolation and metrics with different interpolation methods
                method_results, method_errors = interpolate_and_metrics(src, tgt)
                for method, metrics in method_results.items():
                    all_metric_records.append(
                        MetricRecord(
                            variable=variable_label,
                            panel=p,
                            method=method,
                            rmse=metrics["rmse"],
                            nrmse=metrics["nrmse"],
                            err_l2=metrics["err_abs"],
                            err_l2_rel=metrics["err_rel"],
                            sp_err_abs=metrics["sp_err_abs"],
                            sp_err_rel=metrics["sp_err_rel"],
                        )
                    )


            # Same grid, we take l2 error and l2 spectral error
            else:
                results = evaluate_without_interpolation(data1_var, data2_var)
                metrics = results["direct"]

                # results_by_method["direct"].append(metrics)

                all_metric_records.append(
                    MetricRecord(
                        variable=variable_label,
                        panel=p,
                        method="direct",
                        rmse=metrics["rmse"],
                        nrmse=metrics["nrmse"],
                        err_l2=metrics["err_l2"],
                        err_l2_rel=metrics["err_l2_rel"],
                        sp_err_abs=metrics["sp_err_abs"],
                        sp_err_rel=metrics["sp_err_rel"],
                    )
                )

        # Report stream
        # report_text = build_report_text(variable_label, var_index, results_by_method, errors_by_method)
        # for line in report_text.splitlines():
        #     print(line)
        # all_reports_text.append(report_text)

    # if args.save_path:
    #     try:
    #         with open(args.save_path, "w", encoding="utf-8") as f:
    #             f.write("\n\n".join(all_reports_text))
    #         print(f"\nSaved report to: {args.save_path}")
    #     except Exception as e:
    #         print(f"\nFailed to save report to {args.save_path}: {e}")
            
            
    if args.report_type == "simple":
        return all_metric_records

    elif args.report_type == "verbose":
        return {
            "reports": all_metric_records,
            "errors": all_metric_records,
        }

# Small python wrapper
def run(
    data_file_1: str,
    data_file_2: str,
    *,
    time_index: int = -1,
    save_path: str = "",
    vars: list[str] | None = None,
    report_type: str = "simple",
) -> list[MetricRecord] | dict:

    args = SimpleNamespace(
        data_file_1=data_file_1,
        data_file_2=data_file_2,
        time_index=time_index,
        save_path=save_path,
        vars=vars or ["P", "rho", "theta"],
        report_type=report_type,
    )

    return main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Prints error and spectral error between two datas.
    """
    )
    parser.add_argument("data_file_1", type=str, help="Path to the first output file")
    parser.add_argument("data_file_2", type=str, help="Path to the second output file")

    # Optionals
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
    
    parser.add_argument("--report_type", default="simple", type=str, choices=["simple", "verbose"])

    # Run
    main(parser.parse_args())