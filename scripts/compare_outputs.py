#!/usr/bin/env python3

import os
import sys
import argparse

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
src_dir = os.path.join(root_dir, "wx_factory")
sys.path.append(root_dir)
sys.path.append(src_dir)

import numpy as np
import netCDF4 as nc
from numpy.typing import NDArray

from common.layout_conversion import sv_to_netcdf
from scipy.interpolate import RegularGridInterpolator


# Grid interpolator with method 'quintic'
# Equivalent to finite elements with 5 points
def get_interpolator(dest: NDArray) -> RegularGridInterpolator:

    nz, ny, nx = dest.shape[0], dest.shape[1], dest.shape[2]
    z, y, x = np.linspace(0, 1, nz), np.linspace(0, 1, ny), np.linspace(0, 1, nx)
    interp = RegularGridInterpolator((z, y, x), dest, method="quintic")

    return interp


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

    time_index = args.time_index

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

    vars = args.vars
    if args.input_type == "sv":
        vars = range(
            data1.shape[1]
        )  # since state_vector does not have the names of the variables, we can use the lenght of the variable vector

    # Iterate through each data variable such as rho, P, theta
    for var_index in range(len(vars)):

        err_abs = [None] * 5
        err_rel = [None] * 5
        sp_err_abs = [None] * 5
        sp_err_rel = [None] * 5

        rmse = [None] * 5
        nrmse = [None] * 5

        # Iterate through each panels
        for p in range(0, 5):

            if args.input_type == "netcdf":
                data1_var, data2_var = process_netcdf(data1, data2, vars[var_index], p, time_index)
            elif args.input_type == "sv":
                data1_var, data2_var = process_reshaped_sv(data1, data2, var_index, p)

            # Interpolating one of the grids since different sizes
            if data1_var.shape != data2_var.shape:

                # Interpolating along the grids with more points
                isMin1 = np.sum(data1_var.shape) > np.sum(data2_var.shape)

                interpolator = get_interpolator(data1_var if isMin1 else data2_var)
                projected = project(data2_var if isMin1 else data1_var, interpolator)

                err_abs[p], err_rel[p] = get_error(data2_var if isMin1 else data1_var, projected)
                sp_err_abs[p], sp_err_rel[p] = spectral_error(data2_var if isMin1 else data1_var, projected)
                rmse[p], nrmse[p] = get_error_rmse(data2_var if isMin1 else data1_var, projected)

            # Same grid, we take l2 error and l2 spectral error
            else:
                err_abs[p], err_rel[p] = get_error(data1_var, data2_var)
                sp_err_abs[p], sp_err_rel[p] = spectral_error(data1_var, data2_var)
                rmse[p], nrmse[p] = get_error_rmse(data1_var, data2_var)

        print("")
        print(f"-------------")
        print(f"Report for {vars[var_index]}")

        # print(f"Root mean square error: {np.mean(rmse)}")
        print(f"Normalized root mean square error: {np.max(nrmse)}")
        # print(f"Absolute spectral error: {np.mean(sp_err_abs)}, ")
        print(f"Relative spectral error: {np.max(sp_err_rel)}")
        # print(f"Absolute error: {np.mean(err_abs)}, ")
        # print(f"Relative error (for comparison): {np.max(err_rel)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Prints error and spectral error between two datas at fixed time step.
        Quintic interpolation is used if grids are not the same size.
        Call with "python PATH_TO_SCRIPT_DIR/compare_outputs.py PATH_TO_DATA1.nc PATH_TO_DATA2.nc"
        Add --input_type sv when comparing state_vectors
    """
    )
    parser.add_argument("data_file_1", type=str, help="Path to the first output file netcdf")
    parser.add_argument("data_file_2", type=str, help="Path to the second output file netcdf")

    # Optionals
    parser.add_argument("--input_type", default="netcdf", type=str, help="netcdf or sv")
    parser.add_argument("--time_index", default=-1, type=int, help="Time step to compare. Defaults at last.")
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
