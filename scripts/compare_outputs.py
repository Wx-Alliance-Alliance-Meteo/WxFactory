#!/usr/bin/env python3

import os
import sys

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
src_dir = os.path.join(root_dir, "wx_factory")
sys.path.append(root_dir)
sys.path.append(src_dir)

import matplotlib.pyplot as plt
import numpy as np

import argparse

import netCDF4 as nc

from math import gcd
from wx_factory.common.interpolation import lagrange_poly
from common.interpolation import Interpolator

from scipy.interpolate import RegularGridInterpolator


def get_interpolator(dest):

    nz, ny, nx = dest.shape[0], dest.shape[1], dest.shape[2]
    z, y, x = np.linspace(0, 1, nz), np.linspace(0, 1, ny), np.linspace(0, 1, nx)
    interp = RegularGridInterpolator((z, y, x), dest, method="quintic")

    return interp


def project(data, interpolator: RegularGridInterpolator):
    nz, ny, nx = data.shape[0], data.shape[1], data.shape[2]
    z, y, x = np.linspace(0, 1, nz), np.linspace(0, 1, ny), np.linspace(0, 1, nx)
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

    interp_points = np.array([Z.ravel(), Y.ravel(), X.ravel()]).T

    interp_vals = interpolator(interp_points)
    data_proj = interp_vals.reshape(nz, ny, nx)

    return data_proj


def get_error(grid1, grid2):

    err = np.abs(grid1 - grid2)
    err_l2 = np.linalg.norm(err)
    err_l2_rel = err_l2 / np.linalg.norm(grid1)

    return err_l2, err_l2_rel


def process_netcdf(data1, data2, variable: str, panel, time_index):
    data1_var = data1[variable]
    data2_var = data2[variable]

    data1_ready = data1_var[time_index, panel, ...]
    data2_ready = data2_var[time_index, panel, ...]
    return data1_ready, data2_ready


def spectral_error(grid1, grid2):
    fft1 = np.abs(np.fft.fftn(grid1))
    fft2 = np.abs(np.fft.fftn(grid2))

    err = np.linalg.norm(fft1 - fft2)
    err_rel = err / np.linalg.norm(fft1)
    return err, err_rel


def main(args):

    time_index = args.time_index
    vars = args.vars

    data1 = nc.Dataset(args.data_file_1, "r")
    data2 = nc.Dataset(args.data_file_2, "r")

    # Iterate through each data variable such as rho, P, theta
    for var in vars:

        err_abs = [None] * 5
        err_rel = [None] * 5
        sp_err_abs = [None] * 5
        sp_err_rel = [None] * 5

        # Iterate through each panels
        for i in range(0, 5):
            theta1, theta2 = process_netcdf(data1, data2, var, i, time_index)

            # Interpolating one of the grids since different sizes
            if theta1.shape != theta2.shape:

                # Interpolating along the grids with more points
                isMin1 = np.sum(theta1.shape) > np.sum(theta2.shape)

                interpolator = get_interpolator(theta1 if isMin1 else theta2)
                projected = project(theta2 if isMin1 else theta1, interpolator)

                err_abs[i], err_rel[i] = get_error(theta2 if isMin1 else theta1, projected)
                sp_err_abs[i], sp_err_rel[i] = spectral_error(theta2 if isMin1 else theta1, projected)

            # Same grid, we take l2 error and l2 spectral error
            else:
                err_abs[i], err_rel[i] = get_error(theta1, theta2)
                sp_err_abs[i], sp_err_rel[i] = spectral_error(theta1, theta2)

        # note: might consider rt mean square instead
        print("")
        print(f"-------------")
        print(f"Report for {var}")
        print(f"Absolute error: {np.mean(err_abs)}, ")
        print(f"Relative error: {np.mean(err_rel)}")
        print(f"Absolute spectral error: {np.mean(sp_err_abs)}, ")
        print(f"Relative spectral error: {np.mean(sp_err_rel)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Prints error and spectral error between two datas at fixed time step.
        Quintic interpolation is used if grids are not the same size.
        Call with "python PATH_TO_SCRIPT_DIR/compare_outputs.py PATH_TO_DATA1.nc PATH_TO_DATA2.nc"
    """
    )
    parser.add_argument("data_file_1", type=str, help="Path to the first output file netcdf")
    parser.add_argument("data_file_2", type=str, help="Path to the second output file netcdf")

    # Optionals
    parser.add_argument("--time_index", default=-1, type=int, help="Time step to compare. Defaults at last.")
    parser.add_argument(
        "--vars",
        nargs="+",
        type=str,
        default=["P", "rho", "theta"],
        help="List of variables to estimate the error with. e.g. --vars P rho theta",
    )

    # Run
    main(parser.parse_args())
