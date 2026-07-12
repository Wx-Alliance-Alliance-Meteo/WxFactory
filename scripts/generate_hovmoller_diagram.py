#!/usr/bin/env python3

import sys
import os

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
src_dir = os.path.join(root_dir, "wx_factory")
sys.path.append(root_dir)
sys.path.append(src_dir)

import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import FormatStrFormatter

import argparse
import ast


def _read_data(path):
    data = nc.Dataset(path, "r")

    times = data["time"]
    theta = data["theta"]
    elev = data["elev"]
    lons = data["lons"]
    lats = data["lats"]

    return theta, times, lons, lats, elev


def _process_cube_sphere(theta, times, lons, lats, elevs, lat_index, height_index, time_start_index, time_stop_index):
    """
    _process_cube_sphere:

    theta shape: ntimes, npanels, nelevs, nlats, nlons

    The panels correspond to the cube-sphere layout with 4 and 5 on top and bottom.
    Since we want value along the equator, we only select panels [0, 1, 2, 3] and the latitude corresponding to the equator
    """

    # We take the difference between the last theta and at the start with respect to longitude and time
    # Values are taken at the equator along panels 0 to 3
    theta_diff = theta[:, :, height_index, lat_index, :] - theta[time_start_index, :, height_index, lat_index, :]

    # concatenate the different panels
    theta_merge = np.concatenate([theta_diff[time_start_index:time_stop_index, i, :] for i in [0, 1, 2, 3]], axis=-1)
    lons_merge = np.concatenate([lons[i, lat_index, :] for i in [0, 1, 2, 3]])
    times_merge = times[time_start_index:time_stop_index]

    # Rearrange so that longitudes have the proper order
    sort_idx = np.argsort(lons_merge)
    lons_sort = lons_merge[sort_idx]
    theta_sort = theta_merge[:, sort_idx]

    return theta_sort, lons_sort, times_merge


def _plot_hovmoller(theta, lons, times, output_file, plot_kwargs):

    lons_grid, times_grid = np.meshgrid(lons, times)

    # useful for debug
    # c = plt.pcolormesh(lons_grid, times_grid, theta, cmap = "summer")

    # Show negative levels in dashed, positive solide lines
    levels = np.linspace(np.min(theta), np.max(theta), 10)
    levels_neg = [lvl for lvl in levels if lvl < 0]
    levels_pos = [lvl for lvl in levels if lvl >= 0]

    cont_pos = plt.contour(lons_grid, times_grid, theta, levels=levels_pos, linestyles="solid", colors="black")
    cont_neg = plt.contour(lons_grid, times_grid, theta, levels=levels_neg, linestyles="dashed", colors="black")

    # contour text on line
    plt.clabel(cont_pos, inline=True, fmt="%.3f", colors="blue", fontsize=8)
    plt.clabel(cont_neg, inline=True, fmt="%.3f", colors="blue", fontsize=8)

    # Add the thick red line between (130, 0) and (199.46, 3600)
    x_line = [120, 329.88]  # Longitude values (130 to 199.46)
    y_line = [0, times[-1]]  # Time values (0 to last time)

    x_line2 = [120, 72.088]  # Longitude values (130 to 199.46)
    y_line2 = [0, times[-1]]  # Time values (0 to last time)

    # Plot the red line
    plt.plot(x_line, y_line, color="red", linewidth=2)
    plt.plot(x_line2, y_line2, color="red", linewidth=2)

    # general
    plt.gca().set_xlabel("\phi")
    plt.gca().tick_params(axis="x", labelsize=14)
    plt.gca().tick_params(axis="y", labelsize=14)
    plt.gca().set_ylabel("Time (s)", fontsize=14)
    plt.gca().set_xlabel("Longitude", fontsize=14)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")


def _closest_value_index(arr, val):
    diff = np.abs(arr - val)
    argmin = np.argmin(diff)
    minval = arr[argmin]
    return argmin, minval


def main(args):
    theta, times, lons, lats, elevs = _read_data(args.data_file)

    # For all args, check if index arg is used ** Compared with default -1
    # Otherwise, approximate value and find the closest value index
    height_index = args.height_index
    height_value = args.height_value
    if height_index == -1:
        height_index, height_value_approx = _closest_value_index(elevs[0, :, 0, 0], height_value)
        print(f"Using height value: {height_value_approx} at index {height_index}")
    else:
        print(f"Using height index: {height_index}")

    # time_start index cannot be -1, so using it as reference
    # default index if nothing changed is 0
    time_start_index = args.time_start_index
    time_start_value = args.time_start_value
    if time_start_index == -1:
        if time_start_value != -1:
            time_start_index, time_start_value_approx = _closest_value_index(times[...], time_start_value)
            print(f"Using time start value: {time_start_value_approx} at index {time_start_index}")
        else:
            time_start_index = 0
            print(f"Using time start index: {time_start_index}")
    else:
        print(f"Using time start index: {time_start_index}")

    # time_stop index cannot be 0, so using it as reference
    # default index if nothing changed is -1
    time_stop_index = args.time_stop_index
    time_stop_value = args.time_stop_value
    if time_stop_index == 0:
        if time_stop_value != 0:
            time_stop_index, time_stop_value_approx = _closest_value_index(times[...], time_stop_value)
            print(f"Using time stop value: {time_stop_value_approx} at index {time_stop_index}")
        else:
            time_stop_index = -1  # default index
            print(f"Using time stop index: {time_stop_index}")
    else:
        print(f"Using time stop index: {time_stop_index}")

    lat_index = args.lat_index
    lat_value = args.lat_value
    if lat_index == -1:
        lat_index, lat_value_approx = _closest_value_index(lats[0, :, 0], lat_value)
        print(f"Using lat value: {lat_value_approx} at index {lat_index}")
    else:
        print(f"Using lat index: {lat_index}")

    # process
    theta, lons, times = _process_cube_sphere(
        theta, times, lons, lats, elevs, lat_index, height_index, time_start_index, time_stop_index
    )

    # Plot
    if args.plot_kwargs:
        plot_kwargs = ast.literal_eval(args.plot_kwargs)
    else:
        plot_kwargs = {}
    _plot_hovmoller(theta, lons, times, args.output_file, plot_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Plot the potential temperature perturbation with a fixed latitude
    Call with "python PATH_TO_SCRIPT_DIR/generate_hovmoller_diagram.py PATH_TO_DATA.nc"
    """)

    parser.add_argument("data_file", help="Path to the output file")

    # optional
    parser.add_argument(
        "--output_file", default=os.path.join(root_dir, "results", "hovmoller"), help="Path to the output file"
    )

    parser.add_argument("--time_stop_index", default=0, type=int, help="Time step end to compare perturbation")
    parser.add_argument("--time_stop_value", default=0, type=float, help="Time end to compare perturbation")

    parser.add_argument("--time_start_index", default=-1, type=int, help="Time step start to compare perturbation")
    parser.add_argument("--time_start_value", default=-1, type=float, help="Time start to compare perturbation")

    parser.add_argument("--lat_value", default=0, type=float, help="Fixed latitude approximate value. Default equator")
    parser.add_argument("--lat_index", default=-1, type=int, help="Fixed latitude index.")

    parser.add_argument("--height_index", default=-1, type=int, help="Fixed height index. Default 0")
    parser.add_argument(
        "--height_value", default=5243.55, type=float, help="Fixed height approximate value (m). Default 0."
    )

    parser.add_argument("--plot_kwargs", nargs="*", help="Pyplot keywords arguments")

    main(parser.parse_args())
