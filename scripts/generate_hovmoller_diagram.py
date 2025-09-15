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

    # cubesphere - panels 0, 1, 2, 3
    # TODO theta diff?
    theta_diff = theta[:, :, height_index, lat_index, :] - theta[time_start_index, :, height_index, lat_index, :]
    theta_merge = np.concatenate([theta_diff[time_start_index:time_stop_index, i, :] for i in [0, 1, 2, 3]], axis = -1)
    lons_merge = np.concatenate([lons[i, lat_index, :] for i in [0, 1, 2, 3]])
    times_merge = times[time_start_index:time_stop_index]

    # rearrange longitudes
    sort_idx = np.argsort(lons_merge)
    lons_sort = lons_merge[sort_idx]
    theta_sort = theta_merge[:, sort_idx]

    return theta_sort, lons_sort, times_merge

def _plot_hovmoller(theta, lons, times, output_file, plot_kwargs):

    lons_grid, times_grid = np.meshgrid(lons, times)
    
    # useful for debug
    # c = plt.pcolormesh(lons_grid, times_grid, theta, cmap = "summer")

    # Show negative levels in dashed, positive solide lines
    levels=np.linspace(np.min(theta), np.max(theta), 10)
    levels_neg = [lvl for lvl in levels if lvl < 0]
    levels_pos = [lvl for lvl in levels if lvl >= 0]

    cont_pos = plt.contour(lons_grid, times_grid, theta, levels=levels_pos, linestyles='solid', colors='black')
    cont_neg = plt.contour(lons_grid, times_grid, theta, levels=levels_neg, linestyles='dashed', colors='black')
   
    # contour text on line
    plt.clabel(cont_pos, inline=True, fmt="%.2f", colors='blue', fontsize=8)
    plt.clabel(cont_neg, inline=True, fmt="%.2f", colors='blue', fontsize=8)

    # Add the thick red line between (130, 0) and (199.46, 3600)
    x_line = [120, 329.88]  # Longitude values (130 to 199.46)
    y_line = [0, times[-1]]  # Time values (0 to last time)

    x_line2 = [120, 72.088]  # Longitude values (130 to 199.46)
    y_line2 = [0, times[-1]]  # Time values (0 to last time)

    # Plot the red line
    plt.plot(x_line, y_line, color='red', linewidth=3)
    plt.plot(x_line2, y_line2, color='red', linewidth=3)

    # general
    plt.gca().set_xlabel("\phi")
    plt.gca().tick_params(axis="x", labelsize=14)
    plt.gca().tick_params(axis="y", labelsize=14)
    plt.gca().set_ylabel("Time (s)", fontsize=14)
    plt.gca().set_xlabel("Longitude", fontsize=14)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")

def _closest_value_index(arr, val):
    diff = np.abs(arr - val)
    return np.argmin(diff)

def main(args):
    theta, times, lons, lats, elevs = _read_data(args.data_file)

    height_index = args.height_index
    height_value = args.height_value
    if (args.height_value != -1):
        height_index = _closest_value_index(elevs[0,:,0,0], height_value)

    time_start_index = args.time_start_index
    time_start_value = args.time_start_value
    if (args.time_start_value != -1):
        print("using time start value")
        time_start_index = _closest_value_index(times, time_start_value) + 1

    time_stop_index = times.shape[0] if -1 else args.time_stop_index # non inclusive time stop index
    time_stop_value = args.time_stop_value
    if (args.time_stop_value != -1):
        print("using time stop value")
        time_stop_index = _closest_value_index(times, time_stop_value)

    lat_index = args.lat_index
    lat_value = args.lat_value
    if (args.time_stop_value != -1):
        lat_index = _closest_value_index(lats[0,:,0], lat_value)

    # process
    theta, lons, times = _process_cube_sphere(theta, times, lons, lats, elevs,
lat_index, height_index, time_start_index, time_stop_index)

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
    parser.add_argument("--output_file", default=os.path.join(root_dir, "results", "hovmoller"), help="Path to the output file")

    parser.add_argument("--time_stop_index", default=-1, type=int, help="Time step end to compare perturbation")
    parser.add_argument("--time_stop_value", default=-1, type=float, help="Time end to compare perturbation")

    parser.add_argument("--time_start_index", default=0, type=int, help="Time step start to compare perturbation")
    parser.add_argument("--time_start_value", default=-1, type=float, help="Time start to compare perturbation")

    parser.add_argument("--lat_index", default=0, type=int, help="Fixed latitude index. Default equator")
    parser.add_argument("--lat_value", default=-1, type=float, help="Fixed latitude approximate value.")

    parser.add_argument("--height_index", default=0, type=int, help="Fixed height index. Default 0")
    parser.add_argument("--height_value", default=-1, type=float, help="Fixed height approximate value. Default 0.")

    parser.add_argument("--plot_kwargs", nargs="*", help="Pyplot keywords arguments")

    main(parser.parse_args())