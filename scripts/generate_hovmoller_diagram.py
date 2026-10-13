#!/usr/bin/env python3

import argparse
import ast
import os
from contextlib import contextmanager

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")


@contextmanager
def _read_data(path):
    with nc.Dataset(path, "r") as data:
        yield tuple(data[name] for name in ("theta", "time", "lons", "lats", "elev"))


def _process_cube_sphere(theta, times, lons, lats, elevs, lat_index, height_index, time_start_index, time_stop_index):
    """Sample one latitude row on equatorial panels 0--3; theta axes are (time, panel, z, y, x)."""

    theta_diff = theta[:, :, height_index, lat_index, :] - theta[time_start_index, :, height_index, lat_index, :]

    merged_theta = np.concatenate([theta_diff[time_start_index:time_stop_index, i, :] for i in [0, 1, 2, 3]], axis=-1)
    merged_longitudes = np.concatenate([lons[i, lat_index, :] for i in [0, 1, 2, 3]])
    times_merge = times[time_start_index:time_stop_index]

    longitude_order = np.argsort(merged_longitudes)
    sorted_longitudes = merged_longitudes[longitude_order]
    sorted_theta = merged_theta[:, longitude_order]

    return sorted_theta, sorted_longitudes, times_merge


def _plot_hovmoller(theta, lons, times, output_file, plot_kwargs):

    lons_grid, times_grid = np.meshgrid(lons, times)

    levels = np.linspace(np.min(theta), np.max(theta), 10)
    levels_neg = [lvl for lvl in levels if lvl < 0]
    levels_pos = [lvl for lvl in levels if lvl >= 0]

    positive_options = {"linestyles": "solid", "colors": "black", **plot_kwargs}
    negative_options = {"linestyles": "dashed", "colors": "black", **plot_kwargs}
    cont_pos = plt.contour(lons_grid, times_grid, theta, levels=levels_pos, **positive_options)
    cont_neg = plt.contour(lons_grid, times_grid, theta, levels=levels_neg, **negative_options)

    plt.clabel(cont_pos, inline=True, fmt="%.3f", colors="blue", fontsize=8)
    plt.clabel(cont_neg, inline=True, fmt="%.3f", colors="blue", fontsize=8)

    eastward_longitudes = [120, 329.88]
    guide_times = [0, times[-1]]

    westward_longitudes = [120, 72.088]

    plt.plot(eastward_longitudes, guide_times, color="red", linewidth=2)
    plt.plot(westward_longitudes, guide_times, color="red", linewidth=2)

    plt.gca().set_xlabel(r"\phi")
    plt.gca().tick_params(axis="x", labelsize=14)
    plt.gca().tick_params(axis="y", labelsize=14)
    plt.gca().set_ylabel("Time (s)", fontsize=14)
    plt.gca().set_xlabel("Longitude", fontsize=14)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def _closest_value_index(values, target):
    index = int(np.argmin(np.abs(values - target)))
    return index, values[index]


def main(args):
    with _read_data(args.data_file) as (theta, times, lons, lats, elevs):

        height_index = args.height_index
        height_value = args.height_value
        if height_index == -1:
            height_index, height_value_approx = _closest_value_index(elevs[0, :, 0, 0], height_value)
            print(f"Using height value: {height_value_approx} at index {height_index}")
        else:
            print(f"Using height index: {height_index}")

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

        time_stop_index = args.time_stop_index
        time_stop_value = args.time_stop_value
        if time_stop_index == 0:
            if time_stop_value != 0:
                time_stop_index, time_stop_value_approx = _closest_value_index(times[...], time_stop_value)
                print(f"Using time stop value: {time_stop_value_approx} at index {time_stop_index}")
            else:
                time_stop_index = -1
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

        theta, lons, times = _process_cube_sphere(
            theta, times, lons, lats, elevs, lat_index, height_index, time_start_index, time_stop_index
        )

        if args.plot_kwargs:
            plot_kwargs = ast.literal_eval(" ".join(args.plot_kwargs))
        else:
            plot_kwargs = {}
        _plot_hovmoller(theta, lons, times, args.output_file, plot_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot potential-temperature change against longitude and time.")

    parser.add_argument("data_file", help="Path to the output file")

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
