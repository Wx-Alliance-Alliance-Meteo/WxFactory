#!/usr/bin/env python3

import argparse
import ast
import os
from contextlib import contextmanager

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from matplotlib.ticker import FormatStrFormatter

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")


@contextmanager
def _read_data(path):
    with nc.Dataset(path, "r") as data:
        yield tuple(data[name] for name in ("theta", "time", "lons", "lats", "elev"))


def _process_cube_sphere(theta, times, lons, lats, elevs, lat_index, time_start_index, time_stop_index):
    """Sample one latitude row on equatorial panels 0--3; theta axes are (time, panel, z, y, x)."""

    theta_diff = theta[time_stop_index, 0:4, :, lat_index, :] - theta[time_start_index, 0:4, :, lat_index, :]

    merged_theta = np.concatenate([theta_diff[i, :, :] for i in [0, 1, 2, 3]], axis=-1)
    merged_longitudes = np.concatenate([lons[i, lat_index, :] for i in [0, 1, 2, 3]])
    elevs_merge = elevs[0, :, 0, 0] / 1e3

    longitude_order = np.argsort(merged_longitudes)
    sorted_longitudes = merged_longitudes[longitude_order]
    sorted_theta = merged_theta[:, longitude_order]

    return sorted_theta, sorted_longitudes, elevs_merge, times[time_stop_index]


def _plot_potential(theta, lons, elevs, time, output_file, plot_kwargs):

    lons_grid, elev_grid = np.meshgrid(lons, elevs)

    plt.figure(figsize=(15, 5))

    c = plt.contourf(lons_grid, elev_grid, theta, cmap="summer")
    cb = plt.colorbar(c)

    dthetamin = np.amin(theta)
    dthetamax = np.amax(theta)
    labels = np.linspace(dthetamin, dthetamax, 15)

    plt.contour(lons_grid, elev_grid, theta, levels=labels, colors=["black"], linewidths=0.8, **plot_kwargs)

    cb.set_ticks(ticks=labels, labels=[str(pos) for pos in labels])
    cb.ax.set_ylabel("Δθ", fontsize=14)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    plt.gca().set_xlabel(r"\phi")
    plt.gca().set_xticks([0, 90, 180, 270, 360])
    plt.gca().set_title(f"t = {time:g} s")
    plt.gca().tick_params(axis="x", labelsize=14)
    plt.gca().tick_params(axis="y", labelsize=14)
    plt.gca().set_ylabel("H (km)", fontsize=14)
    plt.gca().set_xlabel("Longitude", fontsize=14)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def _closest_value_index(values, target):
    index = int(np.argmin(np.abs(values - target)))
    return index, values[index]


def main(args):

    with _read_data(args.data_file) as (theta, times, lons, lats, elevs):

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

        theta, lons, elevs, time = _process_cube_sphere(
            theta, times, lons, lats, elevs, lat_index, time_start_index, time_stop_index
        )

        if args.plot_kwargs:
            plot_kwargs = ast.literal_eval(" ".join(args.plot_kwargs))
        else:
            plot_kwargs = {}
        _plot_potential(theta, lons, elevs, time, args.output_file, plot_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot potential-temperature change along a fixed latitude row.")

    parser.add_argument("data_file", help="Path to the output file")

    parser.add_argument(
        "--output_file",
        default=os.path.join(root_dir, "results", "potential_temperature"),
        help="Path to the output file",
    )

    parser.add_argument("--lat_value", default=0, type=float, help="Fixed latitude approximate value. Default equator")
    parser.add_argument("--lat_index", default=-1, type=int, help="Fixed latitude index")

    parser.add_argument("--time_stop_index", default=0, type=int, help="Time step end to compare perturbation")
    parser.add_argument("--time_stop_value", default=0, type=float, help="Time end to compare perturbation")

    parser.add_argument("--time_start_index", default=-1, type=int, help="Time step start to compare perturbation")
    parser.add_argument("--time_start_value", default=-1, type=float, help="Time start to compare perturbation")

    parser.add_argument("--plot_kwargs", nargs="*", help="Pyplot keywords arguments")

    main(parser.parse_args())
