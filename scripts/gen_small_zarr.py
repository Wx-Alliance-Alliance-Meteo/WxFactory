#!/usr/bin/env python3

import argparse

import numcodecs
import xarray as xr
import zarr


def main():
    parser = argparse.ArgumentParser(description="Copy one time slice into a small Zarr dataset.")
    parser.add_argument(
        "input_zarr", nargs="?", default="/fs/site7/eccc/mrd/rpna/cap003/datasets/era5_0.25deg_13level_lq/2020"
    )
    parser.add_argument("--output", default="results/2020_small.zarr")
    parser.add_argument("--time-index", type=int, default=0)
    args = parser.parse_args()

    print("xarray:", xr.__version__)
    print("zarr:", zarr.__version__)
    print("numcodecs:", numcodecs.__version__)
    with xr.open_zarr(args.input_zarr) as dataset:
        subset = dataset.isel(time=slice(args.time_index, args.time_index + 1))
        # Source chunk encodings may not fit the shortened time axis.
        for variable in subset.variables:
            subset[variable].encoding = {}
        subset.to_zarr(args.output, mode="w")
        print(subset)
    print(f"Saved subset to {args.output}")


if __name__ == "__main__":
    main()
