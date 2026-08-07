#!/usr/bin/env python3

import argparse
import glob
import multiprocessing as mp
import os
import shutil
import subprocess

import xarray as xr


def run_job(config_file, mpi_ranks=6):
    cmd = ["mpirun", "-n", str(mpi_ranks), "./WxFactory", config_file]

    print(f"Starting: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"Job failed for {config_file}")


def find_zarr_files(year, search_dir="results"):
    pattern = os.path.join(search_dir, f"{year}-*.zarr")

    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No Zarr files found matching {pattern}")

    print(f"Found {len(files)} Zarr files:")
    for f in files:
        print(f"  {f}")

    return files


def merge_zarr(year, search_dir="results"):
    files = find_zarr_files(year, search_dir)

    output_zarr = os.path.join(search_dir, f"{year}.zarr")

    if os.path.exists(output_zarr):
        print(f"Removing existing {output_zarr}")
        shutil.rmtree(output_zarr)

    datasets = [xr.open_zarr(f) for f in files]

    combined = xr.concat(datasets, dim="time")
    combined = combined.sortby("time")

    print(f"Writing merged dataset to {output_zarr}")
    combined.to_zarr(output_zarr, mode="w")

    print("Merge complete.")


def main():
    parser = argparse.ArgumentParser(description="Generate yearly Zarr output.")

    parser.add_argument("configs", nargs="+", help="Configuration files")

    parser.add_argument("--year", type=int, required=True)

    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--mpi-ranks", type=int, default=6)

    parser.add_argument("--results-dir", default="results", help="Directory containing zarr outputs")

    args = parser.parse_args()

    with mp.Pool(args.workers) as pool:
        pool.starmap(run_job, [(cfg, args.mpi_ranks) for cfg in args.configs])

    merge_zarr(args.year, args.results_dir)


if __name__ == "__main__":
    main()
