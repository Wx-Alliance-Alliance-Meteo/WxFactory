#!/usr/bin/env python3

import sys
import numpy as np
import xarray as xr
import zarr


def compare_outputs(nc_file, zarr_store):
    ds_nc = xr.open_dataset(nc_file)
    ds_zarr = zarr.open_group(zarr_store, mode="r")

    nc_vars = set(ds_nc.variables)
    zarr_vars = set(ds_zarr.array_keys())
    if nc_vars != zarr_vars:
        print("ERROR: NetCDF and Zarr variable lists differ")
        print("Only in NetCDF:", sorted(nc_vars - zarr_vars))
        print("Only in Zarr  :", sorted(zarr_vars - nc_vars))
    success = False

    try:
        success = True

        for variable_name in ds_nc.variables:

            if variable_name not in zarr_vars:
                print(f"ERROR: Variable '{variable_name}' missing from Zarr")
                success = False
                continue

            nc_values = ds_nc[variable_name].values
            zarr_values = ds_zarr[variable_name][:]

            if nc_values.shape != zarr_values.shape:
                print(
                    f"ERROR: Shape mismatch for '{variable_name}'\n"
                    f"  NetCDF: {nc_values.shape}\n"
                    f"  Zarr  : {zarr_values.shape}"
                )
                success = False
                continue

            if not np.array_equal(nc_values, zarr_values):
                diff = np.abs(nc_values - zarr_values)

                mismatch_locations = np.argwhere(nc_values != zarr_values)

                first_idx = tuple(mismatch_locations[0])

                print(
                    f"\nERROR: Variable '{variable_name}' differs\n"
                    f"  Number of mismatches : {len(mismatch_locations)}\n"
                    f"  Maximum difference   : {np.max(diff):.16e}\n"
                    f"  First mismatch index : {first_idx}\n"
                    f"  NetCDF value         : {nc_values[first_idx]}\n"
                    f"  Zarr value           : {zarr_values[first_idx]}\n"
                    f"  Difference           : {diff[first_idx]}"
                )

                success = False
            else:
                print(f"OK: {variable_name}")

        return success

    finally:
        ds_nc.close()


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage:\n" "python scripts/compare_zarr_to_nc.py results/out.nc results/out.zarr")
        sys.exit(1)

    nc_file = sys.argv[1]
    zarr_store = sys.argv[2]

    success = compare_outputs(
        nc_file,
        zarr_store,
    )

    if success:
        print("\nSUCCESS: NetCDF and Zarr outputs are identical")
        sys.exit(0)

    print("\nFAILURE: Differences detected")
    sys.exit(2)
