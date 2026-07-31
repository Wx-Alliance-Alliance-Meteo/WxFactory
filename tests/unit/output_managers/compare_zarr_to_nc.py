import os
import unittest
import subprocess
import numpy as np
import xarray as xr


class CompareZarrToNcTestCase(unittest.TestCase):
    """
    Compare NetCDF and Zarr outputs generated from the same ERA5 export.

    NetCDF variables:
        h, U, V, RV, PV

    Zarr variables:
        data(time, equations, z, faces, y, x)

    Equation mapping:
        0 -> h
        1 -> U
        2 -> V
        3 -> RV
        4 -> PV
    """

    NC_FILE = "results/out.nc"
    ZARR_STORE = "results/out.zarr"

    def test_compare_zarr_to_nc(self):
        if not os.path.exists(self.NC_FILE):
            self.skipTest(f"Missing NetCDF file: {self.NC_FILE}")

        if not os.path.exists(self.ZARR_STORE):
            self.skipTest(f"Missing Zarr store: {self.ZARR_STORE}")

        ds_nc = xr.open_dataset(self.NC_FILE)
        ds_zarr = xr.open_zarr(self.ZARR_STORE)

        try:
            for variable_name in ds_nc.data_vars:

                self.assertIn(
                    variable_name,
                    ds_zarr.data_vars,
                    f"Variable '{variable_name}' missing from Zarr",
                )

                nc_values = ds_nc[variable_name].values
                zarr_values = ds_zarr[variable_name].values

                self.assertEqual(
                    nc_values.shape,
                    zarr_values.shape,
                    (
                        f"Shape mismatch for variable '{variable_name}'. "
                        f"NetCDF={nc_values.shape}, "
                        f"Zarr={zarr_values.shape}"
                    ),
                )

                if not np.array_equal(nc_values, zarr_values):

                    diff = np.abs(nc_values - zarr_values)

                    mismatch_locations = np.argwhere(nc_values != zarr_values)

                    first_idx = tuple(mismatch_locations[0])

                    self.fail(
                        f"Variable '{variable_name}' differs.\n"
                        f"Number of mismatches : {len(mismatch_locations)}\n"
                        f"Maximum difference : {np.max(diff):.16e}\n"
                        f"First mismatch idx : {first_idx}\n"
                        f"NetCDF value       : {nc_values[first_idx]}\n"
                        f"Zarr value         : {zarr_values[first_idx]}\n"
                        f"Difference         : {diff[first_idx]}"
                    )

        finally:
            ds_nc.close()
            ds_zarr.close()
