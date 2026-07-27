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
            self.assertIn(
                "data",
                ds_zarr.data_vars,
                "Variable 'data' not found in Zarr dataset",
            )

            #expected_equations = [str(var) for var in ds_nc.data_vars]
            expected_equations = [
                "U",
                "V",
                "W",
                "rho",
                "theta",
                "P",
            ]

            self.assertIn(
                "equations",
                ds_zarr.coords,
                "Missing 'equations' coordinate in Zarr dataset",
            )

            actual_equations = [str(v) for v in ds_zarr["equations"].values.tolist()]

            self.assertEqual(
                actual_equations,
                expected_equations,
                ("Unexpected equation ordering.\n" f"Expected: {expected_equations}\n" f"Found:    {actual_equations}"),
            )
            
            if "elev" in ds_nc:
                nc_elev = ds_nc["elev"].values
                zarr_elev = ds_zarr["elev"].values

                self.assertEqual(
                    nc_elev.shape,
                    zarr_elev.shape,
                )

            if "topo" in ds_nc:
                nc_topo = ds_nc["topo"].values
                zarr_topo = ds_zarr["topo"].values

                self.assertEqual(
                    nc_topo.shape,
                    zarr_topo.shape,
                )

            for equation_index, variable_name in enumerate(actual_equations):

                self.assertIn(
                    variable_name,
                    ds_nc.data_vars,
                    f"Variable '{variable_name}' not found in NetCDF dataset",
                )

                nc_values = ds_nc[variable_name].values

                zarr_values = ds_zarr["data"].isel(equations=equation_index).values

                self.assertEqual(
                    nc_values.shape,
                    zarr_values.shape,
                    (
                        f"Shape mismatch for variable '{variable_name}'. "
                        f"NetCDF={nc_values.shape}, "
                        f"Zarr={zarr_values.shape}"
                    ),
                )

                diff = np.abs(nc_values - zarr_values)

                """if not np.allclose(
                    nc_values,
                    zarr_values,
                    atol=self.ATOL,
                    rtol=0.0,
                ):"""
                if not np.array_equal(
                    nc_values,
                    zarr_values,
                ):
                    diff = np.abs(nc_values - zarr_values)
                    max_diff = np.max(diff)

                    mismatch_locations = np.argwhere(nc_values != zarr_values)

                    first_idx = tuple(mismatch_locations[0])

                    self.fail(
                        f"Variable '{variable_name}' differs.\n"
                        f"Number of mismatches : {len(mismatch_locations)}\n"
                        f"Maximum difference : {max_diff:.16e}\n"
                        f"First mismatch idx : {first_idx}\n"
                        f"NetCDF value       : {nc_values[first_idx]}\n"
                        f"Zarr value         : {zarr_values[first_idx]}\n"
                        f"Difference         : {diff[first_idx]}",
                    )

        finally:
            ds_nc.close()
            ds_zarr.close()
