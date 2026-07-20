import zarr
import xarray as xr
import numcodecs

print("xarray:", xr.__version__)
print("zarr:", zarr.__version__)
print("numcodecs:", numcodecs.__version__)

# Input and output Zarr stores
input_zarr = "/fs/site7/eccc/mrd/rpna/cap003/datasets/era5_0.25deg_13level_lq/2020"
output_zarr = "results/2020_small.zarr"

# Open dataset
ds = xr.open_zarr(input_zarr)

# Option 1: Select a date range
subset = ds.isel(time=slice(0, 1))

# Remove inherited zarr encoding
for var in subset.variables:
    subset[var].encoding = {}

# Save to new Zarr
subset.to_zarr(output_zarr, mode="w")

print(subset)
print(f"Saved subset to {output_zarr}")
