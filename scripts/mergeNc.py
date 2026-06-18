import xarray as xr
import glob
import os

base_dir = os.getcwd()
input_dir = os.path.join(base_dir, "results")
output_dir = os.path.join(input_dir, "output")
pattern = os.path.join(input_dir, "*.nc")

os.makedirs(output_dir, exist_ok=True)

files = sorted(glob.glob(pattern))

if not files:
    raise RuntimeError(f"No NetCDF files found in {input_dir}")

print("Found files:")
for f in files:
    print("  ", f)

valid_files = []
for f in files:
    try:
        with xr.open_dataset(f) as ds:
            # simple check: has time variable
            if "time" not in ds.variables:
                print(f"Skipping file (no time variable): {f}")
                continue

        valid_files.append(f)

    except Exception as e:
        print(f"Skipping corrupted file {f}: {e}")

if not valid_files:
    raise RuntimeError("No valid NetCDF files to merge")

print(f"Using {len(valid_files)} valid files")

ds = xr.open_mfdataset(valid_files, combine="by_coords")

ds.load()

output_file = os.path.join(output_dir, "merged.nc")

if os.path.exists(output_file):
    print("Removing existing file:", output_file)
    os.remove(output_file)

ds.to_netcdf(output_file)

print("Merged file written to:", output_file)
