import xarray as xr
import matplotlib.pyplot as plt

ds = xr.open_zarr("/fs/site7/eccc/mrd/rpna/cap003/datasets/era5_0.25deg_13level_lq/2020", consolidated=True)

# Convert longitude from [0, 360] → [-180, 180]
ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))

ds = ds.sortby("longitude")

ds = ds.chunk(
    {
        "time": 1,
        "latitude": ds.sizes["latitude"],
        "longitude": ds.sizes["longitude"],
        "features": ds.sizes["features"],
    }
)

print(ds.time.values)

data_var = ds[list(ds.data_vars)[0]]

z = data_var.sel(features="geopotential_h500")

z = z.isel(time=0)

z = z / 9.81

z = z.compute()

plt.figure(figsize=(10, 6))

z.plot(cmap="viridis", cbar_kwargs={"label": "Fluid Height (m)"})

z.plot.contour(colors="black", linewidths=0.5, levels=15)

plt.title("Fluid Height (Geopotential Height at 500 hPa)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.tight_layout()
plt.savefig("results/ERA5.png")
