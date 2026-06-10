import matplotlib

matplotlib.use("Agg")

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---- Load dataset ----
ds = xr.open_zarr("/fs/site7/eccc/mrd/rpna/cap003/datasets/era5_0.25deg_13level_lq/2020", consolidated=True)


# ---- Fix longitude ----
ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby("longitude")

# ---- Chunking ----
ds = ds.chunk(
    {
        "time": 1,
        "latitude": ds.sizes["latitude"],
        "longitude": ds.sizes["longitude"],
        "features": ds.sizes["features"],
    }
)

# ---- Select geopotential height ----
data_var = ds[list(ds.data_vars)[0]]
z = data_var.sel(features="geopotential_h1000") / 9.81

# ---- Prepare first frame ----
z0 = z.isel(time=0).compute()

fig, ax = plt.subplots(figsize=(10, 6))
img = z0.plot(ax=ax, cmap="viridis", add_colorbar=True)
title = ax.set_title("Time index: 0")


# ---- Update function ----
def animate(i):
    frame = z.isel(time=i).compute()
    img.set_array(frame.values.ravel())
    title.set_text(f"Time index: {i}")
    return [img]


# ---- Create animation ----
ani = animation.FuncAnimation(fig, animate, frames=min(100, z.sizes["time"]), interval=200)

ani.save("results/ERA5.gif", writer="pillow", fps=5)

print("Saved animation as fluid_height.gif")
