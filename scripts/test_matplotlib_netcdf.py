import netCDF4
import matplotlib.pyplot as plt

ds = netCDF4.Dataset("results/out.nc")

h = ds["h"][0]

fig, axs = plt.subplots(2, 3, figsize=(10, 6))

for i, ax in enumerate(axs.flat):
    im = ax.imshow(h[i])
    ax.set_title(f"Face {i}")

plt.colorbar(im, ax=axs)
plt.savefig("results/Netcdf.png")
