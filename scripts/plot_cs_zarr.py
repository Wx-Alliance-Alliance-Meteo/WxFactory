import matplotlib

matplotlib.use("Agg")

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import argparse


def load_dataset(file):
    ds = xr.open_zarr(file)
    return ds


def get_h_equations(ds, z_level=0):
    # equations dimension contains ["h", "U", "V", ...]
    h = ds["data"].sel(equations="h")

    # select vertical level
    h = h.isel(z=z_level)

    return h


def plot_faces(h, time_index, output):

    # extract frame (time, faces, y, x)
    frame = h.isel(time=time_index)

    n_faces = frame.sizes["faces"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)

    axes = axes.flatten()

    for p in range(n_faces):
        ax = axes[p]

        data = frame.isel(faces=p)

        im = ax.imshow(data.values, cmap="viridis")

        ax.set_title(f"faces {p}")
    fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02)

    plt.suptitle(f"h at time index {time_index}")

    plt.savefig("results/" + output)
    print(f"Saved: results/{output}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", required=True)
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--z", type=int, default=0)
    parser.add_argument("--output", default="Zarr.png")

    args = parser.parse_args()

    ds = load_dataset(args.file)
    h = get_h_equations(ds, args.z)

    plot_faces(h, args.time_index, args.output)


if __name__ == "__main__":
    main()
