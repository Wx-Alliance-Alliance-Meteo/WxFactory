import matplotlib

matplotlib.use("Agg")

import argparse

import matplotlib.pyplot as plt
import xarray as xr


def load_dataset(file):
    return xr.open_zarr(file)


def get_h_equations(ds, z_level=0):

    h = ds["data"].sel(equations="h")

    h = h.isel(z=z_level)

    return h


def plot_faces(h, time_index, output):

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
    plt.close(fig)
    print(f"Saved: results/{output}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", required=True)
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--z", type=int, default=0)
    parser.add_argument("--output", default="Zarr.png")

    args = parser.parse_args()

    with load_dataset(args.file) as dataset:
        height = get_h_equations(dataset, args.z)
        plot_faces(height, args.time_index, args.output)


if __name__ == "__main__":
    main()
