import matplotlib

matplotlib.use("Agg")

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse


def load_dataset(path):
    ds = xr.open_zarr(path, consolidated=True)

    # Fix longitude
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    ds = ds.sortby("longitude")

    # Chunking
    ds = ds.chunk(
        {
            "time": 1,
            "latitude": ds.sizes["latitude"],
            "longitude": ds.sizes["longitude"],
            "features": ds.sizes["features"],
        }
    )
    return ds


def get_variable(ds, level):
    data_var = ds[list(ds.data_vars)[0]]
    z = data_var.sel(features=level) / 9.81
    return z


def save_image(z, time_index, output):
    frame = z.isel(time=time_index).compute()

    plt.figure(figsize=(10, 6))

    frame.plot(cmap="viridis", cbar_kwargs={"label": "Height (m)"})
    frame.plot.contour(colors="black", linewidths=0.5, levels=15)

    plt.title(f"Geopotential Height ({frame.time.values})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved image: {output}")


def save_gif(z, output, max_frames):
    fig, ax = plt.subplots(figsize=(10, 6))

    first = z.isel(time=0).compute()
    img = first.plot(ax=ax, cmap="viridis", add_colorbar=True)
    title = ax.set_title("Time index: 0")

    def animate(i):
        frame = z.isel(time=i).compute()
        img.set_array(frame.values.ravel())
        title.set_text(f"Time index: {i}")
        return [img]

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=min(max_frames, z.sizes["time"]),
        interval=200,
    )

    ani.save(output, writer="pillow", fps=5)
    print(f"Saved GIF: {output}")


def main():
    parser = argparse.ArgumentParser(description="ERA5 visualization tool")

    parser.add_argument(
        "--path",
        required=True,
        help="Path to Zarr dataset",
    )

    parser.add_argument(
        "--mode",
        choices=["image", "gif"],
        required=True,
        help="Output type: image or gif",
    )

    parser.add_argument(
        "--level",
        default="geopotential_h500",
        help="Feature level (e.g., geopotential_h500)",
    )

    parser.add_argument(
        "--time-index",
        type=int,
        default=0,
        help="Time index for image mode",
    )

    parser.add_argument(
        "--output",
        default="ERA5",
        help="Output file name",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum frames for GIF",
    )

    args = parser.parse_args()

    ds = load_dataset(args.path)
    z = get_variable(ds, args.level)

    if args.mode == "image":
        output_file = "results/" + args.output + ".png"
    elif args.mode == "gif":
        output_file = "results/" + args.output + ".gif"

    if args.mode == "image":
        save_image(z, args.time_index, output_file)

    elif args.mode == "gif":
        save_gif(z, output_file, args.max_frames)


if __name__ == "__main__":
    main()
