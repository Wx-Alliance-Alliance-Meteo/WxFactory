import netCDF4
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np


def save_image(nc_path, time_index, height_index, output_base):
    ds = netCDF4.Dataset(nc_path)

    # Select time + height → (face, lat, lon)
    h = ds["h"][time_index, height_index, :, :, :]
    h = h[:]

    fig, axs = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)

    for i, ax in enumerate(axs.flat):
        if i < h.shape[0]:
            im = ax.imshow(h[i, :, :])
            ax.set_title(f"Face {i}")
        else:
            ax.axis("off")

    plt.colorbar(im, ax=axs)

    output_file = output_base + ".png"
    plt.savefig(output_file)

    print(f"Saved image: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="NetCDF cube visualization")

    parser.add_argument(
        "--file",
        default="results/out.nc",
        help="Path to NetCDF file",
    )

    parser.add_argument(
        "--time-index",
        type=int,
        default=0,
        help="Time index to plot",
    )

    parser.add_argument(
        "--height-index",
        type=int,
        default=0,
        help="Height level index",
    )

    parser.add_argument(
        "--output",
        default="results/Netcdf",
        help="Output filename (without extension)",
    )

    args = parser.parse_args()

    base, _ = os.path.splitext(args.output)

    save_image(args.file, args.time_index, args.height_index, base)


if __name__ == "__main__":
    main()
