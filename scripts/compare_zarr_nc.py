#!/usr/bin/env python3

import sys
import numpy as np
import xarray as xr


def load_netcdf(nc_file):
    ds = xr.open_dataset(nc_file)

    h = ds["PV"]

    lats = ds["lats"][:].values if "lats" in ds else None
    lons = ds["lons"][:].values if "lons" in ds else None

    return h, lats, lons


def load_zarr(zarr_store):
    ds = xr.open_zarr(zarr_store)

    h = ds["data"].isel(equations=4)

    return h


def report_difference(name, a, b, atol):

    diff = np.abs(a - b)

    max_diff = np.max(diff)
    n_diff = np.count_nonzero(diff > atol)

    print(f"\n{name}")
    print("-" * len(name))
    print("shape:", a.shape)
    print("max diff:", max_diff)
    print("num different:", n_diff)

    if n_diff > 0:
        idx = tuple(np.argwhere(diff > atol)[0])

        print("first mismatch index:", idx)
        print("A value:", a[idx])
        print("B value:", b[idx])
        print("diff   :", diff[idx])

    return n_diff == 0


def compare_h(nc_file, zarr_store, atol=1e-12):

    h_nc, lats_nc, lons_nc = load_netcdf(nc_file)
    h_zarr = load_zarr(zarr_store)

    h_nc = h_nc.values
    h_zarr = h_zarr.values

    print("\n=== SHAPE CHECK ===")
    print("NetCDF :", h_nc.shape)
    print("Zarr   :", h_zarr.shape)

    #
    # Direct comparison
    #
    if h_nc.shape == h_zarr.shape:
        if report_difference("DIRECT COMPARISON", h_nc, h_zarr, atol):
            print("\nPERFECT MATCH")
            return

    #
    # Check transpose of last two dimensions
    #
    print("\n=== TRYING X/Y TRANSPOSE ===")

    if h_nc.ndim == h_zarr.ndim:

        perm = list(range(h_zarr.ndim))
        perm[-1], perm[-2] = perm[-2], perm[-1]

        h_trans = np.transpose(h_zarr, perm)

        if h_trans.shape == h_nc.shape:

            if report_difference(
                "TRANSPOSED X/Y COMPARISON",
                h_nc,
                h_trans,
                atol,
            ):
                print("\nDATA MATCHES AFTER X/Y TRANSPOSE")
                return

    #
    # Face-by-face analysis
    #
    print("\n=== FACE ANALYSIS ===")

    if h_nc.ndim == 5 and h_zarr.ndim == 5:

        nfaces = min(h_nc.shape[2], h_zarr.shape[2])

        for face in range(nfaces):

            diff = np.abs(h_nc[:, :, face, :, :] - h_zarr[:, :, face, :, :])

            print(f"Face {face}: " f"max diff={np.max(diff):.16e}")

    print("\n=== TIME ANALYSIS ===")

    for t in range(h_nc.shape[0]):

        diff = np.abs(h_nc[t] - h_zarr[t])

        max_diff = np.max(diff)

        if max_diff > 0:
            print(f"time={t:2d} " f"max_diff={max_diff:.6e}")

    for t in range(h_nc.shape[0]):

        diff = np.abs(h_nc[t] - h_zarr[t])

        n_diff = np.count_nonzero(diff > 1e-12)

        print(f"time={t:2d} " f"num_diff={n_diff}")

    for t in range(h_nc.shape[0]):

        print(f"\nTIME {t}")

        print("NC  min/max:", np.min(h_nc[t]), np.max(h_nc[t]))

        print("ZR  min/max:", np.min(h_zarr[t]), np.max(h_zarr[t]))

    #
    # Coordinate diagnostics
    #
    if lats_nc is not None and lons_nc is not None:

        print("\n=== COORDINATE INFO ===")

        print("Latitude shape :", lats_nc.shape)

        print("Longitude shape:", lons_nc.shape)

        print("Latitude range :", np.nanmin(lats_nc), np.nanmax(lats_nc))

        print("Longitude range:", np.nanmin(lons_nc), np.nanmax(lons_nc))

    print("\nFILES DO NOT APPEAR IDENTICAL")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage:\n" "python compare_h.py output.nc output.zarr")
        sys.exit(1)

    compare_h(
        sys.argv[1],
        sys.argv[2],
        atol=1e-12,
    )
