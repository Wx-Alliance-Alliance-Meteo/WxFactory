import xarray as xr
import numpy
from .shallow_water import sw_from_ERA5

from ..common.definitions import idx_h, idx_hu1, idx_hu2
from ..geometry import CubedSphere2D


def export_era5_all_timesteps(sim, input_path):

    ds = xr.open_zarr(input_path, consolidated=True)
    time_start = str(sim.config.time_start)
    time_end = str(sim.config.time_end)

    ds_subset = ds.sel(time=slice(time_start, time_end))
    geom = sim.geometry
    xp = geom.device.xp

    for i in range(ds_subset.sizes["time"]):
        if i != 0:
            print("Timestamp: ", i)
            u1, u2, h = sw_from_ERA5(geom, ds_subset, i)

            Q = xp.zeros((3,) + h.shape)

            Q[idx_h] = h
            Q[idx_hu1] = h * u1
            Q[idx_hu2] = h * u2

            sim.output.__write_result__(Q, i)

    sim.output.__finalize__()
