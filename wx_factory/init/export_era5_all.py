from time import time

import torch
import xarray as xr

from ..common.configuration import Configuration
from ..common.definitions import idx_h, idx_hu1, idx_hu2
from .shallow_water import sw_from_ERA5


def export_era5_all_timesteps(sim, config: Configuration):

    t0 = time()
    geom = sim.geometry
    base_shape = geom.lon.shape
    num_equations = 3
    dtype = torch.float64

    ds = xr.open_zarr(config.initial_condition, consolidated=True)
    time_start = str(config.time_start)
    time_end = str(config.time_end)

    dataset = ds.sel(time=slice(time_start, time_end))

    features = list(dataset["features"].values)
    feature_map = {str(f): i for i, f in enumerate(features)}
    NZ = len(geom.z_levels)

    for i in range(dataset.sizes["time"] - 1):
        if i != 0:
            u1_contra, u2_contra, fluid_height = sw_from_ERA5(geom, dataset, i, geom.z_levels, feature_map)

            Q = torch.zeros((num_equations, NZ) + base_shape, dtype=dtype)

            Q[idx_h, ...] = fluid_height
            Q[idx_hu1, ...] = fluid_height * u1_contra
            Q[idx_hu2, ...] = fluid_height * u2_contra

            sim.output.step(Q, i)

    sim.output.finalize(time() - t0)
