from .shallow_water import sw_from_ERA5
from time import time
import xarray as xr

from ..common.definitions import idx_h, idx_hu1, idx_hu2
from ..common.configuration import Configuration


def export_era5_all_timesteps(sim, config: Configuration, dataset):

    t0 = time()
    geom = sim.geometry
    xp = geom.device.xp

    base_shape = geom.lon.shape
    num_equations = 3
    dtype = xp.float64

    features = list(dataset["features"].values)
    feature_map = {str(f): i for i, f in enumerate(features)}
    NZ = len(geom.z_levels)

    for i in range(dataset.sizes["time"]):
        if i != 0:
            u1_contra, u2_contra, fluid_height = sw_from_ERA5(geom, dataset, i, geom.z_levels, feature_map)

            Q = xp.zeros((num_equations, NZ) + base_shape, dtype=dtype)

            Q[idx_h, ...] = fluid_height
            Q[idx_hu1, ...] = fluid_height * u1_contra
            Q[idx_hu2, ...] = fluid_height * u2_contra

            sim.output.step(Q, i)

    sim.output.finalize(time() - t0)
