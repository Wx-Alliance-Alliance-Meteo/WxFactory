from .shallow_water import sw_from_ERA5

from ..common.definitions import idx_h, idx_hu1, idx_hu2
from ..geometry import CubedSphere2D
from ..common.configuration import Configuration


def export_era5_all_timesteps(sim, config: Configuration):
    ds_subset = config.ds_subset
    geom = sim.geometry
    xp = geom.device.xp

    base_shape = geom.lon.shape
    num_equations = 3
    dtype = xp.float64
    NZ = len(config.z_levels)

    for i in range(ds_subset.sizes["time"] - 1):
        if i != 0:
            print("Timestamp: ", i)
            u1_contra, u2_contra, fluid_height = sw_from_ERA5(geom, ds_subset, i, config.z_levels, config.feature_map)

            Q = xp.zeros((NZ, num_equations) + base_shape, dtype=dtype)

            Q[:, idx_h, ...] = fluid_height
            Q[:, idx_hu1, ...] = fluid_height * u1_contra
            Q[:, idx_hu2, ...] = fluid_height * u2_contra

            sim.output.__write_result__(Q, ds_subset.data["time"][i])

    sim.output.__finalize__()
