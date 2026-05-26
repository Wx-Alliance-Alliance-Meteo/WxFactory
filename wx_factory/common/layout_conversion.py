from ..output.state import load_state
import numpy as np


def sv_to_netcdf(sv_file: str):
    """
    Convert state_vector file to netcdf geometry
    State vector shape: (panels, variables, elevs, verticals, horizontal, num_points ** 2 (for 2D) or num_points ** 3 (for 3D))
    Netcdf shape: (times, panels, elevs, verticals, horizontal)
    See geometry.to_single_block for reference
    """

    vector, config = load_state(sv_file)
    num_elements_horizontal = config.num_elements_horizontal
    num_elements_vertical = config.num_elements_vertical
    num_solpts = config.num_solpts

    grid_type = config.grid_type

    types_2d = ["cartesian_2d"]
    types_3d = ["cubed_sphere"]

    if grid_type in types_3d:

        nk = num_elements_vertical * num_solpts
        nj = num_elements_horizontal * num_solpts
        ni = num_elements_horizontal * num_solpts

        tmp_shape = vector.shape[:2] + (
            num_elements_vertical,
            num_elements_horizontal,
            num_elements_horizontal,
            num_solpts,
            num_solpts,
            num_solpts,
        )
        block_shape = (nk, nj, ni)
        new_shape = tmp_shape[:2] + block_shape

        vector_tmp = vector.reshape(tmp_shape)

        # get shape k, numsolpts, j, numsolpts, i, numsolpts
        np.moveaxis(vector_tmp, (-3, -2), (-5, -3))
        vector_new = np.reshape(vector_tmp, new_shape)

    elif grid_type in types_2d:

        nj = num_elements_vertical * num_solpts
        ni = num_elements_horizontal * num_solpts

        tmp_shape = vector.shape[:2] + (num_elements_vertical, num_elements_horizontal, num_solpts, num_solpts)
        new_shape = vector.shape[:2] + (nj, ni)

        vector_tmp = np.reshape(vector, tmp_shape)
        np.swapaxes(vector_tmp, -2, -3)
        vector_new = np.reshape(vector_tmp, new_shape)

    return vector_new
