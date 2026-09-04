import numpy
import torch
import xarray
from mpi4py import MPI
from torch import Tensor

from ..common import (
    Configuration,
    ConfigurationSchema,
    MissingModule,
    angle24,
    decode_ig4,
    default_schema_path,
    readfile,
)
from ..geometry import CubedSphere2D, CubedSphere3D
from ..wx_mpi import Conditional, SingleProcess, do_once
from .state import load_state

try:
    import rmn

except ModuleNotFoundError as e:
    rmn = MissingModule(e)

try:
    import georef
except (ModuleNotFoundError, OSError) as e:
    georef = MissingModule(e)


def extract_available_levels(ds: xarray.Dataset):
    features = list(ds["features"].values)
    feature_set = {str(f) for f in features}

    levels: list[int] = []

    for f in features:
        name = str(f)

        if name.startswith("geopotential_h"):
            level = name.split("_h")[-1]

            geo = f"geopotential_h{level}"
            u = f"u_component_of_wind_h{level}"
            v = f"v_component_of_wind_h{level}"

            if geo in feature_set and u in feature_set and v in feature_set:
                levels.append(int(level))

    return sorted(set(levels))


class InputManager:
    schema: ConfigurationSchema | None

    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.schema = None

    @staticmethod
    def read_config(config_file: str, comm: MPI.Comm, schema: ConfigurationSchema | None = None) -> Configuration:
        if schema is None:
            schema = ConfigurationSchema(do_once(readfile, default_schema_path, comm=comm))
        return Configuration(do_once(readfile, config_file, comm=comm), schema)

    @staticmethod
    def read_config_from_save_file(save_file: str, comm: MPI.Comm) -> tuple[Configuration, Tensor]:
        config_str = None
        schema_str = None
        vector = None
        with SingleProcess(comm) as s, Conditional(s):
            vector, config = load_state(save_file)
            config_str = config.config_content
            schema_str = config.schema.raw_string

        config_str = comm.bcast(config_str)
        schema_str = comm.bcast(schema_str)

        schema = ConfigurationSchema(schema_str)
        return Configuration(config_str, schema), vector

    @staticmethod
    def read_grid_params(
        grid_file_name: str, comm: MPI.Comm
    ) -> tuple[int, int, angle24.angle24, angle24.angle24, angle24.angle24]:

        with SingleProcess(comm) as s, Conditional(s), rmn.fst24_file(grid_file_name) as grid_file:
            for record in grid_file.new_query(typvar="X", grtyp="Q"):
                s.return_value = decode_ig4(record.ig4) + (
                    angle24.decode(record.ig1),
                    angle24.decode(record.ig2),
                    angle24.decode(record.ig3),
                )

        return s.return_value

    @staticmethod
    def read_mountain(mountain_file_name: str, geometry: CubedSphere2D | CubedSphere3D) -> Tensor:
        """Read the surface topography (ME field) from an FST file onto the model grid.

        The field is read on a single process. If the file's grid does not match the
        model's cubed-sphere grid, it is interpolated onto it; otherwise the data is
        used as-is. The full 6-face field is then distributed across the horizontal
        processes and returned in the element-wise ("new") memory layout: the floor
        layout for a 3D geometry, the full field for a 2D geometry.

        Parameters:
        -----------
        mountain_file_name : str
            Path to the FST file containing the surface topography
        geometry : CubedSphere2D | CubedSphere3D
            Geometry object, used for the grid reference, the interpolation, the
            process distribution and the memory layout

        Returns:
        --------
        Tensor
            Surface height on the model grid, in the element-wise ("new") layout
        """
        mountain_field = None
        comm = geometry.context.comm
        with SingleProcess(comm) as s, Conditional(s), rmn.fst24_file(mountain_file_name) as mountain_file:
            mountain_rec = next(mountain_file.new_query(nomvar="ME"))

            num_points = geometry.total_num_elements_horizontal * geometry.num_solpts
            target_shape = (6,) + (num_points, num_points)
            total_points = 6 * num_points * num_points
            mountain_ref = georef.GeoRef.fromrecord(mountain_rec)
            if mountain_rec.ni * mountain_rec.nj * mountain_rec.nk != total_points or mountain_rec.grtyp != "Q":
                # Not the same grid, need to interpolate
                cs_ref = georef.CubedSphereRef(
                    geometry.lambda0,
                    geometry.phi0,
                    geometry.alpha0,
                    geometry.total_num_elements_horizontal,
                    geometry.num_solpts,
                )
                field_tmp = mountain_ref.interpolate(mountain_rec.data, cs_ref)
                mountain_field = torch.asarray(field_tmp.T.reshape(target_shape))
            else:
                mountain_field = torch.asarray(mountain_rec.data.T.reshape(target_shape))

        mountain_field = geometry.process_topology.distribute_cube(mountain_field, 2)

        result = (
            geometry._to_new(mountain_field)
            if isinstance(geometry, CubedSphere2D)
            else geometry.to_new_floor(mountain_field)
        )

        return result

    @staticmethod
    def read_fields(data_file_name: str, field_names: list[str], geometry: CubedSphere2D) -> Tensor:
        comm = geometry.context.comm
        fields = [None for _ in field_names]
        with SingleProcess(comm) as s, Conditional(s):
            num_points = geometry.total_num_elements_horizontal * geometry.num_solpts
            target_shape = (6,) + (num_points, num_points)
            with rmn.fst24_file(data_file_name) as data_file:
                fields = [next(data_file.new_query(nomvar=var)).data.T.reshape(target_shape) for var in field_names]

        fields = [geometry.process_topology.distribute_cube(f, 2) for f in fields]

        return torch.asarray(geometry._to_new(numpy.stack(fields)))
