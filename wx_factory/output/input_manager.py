import numpy
import torch
import xarray
from mpi4py import MPI
from torch import Tensor

from ..common import Configuration, ConfigurationSchema, angle24, decode_ig4, default_schema_path, readfile
from ..geometry import CubedSphere2D
from ..wx_mpi import Conditional, SingleProcess, do_once
from .state import load_state

try:
    import rmn

    rmn_available = True
except ModuleNotFoundError:
    rmn_available = False


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
        if not rmn_available:
            raise ModuleNotFoundError("rmn")

        with SingleProcess(comm) as s, Conditional(s):
            with rmn.fst24_file(grid_file_name) as grid_file:
                for record in grid_file.new_query(typvar="X", grtyp="Q"):
                    s.return_value = decode_ig4(record.ig4) + (
                        angle24.decode(record.ig1),
                        angle24.decode(record.ig2),
                        angle24.decode(record.ig3),
                    )

        return s.return_value

    @staticmethod
    def read_mountain(mountain_file_name: str, geometry: CubedSphere2D) -> Tensor:
        mountain_field = None
        comm = geometry.context.comm
        with SingleProcess(comm) as s, Conditional(s):
            num_points = geometry.total_num_elements_horizontal * geometry.num_solpts
            target_shape = (6,) + (num_points, num_points)
            with rmn.fst24_file(mountain_file_name) as mountain_file:
                for record in mountain_file.new_query(nomvar="ME"):
                    mountain_field = record.data.T.reshape(target_shape)
                    break

        mountain_field = geometry.process_topology.distribute_cube(mountain_field, 2)

        return torch.asarray(geometry._to_new(mountain_field))

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
