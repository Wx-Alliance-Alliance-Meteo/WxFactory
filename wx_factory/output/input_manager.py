from typing import Optional

from mpi4py import MPI
import numpy
from numpy.typing import NDArray

from .state import load_state

from common import angle24, Configuration, ConfigurationSchema, default_schema_path, decode_ig4, readfile
from wx_mpi import do_once, SingleProcess, Conditional
from process_topology import ProcessTopology
from geometry import CubedSphere2D

from common.graphx import plot_array

try:
    import rmn

    rmn_available = True
except ModuleNotFoundError:
    rmn_available = False


class InputManager:
    schema: Optional[ConfigurationSchema]

    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.schema = None

    @staticmethod
    def read_config(config_file: str, comm: MPI.Comm, schema: Optional[ConfigurationSchema] = None) -> Configuration:
        if schema is None:
            schema = ConfigurationSchema(do_once(readfile, default_schema_path, comm=comm))
        return Configuration(do_once(readfile, config_file, comm=comm), schema)

    @staticmethod
    def read_config_from_save_file(save_file: str, comm: MPI.Comm) -> tuple[Configuration, NDArray]:
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
                for record in grid_file.new_query(typvar="X", grtyp="C"):
                    s.return_value = decode_ig4(record.ig4) + (
                        angle24.decode(record.ig1),
                        angle24.decode(record.ig2),
                        angle24.decode(record.ig3),
                    )

        return s.return_value

    @staticmethod
    def read_mountain(mountain_file_name: str, geometry: CubedSphere2D) -> NDArray:
        mountain_field = None
        comm = geometry.device.comm
        with SingleProcess(comm) as s, Conditional(s):
            num_points = geometry.total_num_elements_horizontal * geometry.num_solpts
            target_shape = (6,) + (num_points, num_points)
            with rmn.fst24_file(mountain_file_name) as mountain_file:
                for record in mountain_file.new_query(nomvar="ME"):
                    mountain_field = record.data.T.reshape(target_shape)
                    break

        mountain_field = geometry.process_topology.distribute_cube(mountain_field, 2)
        # plot_array(mountain_field, "mountain.png", comm=comm, background_value=-100)

        return geometry.device.xp.asarray(geometry._to_new(mountain_field))

    @staticmethod
    def read_fields(data_file_name: str, field_names: list[str], geometry: CubedSphere2D) -> NDArray:
        comm = geometry.device.comm
        fields = [None for _ in field_names]
        with SingleProcess(comm) as s, Conditional(s):
            num_points = geometry.total_num_elements_horizontal * geometry.num_solpts
            target_shape = (6,) + (num_points, num_points)
            with rmn.fst24_file(data_file_name) as data_file:
                fields = [next(data_file.new_query(nomvar=var)).data.T.reshape(target_shape) for var in field_names]

        # print(f"fields = {fields}")
        fields = [geometry.process_topology.distribute_cube(f, 2) for f in fields]

        # for name, f in zip(field_names, fields):
        #     plot_array(f, f"{name}.png", comm=comm, background_value=f.min() - 10.0)

        xp = geometry.device.xp
        return xp.asarray(geometry._to_new(numpy.stack(fields)))
