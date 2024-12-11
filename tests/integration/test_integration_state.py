import unittest
from typing import Optional
from common.simulation import Simulation
import sys
from mpi4py import MPI
from output import state
from common.configuration import Configuration
from common.configuration_schema import load_default_schema
import numpy
from configparser import ConfigParser, NoSectionError, NoOptionError
import os
import common.wx_mpi

from typing import Optional, Type, TypeVar, Union

OptionType = TypeVar("OptionType", bound=Union[int, float, str, bool])


def _get_opt_from_parser(
    parser: ConfigParser, section_name: str, option_name: str, option_type: Type[OptionType]
) -> OptionType:
    value: Optional[OptionType] = None
    if option_type == float:
        value = parser.getfloat(section_name, option_name)
    elif option_type == int:
        value = parser.getint(section_name, option_name)
    elif option_type == str:
        value = parser.get(section_name, option_name)
    elif option_type == bool:
        value = parser.getint(section_name, option_name) > 0

    else:
        raise ValueError(f"Cannot get this option type (not implemented): {option_type}")

    assert value is not None
    return value


def _validate_option(
    option_name: str,
    value: OptionType,
    valid_values: Optional[list[OptionType]],
    min_value: Optional[OptionType],
    max_value: Optional[OptionType],
) -> OptionType:

    if valid_values is not None and value not in valid_values:
        raise ValueError(
            f'"{value}" is not considered a valid value for option "{option_name}".'
            f" Available values are {valid_values}"
        )
    if min_value is not None:
        if value < min_value:
            print(f'WARNING: Adjusting "{option_name}" to min value "{min_value}"')
            value = min_value
    if max_value is not None:
        if value > max_value:
            print(f'WARNING: Adjusting "{option_name}" to max value "{max_value}"')
            value = max_value

    return value


def _get_option(
    parser: ConfigParser,
    filename: str,
    section_name: str,
    option_name: str,
    option_type: Type[OptionType],
    default_value: Optional[OptionType],
    valid_values: Optional[list[OptionType]] = None,
    min_value: Optional[OptionType] = None,
    max_value: Optional[OptionType] = None,
) -> OptionType:
    value: Optional[OptionType] = None

    try:
        value = _get_opt_from_parser(parser, section_name, option_name, option_type)
        value = _validate_option(option_name, value, valid_values, min_value, max_value)
    except (NoOptionError, NoSectionError) as e:
        if default_value is None:
            e.message += f"\nMust specify a value for option '{option_name}' in file {filename}"
            raise
        value = default_value

    return value


class StateIntegrationTestCases(unittest.TestCase):
    config_dir_path: str
    num_process_required: int

    def __init__(self, config_dir_path: str):
        super().__init__("test_state")
        self.config_dir_path = config_dir_path
        self.num_process_required = 0
        requirement_filename = f"{self.config_dir_path}/requirement.ini"

        if os.path.exists(requirement_filename):
            parser = ConfigParser()
            parser.read(requirement_filename, encoding="utf-8")

            self.num_process_required = _get_option(
                parser, requirement_filename, "System", "processes", int, 1, min_value=1
            )
        else:
            raise FileNotFoundError()

    def setUp(self):
        if MPI.COMM_WORLD.size != self.num_process_required:
            self.fail("You do not have the required number of process")

    def test_state(self):
        has_exited: bool = False
        exit_code: Optional[sys._ExitCode] = None

        config_file: str = f"{self.config_dir_path}/config.ini"

        
        config_content = ""
        config_schema_content = ""
        if MPI.COMM_WORLD.rank == 0:
            with open(config_file) as cf:
                config_content = "\n".join(cf.readlines())
            with open("config/config-format.json") as schema_file:
                config_schema_content = "\n".join(schema_file.readlines())

            common.wx_mpi.bcast_string(config_content, MPI.COMM_WORLD)
            common.wx_mpi.bcast_string(config_schema_content, MPI.COMM_WORLD)
        else:
            config_content = common.wx_mpi.rcv_bcast_string(0, MPI.COMM_WORLD)
            config_schema_content = common.wx_mpi.rcv_bcast_string(0, MPI.COMM_WORLD)

        try:
            sim = Simulation(config_content, config_schema_content)
            sim.run()
        except SystemExit as e:
            has_exited = True
            exit_code = e.code
        except Exception as e:
            print(e, flush=True)
            raise e

        if has_exited:
            print(f"Process {MPI.COMM_WORLD.rank} has exited prematurely")
            exit(exit_code)

        
        
        schema = load_default_schema()
        conf = Configuration(config_content, schema)

        state_params = (
            conf.dt,
            conf.num_elements_horizontal,
            conf.num_elements_vertical,
            conf.num_solpts,
            MPI.COMM_WORLD.size,
        )
        config_hash = state_params.__hash__() & 0xFFFFFFFFFFFF

        base_name = f"state_vector_{config_hash:012x}_{MPI.COMM_WORLD.rank:03d}"
        state_vector_file: str = f"{conf.output_dir}/{base_name}.{conf.save_state_freq:08d}.npy"
        true_state_vector_file: str = f"{self.config_dir_path}/{base_name}.{conf.save_state_freq:08d}.npy"
        

        [data, _] = state.load_state(state_vector_file, schema)
        [true_data, _] = state.load_state(true_state_vector_file, schema)

        if len(data.shape) != len(true_data.shape):
            self.fail("The result tuple and solution tuple aren't the same shape")

        for it_length in range(len(data.shape)):
            if data.shape[it_length] != true_data.shape[it_length]:
                self.fail("The result problem is not the same size as the solution's")

        delta = true_data - data

        diff = numpy.linalg.norm(delta).item()
        true_value = numpy.linalg.norm(true_data).item()

        relative_diff = diff / true_value

        self.assertLessEqual(relative_diff, conf.tolerance, f"The relative difference ({relative_diff:.2e}) is too big")
