from configparser import ConfigParser, NoSectionError, NoOptionError
import glob
import os
import sys
from typing import Optional, Type, TypeVar, Union
import unittest

from mpi4py import MPI
import numpy

from common import Configuration, load_default_schema
from output import state
from simulation import Simulation
import wx_mpi


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

        if not os.path.exists(self.config_dir_path):
            self.fail(f"Could not find test case {self.config_dir_path}")

        requirement_filename = f"{self.config_dir_path}/requirement.ini"
        if not os.path.exists(requirement_filename):
            raise FileNotFoundError(requirement_filename)

        parser = ConfigParser()
        parser.read(requirement_filename, encoding="utf-8")

        self.num_process_required = _get_option(
            parser, requirement_filename, "System", "processes", int, 1, min_value=1
        )

    def setUp(self):
        if MPI.COMM_WORLD.size != self.num_process_required:
            self.fail(
                f"We are using {MPI.COMM_WORLD.size} process(es), but the test requires {self.num_process_required}"
            )

        self.schema = load_default_schema()
        self.config_files = glob.glob(f"{self.config_dir_path}/config*.ini")
        print(f"Config files: {self.config_files}")

    def test_state(self):
        has_exited: bool = False
        exit_code: Optional[sys._ExitCode] = None

        for config_file in self.config_files:
            config_content = wx_mpi.readfile(config_file)

            config = Configuration(config_content, self.schema)

            try:
                sim = Simulation(config)
                sim.run()
            except SystemExit as e:
                has_exited = True
                exit_code = e.code
            except Exception as e:
                print(e, flush=True)
                raise e

            if has_exited:
                print(f"Process {MPI.COMM_WORLD.rank} has exited prematurely")
                sys.exit(exit_code)

            conf = sim.config

            state_vector_file = sim.output.state_file_name(conf.save_state_freq)
            base_name = os.path.split(state_vector_file)[-1]
            true_state_vector_file: str = f"{self.config_dir_path}/{base_name}"

            [data, _] = state.load_state(state_vector_file, self.schema)
            [true_data, _] = state.load_state(true_state_vector_file, self.schema)

            if data.shape != true_data.shape:
                self.fail("The result problem is not the same size as the solution's")

            delta = true_data - data

            diff = numpy.linalg.norm(delta).item()
            true_value = numpy.linalg.norm(true_data).item()

            relative_diff = diff / true_value

            self.assertLessEqual(
                relative_diff, conf.tolerance, f"The relative difference ({relative_diff:.2e}) is too big"
            )
