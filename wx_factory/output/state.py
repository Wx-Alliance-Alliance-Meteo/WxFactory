from typing import Optional, Tuple

import numpy
from numpy.typing import NDArray

from common import Configuration, ConfigurationSchema, load_default_schema


def get_array_module(a):
    try:
        import cupy

        if isinstance(a, cupy.ndarray):
            return cupy
    except:
        return numpy

    return numpy


_SAVE_VERSION_ID = 1
_CONFIG_CONTENT_MARKER = "----- config_content ------\n"


def save_state(state: NDArray, param: Configuration, output_file_name: str) -> None:
    """Save simulation state into a file, along with its configuration.

    Exact content:
        1. The array itself
        2. Version
        3. Configuration schema (as a string)
        4. Separator
        5. Configuration content
    """
    with open(output_file_name, "wb+") as output_file:
        xp = get_array_module(state)
        xp.save(output_file, state)
        output_file.write(bytes(f"{_SAVE_VERSION_ID}\n", "utf-8"))
        output_file.write(bytes(param.schema.raw_string, "utf-8"))
        output_file.write(bytes(f"{_CONFIG_CONTENT_MARKER}", "utf-8"))
        output_file.write(bytes(param.config_content, "utf-8"))


def load_state(input_file_name: str) -> Tuple[numpy.ndarray, Configuration]:
    """Retrieve simulation state from file, along with its configuration.

    There are several components to the save file. They are retrieved in the same
    """
    with open(input_file_name, "rb") as input_file:
        state = numpy.load(input_file)

        try:
            version_str = str(input_file.readline(), "utf-8")
            save_version = int(version_str)
        except:
            save_version = 0

        if save_version >= 1:
            schema_lines = []
            for line in input_file:
                line_str = str(line, "utf-8")
                if _CONFIG_CONTENT_MARKER == line_str:
                    break
                schema_lines.append(line_str)
            config_schema = ConfigurationSchema("\n".join(schema_lines))
        else:
            config_schema = load_default_schema()

        config_content = "".join([str(line, "utf-8") for line in input_file.readlines()]).strip()

        conf = Configuration(config_content, config_schema)

        return state, conf
