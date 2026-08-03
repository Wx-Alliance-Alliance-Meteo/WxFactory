from typing import Optional, Tuple
import re

import numpy
from numpy.typing import NDArray
import torch
from torch import Tensor

from ..common import Configuration, ConfigurationSchema, load_default_schema

_SAVE_VERSION_ID = 1
_CONFIG_CONTENT_MARKER = "----- config_content ------\n"


def save_state(state: Tensor, param: Configuration, output_file_name: str) -> None:
    """Save simulation state into a file, along with its configuration.

    Exact content:
        1. The array itself
        2. Version
        3. Configuration schema (as a string)
        4. Separator
        5. Configuration content
    """
    with open(output_file_name, "wb+") as output_file:
        numpy.save(output_file, state.cpu().numpy())
        output_file.write(bytes(f"{_SAVE_VERSION_ID}\n", "utf-8"))
        output_file.write(bytes(param.schema.raw_string, "utf-8"))
        output_file.write(bytes(f"{_CONFIG_CONTENT_MARKER}", "utf-8"))
        output_file.write(bytes(param.config_content, "utf-8"))


def load_state(input_file_name: str, device: Optional[torch.device] = None) -> Tuple[Tensor, Configuration]:
    """Retrieve simulation state from file, along with its configuration.

    There are several components to the save file. They are retrieved in the same
    """
    with open(input_file_name, "rb") as input_file:
        state = torch.tensor(numpy.load(input_file), device=device)

        try:
            version_str = str(input_file.readline(), "utf-8")
            save_version = int(version_str)
        except:
            save_version = 0

        stored_schema = None
        if save_version >= 1:
            schema_lines = []
            for line in input_file:
                line_str = str(line, "utf-8")
                if _CONFIG_CONTENT_MARKER == line_str:
                    break
                schema_lines.append(line_str)
            stored_schema = ConfigurationSchema("\n".join(schema_lines))

        default_schema = load_default_schema()

        content_list = [str(line, "utf-8").strip() for line in input_file.readlines()]
        content_list = [a for a in content_list if a != ""]
        config_content = "\n".join(content_list)

        try:
            conf = Configuration(config_content, default_schema)
        except:
            if stored_schema is not None:
                conf = Configuration(config_content, stored_schema)
            else:
                raise

        return state, conf
