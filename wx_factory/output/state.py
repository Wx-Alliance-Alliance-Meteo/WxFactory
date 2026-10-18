from configparser import ConfigParser
from io import StringIO

import numpy
import torch
from torch import Tensor

from ..common import Configuration, ConfigurationSchema, load_default_schema
from ..common.euler_state import state_representation

_SAVE_VERSION_ID = 1
_CONFIG_CONTENT_MARKER = "----- config_content ------\n"


def save_state(state: Tensor, param: Configuration, output_file_name: str) -> None:
    """Save simulation state into a file, along with its configuration.

    Exact content:
        1. The array itself
        2. Version
        3. State representation (version 3, Euler only)
        4. Configuration schema (as a string)
        5. Separator
        6. Configuration content
    """
    with open(output_file_name, "wb+") as output_file:
        numpy.save(output_file, state.cpu().numpy())
        version = 3 if param.equations == "euler" else _SAVE_VERSION_ID
        output_file.write(bytes(f"{version}\n", "utf-8"))
        if version == 3:
            output_file.write(bytes(f"{state_representation(param)}\n", "utf-8"))
        output_file.write(bytes(param.schema.raw_string, "utf-8"))
        output_file.write(bytes(f"{_CONFIG_CONTENT_MARKER}", "utf-8"))
        output_file.write(bytes(param.config_content, "utf-8"))


def load_state(input_file_name: str, device: torch.device | None = None) -> tuple[Tensor, Configuration]:
    """Retrieve simulation state from file, along with its configuration.

    There are several components to the save file. They are retrieved in the same
    """
    with open(input_file_name, "rb") as input_file:
        state = torch.tensor(numpy.load(input_file), device=device)

        try:
            version_str = str(input_file.readline(), "utf-8")
            save_version = int(version_str)
        except (ValueError, UnicodeDecodeError):
            save_version = 0

        stored_form = input_file.readline().decode("utf-8").strip() if save_version in (2, 3) else "conservative"
        if save_version == 2 and stored_form == "lamb":
            stored_form = "relative_covariant"
        if save_version > 3 or stored_form not in ("conservative", "relative_covariant"):
            raise ValueError("Unsupported restart state representation")

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

        config_content = "".join([str(line, "utf-8") for line in input_file]).strip()
        if save_version == 2:
            parser = ConfigParser()
            parser.read_string(config_content)
            changed = parser.remove_option("Test_case", "momentum_form") if parser.has_section("Test_case") else False
            if parser.has_option("Spatial_discretization", "lamb_jacobian"):
                parser.set("Spatial_discretization", "jacobian", parser.get("Spatial_discretization", "lamb_jacobian"))
                parser.remove_option("Spatial_discretization", "lamb_jacobian")
                changed = True
            if changed:
                content = StringIO()
                parser.write(content)
                config_content = content.getvalue()

        try:
            conf = Configuration(config_content, default_schema)
        except Exception:
            if stored_schema is not None:
                conf = Configuration(config_content, stored_schema)
            else:
                raise

        if conf.equations == "euler" and stored_form != "relative_covariant":
            raise ValueError(
                "Legacy Euler restart contains conservative momentum; relative covariant velocity is required"
            )
        if conf.equations == "shallow_water" and stored_form != "conservative":
            raise ValueError("Shallow-water restart requires conservative variables")
        conf.state_representation = stored_form
        return state, conf
