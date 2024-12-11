from typing import Tuple, Any
import numpy

from common.configuration import Configuration
from common.configuration_schema import ConfigurationSchema, load_default_schema
from common.device import Device, default_device

def save_state(
    state: numpy.ndarray, param: Configuration, output_file_name: str, device: Device = default_device
) -> None:
    """Save simulation state into a file, along with its configuration."""
    with open(output_file_name, "wb+") as output_file:
        device.xp.save(output_file, state)
        output_file.write(bytes(f"{param.state_version}\n", "utf-8"))
        output_file.write(bytes(param.config_content, "utf-8"))


def load_state(input_file_name: str, schema: ConfigurationSchema = load_default_schema(), device: Device = default_device) -> Tuple[numpy.ndarray, Configuration]:
    """Retrieve simulation state from file, along with its configuration."""
    with open(input_file_name, "rb") as input_file:
        state = device.xp.load(input_file)
        state_version = str(input_file.readline(), "utf-8")
        config_content = "\n".join([str(line, 'utf-8') for line in input_file.readlines()]).strip()
        conf = Configuration(config_content, schema)

        return state, conf
