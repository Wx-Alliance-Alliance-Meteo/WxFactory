from typing import Optional, Tuple

from numpy.typing import NDArray

from common import Configuration, ConfigurationSchema, load_default_schema
from device import Device


def save_state(state: NDArray, param: Configuration, output_file_name: str, device: Device = None) -> None:
    """Save simulation state into a file, along with its configuration."""
    with open(output_file_name, "wb+") as output_file:
        if device is None:
            device = Device.get_default()
        device.xp.save(output_file, state)
        output_file.write(bytes(f"{param.state_version}\n", "utf-8"))
        output_file.write(bytes(param.config_content, "utf-8"))


def load_state(
    input_file_name: str, schema: Optional[ConfigurationSchema] = None, device: Device = None
) -> Tuple[NDArray, Configuration]:
    """Retrieve simulation state from file, along with its configuration."""
    if device is None:
        device = Device.get_default()
    if schema is None:
        schema = load_default_schema()
    with open(input_file_name, "rb") as input_file:
        state = device.xp.load(input_file)
        state_version = str(input_file.readline(), "utf-8")
        config_content = "\n".join([str(line, "utf-8") for line in input_file.readlines()]).strip()
        conf = Configuration(config_content, schema)

        return state, conf
