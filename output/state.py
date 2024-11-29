import json
from typing import Tuple, Any
import os
import numpy

from common.configuration import Configuration
from common.device import Device, default_device


def save_state(
    state: numpy.ndarray, param: Configuration, output_file_name: str, device: Device = default_device
) -> None:
    """Save simulation state into a file, along with its configuration."""
    with open(output_file_name, "wb") as output_file:
        device.xp.save(output_file, state)
        output_file.write(bytes(json.dumps(param.pack()), "utf-8"))


def load_state(input_file_name: str, device: Device = default_device) -> Tuple[numpy.ndarray, dict[str, dict[str, Any]]]:
    """Retrieve simulation state from file, along with its configuration."""
    with open(input_file_name, "rb") as input_file:
        state = device.xp.load(input_file)
        sections = json.loads("".join([str(line, 'utf-8') for line in input_file.readlines()]))

        temp_file_name = "temp.ini"

        with open(temp_file_name, "wt") as temp_file:
            for section, options in sections.items():
                temp_file.write(f"[{section}]\n")
                for option, value in options.items():
                    if list == type(value) or dict == type(value):
                        temp_file.write(f"{option} = {json.dumps(value)}\n")
                    elif bool == type(value):
                        temp_file.write(f"{option} = {1 if value else 0}\n")
                    else:
                        temp_file.write(f"{option} = {value}\n")


        conf = Configuration(temp_file_name, False)
        os.remove(temp_file_name)

        return state, conf
