import pickle
import json
from typing import Tuple, Any

import numpy

from common.configuration import Configuration
from common.device import Device, default_device


def save_state(
    state: numpy.ndarray, param: Configuration, output_file_name: str, device: Device = default_device
) -> None:
    """Save simulation state into a file, along with its configuration."""
    with open(output_file_name, "wb") as output_file:
        device.xp.save(output_file, state)
        output_file.write(bytes(json.dumps(param.sections), "utf-8"))


def load_state(input_file_name: str, device: Device = default_device) -> Tuple[numpy.ndarray, dict[str, dict[str, Any]]]:
    """Retrieve simulation state from file, along with its configuration."""
    with open(input_file_name, "rb") as input_file:
        state = device.xp.load(input_file)
        sections = json.loads("".join([str(line, 'utf-8') for line in input_file.readlines()]))

        return state, sections
