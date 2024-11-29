import json
from typing import Tuple, Any
import os
import numpy
import mpi4py

from common.configuration import Configuration
from common.device import Device, default_device


def save_state(
    state: numpy.ndarray, param: Configuration, output_file_name: str, device: Device = default_device
) -> None:
    """Save simulation state into a file, along with its configuration."""
    with open(output_file_name, "wb") as output_file:
        device.xp.save(output_file, state)
        output_file.write(bytes(param.config_content, "utf-8"))


def load_state(input_file_name: str, device: Device = default_device) -> Tuple[numpy.ndarray, dict[str, dict[str, Any]]]:
    """Retrieve simulation state from file, along with its configuration."""
    with open(input_file_name, "rb") as input_file:
        state = device.xp.load(input_file)
        config_content = "\n".join([str(line, 'utf-8') for line in input_file.readlines()]).strip()

        temp_file_name = f"temp-{mpi4py.MPI.COMM_WORLD.rank}.ini"

        with open(temp_file_name, "wt") as temp_file:
            temp_file.write(config_content)
            temp_file.flush()


        conf = Configuration(temp_file_name, False)
        os.remove(temp_file_name)

        return state, conf
