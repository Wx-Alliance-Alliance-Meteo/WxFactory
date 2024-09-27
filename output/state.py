import pickle
from typing import Tuple

import numpy

from common.configuration import Configuration
from common.device import Device, default_device

def save_state(state: numpy.ndarray, param: Configuration, output_file_name: str, device: Device=default_device) -> None:
   '''Save simulation state into a file, along with its configuration.'''
   with open(output_file_name, 'wb') as output_file:
      device.xp.save(output_file, state)
      pickle.dump(param, output_file)

def load_state(input_file_name: str, device: Device=default_device) -> Tuple[numpy.ndarray, Configuration]:
   '''Retrieve simulation state from file, along with its configuration.'''
   with open(input_file_name, 'rb') as input_file:
      state = device.xp.load(input_file)
      param = pickle.load(input_file)

      return state, param
