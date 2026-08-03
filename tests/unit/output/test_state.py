import os
import random

from mpi4py import MPI
import numpy
from torch import Tensor

import wx_factory.common.configuration
import wx_factory.common.configuration_schema
from wx_factory.device import PytorchDevice
import wx_factory.output.state

import tests.unit.array_generator as array_generator
import tests.unit.common.config_pack
from wx_test import WxTestCase

state_input_dir = "tests/data/unit/state_tests"
state_tmp_dir = "tests/data/temp"


class StateTestCases(WxTestCase):
    def setUp(self):
        super().setUp()
        self.cpu_device = PytorchDevice(MPI.COMM_WORLD, "cpu")
        if not os.path.exists(state_tmp_dir):
            os.mkdir(state_tmp_dir)

    def test_save_load_works(self):
        schema_path = "config/config-format.json"
        schema_text: str
        with open(schema_path) as f:
            schema_text = "\n".join(f.readlines())
        schema = wx_factory.common.configuration_schema.ConfigurationSchema(schema_text)

        config_path = os.path.join(state_input_dir, "config.ini")
        with open(config_path) as f:
            config_text = "\n".join(f.readlines())

        output_path = os.path.join(state_tmp_dir, "test_state_data")
        seed: int = 5646459
        rand = random.Random(seed)
        number_of_data = 5
        [arr] = array_generator.generate_vectors(number_of_data, rand, -10, 10, [self.cpu_device])

        conf = wx_factory.common.configuration.Configuration(config_text, schema)

        wx_factory.output.state.save_state(arr, conf, output_path)

        data, loaded_conf = wx_factory.output.state.load_state(output_path)
        safe_conf = tests.unit.common.config_pack.pack(loaded_conf)

        self.assertEqual(len(arr.shape), len(data.shape), "The shape of the data has changed between a save and a load")
        self.assertEqual(len(data.shape), 1, "The data is not a vector anymore")
        self.assertEqual(arr.shape[0], data.shape[0], "The lenght of the vector has changed between a save and a load")
        self.assertEqual(
            data.shape[0], number_of_data, "The lenght of the vector has changed between a save and a load"
        )

        for it in range(number_of_data):
            self.assertEqual(arr[it], data[it], f"Data at {it} has changed between a save and a load")

        for section, values in safe_conf.items():
            for key, value in values.items():
                initial_conf_value = getattr(conf, key)
                if list == type(initial_conf_value):
                    self.assertListEqual(
                        initial_conf_value, value, f"Configuration value {key} in section {section} has changed"
                    )
                elif dict == type(initial_conf_value):
                    self.assertDictEqual(
                        initial_conf_value, value, f"Configuration value {key} in section {section} has changed"
                    )
                else:
                    self.assertEqual(
                        initial_conf_value, value, f"Configuration value {key} in section {section} has changed"
                    )
