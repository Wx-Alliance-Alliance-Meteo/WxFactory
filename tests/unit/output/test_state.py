import os
import random
import unittest

from mpi4py import MPI
import numpy

import common.configuration
import common.configuration_schema
from device import CpuDevice
import output.state

import tests.unit.ndarray_generator as ndarray_generator
import tests.unit.common.config_pack

state_input_dir = "tests/data/unit/state_tests"
state_tmp_dir = "tests/data/temp"


class StateTestCases(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.cpu_device = CpuDevice(MPI.COMM_WORLD)
        if not os.path.exists(state_tmp_dir):
            os.mkdir(state_tmp_dir)

    def test_save_load_works(self):
        schema_path = "config/config-format.json"
        schema_text: str
        with open(schema_path) as f:
            schema_text = "\n".join(f.readlines())
        schema = common.configuration_schema.ConfigurationSchema(schema_text)

        config_path = os.path.join(state_input_dir, "config.ini")
        config_text: str
        with open(config_path) as f:
            config_text = "\n".join(f.readlines())

        output_path = os.path.join(state_tmp_dir, "test_state_data")
        seed: int = 5646459
        rand = random.Random(seed)
        number_of_data = 5
        [arr] = ndarray_generator.generate_vectors(number_of_data, rand, -10, 10, [self.cpu_device])

        conf = common.configuration.Configuration(config_text, schema)

        output.state.save_state(arr, conf, output_path)

        data, loaded_conf = output.state.load_state(output_path)
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

    def test_load_old_state(self):
        state, config = output.state.load_state(os.path.join(state_input_dir, "old_save_file.wx"))
        self.assertTrue(isinstance(state, numpy.ndarray))
        self.assertEqual(state.shape, (4, 8, 8, 4))
        self.assertTrue(isinstance(config, common.configuration.Configuration))
        self.assertEqual(config.num_solpts, 2)
        self.assertEqual(config.num_elements_horizontal, 8)
        self.assertEqual(config.num_elements_vertical, 8)
        self.assertEqual(config.equations, "euler")
        self.assertEqual(config.grid_type, "cartesian2d")
        self.assertEqual(config.dt, 5)
        self.assertEqual(config.t_end, 150)
