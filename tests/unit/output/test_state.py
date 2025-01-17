import os
import random

import common.configuration
import common.configuration_schema
import output.state

import tests.unit.cpu_test as cpu_test
import tests.unit.ndarray_generator as ndarray_generator
import tests.unit.common.config_pack


class StateTestCases(cpu_test.CpuTestCases):
    def setUp(self):
        super().setUp()
        if not os.path.exists("tests/data/temp"):
            os.mkdir("tests/data/temp")

    def test_save_load_works(self):
        schema_path = "config/config-format.json"
        schema_text: str
        with open(schema_path) as f:
            schema_text = "\n".join(f.readlines())
        schema = common.configuration_schema.ConfigurationSchema(schema_text)

        config_path = "tests/data/state_tests/config.ini"
        config_text: str
        with open(config_path) as f:
            config_text = "\n".join(f.readlines())

        output_path = "tests/data/temp/test_state_data"
        seed: int = 5646459
        rand = random.Random(seed)
        number_of_data = 5
        [arr] = ndarray_generator.generate_vectors(number_of_data, rand, -10, 10, [self.cpu_device])

        conf = common.configuration.Configuration(config_text, schema)

        output.state.save_state(arr, conf, output_path, self.cpu_device)

        data, loaded_conf = output.state.load_state(output_path, schema, self.cpu_device)
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
