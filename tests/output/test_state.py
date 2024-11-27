import cpu_test
import output.state
import common.configuration
import ndarray_generator
import random

class StateTestCases(cpu_test.CpuTestCases):
    def setUp(self):
        super().setUp()
    
    def test_save_load_works(self):
        output_path = "tests/data/temp/test_state_data"
        seed: int = 5646459
        rand = random.Random(seed)
        number_of_data = 5
        [arr] = ndarray_generator.generate_vectors(number_of_data, rand, -10, 10, [self.cpu_device])
        
        conf = common.configuration.Configuration("tests/data/state_tests/config.ini", False)
        
        output.state.save_state(arr, conf, output_path, self.cpu_device)

        data, safe_conf = output.state.load_state(output_path, self.cpu_device)
        
        self.assertEqual(len(arr.shape), len(data.shape), "The shape of the data has changed between a save and a load")
        self.assertEqual(len(data.shape), 1, "The data is not a vector anymore")
        self.assertEqual(arr.shape[0], data.shape[0], "The lenght of the vector has changed between a save and a load")
        self.assertEqual(data.shape[0], number_of_data, "The lenght of the vector has changed between a save and a load")

        for it in range(number_of_data):
            self.assertEqual(arr[it], data[it], f"Data at {it} has changed between a save and a load")

        for section, values in safe_conf.items():
            for key, value in values.items():
                initial_conf_value = getattr(conf, key)
                if type(list) == type(initial_conf_value):
                    self.assertListEqual(initial_conf_value, value, f"Configuration value {key} in section {section} has changed")
                elif type(dict) == type(initial_conf_value):
                    self.assertDictEqual(initial_conf_value, value, f"Configuration value {key} in section {section} has changed")
                else:
                    self.assertEqual(initial_conf_value, value, f"Configuration value {key} in section {section} has changed")

