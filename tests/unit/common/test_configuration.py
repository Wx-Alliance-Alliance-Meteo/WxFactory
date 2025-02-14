import os
import unittest

import common.configuration
import common.configuration_schema


config_test_dir = "tests/data/unit/configuration_tests"


class ConfigurationTestCases(unittest.TestCase):
    def test_load_configuration_with_schema_default(self):
        schema_str: str
        configuration_str: str
        with open(os.path.join(config_test_dir, "config-format-1.json"), "rt") as f:
            schema_str = "\n".join(f.readlines())

        with open(os.path.join(config_test_dir, "config-1.ini")) as f:
            configuration_str = "\n".join(f.readlines())

        schema = common.configuration_schema.ConfigurationSchema(schema_str)
        conf = common.configuration.Configuration(configuration_str, schema, load_post_config=False)
        self.assertEqual(conf.size, 1)
        self.assertEqual(conf.time, 0.0)
        self.assertEqual(conf.end_time, 2.0)
        self.assertEqual(conf.verbose, True)
        self.assertListEqual(conf.values1, [1, 2, 3])
        self.assertListEqual(conf.values2, [1.0, 2.0])
        self.assertEqual(getattr(conf, "config-name"), "potato inc")
        self.assertEqual(conf.path, "./Potato")

    def test_load_configuration_with_valid_values(self):
        schema_str: str
        configuration_str: str
        with open(os.path.join(config_test_dir, "config-format-2.json"), "rt") as f:
            schema_str = "\n".join(f.readlines())

        with open(os.path.join(config_test_dir, "config-2.ini")) as f:
            configuration_str = "\n".join(f.readlines())

        schema = common.configuration_schema.ConfigurationSchema(schema_str)
        conf = common.configuration.Configuration(configuration_str, schema, load_post_config=False)
        self.assertEqual(conf.numeric1, 0)
        self.assertEqual(conf.numeric2, 0)
        self.assertEqual(conf.numeric3, 2)
        self.assertListEqual(conf.numeric4, [0, 1])
        self.assertListEqual(conf.numeric5, [-1])
        self.assertListEqual(conf.numeric6, [0, 2])
        self.assertEqual(conf.string1, "1")

    def test_load_configuration_with_dependency(self):
        schema_str: str
        configuration_str: str
        with open(os.path.join(config_test_dir, "config-format-3.json"), "rt") as f:
            schema_str = "\n".join(f.readlines())

        with open(os.path.join(config_test_dir, "config-3.1.ini")) as f:
            configuration_str = "\n".join(f.readlines())

        schema = common.configuration_schema.ConfigurationSchema(schema_str)
        conf = common.configuration.Configuration(configuration_str, schema, load_post_config=False)
        self.assertEqual(conf.x, 1)
        self.assertEqual(conf.y, 2)

        with open(os.path.join(config_test_dir, "config-3.2.ini")) as f:
            configuration_str = "\n".join(f.readlines())

        conf = common.configuration.Configuration(configuration_str, schema, load_post_config=False)
        self.assertEqual(conf.x, 5)
        self.assertFalse(hasattr(conf, "y"))
