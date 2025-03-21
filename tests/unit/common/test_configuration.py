import os
import glob
import unittest

from common import Configuration, ConfigurationSchema, readfile, ConfigValueError


self_dir = os.path.dirname(os.path.realpath(__file__))
config_test_dir = self_dir


class ConfigurationTestCases(unittest.TestCase):
    def test_load_configuration_with_schema_default(self):
        schema_str: str
        configuration_str: str
        with open(os.path.join(config_test_dir, "config-format-1.json"), "rt") as f:
            schema_str = "\n".join(f.readlines())

        with open(os.path.join(config_test_dir, "config-1.ini")) as f:
            configuration_str = "\n".join(f.readlines())

        schema = ConfigurationSchema(schema_str)
        conf = Configuration(configuration_str, schema, load_post_config=False)
        self.assertEqual(conf.size, 1)
        self.assertEqual(conf.time, 0.0)
        self.assertEqual(conf.end_time, 2.0)
        self.assertEqual(conf.verbose, True)
        self.assertListEqual(conf.values1, [1, 2, 3])
        self.assertListEqual(conf.values2, [1.0, 2.0])
        self.assertEqual(getattr(conf, "config-name"), "potato inc")
        self.assertEqual(conf.path, "./Potato")

    def test_load_configuration_with_valid_values(self):
        schema_str = readfile(os.path.join(config_test_dir, "config-format-2.json"))
        configuration_str = readfile(os.path.join(config_test_dir, "config-2.ini"))

        schema = ConfigurationSchema(schema_str)
        conf = Configuration(configuration_str, schema, load_post_config=False)
        self.assertEqual(conf.int1, 0)
        self.assertEqual(conf.int2, 0)
        self.assertEqual(conf.int3, 2)
        self.assertListEqual(conf.intlist4, [0, 1])
        self.assertListEqual(conf.intlist5, [-1])
        self.assertListEqual(conf.intlist6, [0, 2])
        self.assertEqual(conf.string1, "1")
        self.assertFalse(conf.bool1)
        self.assertFalse(conf.bool2)
        self.assertTrue(conf.bool3)

    def test_load_configuration_with_invalid_values(self):
        schema = ConfigurationSchema(readfile(os.path.join(config_test_dir, "config-format-2.json")))

        fail_files = sorted(glob.glob(config_test_dir + "/config-2_fail-*.ini"))

        for conf_file in fail_files:
            print(f"testing config file {conf_file}")
            content = readfile(conf_file)
            try:
                Configuration(content, schema, load_post_config=False)
                self.fail("Should have failed!")
            except ConfigValueError as e:
                print(e)

    def test_load_configuration_with_dependency(self):
        schema_str: str
        configuration_str: str
        with open(os.path.join(config_test_dir, "config-format-3.json"), "rt") as f:
            schema_str = "\n".join(f.readlines())

        with open(os.path.join(config_test_dir, "config-3.1.ini")) as f:
            configuration_str = "\n".join(f.readlines())

        schema = ConfigurationSchema(schema_str)
        conf = Configuration(configuration_str, schema, load_post_config=False)
        self.assertEqual(conf.x, 1)
        self.assertEqual(conf.y, 2)

        with open(os.path.join(config_test_dir, "config-3.2.ini")) as f:
            configuration_str = "\n".join(f.readlines())

        conf = Configuration(configuration_str, schema, load_post_config=False)
        self.assertEqual(conf.x, 5)
        self.assertFalse(hasattr(conf, "y"))
