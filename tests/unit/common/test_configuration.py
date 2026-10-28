import glob
import os
import warnings
from configparser import ConfigParser
from pathlib import Path
from tempfile import TemporaryDirectory

from wx_factory.common.eval_expr import _math_constants

_math_constants["e"] = 1
_math_constants["f"] = 5
from wx_test import WxTestCase

from wx_factory.common import (
    Configuration,
    ConfigurationSchema,
    ConfigValueError,
    load_default_schema,
    readfile,
)

self_dir = os.path.dirname(os.path.realpath(__file__))
config_test_dir = self_dir


class ConfigurationTestCases(WxTestCase):
    def test_unknown_and_misplaced_options_are_reported_with_suggestions(self):
        content = readfile(str(Path(__file__).resolve().parents[3] / "config/case5.ini"))
        content = content.replace("[Test_case]", "[Test_case]\ninitial_condition = input.nc\nunrecognized_key = 1")
        content = content.replace("[Grid]", "[Grid]\ndt = 7")
        with self.assertWarns(UserWarning) as caught:
            config = Configuration(content, load_default_schema())
        message = str(caught.warning)
        self.assertIn("[Test_case] initial_condition; did you mean [Test_case] initial_conditions?", message)
        self.assertIn("[Test_case] unrecognized_key", message)
        self.assertIn("[Grid] dt; did you mean [Time_integration] dt?", message)
        self.assertEqual(config.dt, 1800)
        self.assertFalse(hasattr(config, "initial_condition"))

    def test_unknown_option_is_reported_before_a_required_value_fails(self):
        content = readfile(str(Path(__file__).resolve().parents[3] / "config/case5.ini"))
        content = content.replace("dt = 1800", "dtt = 1800")
        with (
            self.assertWarnsRegex(UserWarning, r"dtt; did you mean \[Time_integration\] dt"),
            self.assertRaises(ConfigValueError),
        ):
            Configuration(content, load_default_schema())

    def test_inactive_options_and_case_insensitive_names_are_recognized(self):
        content = readfile(str(Path(__file__).resolve().parents[3] / "config/case5.ini"))
        content = content.replace("[Test_case]", "[Test_case]\nINITIAL_CONDITIONS = input.nc")
        content = content.replace("dt = 1800", "DT = 1800")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = Configuration(content, load_default_schema())
        self.assertEqual(caught, [])
        self.assertEqual(config.dt, 1800)
        self.assertFalse(hasattr(config, "initial_conditions"))

    def test_default_options_are_checked_without_inherited_duplicates(self):
        content = readfile(str(Path(__file__).resolve().parents[3] / "config/case5.ini"))
        content = "[DEFAULT]\ntolerance = 1e-6\nmisspelled_option = 1\n" + content
        content = content.replace("tolerance = 1e-7\n", "")
        with self.assertWarns(UserWarning) as caught:
            config = Configuration(content, load_default_schema())
        self.assertEqual(str(caught.warning).count("misspelled_option"), 1)
        self.assertNotIn("tolerance", str(caught.warning))
        self.assertEqual(config.tolerance, 1e-6)

    def test_era5_examples_load_the_configured_initial_conditions(self):
        config_dir = Path(__file__).resolve().parents[3] / "config"
        for name in ("ERA5_2_CS.ini", *(f"ERA5_CS_2020{month:02d}.ini" for month in range(1, 5))):
            with self.subTest(config=name):
                content = readfile(str(config_dir / name))
                parser = ConfigParser()
                parser.read_string(content)
                expected = parser.get("Test_case", "initial_conditions")
                field = next(field for field in load_default_schema().fields if field.name == "initial_conditions")
                self.assertTrue(expected)
                self.assertEqual(field.read(parser), expected)

    def test_default_schema_outside_source_directory(self):
        previous = os.getcwd()
        with TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.assertTrue(load_default_schema().fields)
            finally:
                os.chdir(previous)

    def test_main_precision_option(self):
        schema_file = os.path.join(config_test_dir, "../../../wx_factory/common/config-format.json")
        schema = ConfigurationSchema(readfile(schema_file))
        precision = next(field for field in schema.fields if field.name == "precision")

        self.assertEqual(precision.section, "Device")
        self.assertTrue(precision.validate("double"))
        self.assertTrue(precision.validate("mixed"))
        self.assertFalse(precision.validate("single"))

    def test_load_configuration_with_schema_default(self):
        schema_str: str
        configuration_str: str
        with open(os.path.join(config_test_dir, "config-format-1.json"), "rt") as f:
            schema_str = "\n".join(f.readlines())

        with open(os.path.join(config_test_dir, "config-1.ini")) as f:
            configuration_str = "\n".join(f.readlines())

        schema = ConfigurationSchema(schema_str)
        conf = Configuration(configuration_str, schema)
        self.assertEqual(conf.size, 1)
        self.assertEqual(conf.time, 0.0)
        self.assertEqual(conf.end_time, 2.0)
        self.assertEqual(conf.verbose, True)
        self.assertEqual(conf.bool1, True)
        self.assertEqual(conf.bool2, True)
        self.assertEqual(conf.bool3, True)
        self.assertEqual(conf.bool4, True)
        self.assertEqual(conf.bool5, False)
        self.assertEqual(conf.bool6, False)
        self.assertEqual(conf.bool7, False)
        self.assertEqual(conf.bool8, False)
        self.assertListEqual(conf.values1, [1, 2, 3])
        self.assertListEqual(conf.values2, [1.0, 2.0])
        self.assertEqual(getattr(conf, "config-name"), "potato inc")
        self.assertEqual(conf.path, "./Potato")

    def test_load_configuration_with_valid_values(self):
        schema_file = os.path.join(config_test_dir, "config-format-2.json")
        config_file = os.path.join(config_test_dir, "config-2.ini")

        try:
            schema = ConfigurationSchema(readfile(schema_file))
        except (OSError, ValueError):
            raise ValueError(f"Could not read and parse schema file {schema_file}")
        try:
            conf = Configuration(readfile(config_file), schema)
        except Exception as e:
            raise ValueError(f"Could not read and parse configuration file {config_file}") from e

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
                Configuration(content, schema)
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
        conf = Configuration(configuration_str, schema)
        self.assertEqual(conf.x, 1)
        self.assertEqual(conf.y, 2)

        with open(os.path.join(config_test_dir, "config-3.2.ini")) as f:
            configuration_str = "\n".join(f.readlines())

        conf = Configuration(configuration_str, schema)
        self.assertEqual(conf.x, 5)
        self.assertFalse(hasattr(conf, "y"))

    def test_load_configuration_with_expression(self):
        schema_str: str
        configuration_str: str
        with open(os.path.join(config_test_dir, "config-format-4.json"), "rt") as f:
            schema_str = "\n".join(f.readlines())

        with open(os.path.join(config_test_dir, "config-4.ini")) as f:
            configuration_str = "\n".join(f.readlines())

        schema = ConfigurationSchema(schema_str)
        conf = Configuration(configuration_str, schema)
        self.assertEqual(conf.x, _math_constants["pi"])
        self.assertEqual(conf.x2, _math_constants["pi"])
        self.assertEqual(conf.y, _math_constants["pi"] + 1)
        self.assertEqual(conf.z, _math_constants["pi"] * 2)
        self.assertEqual(conf.a, _math_constants["e"] + 4)
        self.assertEqual(conf.b, [_math_constants["e"] + 1, _math_constants["f"] + 2])
        self.assertEqual(conf.c, [_math_constants["pi"] + 1, _math_constants["pi"] - 1])
        self.assertEqual(conf.c2, [_math_constants["pi"], _math_constants["pi"] + 1])
        self.assertEqual(conf.d, _math_constants["pi"])
        self.assertEqual(conf.e, 2)
