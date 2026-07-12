import unittest

from wx_factory.common import config_hints


class ConfigHintsTestCases(unittest.TestCase):
    def test_type_hints_are_up_to_date(self):
        """The generated type-hint block in configuration.py must match the schema.

        If this fails, the schema was changed without regenerating the annotations. Run
        `python -m wx_factory.common.config_hints --write` and commit the result."""
        schema = config_hints.load_schema()
        diff = config_hints.check(schema)
        self.assertEqual(
            diff,
            [],
            msg="Configuration type hints are out of date; regenerate with "
            "`python -m wx_factory.common.config_hints --write`.\n" + "".join(diff),
        )


if __name__ == "__main__":
    unittest.main()
