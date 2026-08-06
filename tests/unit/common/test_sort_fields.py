import unittest

from wx_test import WxTestCase

from wx_factory.common.configuration_schema import (
    ConfigFieldRange,
    ConfigurationField,
    ConfigValueError,
    sort_fields_by_dependency,
)


def make_field(name, dependency=None):
    """Build a minimal ConfigurationField for ordering tests."""
    return ConfigurationField(
        field_name=name,
        field_section="test",
        field_default=None,
        field_type=int,
        is_list=False,
        valid_range=ConfigFieldRange(),
        dependency=dependency,
        description="",
    )


def names(fields):
    return [f.name for f in fields]


class SortFieldsByDependencyTestCases(WxTestCase):
    def test_no_dependencies_preserves_order(self):
        fields = [make_field("a"), make_field("b"), make_field("c")]
        self.assertEqual(names(sort_fields_by_dependency(fields)), ["a", "b", "c"])

    def test_dependent_field_placed_after_target(self):
        fields = [make_field("gated", dependency=("base", ["x"])), make_field("base")]
        out = names(sort_fields_by_dependency(fields))
        self.assertEqual(set(out), {"base", "gated"})
        self.assertLess(out.index("base"), out.index("gated"))

    def test_multi_level_chain_is_ordered(self):
        # c depends on b, b depends on a; provided in reverse order
        fields = [
            make_field("c", dependency=("b", ["x"])),
            make_field("b", dependency=("a", ["x"])),
            make_field("a"),
        ]
        out = names(sort_fields_by_dependency(fields))
        self.assertLess(out.index("a"), out.index("b"))
        self.assertLess(out.index("b"), out.index("c"))

    def test_no_field_is_emitted_more_than_once(self):
        # Reproduces the historical duplication bug: a gated field whose target is itself gated.
        fields = [
            make_field("leaf", dependency=("mid", ["x"])),
            make_field("mid", dependency=("root", ["x"])),
            make_field("root"),
            make_field("plain"),
        ]
        out = names(sort_fields_by_dependency(fields))
        self.assertEqual(sorted(out), ["leaf", "mid", "plain", "root"])
        self.assertEqual(len(out), len(set(out)))

    def test_unresolvable_dependency_is_kept_not_dropped(self):
        # 'orphan' depends on a field that does not exist; it must still appear in the output.
        fields = [make_field("a"), make_field("orphan", dependency=("missing", ["x"]))]
        out = names(sort_fields_by_dependency(fields))
        self.assertIn("orphan", out)
        self.assertEqual(set(out), {"a", "orphan"})

    def test_dependency_cycle_does_not_hang_and_keeps_fields(self):
        fields = [
            make_field("x", dependency=("y", ["v"])),
            make_field("y", dependency=("x", ["v"])),
            make_field("free"),
        ]
        out = names(sort_fields_by_dependency(fields))
        self.assertEqual(set(out), {"x", "y", "free"})
        self.assertEqual(len(out), 3)

    def test_duplicate_name_raises(self):
        fields = [make_field("dup"), make_field("dup")]
        with self.assertRaises(ConfigValueError):
            sort_fields_by_dependency(fields)


if __name__ == "__main__":
    unittest.main()
