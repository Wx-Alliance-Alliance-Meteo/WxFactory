import types
import unittest

from wx_test import WxTestCase

from wx_factory.output.registry import OUTPUT_REGISTRY, register_output, resolve_output


def make_ctx(output_family, output_format):
    """Minimal context: resolve_output only reads geometry.output_family and config.output_format."""
    geometry = types.SimpleNamespace(output_family=output_family)
    config = types.SimpleNamespace(output_format=output_format)
    return types.SimpleNamespace(geometry=geometry, config=config)


class OutputRegistryTestCases(WxTestCase):
    def test_expected_combinations_are_registered(self):
        keys = set(OUTPUT_REGISTRY)
        self.assertIn(("cartesian", None), keys)
        self.assertIn(("cubesphere", "netcdf"), keys)
        self.assertIn(("cubesphere", "fst"), keys)

    def test_format_independent_family_uses_default_entry(self):
        # The cartesian family registers with format None and must match any requested format.
        marker = object()
        original = OUTPUT_REGISTRY[("cartesian", None)]
        OUTPUT_REGISTRY[("cartesian", None)] = lambda ctx: marker
        try:
            self.assertIs(resolve_output(make_ctx("cartesian", "netcdf")), marker)
            self.assertIs(resolve_output(make_ctx("cartesian", "whatever")), marker)
        finally:
            OUTPUT_REGISTRY[("cartesian", None)] = original

    def test_format_specific_dispatch(self):
        m_netcdf, m_fst = object(), object()
        saved = {k: OUTPUT_REGISTRY[k] for k in [("cubesphere", "netcdf"), ("cubesphere", "fst")]}
        OUTPUT_REGISTRY[("cubesphere", "netcdf")] = lambda ctx: m_netcdf
        OUTPUT_REGISTRY[("cubesphere", "fst")] = lambda ctx: m_fst
        try:
            self.assertIs(resolve_output(make_ctx("cubesphere", "netcdf")), m_netcdf)
            self.assertIs(resolve_output(make_ctx("cubesphere", "fst")), m_fst)
        finally:
            OUTPUT_REGISTRY.update(saved)

    def test_unknown_combination_raises_helpful_error(self):
        with self.assertRaises(ValueError) as cm:
            resolve_output(make_ctx("cubesphere", "hdf5"))
        self.assertIn("Registered combinations", str(cm.exception))

    def test_duplicate_registration_raises(self):
        @register_output("dup_family", "fmt")
        def _first(ctx):
            return None

        try:
            with self.assertRaises(ValueError):

                @register_output("dup_family", "fmt")
                def _second(ctx):
                    return None

        finally:
            del OUTPUT_REGISTRY[("dup_family", "fmt")]


if __name__ == "__main__":
    unittest.main()
