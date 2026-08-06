import types
import unittest

from wx_test import WxTestCase

from wx_factory import integrators
from wx_factory.geometry import Cartesian3D, CubedSphere2D, CubedSphere3D
from wx_factory.rhs.rhs_selector import (
    RHS_REGISTRY,
    RhsBundle,
    register_rhs,
    resolve_rhs,
)


def make_ctx(equations, geom, discretization="dfr"):
    """Minimal RhsContext-like object: resolve_rhs only reads param and type(geom)."""
    param = types.SimpleNamespace(equations=equations, discretization=discretization)
    return types.SimpleNamespace(param=param, geom=geom)


class DummyGeom:
    pass


class RhsBundleTestCases(WxTestCase):
    def test_full_and_shape_are_stored(self):
        sentinel = object()
        bundle = RhsBundle(full=sentinel, shape=(2, 3))
        self.assertIs(bundle.full, sentinel)
        self.assertEqual(bundle.shape, (2, 3))

    def test_missing_partitions_raise_when_called(self):
        bundle = RhsBundle(full=lambda q: q, shape=(1,))
        with self.assertRaises(NotImplementedError):
            bundle.explicit(0)
        with self.assertRaises(NotImplementedError):
            bundle.implicit(0)

    def test_provided_partitions_are_used(self):
        exp = lambda q: "e"
        imp = lambda q: "i"
        bundle = RhsBundle(full=lambda q: q, shape=(1,), explicit=exp, implicit=imp)
        self.assertIs(bundle.explicit, exp)
        self.assertIs(bundle.implicit, imp)

    def test_has_partition_reports_availability(self):
        with_partition = RhsBundle(full=lambda q: q, shape=(1,), explicit=lambda q: q, implicit=lambda q: q)
        self.assertTrue(with_partition.has_partition)
        self.assertIsNone(with_partition.partition_reason)

        without = RhsBundle(full=lambda q: q, shape=(1,))
        self.assertFalse(without.has_partition)
        self.assertTrue(without.partition_reason)

    def test_reason_reaches_the_user(self):
        reason = "my equations have no stiff part to separate"
        bundle = RhsBundle(full=lambda q: q, shape=(1,), partition_reason=reason)
        self.assertEqual(bundle.partition_reason, reason)
        with self.assertRaises(NotImplementedError) as raised:
            bundle.implicit(0)
        self.assertIn(reason, str(raised.exception))

    def test_half_a_partition_is_rejected(self):
        with self.assertRaises(ValueError):
            RhsBundle(full=lambda q: q, shape=(1,), explicit=lambda q: q)
        with self.assertRaises(ValueError):
            RhsBundle(full=lambda q: q, shape=(1,), implicit=lambda q: q)


class RhsRegistryTestCases(WxTestCase):
    def test_expected_combinations_are_registered(self):
        keys = set(RHS_REGISTRY)
        self.assertIn(("euler", CubedSphere3D), keys)
        self.assertIn(("euler", Cartesian3D), keys)
        self.assertIn(("shallow_water", CubedSphere2D), keys)

    def test_unknown_discretization_raises(self):
        ctx = make_ctx("euler", Cartesian3D.__new__(Cartesian3D), discretization="finite_volume")
        with self.assertRaises(ValueError):
            resolve_rhs(ctx)

    def test_unregistered_combination_raises_helpful_error(self):
        ctx = make_ctx("shallow_water", DummyGeom())
        with self.assertRaises(ValueError) as cm:
            resolve_rhs(ctx)
        # The message should list what is available, to guide the user.
        self.assertIn("Registered combinations", str(cm.exception))

    def test_resolve_dispatches_to_registered_factory(self):
        marker = object()

        @register_rhs("test_eq", DummyGeom)
        def _factory(ctx):
            return marker

        try:
            result = resolve_rhs(make_ctx("test_eq", DummyGeom()))
            self.assertIs(result, marker)
        finally:
            del RHS_REGISTRY[("test_eq", DummyGeom)]

    def test_duplicate_registration_raises(self):
        @register_rhs("dup_eq", DummyGeom)
        def _first(ctx):
            return None

        try:
            with self.assertRaises(ValueError):

                @register_rhs("dup_eq", DummyGeom)
                def _second(ctx):
                    return None

        finally:
            del RHS_REGISTRY[("dup_eq", DummyGeom)]


class PartitionedIntegratorTestCases(WxTestCase):
    def test_registries_are_disjoint(self):
        self.assertEqual(set(integrators.REGISTRY) & set(integrators.PARTITIONED_REGISTRY), set())

    def test_known_partitioned_schemes(self):
        self.assertIn("partrosexp2", integrators.PARTITIONED_REGISTRY)
        self.assertIn("imex2", integrators.PARTITIONED_REGISTRY)
        self.assertIn("tvdrk3", integrators.REGISTRY)

    def test_partitioned_scheme_without_partition_is_refused(self):
        reason = "the partitioned right-hand side is not implemented for these equations"
        bundle = RhsBundle(full=lambda q: q, shape=(1,), partition_reason=reason)
        with self.assertRaises(ValueError) as raised:
            integrators.resolve("partrosexp2", None, bundle, None, None)
        message = str(raised.exception)
        self.assertIn("partrosexp2", message)
        self.assertIn(reason, message)
        self.assertIn("tvdrk3", message)

    def test_unknown_scheme_lists_both_registries(self):
        bundle = RhsBundle(full=lambda q: q, shape=(1,), partition_reason="none here")
        with self.assertRaises(ValueError) as raised:
            integrators.resolve("no_such_scheme", None, bundle, None, None)
        message = str(raised.exception)
        self.assertIn("tvdrk3", message)
        self.assertIn("partrosexp2", message)


if __name__ == "__main__":
    unittest.main()
