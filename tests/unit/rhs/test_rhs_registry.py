import types
import unittest

from wx_factory.geometry import Cartesian2D, CubedSphere2D, CubedSphere3D
from wx_factory.rhs import rhs_selector
from wx_factory.rhs.rhs_selector import RHS_REGISTRY, RhsBundle, register_rhs, resolve_rhs


def make_ctx(equations, geom, discretization="dfr"):
    """Minimal RhsContext-like object: resolve_rhs only reads param and type(geom)."""
    param = types.SimpleNamespace(equations=equations, discretization=discretization)
    return types.SimpleNamespace(param=param, geom=geom)


class DummyGeom:
    pass


class RhsBundleTestCases(unittest.TestCase):
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


class RhsRegistryTestCases(unittest.TestCase):
    def test_expected_combinations_are_registered(self):
        keys = set(RHS_REGISTRY)
        self.assertIn(("euler", CubedSphere3D), keys)
        self.assertIn(("euler", Cartesian2D), keys)
        self.assertIn(("shallow_water", CubedSphere2D), keys)

    def test_unknown_discretization_raises(self):
        ctx = make_ctx("euler", Cartesian2D.__new__(Cartesian2D), discretization="finite_volume")
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


if __name__ == "__main__":
    unittest.main()
