import types
import unittest

from wx_factory.geometry import registry
from wx_factory.geometry.registry import GEOMETRY_REGISTRY, register_geometry, resolve_geometry


def make_ctx(grid_type, equations, grid_file=""):
    """Minimal context: resolve_geometry only reads config.grid_file / grid_type / equations."""
    config = types.SimpleNamespace(grid_file=grid_file, grid_type=grid_type, equations=equations)
    return types.SimpleNamespace(config=config)


class GeometryRegistryTestCases(unittest.TestCase):
    def test_expected_combinations_are_registered(self):
        keys = set(GEOMETRY_REGISTRY)
        self.assertIn(("cubed_sphere", "euler"), keys)
        self.assertIn(("cubed_sphere", "shallow_water"), keys)
        self.assertIn(("cartesian2d", "euler"), keys)

    def test_unregistered_combination_raises_helpful_error(self):
        with self.assertRaises(ValueError) as cm:
            resolve_geometry(make_ctx("klein_bottle", "euler"))
        self.assertIn("Registered combinations", str(cm.exception))

    def test_resolve_dispatches_to_registered_factory(self):
        marker = object()

        @register_geometry("test_grid", "test_eq")
        def _factory(ctx):
            return marker

        try:
            self.assertIs(resolve_geometry(make_ctx("test_grid", "test_eq")), marker)
        finally:
            del GEOMETRY_REGISTRY[("test_grid", "test_eq")]

    def test_grid_file_forces_cubed_sphere_2d(self):
        # A grid file must route to the 2D cubed-sphere factory regardless of grid_type/equations.
        marker = object()
        key = registry._GRID_FILE_KEY
        original = GEOMETRY_REGISTRY[key]
        GEOMETRY_REGISTRY[key] = lambda ctx: marker
        try:
            result = resolve_geometry(make_ctx("cartesian2d", "euler", grid_file="my_grid.nc"))
            self.assertIs(result, marker)
        finally:
            GEOMETRY_REGISTRY[key] = original

    def test_duplicate_registration_raises(self):
        @register_geometry("dup_grid", "dup_eq")
        def _first(ctx):
            return None

        try:
            with self.assertRaises(ValueError):

                @register_geometry("dup_grid", "dup_eq")
                def _second(ctx):
                    return None

        finally:
            del GEOMETRY_REGISTRY[("dup_grid", "dup_eq")]


if __name__ == "__main__":
    unittest.main()
