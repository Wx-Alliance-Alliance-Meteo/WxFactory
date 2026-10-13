import types
import unittest

from wx_test import WxTestCase

from wx_factory.geometry import registry
from wx_factory.geometry.registry import (
    GEOMETRY_REGISTRY,
    resolve_geometry,
)


def make_ctx(grid_type, equations, grid_file=""):
    """Minimal context: resolve_geometry only reads config.grid_file / grid_type / equations."""
    config = types.SimpleNamespace(grid_file=grid_file, grid_type=grid_type, equations=equations)
    return types.SimpleNamespace(config=config)


class GeometryRegistryTestCases(WxTestCase):

    def test_unregistered_combination_raises_helpful_error(self):
        with self.assertRaises(ValueError) as cm:
            resolve_geometry(make_ctx("klein_bottle", "euler"))
        self.assertIn("Registered combinations", str(cm.exception))

    def test_grid_file_forces_cubed_sphere_2d(self):
        # A grid file must route to the 2D cubed-sphere factory regardless of grid_type/equations.
        marker = object()
        key = registry._GRID_FILE_KEY
        original = GEOMETRY_REGISTRY[key]
        GEOMETRY_REGISTRY[key] = lambda ctx: marker
        try:
            result = resolve_geometry(make_ctx("cartesian3d", "euler", grid_file="my_grid.nc"))
            self.assertIs(result, marker)
        finally:
            GEOMETRY_REGISTRY[key] = original


if __name__ == "__main__":
    unittest.main()
