"""Flat 3D cartesian slab geometry, built as the identity-metric limit of the cubed sphere.

The PartRosExp2 partition and its analytic Jacobians are metric-array-driven, not geometry-aware:
they read sqrtG / h_contra / christoffel and multiply. A cartesian slab is exactly the special case
of the cubed-sphere metric with the gnomonic projection removed (X = Y = 0, so the panel-distortion
factor delta^2 = 1 + X^2 + Y^2 collapses to 1), the sphere flattened (shallow metric with A = 1, so
there is no R = height + radius term to lose to round-off in single precision), and no rotation.

A flat slab is not, conceptually, *a kind of* cubed sphere, so Cartesian3D does NOT inherit
CubedSphere3D. Instead it **composes** one: a cubed-sphere scaffold is built purely to reuse the
~1300 lines of DG layout / interface / coordinate / topology construction, then its metric-defining
coordinates are flattened to the identity-metric limit. Cartesian3D delegates the geometry interface
(``_to_new``, ``gnomonic``, shapes, ...) to that flattened scaffold. The metric (built afterwards, in
metric3d) evaluates its existing closed form at the flat limit and produces the exact terrain-
following cartesian metric -- christoffel == 0, h^12 == 0, h^11 == 4 / delta_x1^2 -- with no new math.
"""

import torch
from .cubed_sphere_3d import CubedSphere3D
from .geometry import Geometry


class Cartesian3D(Geometry):
    """A flat cartesian slab. Its own Geometry, composing (not inheriting) a flattened cubed sphere."""

    # Marks a 3D Euler DG grid (as CubedSphere3D does) for the operator / output code that needs to
    # distinguish 3D from 2D without depending on the concrete class.
    is_3d_euler_grid = True
    output_family = "cartesian"  # produces x-z images (a cartesian slab has no cube panels to write)

    def __init__(
        self,
        num_elements_horizontal: int,
        num_elements_vertical: int,
        num_solpts: int,
        total_num_elements_horizontal: int,
        ztop: float,
        process_topology,
        param,
        x_extent=None,
        y_extent=None,
    ):
        # CubedSphere3D.__init__ reads a few sphere-only config fields; a flat slab has trivial values
        # for them (no terrain-following coordinate, shallow-atmosphere metric). Fill them in if the
        # cartesian config omitted them. (They are discarded by _flatten anyway.)
        if getattr(param, "vertical_coord", None) is None:
            param.vertical_coord = "gal_chen"
        if getattr(param, "depth_approx", None) is None:
            param.depth_approx = "shallow"

        # Build the cubed-sphere scaffold (DG layout, interfaces, _to_new, coordinates, topology). The
        # rotation angles are irrelevant -- _flatten discards the sphere.
        sphere = CubedSphere3D(
            num_elements_horizontal,
            num_elements_vertical,
            num_solpts,
            total_num_elements_horizontal,
            0.0,  # lambda0
            0.0,  # phi0
            0.0,  # alpha0
            ztop,
            process_topology,
            param,
            num_elements_x2=1,  # a 2D (x, z) problem is a thin slab: a single empty element in y
        )
        _flatten(sphere, x_extent, y_extent)
        _build_physical_coords(sphere, x_extent, y_extent)
        # Composition, not inheritance: delegate the geometry interface to the flattened scaffold.
        object.__setattr__(self, "_scaffold", sphere)

    def __getattr__(self, name):
        # Called only when normal attribute lookup fails; forward to the composed scaffold.
        if name == "_scaffold":
            raise AttributeError(name)
        return getattr(self._scaffold, name)

    def to_single_block(self, a):
        """Geometry's one abstract method; forwarded to the scaffold's layout implementation."""
        return self._scaffold.to_single_block(a)


def _build_physical_coords(g, x_extent, y_extent):
    """Physical cartesian coordinates X1 (x), X2 (y), X3 (z) in the new element layout, on ``g``.

    The cubed-sphere scaffold builds angular x1/x2; the flat slab needs true metres for initial
    conditions and output. Each element spans a physical width, and within it the solution points sit
    at the Gauss-Legendre nodes (mapped from the reference [-1, 1] to [0, delta])."""
    ns = g.num_solpts
    x0, x1 = (0.0, 1.0) if x_extent is None else x_extent
    y0, y1 = (0.0, 1.0) if y_extent is None else y_extent
    dx = (x1 - x0) / g.num_elements_x1
    dy = (y1 - y0) / g.num_elements_x2
    dz = g.ztop / g.num_elements_x3

    # reference solution points mapped to [0, delta] within one element
    node = 0.5 * (1.0 + g.solutionPoints)  # in [0, 1]
    xe = x0 + (torch.arange(g.num_elements_x1)[:, None] + node[None, :]) * dx  # (ne1, ns)
    ye = y0 + (torch.arange(g.num_elements_x2)[:, None] + node[None, :]) * dy  # (ne2, ns)
    ze = (torch.arange(g.num_elements_x3)[:, None] + node[None, :]) * dz        # (ne3, ns)

    # ns**3 solution points within an element are ordered (k, j, i) -> flat index
    idx = torch.arange(ns**3)
    i_idx, j_idx, k_idx = idx % ns, (idx // ns) % ns, idx // ns**2
    ne3, ne2, ne1 = g.num_elements_x3, g.num_elements_x2, g.num_elements_x1
    g.X1 = torch.broadcast_to(xe[:, i_idx][None, None, :, :], g.grid_shape_3d_new).copy()
    g.X2 = torch.broadcast_to(ye[:, j_idx][None, :, None, :], g.grid_shape_3d_new).copy()
    g.X3 = torch.broadcast_to(ze[:, k_idx][:, None, None, :], g.grid_shape_3d_new).copy()

    # 2D (x, z) plane in the single-block layout (nk, ni), for x-z image output. The slab is
    # y-invariant, so any y-plane is representative. x runs along ni (element-major), z along nk.
    ni, nk = ne1 * ns, ne3 * ns
    g.X1_cartesian = torch.broadcast_to(xe.reshape(-1)[None, :], (nk, ni)).copy()
    g.X3_cartesian = torch.broadcast_to(ze.reshape(-1)[:, None], (nk, ni)).copy()
    g.x0, g.x1 = x0, x1
    g.z0, g.z1 = 0.0, g.ztop


def _flatten(g, x_extent, y_extent):
    """Replace the sphere's curvature/rotation with the flat cartesian limit, in place on ``g``."""
    # A flat slab: unit "radius" (shallow metric uses A = earth_radius, so A = 1 keeps the horizontal
    # metric height-independent and single-precision clean), and no rotation.
    g.earth_radius = 1.0
    g.rotation_speed = 0.0
    g.deep = False

    # Identity grid rotation -> the christoffel "rotation" (zero-index) terms vanish.
    g.lat_p = 0.0
    g.lon_p = 0.0
    g.angle_p = 0.0

    # Physical element widths, so h^11 = 4 / delta_x1^2 = (2 / dx_phys)^2.
    if x_extent is not None:
        g.delta_x1 = (x_extent[1] - x_extent[0]) / g.num_elements_x1
    if y_extent is not None:
        g.delta_x2 = (y_extent[1] - y_extent[0]) / g.num_elements_x2

    # Zero the gnomonic horizontal coordinates X, Y (keep Z = height) everywhere the metric reads
    # them, so delta^2 = 1 + X^2 + Y^2 collapses to 1 and every horizontal derivative of the metric is
    # exactly zero -> no curvature christoffels.
    for name in (
        "gnomonic", "gnomonic_itf_i", "gnomonic_itf_j", "gnomonic_itf_k",
        "coordVec_gnom", "coordVec_gnom_itf_i", "coordVec_gnom_itf_j", "coordVec_gnom_itf_k",
    ):
        arr = getattr(g, name, None)
        if arr is not None:
            arr[0] = 0.0
            arr[1] = 0.0

    # Block coordinates feed the (rotation) christoffels and Coriolis; zero them, keep delta_block
    # nonzero to avoid a divide-by-zero in the (now identically zero) Coriolis term.
    if hasattr(g, "X_block"):
        g.X_block = torch.zeros_like(g.X_block)
        g.Y_block = torch.zeros_like(g.Y_block)
        g.delta_block = torch.ones_like(g.delta_block)
        g.boundary_sn = torch.zeros_like(g.boundary_sn)
        g.boundary_we = torch.zeros_like(g.boundary_we)
        g.boundary_sn_new = torch.zeros_like(g.boundary_sn_new)
        g.boundary_we_new = torch.zeros_like(g.boundary_we_new)
