"""Flat Cartesian slab with reference-element layouts shared with the cubed sphere.

The scaffold supplies nodal and face layouts. CartesianMetric5D constructs the
independent five-dimensional metric and physical wind conversions.
"""

import torch

from .cubed_sphere_3d import CubedSphere3D
from .geometry import Geometry


class Cartesian3D(Geometry):
    """Reuse cubed-sphere element layouts without inheriting spherical geometry."""

    is_3d_euler_grid = True
    output_family = "cartesian"

    is_y_invariant_slab = True

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
        # The layout scaffold requires these sphere-only settings even for a Cartesian grid.
        if getattr(param, "vertical_coord", None) is None:
            param.vertical_coord = "gal_chen"
        if getattr(param, "depth_approx", None) is None:
            param.depth_approx = "shallow"

        sphere = CubedSphere3D(
            num_elements_horizontal,
            num_elements_vertical,
            num_solpts,
            total_num_elements_horizontal,
            0.0,
            0.0,
            0.0,
            ztop,
            process_topology,
            param,
            num_elements_x2=1,  # a 2D (x, z) problem is a thin slab: a single empty element in y
        )
        _flatten(sphere, x_extent, y_extent)
        _build_physical_coords(sphere, x_extent, y_extent)
        object.__setattr__(self, "_scaffold", sphere)

    def __getattr__(self, name):
        if name == "_scaffold":
            raise AttributeError(name)
        return getattr(self._scaffold, name)

    def to_single_block(self, a):
        return self._scaffold.to_single_block(a)


def _build_physical_coords(g, x_extent, y_extent):
    """Initial conditions and output need Cartesian metres instead of scaffold panel angles."""
    ns = g.num_solpts
    x0, x1 = (0.0, 1.0) if x_extent is None else x_extent
    y0, y1 = (0.0, 1.0) if y_extent is None else y_extent
    dx = (x1 - x0) / g.num_elements_x1
    dy = (y1 - y0) / g.num_elements_x2
    dz = g.ztop / g.num_elements_x3

    node = 0.5 * (1.0 + g.solutionPoints)
    xe = x0 + (torch.arange(g.num_elements_x1)[:, None] + node[None, :]) * dx
    ye = y0 + (torch.arange(g.num_elements_x2)[:, None] + node[None, :]) * dy
    ze = (torch.arange(g.num_elements_x3)[:, None] + node[None, :]) * dz

    # ns**3 solution points within an element are ordered (k, j, i) -> flat index
    idx = torch.arange(ns**3)
    i_idx, j_idx, k_idx = idx % ns, (idx // ns) % ns, idx // ns**2
    ne3, _, ne1 = g.num_elements_x3, g.num_elements_x2, g.num_elements_x1
    g.X1 = torch.broadcast_to(xe[:, i_idx][None, None, :, :], g.grid_shape_3d_new).copy()
    g.X2 = torch.broadcast_to(ye[:, j_idx][None, :, None, :], g.grid_shape_3d_new).copy()
    g.X3 = torch.broadcast_to(ze[:, k_idx][:, None, None, :], g.grid_shape_3d_new).copy()

    # Any y-plane represents the invariant slab for x-z output.
    ni, nk = ne1 * ns, ne3 * ns
    g.X1_cartesian = torch.broadcast_to(xe.reshape(-1)[None, :], (nk, ni)).copy()
    g.X3_cartesian = torch.broadcast_to(ze.reshape(-1)[:, None], (nk, ni)).copy()
    g.x0, g.x1 = x0, x1
    g.z0, g.z1 = 0.0, g.ztop


def _flatten(g, x_extent, y_extent):
    g.earth_radius = 1.0
    g.rotation_speed = 0.0
    g.deep = False

    g.lat_p = 0.0
    g.lon_p = 0.0
    g.angle_p = 0.0

    if x_extent is not None:
        g.delta_x1 = (x_extent[1] - x_extent[0]) / g.num_elements_x1
    if y_extent is not None:
        g.delta_x2 = (y_extent[1] - y_extent[0]) / g.num_elements_x2

    for name in (
        "gnomonic",
        "gnomonic_itf_i",
        "gnomonic_itf_j",
        "gnomonic_itf_k",
        "coordVec_gnom",
        "coordVec_gnom_itf_i",
        "coordVec_gnom_itf_j",
        "coordVec_gnom_itf_k",
    ):
        arr = getattr(g, name, None)
        if arr is not None:
            arr[0] = 0.0
            arr[1] = 0.0

    # Keep block coordinates flat while retaining the layout field delta_block
    # nonzero to avoid a divide-by-zero in the (now identically zero) Coriolis term.
    if hasattr(g, "X_block"):
        g.X_block = torch.zeros_like(g.X_block)
        g.Y_block = torch.zeros_like(g.Y_block)
        g.delta_block = torch.ones_like(g.delta_block)
        g.boundary_sn = torch.zeros_like(g.boundary_sn)
        g.boundary_we = torch.zeros_like(g.boundary_we)
        g.boundary_sn_new = torch.zeros_like(g.boundary_sn_new)
        g.boundary_we_new = torch.zeros_like(g.boundary_we_new)
