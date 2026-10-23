"""Freeze a valid 5D cubed-sphere metric for local principal-operator tests.

This is a periodic unit fixture, not a Cartesian geometry implementation.
The full production RhsEuler and its interface solver operate on frozen fields.
"""

from dataclasses import fields
from types import SimpleNamespace

import sympy
import torch

from wx_factory.geometry.metric5d import Metric5DFields, evaluate_metric5d, panel_frame
from wx_factory.geometry.operators import DFROperators
from wx_factory.geometry.quadrature import gauss_legendre
from wx_factory.rhs.rhs_euler import RhsEuler


class PeriodicUnitExchange:
    def start_exchange_euler_3d(self, south, north, west, east, *_args, **_kwargs):
        return SimpleNamespace(wait=lambda: (north, south, east, west))


def frozen_problem(elements=24, nodes=4, length=80000.0, speed=20.0, vertical_elements=1, terrain_slope=0.0):
    symbolic, points, weights = gauss_legendre(nodes)
    grid = SimpleNamespace(
        num_solpts=nodes,
        itf_size=nodes**2,
        solutionPoints=points,
        glweights=weights,
        extension_sym=[sympy.sympify(-1), *symbolic, sympy.sympify(1)],
        is_3d_euler_grid=True,
        boundary_sn_new=None,
        boundary_we_new=None,
    )
    context = SimpleNamespace(real_dtype=torch.float64)
    grid.context = context
    operators = DFROperators(grid, context)
    half_width = length / (2 * elements)
    point = torch.zeros((3, 1), dtype=torch.float64)
    template = evaluate_metric5d(
        point,
        lambda q: q[2] + terrain_slope * q[0],
        radius=1.0,
        rotation_speed=0.0,
        frame=panel_frame(0.0, 0.0, 0.0, point),
        scales=(half_width, 1.0, 1.0),
        deep=False,
        gravity_surface=0.0,
    )

    def expand(shape):
        values = {}
        for field in fields(Metric5DFields):
            value = getattr(template, field.name)[..., 0]
            values[field.name] = value.reshape(value.shape + (1,) * len(shape)).expand(value.shape + shape).clone()
        return Metric5DFields(**values)

    metric = SimpleNamespace(
        volume=expand((vertical_elements, 1, elements, nodes**3)),
        faces=(
            expand((vertical_elements, 1, elements + 2, 2 * nodes**2)),
            expand((vertical_elements, 3, elements, 2 * nodes**2)),
            expand((vertical_elements + 2, 1, elements, 2 * nodes**2)),
        ),
    )
    config = SimpleNamespace(case_number=0, riemann_mach_cutoff=0.3, vertical_boundary="wall")
    rhs = RhsEuler(grid, operators, metric, PeriodicUnitExchange(), config)
    state = torch.zeros((5, vertical_elements, 1, elements, nodes**3), dtype=torch.float64)
    state[0], state[4] = 1.0, 300.0
    state[1] = half_width * speed
    centers = (torch.arange(elements, dtype=torch.float64) + 0.5) * 2 * half_width
    positions = centers[:, None] + half_width * points[None, :]
    return rhs, state, positions, points, weights, half_width


def transverse_direction(profile, state):
    direction = torch.zeros_like(state)
    direction[2] = profile.repeat(1, state.shape[-1] // profile.shape[-1]).reshape(state.shape[1:])
    return direction


def central_shear_matrix(rhs, elements, speed, half_width):
    """Independent one-dimensional periodic DFR advection matrix (column action)."""
    nodes = rhs.geom.num_solpts
    identity = torch.eye(elements * nodes, dtype=torch.float64)
    profiles = identity.reshape(elements * nodes, elements, nodes)
    west = profiles @ rhs.ops.extrap_west
    east = profiles @ rhs.ops.extrap_east
    common = (east + west.roll(-1, dims=1)) / 2
    faces = torch.stack((common.roll(1, dims=1), common), dim=-1)
    action = -speed / half_width * (profiles @ rhs.ops.diff_solpt_tr + faces @ rhs.ops.correction_tr)
    return action.reshape(elements * nodes, elements * nodes).T.contiguous()
