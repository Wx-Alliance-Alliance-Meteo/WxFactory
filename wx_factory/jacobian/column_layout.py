"""Conversions between grid and vertical-column layouts.

The column layout has shape ``(num_columns, num_elem_z, num_var * num_solpts)``.
"""

from typing import NamedTuple

import torch

from ..common.definitions import idx_rho, idx_rho_theta, idx_rho_u1, idx_rho_u2, idx_rho_u3


class ColumnDims(NamedTuple):
    """Sizes needed to move between the grid layout and the column layout."""

    num_solpts: int  # solution points along one element edge
    num_solpts_2d: int  # solution points on one horizontal slab of an element (num_solpts**2)
    num_var: int  # conserved variables (5 for the Euler equations)
    num_elem_z: int  # vertical elements, i.e. block rows per column
    num_elem_y: int
    num_elem_x: int
    block_size: int  # unknowns per element of a column: num_var * num_solpts
    num_columns: int  # independent vertical columns on this rank


def column_dims(rhsobj, q) -> ColumnDims:
    """Return the :class:`ColumnDims` of a state array ``q`` in grid layout."""
    num_solpts = rhsobj.geom.num_solpts
    num_var, num_elem_z, num_elem_y, num_elem_x, _ = q.shape
    return ColumnDims(
        num_solpts=num_solpts,
        num_solpts_2d=num_solpts * num_solpts,
        num_var=num_var,
        num_elem_z=num_elem_z,
        num_elem_y=num_elem_y,
        num_elem_x=num_elem_x,
        block_size=num_var * num_solpts,
        num_columns=num_elem_y * num_elem_x * num_solpts * num_solpts,
    )


def state_to_columns(rhsobj, x):
    """Convert a state array from grid layout to column layout."""
    d = column_dims(rhsobj, x)
    # Move horizontal element and point indices ahead of the vertical indices.
    split = x.reshape(d.num_var, d.num_elem_z, d.num_elem_y, d.num_elem_x, d.num_solpts, d.num_solpts_2d)
    permuted = torch.permute(split, (2, 3, 5, 1, 0, 4))
    return permuted.reshape(d.num_columns, d.num_elem_z, d.block_size)


def columns_to_state(rhsobj, x_col, reference):
    """Inverse of :func:`state_to_columns`; ``reference`` supplies the grid-layout shape."""
    d = column_dims(rhsobj, reference)
    unflattened = x_col.reshape(d.num_elem_y, d.num_elem_x, d.num_solpts_2d, d.num_elem_z, d.num_var, d.num_solpts)
    permuted = torch.permute(unflattened, (4, 3, 0, 1, 5, 2))
    return permuted.reshape(reference.shape)


def horizontal_momentum_rows(num_solpts: int) -> slice:
    """Return the horizontal-momentum block rows."""
    return slice(idx_rho_u1 * num_solpts, (idx_rho_u2 + 1) * num_solpts)


def stiff_variable_rows(num_solpts: int, device):
    """Return the non-contiguous block rows retained by the stiff partition."""
    return torch.cat(
        (
            torch.arange(idx_rho * num_solpts, (idx_rho + 1) * num_solpts, device=device),
            torch.arange(idx_rho_u3 * num_solpts, (idx_rho_theta + 1) * num_solpts, device=device),
        )
    )


def scalar_to_columns(array, dims: ColumnDims):
    """Convert a scalar grid array to column layout."""
    num_elem_z = array.shape[0]
    split = array.reshape(num_elem_z, dims.num_elem_y, dims.num_elem_x, dims.num_solpts, dims.num_solpts_2d)
    return torch.permute(split, (1, 2, 4, 0, 3)).reshape(
        dims.num_elem_y * dims.num_elem_x * dims.num_solpts_2d, num_elem_z, dims.num_solpts
    )


def scalar_interface_to_columns(array, dims: ColumnDims):
    """Convert scalar vertical traces to column layout with a trailing face axis."""
    num_planes = array.shape[0]
    split = array.reshape(num_planes, dims.num_elem_y, dims.num_elem_x, 2, dims.num_solpts_2d)
    return torch.permute(split, (1, 2, 4, 0, 3)).reshape(
        dims.num_elem_y * dims.num_elem_x * dims.num_solpts_2d, num_planes, 2
    )


def state_interface_to_columns(array, dims: ColumnDims):
    """Convert state traces to column layout with face then variable axes."""
    num_var, num_planes = array.shape[0], array.shape[1]
    split = array.reshape(num_var, num_planes, dims.num_elem_y, dims.num_elem_x, 2, dims.num_solpts_2d)
    return torch.permute(split, (2, 3, 5, 1, 4, 0)).reshape(
        dims.num_elem_y * dims.num_elem_x * dims.num_solpts_2d, num_planes, 2, num_var
    )
