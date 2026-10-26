"""Conversions between grid and vertical-column layouts.

The column layout has shape ``(num_columns, num_elem_z, num_var * num_solpts)``.
"""

from typing import NamedTuple

import torch

from ..common.definitions import idx_rho, idx_v3


class ColumnDims(NamedTuple):
    """Sizes needed to move between the grid layout and the column layout."""

    num_solpts: int  # solution points along one element edge
    num_solpts_2d: int  # solution points on one horizontal slab of an element (num_solpts**2)
    num_var: int  # prognostic variables, including passive tracer densities
    num_elem_z: int  # vertical elements, i.e. block rows per column
    num_elem_y: int
    num_elem_x: int
    block_size: int  # unknowns per element of a column: num_var * num_solpts
    num_columns: int  # independent vertical columns on this rank


def column_dims(rhs, state) -> ColumnDims:
    num_solpts = rhs.geom.num_solpts
    num_var, num_elem_z, num_elem_y, num_elem_x, _ = state.shape
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


def state_to_columns(rhs, state):
    dimensions = column_dims(rhs, state)
    split = state.reshape(
        dimensions.num_var,
        dimensions.num_elem_z,
        dimensions.num_elem_y,
        dimensions.num_elem_x,
        dimensions.num_solpts,
        dimensions.num_solpts_2d,
    )
    permuted = torch.permute(split, (2, 3, 5, 1, 0, 4))
    return permuted.reshape(dimensions.num_columns, dimensions.num_elem_z, dimensions.block_size)


def columns_to_state(rhs, columns, reference):
    """Inverse of :func:`state_to_columns`; ``reference`` supplies the grid-layout shape."""
    dimensions = column_dims(rhs, reference)
    unflattened = columns.reshape(
        dimensions.num_elem_y,
        dimensions.num_elem_x,
        dimensions.num_solpts_2d,
        dimensions.num_elem_z,
        dimensions.num_var,
        dimensions.num_solpts,
    )
    permuted = torch.permute(unflattened, (4, 3, 0, 1, 5, 2))
    return permuted.reshape(reference.shape)


def stiff_variable_rows(num_solpts: int, device, num_var=5):
    """Return the non-contiguous block rows retained by the stiff partition."""
    return torch.cat(
        (
            torch.arange(idx_rho * num_solpts, (idx_rho + 1) * num_solpts, device=device),
            torch.arange(idx_v3 * num_solpts, num_var * num_solpts, device=device),
        )
    )


def scalar_to_columns(array, dims: ColumnDims):
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
