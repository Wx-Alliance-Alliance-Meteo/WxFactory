"""Finite-difference and analytic Jacobians of the right-hand side."""

from .block_solve import (
    block_thomas_solve,
    blocks_matvec,
    horizontal_momentum_matvec,
    solve_stiff_columns,
)
from .column_layout import (
    ColumnDims,
    column_dims,
    columns_to_state,
    horizontal_momentum_rows,
    state_to_columns,
    stiff_variable_rows,
)
from .finite_difference import (
    FiniteDifferenceJacobian,
    FiniteDifferenceRosenbrock,
    fd_jacobian_matvec,
    fd_norm,
    fd_rosenbrock_matvec,
)
from .flux_jacobian import equation_of_state, flux_jacobian_matrix, flux_jacobian_matvec
from .forcing_jacobian import forcing_jac_prepare, forcing_jvp
from .horizontal_jacobian import j2_flux_matvec, j2_prepare
from .linear_operator import LinearOperator
from .vertical_blocks import (
    assemble_j1_blocks,
    assemble_vertical_blocks,
    j1_prepare,
    split_vertical_blocks,
)

__all__ = [
    "ColumnDims",
    "FiniteDifferenceJacobian",
    "FiniteDifferenceRosenbrock",
    "LinearOperator",
    "assemble_j1_blocks",
    "assemble_vertical_blocks",
    "block_thomas_solve",
    "blocks_matvec",
    "column_dims",
    "columns_to_state",
    "equation_of_state",
    "fd_jacobian_matvec",
    "fd_norm",
    "fd_rosenbrock_matvec",
    "flux_jacobian_matrix",
    "flux_jacobian_matvec",
    "forcing_jac_prepare",
    "forcing_jvp",
    "horizontal_momentum_matvec",
    "horizontal_momentum_rows",
    "j1_prepare",
    "j2_flux_matvec",
    "j2_prepare",
    "solve_stiff_columns",
    "split_vertical_blocks",
    "state_to_columns",
    "stiff_variable_rows",
]
