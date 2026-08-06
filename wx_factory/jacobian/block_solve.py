"""Application and inversion of the block-tridiagonal vertical operator."""

import torch

from .column_layout import columns_to_state, horizontal_momentum_rows, state_to_columns, stiff_variable_rows

# Index letters shared by the einsum calls below:
#   c = column, e = vertical element, i = block row, j = block column.


def blocks_matvec(lower, diag, upper, x_col):
    """Apply the block-tridiagonal operator to a column-layout vector."""
    out = torch.einsum("ceij,cej->cei", diag, x_col)
    out[:, 1:] += torch.einsum("ceij,cej->cei", lower[:, 1:], x_col[:, :-1])
    out[:, :-1] += torch.einsum("ceij,cej->cei", upper[:, :-1], x_col[:, 1:])
    return out


def horizontal_momentum_matvec(rhsobj, momentum_blocks, x):
    """Apply the horizontal-momentum rows and return the result in grid layout."""
    lower, diag, upper = momentum_blocks
    rows = horizontal_momentum_rows(rhsobj.geom.num_solpts)
    x_col = state_to_columns(rhsobj, x)

    out = torch.einsum("ceij,cej->cei", diag, x_col)
    out[:, 1:] += torch.einsum("ceij,cej->cei", lower[:, 1:], x_col[:, :-1])
    out[:, :-1] += torch.einsum("ceij,cej->cei", upper[:, :-1], x_col[:, 1:])

    full = torch.zeros_like(x_col)
    full[..., rows] = out
    return columns_to_state(rhsobj, full, x)


def solve_stiff_columns(rhsobj, stiff_blocks, b_col, dt):
    """Solve the stiff-variable subsystem in column layout."""
    rows = stiff_variable_rows(rhsobj.geom.num_solpts, b_col.device)
    x = b_col.clone()
    x[..., rows] = block_thomas_solve(*stiff_blocks, b_col[..., rows], dt)
    return x


def block_thomas_solve(lower, diag, upper, b, dt):
    """Solve ``(I - dt/2 J1) x = b`` with block Thomas and iterative refinement."""
    half_dt = 0.5 * dt
    _, num_elem_z, block_size, _ = diag.shape
    out_dtype = b.dtype
    refine = out_dtype == torch.float32
    work_dtype = torch.float32 if refine else torch.float64
    identity = torch.eye(block_size, dtype=work_dtype).reshape(1, block_size, block_size)

    # Blocks of M = I - (dt/2) J1, in the working precision of the factorization.
    def diag_block(e):
        return identity - half_dt * diag[:, e].astype(work_dtype)

    def sub_block(e):
        return (-half_dt) * lower[:, e].astype(work_dtype)

    def super_block(e):
        return (-half_dt) * upper[:, e].astype(work_dtype)

    b = b.astype(work_dtype)

    # Retain the LU factors for back substitution and iterative refinement.
    factors = [None] * num_elem_z
    reduced_rhs = [None] * num_elem_z
    factors[0] = torch.linalg.lu_factor(diag_block(0))
    reduced_rhs[0] = b[:, 0]
    for e in range(1, num_elem_z):
        # One solve produces both the eliminated super-block and the eliminated right-hand side.
        augmented = torch.concatenate([super_block(e - 1), reduced_rhs[e - 1][..., None]], dim=-1)
        solved = torch.linalg.lu_solve(factors[e - 1][0], factors[e - 1][1], augmented)
        reduced_upper = solved[..., :block_size]
        reduced_vec = solved[..., block_size]
        factors[e] = torch.linalg.lu_factor(diag_block(e) - sub_block(e) @ reduced_upper)
        reduced_rhs[e] = b[:, e] - (sub_block(e) @ reduced_vec[..., None])[..., 0]

    def solve_with_factors(rhs):
        """Solve a new right-hand side with the existing factors."""
        forward = [None] * num_elem_z
        forward[0] = rhs[0]
        for e in range(1, num_elem_z):
            eliminated = torch.linalg.lu_solve(factors[e - 1][0], factors[e - 1][1], forward[e - 1][..., None])[..., 0]
            forward[e] = rhs[e] - (sub_block(e) @ eliminated[..., None])[..., 0]
        sol = [None] * num_elem_z
        top = num_elem_z - 1
        sol[top] = torch.linalg.lu_solve(factors[top][0], factors[top][1], forward[top][..., None])[..., 0]
        for e in range(num_elem_z - 2, -1, -1):
            rhs_e = forward[e] - (super_block(e) @ sol[e + 1][..., None])[..., 0]
            sol[e] = torch.linalg.lu_solve(factors[e][0], factors[e][1], rhs_e[..., None])[..., 0]
        return sol

    # Back substitution, sweeping downward from the top element.
    top = num_elem_z - 1
    x = [None] * num_elem_z
    x[top] = torch.linalg.lu_solve(factors[top][0], factors[top][1], reduced_rhs[top][..., None])[..., 0]
    for e in range(num_elem_z - 2, -1, -1):
        rhs_e = reduced_rhs[e] - (super_block(e) @ x[e + 1][..., None])[..., 0]
        x[e] = torch.linalg.lu_solve(factors[e][0], factors[e][1], rhs_e[..., None])[..., 0]

    if refine:
        # Build the float64 residual element by element to limit peak memory.
        f64 = torch.float64
        identity64 = torch.eye(block_size, dtype=f64).reshape(1, block_size, block_size)
        residual = [None] * num_elem_z
        for e in range(num_elem_z):
            residual[e] = (
                b[:, e].astype(f64)
                - ((identity64 - half_dt * diag[:, e].astype(f64)) @ x[e].astype(f64)[..., None])[..., 0]
            )
            if e > 0:
                residual[e] += half_dt * (lower[:, e].astype(f64) @ x[e - 1].astype(f64)[..., None])[..., 0]
            if e < num_elem_z - 1:
                residual[e] += half_dt * (upper[:, e].astype(f64) @ x[e + 1].astype(f64)[..., None])[..., 0]
        correction = solve_with_factors([residual[e].astype(work_dtype) for e in range(num_elem_z)])
        x = [x[e] + correction[e] for e in range(num_elem_z)]

    return torch.stack(x, dim=1).astype(out_dtype)
