"""Application and inversion of the vertical column operator."""

import functools

import torch

# Index letters shared by the einsum calls below:
#   c = column, e = vertical element, i = block row, j = block column.


def blocks_matvec(lower, diag, upper, x_col, *, periodic=False):
    """Apply the column blocks, including periodic end couplings when requested."""
    out = torch.einsum("ceij,cej->cei", diag, x_col)
    out[:, 1:] += torch.einsum("ceij,cej->cei", lower[:, 1:], x_col[:, :-1])
    out[:, :-1] += torch.einsum("ceij,cej->cei", upper[:, :-1], x_col[:, 1:])
    if periodic:
        out[:, 0] += torch.einsum("cij,cj->ci", lower[:, 0], x_col[:, -1])
        out[:, -1] += torch.einsum("cij,cj->ci", upper[:, -1], x_col[:, 0])
    return out


#: CUDA compute capabilities with strongly rate-limited FP64 arithmetic.
_RATE_LIMITED_FP64 = frozenset({(5, 0), (5, 2), (6, 1), (7, 5), (8, 6), (8, 9), (12, 0)})


@functools.cache
def _fp64_is_rate_limited(device_index: int) -> bool:
    """Return whether the CUDA device has strongly rate-limited FP64 arithmetic.

    Unknown compute capabilities use the conservative float64 path.
    """
    return torch.cuda.get_device_capability(device_index) in _RATE_LIMITED_FP64


def _factor_in_float32(b) -> bool:
    """Select float32 factorization for rate-limited CUDA devices."""
    if b.dtype != torch.float32 or b.device.type != "cuda":
        return False
    return _fp64_is_rate_limited(b.device.index if b.device.index is not None else torch.cuda.current_device())


def block_thomas_solve(lower, diag, upper, b, dt):
    """Solve ``(I - dt/2 J1) x = b`` per column with the block Thomas algorithm.

    Float32 factorization is followed by one refinement step using a float64 residual.
    """
    half_dt = 0.5 * dt
    _, num_elem_z, block_size, _ = diag.shape
    out_dtype = b.dtype
    refine = _factor_in_float32(b)
    work_dtype = torch.float32 if refine else torch.float64
    identity = torch.eye(block_size, dtype=work_dtype).reshape(1, block_size, block_size)

    def diag_block(e):
        return identity - half_dt * diag[:, e].astype(work_dtype)

    def sub_block(e):
        return (-half_dt) * lower[:, e].astype(work_dtype)

    def super_block(e):
        return (-half_dt) * upper[:, e].astype(work_dtype)

    vector_rhs = b.ndim == 3
    b = b.astype(work_dtype)
    if vector_rhs:
        b = b[..., None]

    # Retain the LU factors for back substitution and iterative refinement.
    factors = [None] * num_elem_z
    reduced_rhs = [None] * num_elem_z
    factors[0] = torch.linalg.lu_factor(diag_block(0))
    reduced_rhs[0] = b[:, 0]
    for e in range(1, num_elem_z):
        augmented = torch.concatenate([super_block(e - 1), reduced_rhs[e - 1]], dim=-1)
        solved = torch.linalg.lu_solve(factors[e - 1][0], factors[e - 1][1], augmented)
        reduced_upper = solved[..., :block_size]
        reduced_vec = solved[..., block_size:]
        factors[e] = torch.linalg.lu_factor(diag_block(e) - sub_block(e) @ reduced_upper)
        reduced_rhs[e] = b[:, e] - sub_block(e) @ reduced_vec

    def solve_with_factors(rhs):
        """Solve a new right-hand side with the existing factors."""
        forward = [None] * num_elem_z
        forward[0] = rhs[0]
        for e in range(1, num_elem_z):
            eliminated = torch.linalg.lu_solve(factors[e - 1][0], factors[e - 1][1], forward[e - 1])
            forward[e] = rhs[e] - sub_block(e) @ eliminated
        sol = [None] * num_elem_z
        top = num_elem_z - 1
        sol[top] = torch.linalg.lu_solve(factors[top][0], factors[top][1], forward[top])
        for e in range(num_elem_z - 2, -1, -1):
            rhs_e = forward[e] - super_block(e) @ sol[e + 1]
            sol[e] = torch.linalg.lu_solve(factors[e][0], factors[e][1], rhs_e)
        return sol

    top = num_elem_z - 1
    x = [None] * num_elem_z
    x[top] = torch.linalg.lu_solve(factors[top][0], factors[top][1], reduced_rhs[top])
    for e in range(num_elem_z - 2, -1, -1):
        rhs_e = reduced_rhs[e] - super_block(e) @ x[e + 1]
        x[e] = torch.linalg.lu_solve(factors[e][0], factors[e][1], rhs_e)

    if refine:
        # Build the float64 residual element by element to limit peak memory.
        f64 = torch.float64
        identity64 = torch.eye(block_size, dtype=f64).reshape(1, block_size, block_size)
        residual = [None] * num_elem_z
        for e in range(num_elem_z):
            residual[e] = (
                b[:, e].astype(f64)
                - ((identity64 - half_dt * diag[:, e].astype(f64)) @ x[e].astype(f64))
            )
            if e > 0:
                residual[e] += half_dt * (lower[:, e].astype(f64) @ x[e - 1].astype(f64))
            if e < num_elem_z - 1:
                residual[e] += half_dt * (upper[:, e].astype(f64) @ x[e + 1].astype(f64))
        correction = solve_with_factors([residual[e].astype(work_dtype) for e in range(num_elem_z)])
        x = [x[e] + correction[e] for e in range(num_elem_z)]

    result = torch.stack(x, dim=1).astype(out_dtype)
    return result[..., 0] if vector_rhs else result


def cyclic_block_solve(lower, diag, upper, b, dt):
    """Solve periodic columns using a Woodbury correction to block Thomas.

    The first lower block couples to the last element; the final upper block
    couples to the first. The correction closes these two boundary block rows.
    """
    dtype = b.dtype
    lower, diag, upper, b = (value.to(torch.float64) for value in (lower, diag, upper, b))
    num_columns, num_elements, block_size = b.shape
    identity = torch.eye(block_size, dtype=b.dtype, device=b.device)
    right = b.new_zeros((num_columns, num_elements, block_size, 1 + 2 * block_size))
    right[..., 0] = b
    right[:, 0, :, 1 : 1 + block_size] = identity
    right[:, -1, :, 1 + block_size :] = identity
    solved = block_thomas_solve(lower, diag, upper, right, dt)
    particular, response = solved[..., :1], solved[..., 1:]
    bottom, top = -dt / 2 * lower[:, 0], -dt / 2 * upper[:, -1]

    # Close both periodic ends through the two boundary block rows.
    closure = torch.cat((bottom @ response[:, -1], top @ response[:, 0]), dim=1)
    closure += torch.eye(2 * block_size, dtype=b.dtype, device=b.device)
    boundary_rhs = torch.cat((bottom @ particular[:, -1], top @ particular[:, 0]), dim=1)
    correction = torch.linalg.solve(closure, boundary_rhs)
    return (particular - response @ correction[:, None])[..., 0].to(dtype)
