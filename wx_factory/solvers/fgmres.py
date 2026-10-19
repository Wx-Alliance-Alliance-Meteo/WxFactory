from collections.abc import Callable
from time import time

import torch
from torch import Tensor

from ..context import Context
from .dense import solve_triangular
from .global_operations import global_allreduce, global_dotprod, global_inf_norm, global_norm

__all__ = ["fgmres"]

MatvecOperator = Callable[[Tensor], Tensor]


def _ortho_1_sync_igs(Q: Tensor, R: Tensor, T: Tensor, K: Tensor, j: int, context: Context):
    """Orthonormalization process that only requires a single synchronization step.

    This processes row j of matrix Q (starting from 1) so that the first j rows are orthogonal, and the first (j-1)
    rows are orthonormal.
    It normalizes row j-1 of matrix Q during that process.

    :param Q: The matrix to orthogonalize. We assume that the first (j-1) columns (or is it j-2?) of that matrix
              are already orthonormal
    :param R: Data from previous iterations of the orthonormalization process (?)
    :param T: Data from previous iterations of the orthonormalization process (?)
    :param K: Data from previous iterations of the orthonormalization process (?)
    :param j: Current number of vectors in the Kyrlov space. We orthogonalize the j^th vector (starting from 1)
              with the previous ones.
              _The matrices may be larger than that, so we can't rely on *.shape_
    """
    if j < 2:
        return -1.0

    # Float64 inner products preserve orthogonality without promoting the full basis.
    acc = R.dtype
    qd = Q.dtype

    if Q.dtype == acc:
        local_tmp = Q[:j, :] @ Q[j - 2 : j, :].T
    else:
        # Accumulate the small Gram matrix without a full float64 copy of the basis.
        local_tmp = torch.zeros((j, 2), dtype=acc, device=Q.device)
        chunk_size = 65536
        for start in range(0, Q.shape[1], chunk_size):
            basis = Q[:j, start : start + chunk_size].to(acc)
            local_tmp += basis @ basis[j - 2 : j].T

    context.synchronize()
    global_tmp = global_allreduce(local_tmp, context)
    small_tmp = global_tmp[: j - 2, 0]

    R[: j - 1, j - 1] = global_tmp[: j - 1, 1]

    norm2 = global_tmp[j - 2, 0] - (small_tmp @ small_tmp)
    norm = torch.sqrt(norm2)
    R[j - 2, j - 2] = norm
    R[j - 2, j - 1] -= small_tmp @ R[: j - 2, j - 1]
    R[j - 2, j - 1] /= norm

    T[: j - 2, j - 2] = small_tmp / norm

    if j > 2:
        L = torch.tril(T[: j - 2, : j - 2].T, -1)
        L_plus = L + torch.eye(j - 2, dtype=acc)
        r3 = torch.linalg.solve(L_plus, small_tmp)

        R[: j - 2, j - 2] = K[: j - 2, j - 3] + r3
        K[: j - 1, j - 2] = (R[: j - 1, j - 1] - (R[: j - 1, 1 : j - 1] @ r3)) / norm

        Q[j - 2, :] -= Q[: j - 2, :].T @ small_tmp.to(qd)
        Q[j - 1, :] -= Q[: j - 2, :].T @ R[: j - 2, j - 1].to(qd)

    else:
        K[: j - 1, j - 2] = R[: j - 1, j - 1] / norm

    inv_norm = 1.0 / float(norm) if norm > 0 else float("inf")
    Q[j - 2, :] *= inv_norm
    Q[j - 1, :] -= Q[j - 2, :] * float(R[j - 2, j - 1])
    Q[j - 1, :] *= inv_norm

    return float(norm)


def rotg(a: float, b: float) -> tuple[float, float, float]:
    anorm = abs(a)
    bnorm = abs(b)
    if bnorm == 0:
        return (1.0, 0.0, a)
    if anorm == 0:
        return (0.0, 1.0, b)
    import math

    scl = min(anorm, bnorm)
    sigma: float
    if anorm > bnorm:
        sigma = math.copysign(1.0, a)
    else:
        sigma = math.copysign(1.0, b)

    r = sigma * (scl * math.sqrt((a / scl) ** 2 + (b / scl) ** 2))
    c = a / r
    s = b / r
    return (c, s, r)


def fgmres(
    A: MatvecOperator,
    b: Tensor,
    x0: Tensor | None = None,
    tol: float = 1e-5,
    restart: int = 20,
    maxiter: int | None = None,
    preconditioner: MatvecOperator | None = None,
    hegedus: bool = False,
    verbose: int = 0,
    prefix: str = "",
    context: Context | None = None,
    krylov_dtype: torch.dtype | None = None,
) -> tuple[Tensor, float, float, int, int, list[tuple[float, float, float]]]:
    """
    Solve the given linear system (Ax = b) for x, using the FGMRES algorithm.

    :param A: System matrix. This may be an operator that when applied to a vector [v] results in A*v. A should not be identity (or a multiplication of identity) when the preconditionner is none or identity
    :param B: The right-hand side of the system to solve. Must be a vector

    :param x0: Initial guess for the solution. The zero vector if absent. Optional
    :param tol: Maximum residual (|b - Ax| / |b|), below which we consider the system solved. Optional
    :param restart: Number of iterations in the inner loop. Optional
    :param maxiter: Maximum number of *outer loop* iterations. If absent, it's going to be a very large number. Optional
    :param preconditioner: Operator [M^-1] that preconditions a given vector [v]. Computes the product (M^-1)*v. Optional
    :param hegedus: Whether to apply the Hegedüs trick (whatever that is). Optional
    :param krylov_dtype: Basis storage dtype; defaults to the right-hand side dtype.

    :return: 1. The result [x]
    :return: 2. The relative residual |b - Ax| / |b|
    :return: 3. The number of (inner loop) iterations performed
    :return: 4. A flag that indicates the convergence status (0 if converged, -1 if not)
    :return: 5. The list of residuals at every iteration
    """
    if context is None:
        context = Context.get_default()
    comm = context.comm

    if len(b) <= restart:
        raise ValueError("The b vector should be longer than the number of restart")

    t_start = time()
    niter = 0

    if preconditioner is None:
        preconditioner = lambda x: x

    num_dofs = len(b)

    if maxiter is None:
        maxiter = num_dofs * 10

    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()

    norm_b = global_norm(b, context=context)
    if norm_b == 0.0:
        return torch.zeros_like(b), 0.0, 0.0, 0, 0, [(0.0, time() - t_start, 0.0)]

    tol_relative = tol * norm_b

    Ax0 = A(x)
    residuals = []

    if hegedus:
        norm_Ax0_2 = global_dotprod(Ax0, Ax0, comm=comm)
        if norm_Ax0_2 != 0.0:
            ksi_min = global_dotprod(b, Ax0, comm=comm) / norm_Ax0_2
            x = ksi_min * x0
            Ax0 = A(x)

    r = b - Ax0
    norm_r = global_norm(r, context=context)

    residuals.append(((norm_r / norm_b).item(), time() - t_start, 0.0))

    # Keep the large basis in working precision; float64 bookkeeping protects orthogonality.
    dtype = b.dtype if krylov_dtype is None else krylov_dtype
    acc_dtype = torch.float64 if "float32" in str(dtype) else dtype

    for outer in range(maxiter):
        # NOTE: We are dealing with row-major matrices, but we store the transpose of H and V.
        H = torch.zeros((restart + 2, restart + 2), dtype=acc_dtype)
        R = torch.zeros((restart + 2, restart + 2), dtype=acc_dtype)
        T = torch.zeros((restart + 2, restart + 2), dtype=acc_dtype)
        K = torch.zeros((restart + 2, restart + 2), dtype=acc_dtype)
        V = torch.zeros((restart + 2, num_dofs), dtype=dtype)
        Z = torch.zeros((restart + 1, num_dofs), dtype=dtype)
        Q = []

        V[0, :] = r / norm_r
        Z[0, :] = preconditioner(V[0, :])
        V[1, :] = A(Z[0, :])
        v_norm = _ortho_1_sync_igs(V, R, T, K, 2, context)

        g = torch.zeros(num_dofs, dtype=acc_dtype)
        g[0] = norm_r
        for inner in range(restart):
            niter += 1

            # Modified Gram-Schmidt process (1-sync version, with lagged normalization)
            Z[inner + 1, :] = preconditioner(V[inner + 1])

            V[inner + 2, :] = A(Z[inner + 1, :] / v_norm) * v_norm
            v_norm = _ortho_1_sync_igs(V, R, T, K, inner + 3, context)
            H[inner, : inner + 2] = R[: inner + 2, inner + 1]
            Z[inner + 1, :] /= v_norm

            if inner > 0:
                _apply_givens(Q, H[inner, :], inner)

            if inner != num_dofs - 1 and H[inner, inner + 1] != 0:
                [c, s, r] = rotg(float(H[inner, inner]), float(H[inner, inner + 1]))

                Qblock = torch.tensor([[c, s], [-s.conjugate(), c]], dtype=acc_dtype)
                Q.append(Qblock)

                g[inner : inner + 2] = Qblock @ g[inner : inner + 2]

                H[inner, inner] = torch.dot(Qblock[0, :], H[inner, inner : inner + 2])
                H[inner, inner + 1] = 0.0

            # Don't update norm_r if last inner iteration, because
            # norm_r is calculated directly after this loop ends.
            if inner < restart - 1:
                norm_r = torch.abs(g[inner + 1])
                residuals.append(((norm_r / norm_b).item(), time() - t_start, 0.0))
                if verbose > 1 and comm.rank == 0:
                    print(f"{prefix}norm_r / b = {residuals[-1][0]:.3e}", flush=True)
                if norm_r < tol_relative:
                    break

        y = solve_triangular(H[0 : inner + 1, 0 : inner + 1].T, g[0 : inner + 1])
        update = torch.ravel(Z[: inner + 1, :].T @ y.astype(Z.dtype).reshape(-1, 1))
        x = x + update
        r = b - A(x)

        norm_r = global_norm(r, context=context)

        residuals.append(((norm_r / norm_b).item(), time() - t_start, 0.0))
        if verbose > 0 and comm.rank == 0:
            print(f"{prefix}res: {norm_r / norm_b:.2e} (iter {niter})", flush=True)

        # All ranks must make the same stagnation decision.
        indices = x != 0
        local_change = torch.max(torch.abs(update[indices] / x[indices])) if indices.any() else x.new_tensor(0.0)
        change = global_inf_norm(local_change, comm=comm)
        if change < 1e-12:
            return x, norm_r.item(), norm_b.item(), niter, -1, residuals

        if norm_r < tol_relative:
            return x, norm_r.item(), norm_b.item(), niter, 0, residuals

        del V, Z

    flag = 0
    if norm_r >= tol_relative:
        flag = -1
    return x, norm_r.item(), norm_b.item(), niter, flag, residuals


def _apply_givens(Q, v, k):
    """Apply the first k Givens rotations in Q to v.

    Parameters
    ----------
    Q : list
       list of consecutive 2x2 Givens rotations
    v : array
       vector to apply the rotations to
    k : int
       number of rotations to apply.

    Returns
    -------
    v is changed in place

    Notes
    -----
    This routine is specialized for GMRES.  It assumes that the first Givens
    rotation is for dofs 0 and 1, the second Givens rotation is for
    dofs 1 and 2, and so on.

    """
    for j in range(k):
        v[j : j + 2] = Q[j] @ v[j : j + 2]
