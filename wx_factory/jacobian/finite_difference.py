"""Finite-difference Jacobian-vector products."""

import math
from collections.abc import Callable

import numpy
import torch
from mpi4py import MPI
from numpy.typing import NDArray

from .linear_operator import LinearOperator


def fd_norm(vec: NDArray, comm: MPI.Comm) -> numpy.floating:
    """Compute a distributed 2-norm in the array's working precision."""
    local_norm = torch.linalg.vector_norm(vec)
    dtype = numpy.float32 if vec.dtype == torch.float32 else numpy.float64
    local = numpy.array([(local_norm * local_norm).item()], dtype=dtype)
    total = numpy.empty_like(local)
    comm.Allreduce(local, total)
    return numpy.sqrt(total[0])


def fd_jacobian_matvec(
    vec: NDArray,
    dt: float,
    Q: NDArray,
    rhs: NDArray,
    rhs_handle: Callable[[NDArray], NDArray],
    fd_norm_q: float | None = None,
) -> numpy.ndarray:
    """Apply ``dt * J`` using the NITSOL finite-difference step."""
    direction = vec.reshape(Q.shape)
    eps_machine = numpy.float64(torch.finfo(Q.dtype).eps)
    norm_q = fd_norm_q if fd_norm_q is not None else fd_norm(Q, MPI.COMM_WORLD)
    norm_v = fd_norm(vec, MPI.COMM_WORLD)
    if norm_v == 0.0:
        epsilon = numpy.sqrt(eps_machine)
    else:
        epsilon = numpy.sqrt((numpy.float64(1.0) + norm_q) * eps_machine) / norm_v

    perturbed = Q + epsilon * direction
    rhs_difference = rhs_handle(perturbed) - rhs
    jac = (rhs_difference * (dt / epsilon)).to(Q.dtype)

    return jac.flatten()


class FiniteDifferenceJacobian(LinearOperator):
    """Represent ``dt * J`` at a fixed state."""

    def __init__(self, dt: float, Q: NDArray, rhs_handle: Callable[[NDArray], NDArray]) -> None:
        rhs_result = rhs_handle(Q)
        # Q is fixed during a Krylov solve, so compute its norm once.
        fd_norm_q = fd_norm(Q, MPI.COMM_WORLD)
        super().__init__(
            lambda vec: fd_jacobian_matvec(vec, dt, Q, rhs_result, rhs_handle, fd_norm_q=fd_norm_q),
            Q.dtype,
            Q.shape,
        )


def fd_rosenbrock_matvec(
    vec: numpy.ndarray,
    dt: float,
    Q: numpy.ndarray,
    rhs: numpy.ndarray,
    rhs_handle: Callable,
) -> numpy.ndarray:
    """Apply ``I - dt/2 J`` using a fixed finite-difference step."""
    epsilon = math.sqrt(numpy.finfo(numpy.float32).eps)
    perturbed = Q + epsilon * vec.reshape(Q.shape)
    jac = dt * (rhs_handle(perturbed) - rhs) / epsilon

    return vec - 0.5 * jac.flatten()


class FiniteDifferenceRosenbrock(LinearOperator):
    """``I - dt/2 J`` at a fixed state, as a linear operator."""

    def __init__(self, dt: float, Q: numpy.ndarray, rhs_vec: numpy.ndarray, rhs_handle: Callable) -> None:
        super().__init__(lambda vec: fd_rosenbrock_matvec(vec, dt, Q, rhs_vec, rhs_handle), Q.dtype, Q.shape)
