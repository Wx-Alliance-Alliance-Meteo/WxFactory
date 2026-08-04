import math
from collections.abc import Callable

import numpy
import torch
from mpi4py import MPI
from numpy.typing import NDArray

from ..common import Configuration


class MatvecOp:
    """
    Matrix operator to apply to a vector
    """

    matvec: Callable[[numpy.ndarray], numpy.ndarray]
    dtype: numpy.dtype
    shape: tuple
    size: int

    def __init__(self, matvec: Callable[[NDArray], NDArray], dtype: numpy.dtype, shape: tuple) -> None:
        self.matvec = matvec
        self.dtype = dtype
        self.shape = shape
        self.size = math.prod([i for i in shape])

    def __call__(self, vec: numpy.ndarray) -> numpy.ndarray:
        """
        :param vec: Vector to apply the operation to
        :return: Result of the `A * vec` operation
        """
        return self.matvec(vec)


def _global_norm(vec: NDArray, comm: MPI.Comm) -> numpy.floating:
    """Compute the distributed 2-norm in the array's working precision."""
    local_norm = torch.linalg.vector_norm(vec)
    dtype = numpy.float32 if vec.dtype == torch.float32 else numpy.float64
    local = numpy.array([(local_norm * local_norm).item()], dtype=dtype)
    total = numpy.empty_like(local)
    comm.Allreduce(local, total)
    return numpy.sqrt(total[0])


class MatvecOpBasic(MatvecOp):
    def __init__(self, dt: float, Q: NDArray, rhs_handle: Callable[[NDArray], NDArray], param: Configuration) -> None:
        rhs_result = rhs_handle(Q)
        fd_norm_q = _global_norm(Q, MPI.COMM_WORLD) if param.jacobian_method.lower() == "fd" else None
        super().__init__(
            lambda vec: matvec_fun(
                vec,
                dt,
                Q,
                rhs_result,
                rhs_handle,
                param.jacobian_method,
                fd_norm_q=fd_norm_q,
            ),
            Q.dtype,
            Q.shape,
        )


def matvec_fun(
    vec: NDArray,
    dt: float,
    Q: NDArray,
    rhs: NDArray,
    rhs_handle: Callable[[NDArray], NDArray],
    method: str,
    fd_norm_q: float | None = None,
) -> numpy.ndarray:
    """
    Basic Matvec operation `A * vec`

    :param vec: Vector to apply the operation to
    :param dt: Delta time
    :param Q: ?
    :param rhs: Last computed RHS
    :param rhs_handle: Right hand side to compute
    :param method: Jacobian-action method: complex step or finite difference.
    :param fd_norm_q: Cached distributed norm of the linearization state.

    :return: Result of the `A * vec` operation
    """
    method_key = method.lower()

    if method_key == "complex":
        # Complex-step approximation
        epsilon = math.sqrt(numpy.finfo(float).eps)
        Qvec = Q + 1j * epsilon * vec.reshape(Q.shape)
        jac = dt * (rhs_handle(Qvec) / epsilon).imag
    elif method_key == "fd":
        # Following the NITSOL approach, see Eq. 14 in the review article by Knoll and Keyes on the JFNK method.
        direction = vec.reshape(Q.shape)
        eps_machine = numpy.float64(torch.finfo(Q.dtype).eps)
        norm_q = fd_norm_q if fd_norm_q is not None else _global_norm(Q, MPI.COMM_WORLD)
        norm_v = _global_norm(vec, MPI.COMM_WORLD)
        if norm_v == 0.0:
            epsilon = numpy.sqrt(eps_machine)
        else:
            epsilon = numpy.sqrt((numpy.float64(1.0) + norm_q) * eps_machine) / norm_v

        Qvec = Q + epsilon * direction
        rhs_difference = rhs_handle(Qvec) - rhs
        jac = (rhs_difference * (dt / epsilon)).to(Q.dtype)
    else:
        raise ValueError(f"Unknown Jacobian method '{method}'")

    return jac.flatten()


class MatvecOpRat(MatvecOp):
    def __init__(self, dt: float, Q: numpy.ndarray, rhs_vec: numpy.ndarray, rhs_handle: Callable) -> None:
        super().__init__(lambda vec: matvec_rat(vec, dt, Q, rhs_vec, rhs_handle), Q.dtype, Q.shape)


def matvec_rat(
    vec: numpy.ndarray,
    dt: float,
    Q: numpy.ndarray,
    rhs: numpy.ndarray,
    rhs_handle: Callable,
) -> numpy.ndarray:

    epsilon = math.sqrt(numpy.finfo(numpy.float32).eps)
    Qvec = Q + epsilon * vec.reshape(Q.shape)
    jac = dt * (rhs_handle(Qvec) - rhs) / epsilon

    return vec - 0.5 * jac.flatten()
