import math
from typing import Callable, Tuple

import numpy
from numpy.typing import NDArray

from ..common import Configuration
from .global_operations import global_inf_norm


class MatvecOp:
    """
    Matrix operator to apply to a vector
    """

    matvec: Callable[[numpy.ndarray], numpy.ndarray]
    dtype: numpy.dtype
    shape: Tuple
    size: int

    def __init__(self, matvec: Callable[[NDArray], NDArray], dtype: numpy.dtype, shape: Tuple) -> None:
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


class MatvecOpBasic(MatvecOp):
    def __init__(self, dt: float, Q: NDArray, rhs_handle: Callable[[NDArray], NDArray], param: Configuration) -> None:
        rhs_result = rhs_handle(Q)
        points_per_panel = (param.num_elements_horizontal * param.num_solpts) ** 2
        epsilon_factor = 1.0 + points_per_panel / 100000.0
        super().__init__(
            lambda vec: matvec_fun(
                vec, dt, Q, rhs_result, rhs_handle, param.jacobian_method, eps_factor=epsilon_factor
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
    eps_factor: float = 1.0,
) -> numpy.ndarray:
    """
    Basic Matvec operation `A * vec`

    :param vec: Vector to apply the operation to
    :param dt: Delta time
    :param Q: ?
    :param rhs: Last computed RHS
    :param rhs_handle: Right hand side to compute
    :param method: Method to use for the calculation

    :return: Result of the `A * vec` operation
    """

    if method == "complex":
        # Complex-step approximation
        epsilon = math.sqrt(numpy.finfo(float).eps)
        Qvec = Q + 1j * epsilon * vec.reshape(Q.shape)
        jac = dt * (rhs_handle(Qvec) / epsilon).imag
    else:
        # Finite difference approximation of the Jacobian-vector product.
        if "32" in str(Q.dtype):
            # Single precision: a fixed absolute step is lost to round-off when the state components
            # are large -- Q + epsilon*vec rounds straight back to Q, so the finite difference returns
            # a corrupted Jacobian and the exponential integrators inject energy and blow up. The
            # solution is to scale the step by the global magnitude of the state.
            q_scale = max(1.0, float(global_inf_norm(Q)))
            epsilon = math.sqrt(numpy.finfo(numpy.float32).eps) * eps_factor * q_scale
        else:
            # Double precision: the fixed step is small relative to the state resolution, so no
            # scaling is needed. This formulation is accurate enough and avoid the global communication
            epsilon = math.sqrt(numpy.finfo(numpy.float32).eps) * eps_factor
        Qvec = Q + epsilon * vec.reshape(Q.shape)
        jac = dt * (rhs_handle(Qvec) - rhs) / epsilon

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
