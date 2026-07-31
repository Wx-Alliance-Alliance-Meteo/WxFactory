import math
from typing import Callable, Tuple

import numpy
import torch
import torch.autograd.forward_ad as fwad
from numpy.typing import NDArray

from ..common import Configuration
from ..device import differentiable_mode
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


def _matvec_ad(vec: NDArray, Q: NDArray, rhs_handle: Callable[[NDArray], NDArray]) -> NDArray:
    """Return ``J . vec`` as the forward-AD tangent of the right-hand side at ``Q``."""
    if not differentiable_mode():
        raise RuntimeError(
            "jacobian_method = ad needs differentiable mode, which is off. It is normally enabled "
            "from the configuration before the device is created; a Simulation built around a "
            "device that already installed an inference-mode guard cannot use it."
        )
    if torch.is_inference_mode_enabled():
        raise RuntimeError("jacobian_method = ad cannot differentiate the right-hand side under torch.inference_mode")

    with fwad.dual_level():
        result = rhs_handle(fwad.make_dual(Q, vec.reshape(Q.shape)))
        tangent = fwad.unpack_dual(result).tangent
        if tangent is None:
            raise RuntimeError(
                "the right-hand side returned no forward-AD tangent; it dropped the derivative "
                "of its input (an in-place write into a cached buffer will do this)"
            )
        return tangent.clone()


def matvec_fun(
    vec: NDArray,
    dt: float,
    Q: NDArray,
    rhs: NDArray,
    rhs_handle: Callable[[NDArray], NDArray],
    method: str,
    eps_factor: float = 1.0,
    q_scale: float | None = None,
) -> numpy.ndarray:
    """
    Basic Matvec operation `A * vec`

    :param vec: Vector to apply the operation to
    :param dt: Delta time
    :param Q: ?
    :param rhs: Last computed RHS
    :param rhs_handle: Right hand side to compute
    :param method: How to obtain the Jacobian action: ``ad`` for forward-mode automatic
                   differentiation, ``complex`` for the complex step, anything else for a one-sided
                   finite difference. The first two are exact; only the finite difference reuses
                   ``rhs``, so it costs a single extra evaluation instead of roughly two.
    :param q_scale: Precomputed max(1, ||Q||_inf) for the single-precision step scaling. Q is fixed
                    across all the matvecs of one Krylov solve, so a caller that issues many of them
                    can compute this once and pass it in, avoiding a global reduction per matvec.

    :return: Result of the `A * vec` operation
    """

    if method == "ad":
        # Forward-mode automatic differentiation: the exact directional derivative, with no step
        # size to choose. This matters most in single precision, where the finite difference loses
        # the derivative to subtractive cancellation.
        jac = dt * _matvec_ad(vec, Q, rhs_handle)
    elif method == "complex":
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
            if q_scale is None:
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
