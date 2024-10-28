import math
from typing import Callable, Tuple

import numpy
from common.device import Device, default_device


class MatvecOp:
    def __init__(self, matvec: Callable[[numpy.ndarray], numpy.ndarray], dtype, shape: Tuple) -> None:
        self.matvec = matvec
        self.dtype = dtype
        self.shape = shape
        self.size = math.prod([i for i in shape])

    def __call__(self, vec: numpy.ndarray) -> numpy.ndarray:
        return self.matvec(vec)


class MatvecOpBasic(MatvecOp):
    def __init__(self, dt: float, Q: numpy.ndarray) -> None:
        super().__init__(lambda vec: matvec_fun(vec, dt, Q), Q.dtype, Q.shape)


def matvec_fun(
    vec: numpy.ndarray,
    dt: float,
    Q: numpy.ndarray,
    rhs: numpy.ndarray,
    rhs_handle,
    method="complex",
    device: Device = default_device,
) -> numpy.ndarray:
    if method == "complex":
        # Complex-step approximation
        epsilon = math.sqrt(device.xp.finfo(float).eps)
        Qvec = Q + 1j * epsilon * device.xp.reshape(vec, Q.shape)
        jac = dt * (rhs_handle(Qvec) / epsilon).imag
    else:
        # Finite difference approximation
        epsilon = math.sqrt(device.xp.finfo(device.xp.float32).eps)
        Qvec = Q + epsilon * device.xp.reshape(vec, Q.shape)
        jac = dt * (rhs_handle(Qvec) - rhs) / epsilon

    return jac.flatten()


class MatvecOpRat(MatvecOp):
    def __init__(
        self, dt: float, Q: numpy.ndarray, rhs_vec: numpy.ndarray, rhs_handle: Callable, device: Device = default_device
    ) -> None:
        super().__init__(lambda vec: matvec_rat(vec, dt, Q, rhs_vec, rhs_handle), Q.dtype, Q.shape)


def matvec_rat(
    vec: numpy.ndarray,
    dt: float,
    Q: numpy.ndarray,
    rhs: numpy.ndarray,
    rhs_handle: Callable,
    device: Device = default_device,
) -> numpy.ndarray:
    xp = device.xp

    epsilon = math.sqrt(xp.finfo(xp.float32).eps)
    Qvec = Q + epsilon * xp.reshape(vec, Q.shape)
    jac = dt * (rhs_handle(Qvec) - rhs) / epsilon

    return vec - 0.5 * jac.flatten()
