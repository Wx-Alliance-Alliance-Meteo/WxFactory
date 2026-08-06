"""Matrix-free operator interface used by the Krylov solvers."""

import math
from collections.abc import Callable

import numpy
from numpy.typing import NDArray


class LinearOperator:
    """A matrix-free linear operator."""

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
        """Return ``A * vec``."""
        return self.matvec(vec)
