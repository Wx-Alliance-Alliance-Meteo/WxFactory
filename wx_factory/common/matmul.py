"""Matrix multiplication backend management with nvmath-python support.

This module provides a configurable matrix multiplication backend that can switch
between default (numpy/cupy) and nvmath-python implementations, with caching of
nvmath Matmul objects to avoid repeated planning overhead.
"""

import logging
from typing import Optional

from numpy.typing import NDArray


# Create a dedicated logger for nvmath that only shows warnings and above
# This is because there is something else in this program that is setting the
# root logger level on rank 0 to INFO, causing a lot of verbose output otherwise.
_nvmath_logger = logging.getLogger("wx_factory.nvmath")
_nvmath_logger.setLevel(logging.WARNING)


class MatmulManager:
    """Manages matrix multiplication backend and nvmath object caching.

    This singleton class handles switching between default (numpy/cupy) and nvmath
    backends for matrix multiplication, with caching of nvmath Matmul objects to
    avoid repeated planning overhead.
    """

    _instance = None

    def __init__(self):
        self.backend = "default"
        self.xp = None
        self._cache = {}
        self._nvmath_available = None

    @classmethod
    def get_instance(cls) -> "MatmulManager":
        if cls._instance is None:
            cls._instance = MatmulManager()
        return cls._instance

    def set_backend(self, backend: str):
        """Set the matmul backend ('default' or 'nvmath')."""
        if backend not in ("default", "nvmath"):
            raise ValueError(f"Unknown matmul backend: {backend}")
        self.backend = backend
        # Clear cache when changing backends
        self._cache.clear()

    def _check_nvmath(self) -> bool:
        """Check if nvmath-python is available."""
        if self._nvmath_available is None:
            try:
                from nvmath.linalg import Matmul
                self._nvmath_available = True
            except ImportError:
                self._nvmath_available = False
        return self._nvmath_available

    def matmul(
        self,
        a: NDArray,
        b: NDArray,
        alpha: float = 1.0,
        beta: float = 0.0,
        out: Optional[NDArray] = None,
    ) -> NDArray:
        """Perform matrix multiplication using the configured backend.

        Computes: out = alpha * (a @ b) + beta * out

        Parameters
        ----------
        a : NDArray
            Left operand matrix
        b : NDArray
            Right operand matrix
        alpha : float
            Scalar multiplier for the matrix product (default 1.0)
        beta : float
            Scalar multiplier for the output matrix (default 0.0)
        out : NDArray, optional
            Output array for in-place operation. If None, a new array is allocated.

        Returns
        -------
        NDArray
            Result of alpha * (a @ b) + beta * out
        """
        # Flatten the arrays to 2D for the matmul
        if self.backend == "nvmath" and self._check_nvmath():
            return self._nvmath_matmul(a, b, alpha, beta, out)
        return self._default_matmul(a, b, alpha, beta, out)

    def _default_matmul(
        self,
        a: NDArray,
        b: NDArray,
        alpha: float = 1.0,
        beta: float = 0.0,
        out: Optional[NDArray] = None,
    ) -> NDArray:
        """Perform matrix multiplication using numpy/cupy."""
        if out is None:
            result = self.xp.matmul(a, b)
            if alpha != 1.0:
                result *= alpha
            return result
        else:
            if beta == 0.0:
                self.xp.matmul(a, b, out=out)
                if alpha != 1.0:
                    out *= alpha
            else:
                if alpha == 1.0 and beta == 1.0:
                    out += self.xp.matmul(a, b)
                else:
                    out *= beta
                    out += alpha * self.xp.matmul(a, b)
            return out

    def _nvmath_matmul(
        self,
        a: NDArray,
        b: NDArray,
        alpha: float = 1.0,
        beta: float = 0.0,
        out: Optional[NDArray] = None,
    ) -> NDArray:
        """Perform matrix multiplication using nvmath-python with caching."""
        from nvmath.linalg import Matmul, MatmulOptions

        # Cache key based on shape, dtype, and whether it's in-place
        # In-place operations may have different plans
        if out is not None:
            key = (a.shape, a.dtype.name, b.shape, b.dtype.name, out.shape, out.dtype.name)
        else:
            key = (a.shape, a.dtype.name, b.shape, b.dtype.name)

        if key not in self._cache:
            if out is not None:
                mm = Matmul(a, b, c=out, alpha=alpha, beta=beta, options=MatmulOptions(logger=_nvmath_logger, inplace=True))
            else:
                mm = Matmul(a, b, alpha=alpha, options=MatmulOptions(logger=_nvmath_logger))
            mm.plan()
            self._cache[key] = mm
        else:
            # Reuse existing Matmul object with new operands
            if out is not None:
                self._cache[key].reset_operands(a, b, alpha=alpha, beta=beta, c=out)
            else:
                self._cache[key].reset_operands(a, b, alpha=alpha)

        return self._cache[key].execute()


_matmul_manager = MatmulManager.get_instance()


def set_matmul_backend(backend: str, xp):
    """Set the matmul backend to use ('default' or 'nvmath').

    Parameters
    ----------
    backend : str
        Either 'default' (use numpy/cupy matmul) or 'nvmath' (use nvmath-python).
    xp : Module
        The xp module to use for the 'default' matmul backend.
    """
    _matmul_manager.set_backend(backend)
    _matmul_manager.xp = xp


def apply_op(
    a: NDArray,
    b: NDArray,
    alpha: float = 1.0,
    beta: float = 0.0,
    out: Optional[NDArray] = None,
) -> NDArray:
    """Apply a matrix operator to the last dimension of a.

    Computes: out = alpha * (a @ b) + beta * out

    Parameters
    ----------
    a : NDArray
        Input array to transform
    b : NDArray
        Operator matrix (typically float64)
    alpha : float
        Scalar multiplier for the matrix product (default 1.0)
    beta : float
        Scalar multiplier for the output array (default 0.0)
    out : NDArray, optional
        Output array for in-place operation. If None, a new array is allocated.
        Must have shape compatible with the result (*a.shape[:-1], b.shape[-1]).

    Returns
    -------
    NDArray
        Result of alpha * (a @ b) + beta * out
    """
    sh = a.shape
    a = a.reshape(-1, sh[-1])

    # Handle output array reshaping for in-place operations
    if out is not None:
        out_reshaped = out.reshape(-1, b.shape[-1])
        _matmul_manager.matmul(a, b, alpha=alpha, beta=beta, out=out_reshaped)
        return out
    else:
        result = _matmul_manager.matmul(a, b, alpha=alpha, beta=beta)
        return result.reshape(*sh[:-1], -1)
