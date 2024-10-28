import functools as fct
import numpy as np
import cupy as cp
from numpy.typing import NDArray


@fct.cache
def _fact(k: int) -> int:
    return k * _fact(k - 1) if k > 1 else 1


_m = 13
_expm_b: NDArray[cp.floating] = np.array(
    [_fact(2 * _m - k) * _fact(_m) / _fact(2 * _m) / _fact(_m - k) / _fact(k) for k in range(_m + 1)]
)
del _fact, _m

_expm_theta13: float = 5.371920351148152


def expm(A: NDArray[cp.floating]) -> NDArray[cp.floating]:
    """
    Compute the matrix exponential using Higham's scaling and squaring algorithm.

    References
    ----------
    J. R. Mannuel Ledoh and R. Pulungan, "Parallelization of Padé Approximation of Matrix Exponential with CUDA-Aware MPI,"
    2019 5th International Conference on Science and Technology (ICST), Yogyakarta, Indonesia, 2019, pp. 1-6,
    doi: 10.1109/ICST47872.2019.9166326.
    """

    if len(A.shape) < 2:
        raise cp.linalg.LinAlgError("A must be at least 2-dimensional")
    n = A.shape[-1]
    if A.shape[-2] != n:
        raise cp.linalg.LinAlgError("A must be square in last two dimensions")

    b = _expm_b
    theta13 = _expm_theta13

    s = cp.ceil(cp.log2(cp.linalg.norm(A, 1) / theta13))
    A = A / 2**s

    A2 = A @ A
    A4 = A2 @ A2
    A6 = A4 @ A2

    U = b[13] * A6 + b[11] * A4 + b[9] * A2
    U = A6 @ U
    U += b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * cp.eye(n)
    U = A @ U

    V = b[12] * A6 + b[10] * A4 + b[8] * A2
    V = A6 @ V
    V += b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * cp.eye(n)

    R = cp.linalg.solve(-U + V, U + V)
    for _ in range(int(s)):
        R = R @ R

    return R
