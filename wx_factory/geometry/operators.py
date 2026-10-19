from typing import TYPE_CHECKING, TypeVar

import numpy
import numpy.linalg
import sympy
import torch
from numpy.typing import NDArray
from torch import Tensor

from ..common.matmul import kron
from ..context import Context
from .geometry import Geometry

if TYPE_CHECKING:
    from .metric5d import Metric5D

T = TypeVar("T", bound=numpy.generic)


class DFROperators:
    """Set of operators used for Direct Flux Reconstruction.

    The relevant internal matrices are:
       * The extrapolation matrices, `extrap_west`, `extrap_east`, `extrap_south`, `extrap_north`, `extrap_down`, and
         `extrap_up`. They are used to compute values at the boundaries of an element.
       * Differentiation matrices: `diff`, `diff_ext`, `diff_tr`, `diff_solpt`, `diff_solpt_tr`
       * Correction matrices: `correction`, `correction_tr`.
    """

    def __init__(self, grd: Geometry, context: Context, dtype: torch.dtype | None = None):
        """Initialize the Direct Flux Reconstruction operators (matrices) based on input grid parameters.

        Parameters
        ----------
        grd : Geometry
           Underlying grid, which must define `solutionPoints`, `extension`, `extension_sym` and
           `num_solpts` as member variables
        context : Context
           Context object containing the device, MPI communicator and other configuration information.
        dtype : DTypeLike, optional
           Tensor dtype. Defaults to the context's working real dtype.
        """

        self.context = context
        self.dtype = context.real_dtype if dtype is None else dtype
        build_dtype = torch.float64

        # Construct in float64 so Vandermonde inversion does not accumulate float32 roundoff.
        V = legvander(grd.solutionPoints, grd.num_solpts - 1).to(build_dtype)
        invV = torch.linalg.inv(V)

        # Note that extrap_neg and extrap_pos should be vectors, not a one-row matrix; numpy
        # treats the two differently.
        extrap_neg = (legvander(torch.tensor([-1.0], dtype=build_dtype), grd.num_solpts - 1) @ invV).reshape((-1,))
        extrap_pos = (legvander(torch.tensor([+1.0], dtype=build_dtype), grd.num_solpts - 1) @ invV).reshape((-1,))

        assert extrap_neg.dtype == build_dtype
        assert extrap_pos.dtype == build_dtype

        self.extrap_west = extrap_neg
        self.extrap_east = extrap_pos
        self.extrap_south = extrap_neg
        self.extrap_north = extrap_pos
        self.extrap_down = extrap_neg
        self.extrap_up = extrap_pos

        V = legvander(grd.solutionPoints, grd.num_solpts - 1).to(build_dtype)
        invV = torch.linalg.inv(V)
        diff = diffmat(grd.extension_sym)
        self.diff_ext = torch.asarray(diff).to(build_dtype)

        assert self.diff_ext.dtype == build_dtype

        if check_skewcentrosymmetry(self.diff_ext) is False:
            raise ValueError("Something horribly wrong has happened in the creation of the differentiation matrix")

        self.diff_solpt = self.diff_ext[1:-1, 1:-1].clone()
        self.correction = torch.column_stack((self.diff_ext[1:-1, 0], self.diff_ext[1:-1, -1]))

        self.diff_solpt_tr = self.diff_solpt.T.clone()
        self.correction_tr = self.correction.T.clone()

        # Ordinary differentiation matrices (used only in diagnostic calculations)
        self.diff = diffmat(grd.solutionPoints)
        self.diff = torch.asarray(self.diff).to(build_dtype)
        self.diff_tr = self.diff.T.clone()

        self.quad_weights = torch.outer(grd.glweights, grd.glweights).to(build_dtype)

        assert self.diff_solpt.dtype == build_dtype
        assert self.correction.dtype == build_dtype
        assert self.diff_solpt_tr.dtype == build_dtype
        assert self.correction_tr.dtype == build_dtype
        assert self.diff.dtype == build_dtype
        assert self.diff_tr.dtype == build_dtype
        assert self.quad_weights.dtype == build_dtype

        if getattr(grd, "is_3d_euler_grid", False):
            I2 = torch.eye(grd.num_solpts, dtype=V.dtype)
            I3 = torch.eye(grd.num_solpts**2, dtype=V.dtype)

            self.extrap_x = torch.vstack((kron(I3, self.extrap_west), kron(I3, self.extrap_east))).T.clone()
            self.extrap_y = torch.vstack(
                (kron(I2, kron(self.extrap_south, I2)), kron(I2, kron(self.extrap_north, I2)))
            ).T.clone()
            self.extrap_z = torch.vstack((kron(self.extrap_down, I3), kron(self.extrap_up, I3))).T.clone()

            self.derivative_x = kron(I3, self.diff_solpt).T.clone()
            self.derivative_y = kron(I2, kron(self.diff_solpt, I2)).T.clone()
            self.derivative_z = kron(self.diff_solpt, I3).T.clone()

            corr_west = self.diff_ext[1:-1, 0]
            corr_east = self.diff_ext[1:-1, -1]
            corr_south = corr_west
            corr_north = corr_east
            corr_down = corr_west
            corr_up = corr_east

            self.correction_WE = torch.vstack((kron(I3, corr_west), kron(I3, corr_east)))
            self.correction_SN = torch.vstack((kron(I2, kron(corr_south, I2)), kron(I2, kron(corr_north, I2))))
            self.correction_DU = torch.vstack((kron(corr_down, I3), kron(corr_up, I3)))

        else:
            ident = torch.eye(grd.num_solpts, dtype=build_dtype)
            self.extrap_x = torch.vstack((kron(ident, self.extrap_west), kron(ident, self.extrap_east))).T.clone()
            self.extrap_y = torch.vstack((kron(self.extrap_south, ident), kron(self.extrap_north, ident))).T.clone()
            self.extrap_z = torch.vstack((kron(self.extrap_down, ident), kron(self.extrap_up, ident))).T.clone()

            self.derivative_x = kron(ident, self.diff_solpt).T.clone()
            self.derivative_y = kron(self.diff_solpt, ident).T.clone()
            self.derivative_z = kron(self.diff_solpt, ident).T.clone()

            corr_down = self.diff_ext[1:-1, 0]
            corr_up = self.diff_ext[1:-1, -1]
            self.correction_DU = torch.vstack((kron(corr_down, ident), kron(corr_up, ident)))

            self.correction_SN = torch.vstack((kron(corr_down, ident), kron(corr_up, ident)))

            corr_west = self.diff_ext[1:-1, 0]
            corr_east = self.diff_ext[1:-1, -1]
            self.correction_WE = torch.vstack((kron(ident, corr_west), kron(ident, corr_east)))
        # Cast only completed operators: runtime products must match the state dtype.
        if self.dtype != build_dtype:
            for name, value in vars(self).items():
                if hasattr(value, "dtype") and value.dtype == build_dtype:
                    setattr(self, name, value.to(self.dtype))

        if check_skewcentrosymmetry(self.diff_ext) is False:
            raise ValueError("The stored differentiation matrix lost skew-centrosymmetry during precision conversion")

        assert self.extrap_x.dtype == self.dtype
        assert self.extrap_y.dtype == self.dtype
        assert self.extrap_z.dtype == self.dtype
        assert self.derivative_x.dtype == self.dtype
        assert self.derivative_y.dtype == self.dtype
        assert self.derivative_z.dtype == self.dtype
        assert self.correction_DU.dtype == self.dtype
        assert self.correction_SN.dtype == self.dtype
        assert self.correction_WE.dtype == self.dtype

    def make_filter_3d(self, strength: float, order: int, cutoff: float, geom: Geometry):
        if not getattr(geom, "is_3d_euler_grid", False):
            raise TypeError("The 3D exponential filter requires a three-dimensional Euler geometry")
        if geom.num_solpts < 2:
            raise ValueError("The 3D exponential filter requires at least two solution points")
        if strength < 0.0:
            raise ValueError("The exponential-filter strength must be non-negative")
        if order <= 0 or order % 2:
            raise ValueError("The exponential-filter order must be a positive even integer")
        if not 0.0 <= cutoff <= 1.0:
            raise ValueError("The exponential-filter cutoff must lie in [0, 1]")

        modes = torch.arange(geom.num_solpts, dtype=self.dtype) / (geom.num_solpts - 1)
        attenuation = torch.ones_like(modes)
        filtered = modes > cutoff
        attenuation[filtered] = torch.exp(-strength * ((modes[filtered] - cutoff) / (1.0 - cutoff)) ** order)

        vandermonde = legvander(geom.solutionPoints, geom.num_solpts - 1).to(self.dtype)
        filter_1d = vandermonde @ torch.diag(attenuation) @ torch.linalg.inv(vandermonde)
        identity_1d = torch.eye(geom.num_solpts, dtype=self.dtype)
        identity_2d = torch.eye(geom.num_solpts**2, dtype=self.dtype)
        filter_x = kron(identity_2d, filter_1d).T
        filter_y = kron(identity_1d, kron(filter_1d, identity_1d)).T
        filter_z = kron(filter_1d, identity_2d).T
        return (filter_x @ filter_y) @ filter_z

    @staticmethod
    def apply_filter_3d(Q: Tensor, metric: "Metric5D", filter_matrix: Tensor):
        """Filter metric-weighted conservative variables, converting Euler velocity before and after."""
        fields = metric.volume
        conservative = Q.clone()
        conservative[1:4] = Q[0] * fields.raise_velocity(Q[1:4])
        filtered = ((fields.volume_jacobian * conservative) @ filter_matrix) / fields.volume_jacobian
        filtered[1:4] = fields.lower_velocity(filtered[1:4] / filtered[0])
        return filtered


def diffmat(points) -> NDArray:
    """Create a 2D differentiation matrix for the given set of points."""
    M = len(points)
    D = numpy.zeros((M, M))

    x = sympy.symbols("x")
    for i in range(M):
        dL = sympy.diff(lagrange_poly(x, M - 1, i, points))
        for j in range(M):
            if i != j:
                D[j, i] = dL.subs(x, points[j])
        D[i, i] = dL.subs(x, points[i])

    return D


def lagrange_poly(x: sympy.Symbol, order: int, i: int, xi):
    """Create a symbolic Lagrange polynomial basis function."""
    index = list(range(order + 1))
    index.pop(i)
    return sympy.prod([(x - xi[j]) / (xi[i] - xi[j]) for j in index])


def check_skewcentrosymmetry(m: Tensor) -> bool:
    if m.ndim != 2:
        raise numpy.linalg.LinAlgError("Input matrix is not 2-dimensional!")

    n, _ = m.shape
    middle_row = 0

    if n % 2 == 0:
        middle_row = int(n / 2)
    else:
        middle_row = int(n / 2 + 1)

        if m[middle_row - 1, middle_row - 1] != 0.0:
            print()
            print(
                f"When the order is odd, the central entry of a skew-centrosymmetric matrix must be zero.\n"
                f"Actual value is {m[middle_row - 1, middle_row - 1]}"
            )
            return False

    for i in range(middle_row):
        for j in range(n):
            if m[i, j] != -m[n - i - 1, n - j - 1]:
                print("Non skew-centrosymmetric entries detected:", (m[i, j], m[n - i - 1, n - j - 1]))
                return False

    return True


def legvander(x: Tensor, deg: int) -> Tensor:
    """
    NumPy's legvander, slightly modified to work with any array type.

    See: https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.legvander.html#numpy-polynomial-legendre-legvander
    """

    dims = (deg + 1,) + x.shape
    v = torch.empty(dims, dtype=x.dtype)

    v[0] = 1
    if deg > 0:
        v[1] = x
        for i in range(2, deg + 1):
            v[i] = (v[i - 1] * x * (2 * i - 1) - v[i - 2] * (i - 1)) / i
    return torch.moveaxis(v, 0, -1)
