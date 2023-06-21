import cupy as cp
import numpy

from .geometry import Geometry
from .cubed_sphere import CubedSphere

from .matrices import DFROperators

# typing
from typing import Self
from numpy.typing import NDArray


class CudaDFROperators(DFROperators):

    def __init__(self: Self, grd: Geometry, filter_apply: bool = False, filter_order: int = 8, filter_cutoff: float = 0.25):
        super().__init__(grd, filter_apply, filter_order, filter_cutoff)

        # note: these arrays are constant, W/S/D are equivalent, and E/N/U are equivalent
        # so to avoid unneeded copies, we only use asarray on one of each matrix
        self.extrap_west: NDArray[cp.float64] = cp.asarray(self.extrap_west)
        self.extrap_east: NDArray[cp.float64] = cp.asarray(self.extrap_east)
        self.extrap_south: NDArray[cp.float64] = self.extrap_west
        self.extrap_north: NDArray[cp.float64] = self.extrap_east
        self.extrap_down: NDArray[cp.float64] = self.extrap_west
        self.extrap_up: NDArray[cp.float64] = self.extrap_east

        self.highfilter: NDArray[cp.float64] = cp.asarray(self.highfilter)

        if filter_apply:
            self.V: NDArray[cp.float64] = cp.asarray(self.V)
            self.invV: NDArray[cp.float64] = cp.asarray(self.invV)
            self.filter: NDArray[cp.float64] = cp.asarray(self.filter)

        self.diff_ext: NDArray[cp.float64] = cp.asarray(self.diff_ext)

        self.diff_solpt: NDArray[cp.float64] = cp.asarray(self.diff_solpt)
        self.correction: NDArray[cp.float64] = cp.asarray(self.correction)

        self.diff_solpt_tr: NDArray[cp.float64] = cp.asarray(self.diff_solpt_tr)
        self.correction_tr: NDArray[cp.float64] = cp.asarray(self.correction_tr)

        self.diff: NDArray[cp.float64] = cp.asarray(self.diff)
        self.diff_tr: NDArray[cp.float64] = cp.asarray(self.diff_tr)

        self.quad_weights: NDArray[cp.float64] = cp.asarray(self.quad_weights)
