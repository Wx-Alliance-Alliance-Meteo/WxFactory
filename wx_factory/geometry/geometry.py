from abc import ABC, abstractmethod
from typing import Optional

from mpi4py import MPI
import numpy
from numpy.typing import NDArray
import sympy

from ..device import Device
from .quadrature import gauss_legendre


def cast_double_arrays(obj, xp, dtype) -> None:
    """Cast every double-precision array attribute of `obj` to `dtype`, in place.

    Geometry and metric terms are built in double precision, where the differencing of the terrain
    is most accurate, and then stored in the working precision. This walks the object's attributes
    and downcasts the double arrays, leaving integer indices, masks and already-single arrays alone.
    Works for both NumPy/CuPy arrays (`astype`) and PyTorch tensors (`to`).
    """
    if dtype == xp.float64:
        return

    for name, value in vars(obj).items():
        if hasattr(value, "dtype") and hasattr(value, "shape") and value.dtype == xp.float64:
            cast = value.astype(dtype) if hasattr(value, "astype") else value.to(dtype)
            setattr(obj, name, cast)


class Geometry(ABC):
    """
    Abstract class that groups different geometries
    """

    def __init__(
        self,
        num_solpts: int,
        num_elements_horizontal: int,
        num_elements_vertical: int,
        total_num_elements_horizontal: int,
        device: Device,
        verbose: Optional[bool] = False,
    ) -> None:
        self.device = device
        xp = self.device.xp
        self.dtype = self.device.real_dtype

        ## Element properties -- solution and extension points
        # Gauss-Legendre solution points
        solutionPoints_sym, solutionPoints, glweights = gauss_legendre(num_solpts, xp)
        if verbose and self.device.comm.rank == 0:
            print(f"Solution points : {solutionPoints}")
            print(f"GL weights : {glweights}")

        # Extend the solution points to include -1 and 1
        extension = xp.append(xp.append(xp.array([-1.0]), solutionPoints), xp.array([1.0]))
        extension_sym = solutionPoints_sym.copy()
        extension_sym.insert(0, sympy.sympify("-1"))
        extension_sym.append(sympy.sympify("1"))

        self.num_solpts = num_solpts
        self.num_elements_horizontal = num_elements_horizontal
        self.num_elements_vertical = num_elements_vertical
        self.total_num_elements_horizontal = total_num_elements_horizontal
        self.solutionPoints = xp.asarray(solutionPoints)
        self.solutionPoints_sym = solutionPoints_sym
        self.glweights = xp.asarray(glweights)
        self.extension = xp.asarray(extension)
        self.extension_sym = extension_sym

    @abstractmethod
    def to_single_block(self, a: NDArray) -> NDArray:
        """Convert an array of values over this grid (which be may organized as a list of elements)
        into a single block of data (2D or 3D)."""
