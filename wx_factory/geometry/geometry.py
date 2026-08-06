from abc import ABC, abstractmethod

import sympy
import torch
from numpy.typing import NDArray

from ..context import Context
from .quadrature import gauss_legendre


def cast_double_arrays(obj, dtype) -> None:
    """Cast every double-precision array attribute of `obj` to `dtype`, in place.

    Geometry and metric terms are built in double precision, where the differencing of the terrain
    is most accurate, and then stored in the working precision. This walks the object's attributes
    and downcasts the double arrays, leaving integer indices, masks and already-single arrays alone.
    """
    if dtype == torch.float64:
        return

    for name, value in vars(obj).items():
        if hasattr(value, "dtype") and hasattr(value, "shape") and value.dtype == torch.float64:
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
        context: Context,
        verbose: bool | None = False,
    ) -> None:
        self.context = context
        self.dtype = self.context.real_dtype

        ## Element properties -- solution and extension points
        # Gauss-Legendre solution points
        solutionPoints_sym, solutionPoints, glweights = gauss_legendre(num_solpts)
        if verbose and self.context.comm.rank == 0:
            print(f"Solution points : {solutionPoints}")
            print(f"GL weights : {glweights}")

        # Extend the solution points to include -1 and 1
        extension = torch.cat([torch.tensor([-1.0]), solutionPoints, torch.tensor([1.0])])
        extension_sym = solutionPoints_sym.copy()
        extension_sym.insert(0, sympy.sympify("-1"))
        extension_sym.append(sympy.sympify("1"))

        self.num_solpts = num_solpts
        self.num_elements_horizontal = num_elements_horizontal
        self.num_elements_vertical = num_elements_vertical
        self.total_num_elements_horizontal = total_num_elements_horizontal
        self.solutionPoints = torch.asarray(solutionPoints)
        self.glweights = torch.asarray(glweights)
        self.extension = torch.asarray(extension)
        self.extension_sym = extension_sym
        self.z_levels = {""}

    @abstractmethod
    def to_single_block(self, a: NDArray) -> NDArray:
        """Convert an array of values over this grid (which be may organized as a list of elements)
        into a single block of data (2D or 3D)."""
