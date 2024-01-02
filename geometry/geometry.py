from typing import Optional

from mpi4py import MPI
import numpy
import sympy

from common.program_options import Configuration
from .quadrature import gauss_legendre

# typing
from typing import Self

from main_gef import array

class Geometry:
   """
   Abstract class that groups different geometries
   """
   def __init__(self, nbsolpts: int, grid_type: str, verbose: Optional[bool] = False) -> None:
      ## Element properties -- solution and extension points

      # Gauss-Legendre solution points
      solutionPoints_sym, solutionPoints, glweights = gauss_legendre(nbsolpts)
      if verbose and MPI.COMM_WORLD.rank == 0:
         print(f'Solution points : {solutionPoints}')
         print(f'GL weights : {glweights}')

      # Extend the solution points to include -1 and 1
      extension = numpy.concatenate(((-1,), solutionPoints, (1,)), dtype=numpy.float64)
      extension_sym = solutionPoints_sym.copy()
      extension_sym.insert(0, sympy.sympify('-1'))
      extension_sym.append(sympy.sympify('1'))

      self.nbsolpts = nbsolpts
      self.solutionPoints = array.asarray(solutionPoints)
      self.solutionPoints_sym = solutionPoints_sym
      self.glweights = array.asarray(glweights)
      self.extension = array.asarray(extension)
      self.extension_sym = extension_sym

      ##
      self.grid_type = grid_type
