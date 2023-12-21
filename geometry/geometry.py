from mpi4py import MPI
import numpy
import sympy

from common.program_options import Configuration
from .quadrature import gauss_legendre

# typing
from typing import Self

class Geometry:
   """
   Abstract class that groups different geometries
   """
   def __init__(self: Self, nbsolpts: int, grid_type: str, param: Configuration) -> None:
      ## Element properties -- solution and extension points

      verbose = False

      # Gauss-Legendre solution points
      solutionPoints_sym, solutionPoints, glweights = gauss_legendre(nbsolpts)
      if MPI.COMM_WORLD.rank == 0 and verbose:
         print(f'Solution points : {solutionPoints}')
         print(f'GL weights : {glweights}')

      # Extend the solution points to include -1 and 1
      extension = numpy.concatenate(((-1,), solutionPoints, (1,)), dtype=numpy.float64)
      extension_sym = solutionPoints_sym.copy()
      extension_sym.insert(0, sympy.sympify('-1'))
      extension_sym.append(sympy.sympify('1'))

      self.nbsolpts = nbsolpts
      self.solutionPoints = param.array_module.asarray(solutionPoints)
      self.solutionPoints_sym = solutionPoints_sym
      self.glweights = param.array_module.asarray(glweights)
      self.extension = param.array_module.asarray(extension)
      self.extension_sym = extension_sym

      ##
      self.grid_type = grid_type
