from typing import Optional

from mpi4py import MPI
import numpy
import sympy

from .quadrature import gauss_legendre

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
      extension = numpy.append(numpy.append([-1], solutionPoints), [1])
      extension_sym = solutionPoints_sym.copy()
      extension_sym.insert(0, sympy.sympify('-1'))
      extension_sym.append(sympy.sympify('1'))

      self.nbsolpts = nbsolpts
      self.solutionPoints = solutionPoints
      self.solutionPoints_sym = solutionPoints_sym
      self.glweights = glweights
      self.extension = extension
      self.extension_sym = extension_sym

      ##
      self.grid_type = grid_type
