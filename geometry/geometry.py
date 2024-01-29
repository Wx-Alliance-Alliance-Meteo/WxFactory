from typing import Optional

from mpi4py import MPI
import numpy
import sympy

from main_gef import module_from_name
from .quadrature import gauss_legendre


class Geometry:
   """
   Abstract class that groups different geometries
   """
   def __init__(self, nbsolpts: int, grid_type: str, array_module: str, verbose: Optional[bool] = False) -> None:
      ## Element properties -- solution and extension points
      self.array_module_name = array_module
      self.array_module = module_from_name(array_module)
      xp = self.array_module

      # Gauss-Legendre solution points
      solutionPoints_sym, solutionPoints, glweights = gauss_legendre(nbsolpts, xp)
      if verbose and MPI.COMM_WORLD.rank == 0:
         print(f'Solution points : {solutionPoints}')
         print(f'GL weights : {glweights}')

      # Extend the solution points to include -1 and 1
      extension = xp.append(xp.append(xp.array([-1.0]), solutionPoints), xp.array([1.0]))
      extension_sym = solutionPoints_sym.copy()
      extension_sym.insert(0, sympy.sympify('-1'))
      extension_sym.append(sympy.sympify('1'))

      self.nbsolpts = nbsolpts
      self.solutionPoints = xp.asarray(solutionPoints)
      self.solutionPoints_sym = solutionPoints_sym
      self.glweights = xp.asarray(glweights)
      self.extension = xp.asarray(extension)
      self.extension_sym = extension_sym

      ##
      self.grid_type = grid_type
