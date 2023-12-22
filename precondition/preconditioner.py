from abc import ABC, abstractmethod
from typing import Tuple, Optional

from mpi4py import MPI
import numpy

from common.program_options import Configuration
from solvers import MatvecOp

class Preconditioner(MatvecOp, ABC):
   """Describes a matrix-like object that can be used to precondition a linear system."""

   def __init__(self, dtype, shape: Tuple, param: Configuration) -> None:
      super().__init__(self.apply, dtype, shape)
      self.verbose = param.verbose_precond if MPI.COMM_WORLD.rank == 0 else 0

   def __call__(self, vec: numpy.ndarray, x0:Optional[numpy.ndarray] = None, verbose:Optional[int] = None):
      return self.apply(vec, x0, verbose)

   def apply(self, vec: numpy.ndarray, x0: Optional[numpy.ndarray] = None, verbose: Optional[int] = None) \
         -> numpy.ndarray:
      if verbose is None: verbose = self.verbose
      return self.__apply__(vec, x0, verbose)

   @abstractmethod
   def __apply__(self, vec: numpy.ndarray, x0: Optional[numpy.ndarray] = None, verbose: Optional[int] = None) \
         -> numpy.ndarray:
      pass

   # def prepare(self, dt: float, field: numpy.ndarray, prev_field:Optional[numpy.ndarray] = None) -> None:
   #    return self.__prepare__(dt, field, prev_field)

   # @abstractmethod
   # def __prepare__(self, dt: float, field: numpy.ndarray, prev_field:Optional[numpy.ndarray] = None) -> None:
   #    pass
