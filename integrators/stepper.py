from abc       import ABC, abstractmethod
from itertools import combinations
import math
from time      import time
from typing    import Optional
import sys

import numpy

from precondition.multigrid import Multigrid

class Stepper(ABC):
   latest_time: float
   preconditioner: Optional[Multigrid]
   def __init__(self, preconditioner: Optional[Multigrid] = None) -> None:
      self.preconditioner = preconditioner

   @abstractmethod
   def __step__(self, Q: numpy.ndarray, dt: float) -> numpy.ndarray:
      pass

   def step(self, Q: numpy.ndarray, dt: float):
      """ Advance the system forward in time """
      t0 = time()

      if self.preconditioner is not None:
         self.preconditioner.prepare(dt, Q)

      result = self.__step__(Q, dt)

      t1 = time()
      self.latest_time = t1 - t0

      return result

class scipy_counter(object): # TODO : tempo
   def __init__(self, disp=False):
      self._disp = disp
      self.niter = 0
      self.res = 0.0
   def __call__(self, rk=None):
      self.niter += 1
      if rk is not None:
         self.res = float(rk)
      if self._disp:
         print(f'iter {self.niter:3d}\trk = {str(rk)}')
         sys.stdout.flush()
   def nb_iter(self):
      return self.niter

# Computes the coefficients for stiffness resilient exponential methods based on node values c
def alpha_coeff(c):
   m = len(c)
   alpha = numpy.zeros((m, m))
   for i in range(m):
      c_no_i = [cc for (j, cc) in enumerate(c) if j != i]
      denom = c[i] ** 2 * math.prod([c[i] - cl for cl in c_no_i])
      for k in range(m):
         sp = sum([math.prod(v) for v in combinations(c_no_i, m - k - 1)])
         alpha[k, i] = (-1) ** (m - k + 1) * math.factorial(k + 2) * sp / denom

   return alpha
