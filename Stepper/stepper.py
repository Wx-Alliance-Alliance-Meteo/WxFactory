from abc       import ABC, abstractmethod
from itertools import combinations
import math
from time      import time
from typing    import Optional

import numpy

from Precondition.multigrid import Multigrid

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

class StrangSplitting(Stepper):
   def __init__(self, scheme1, scheme2):
      super().__init__()
      self.scheme1 = scheme1
      self.scheme2 = scheme2

   def __step__(self, Q, dt):
      Q = self.scheme1.step(Q, dt/2)
      Q = self.scheme2.step(Q, dt)
      return Q

class scipy_counter(object): # TODO : tempo
   def __init__(self, disp=False):
      self._disp = disp
      self.niter = 0
   def __call__(self, rk=None):
      self.niter += 1
      if self._disp:
         print(f'iter {self.niter:3d}\trk = {str(rk)}')
   def nb_iter(self):
      return self.niter

# Computes the coefficients for stiffness resilient exponential methods based on node values c
def alpha_coeff(c):
   m = len(c)
   p = m + 2
   alpha = numpy.zeros((m, m))
   for i in range(m):
      c_no_i = [cc for (j, cc) in enumerate(c) if j != i]
      denom = c[i] ** 2 * math.prod([c[i] - cl for cl in c_no_i])
      for k in range(m):
         sp = sum([math.prod(v) for v in combinations(c_no_i, m - k - 1)])
         alpha[k, i] = (-1) ** (m - k + 1) * math.factorial(k + 2) * sp / denom

   return alpha
