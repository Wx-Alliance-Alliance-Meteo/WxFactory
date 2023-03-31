from typing import Callable

from common.program_options import Configuration
from .integrator            import Integrator, SolverInfo

class Tvdrk3(Integrator):
   def __init__(self, param: Configuration, rhs: Callable):
      super().__init__(param, preconditioner=None)
      self.rhs = rhs

   def __step__(self, Q, dt):
      Q1 = Q + self.rhs(Q) * dt
      #Q = Q + self.rhs(Q) * dt
      Q2 = 0.75 * Q + 0.25 * Q1 + 0.25 * self.rhs(Q1) * dt
      Q = 1.0 / 3.0 * Q + 2.0 / 3.0 * Q2 + 2.0 / 3.0 * self.rhs(Q2) * dt

      self.solver_info = SolverInfo(total_num_it=1)
      return Q
