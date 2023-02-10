from mpi4py import MPI

from solvers.nonlin  import newton_krylov
from .stepper        import Stepper

class Imex2(Stepper):
   def __init__(self, rhs_exp, rhs_imp, tol):
      super().__init__()

      if MPI.COMM_WORLD.size > 1:
         raise ValueError(f'RosExp2 has only been tested with 1 PE. Gotta make sure it works with more than that.')

      self.rhs_exp = rhs_exp
      self.rhs_imp = rhs_imp
      self.tol = tol

   def __step__(self, Q, dt):
      rhs = Q + dt/2 * self.rhs_exp(Q)
      def g(v): return v - dt/2 * self.rhs_imp(v) - rhs
      Y1, _, _ = newton_krylov(g, Q)

      # Update solution
      return Q + dt * (self.rhs_imp(Y1) + self.rhs_exp(Y1))
