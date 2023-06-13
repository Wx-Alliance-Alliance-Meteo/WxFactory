import numpy
import scipy
import math
from time import time
from typing import Callable

from mpi4py import MPI

from common.program_options import Configuration
from .integrator            import Integrator
from solvers                import fgmres, matvec_rat, SolverInfo, newton_krylov


class BackwardEuler(Integrator):
   def __init__(self, param: Configuration, rhs_handle: Callable, preconditioner=None) -> None:
      super().__init__(param, preconditioner)
      self.rhs = rhs_handle
      self.tol = param.tolerance

   def BE_system(self, Q_plus, Q, dt, rhs):
      return (Q_plus - Q) / dt - rhs(Q_plus)

   def __step__(self, Q, dt):
      def BE_fun(Q_plus): return self.BE_system(Q_plus, Q, dt, self.rhs)

      maxiter = None
      if self.preconditioner is not None:
         self.preconditioner.prepare(dt, Q)
         maxiter = 800

      # Update solution
      t0 = time()
      newQ, nb_iter, residuals = newton_krylov(BE_fun, Q, f_tol=self.tol, fgmres_restart=30,
         fgmres_precond=self.preconditioner, verbose=False, maxiter=maxiter)
      t1 = time()

      self.solver_info = SolverInfo(0, t1 - t0, nb_iter, residuals)

      return numpy.reshape(newQ, Q.shape)
