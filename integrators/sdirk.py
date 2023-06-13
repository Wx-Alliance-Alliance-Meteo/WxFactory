import numpy
import scipy
import math
from time import time
from typing import Callable

from mpi4py import MPI

from common.program_options import Configuration
from .integrator            import Integrator
from solvers                import fgmres, matvec_rat, SolverInfo, newton_krylov


class SDIRKLstable(Integrator):
   def __init__(self, param: Configuration, rhs_handle: Callable, preconditioner=None) -> None:
      super().__init__(param, preconditioner)
      self.rhs = rhs_handle
      self.tol = param.tolerance
      self.sdirkparam = 1.0+1.0/math.sqrt(2.0)

   def SDIRKLstable_system1(self, Q1, Q, dt, rhs):
      return (Q1 - Q) / dt - self.sdirkparam * rhs(Q1)
   def SDIRKLstable_system2(self, Q2, Q, Q1, dt, rhs):
      return (Q2 - Q) / dt - (1.0-2.0*self.sdirkparam) * rhs(Q1) - self.sdirkparam*rhs(Q2)
  # def SDIRKLstable_system(self, Q_plus, Q, Q1, Q2, dt, rhs):
  #    return (Q_plus - Q)/dt - 0.5*rhs(Q1) - 0.5*rhs(Q2)

   def __step__(self, Q, dt):
      def SDIRK_fun1(Q1): return self.SDIRKLstable_system1(Q1, Q, dt, self.rhs)
      def SDIRK_fun2(Q2): return self.SDIRKLstable_system2(Q2, Q, Q1, dt, self.rhs)
   #   def SDIRK_fun(Q_plus): return self.SDIRKLstable_system(Q_plus, Q, Q1, Q2, dt, self.rhs)

      maxiter = None
      if self.preconditioner is not None:
         self.preconditioner.prepare(dt, Q)
         maxiter = 800

      # Update solution
      t0 = time()
      Q1, nb_iter, residuals = newton_krylov(SDIRK_fun1, Q, f_tol=self.tol, fgmres_restart=30,
         fgmres_precond=self.preconditioner, verbose=False, maxiter=maxiter)
      Q2, nb_iter, residuals = newton_krylov(SDIRK_fun2, Q, f_tol=self.tol, fgmres_restart=30,
                                             fgmres_precond=self.preconditioner, verbose=False, maxiter=maxiter)
      newQ = Q + dt*(0.5*self.rhs(Q1) + 0.5*self.rhs(Q2))
      t1 = time()

      self.solver_info = SolverInfo(0, t1 - t0, nb_iter, residuals)

      return newQ

