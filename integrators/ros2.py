from time import time
from typing import Callable

import numpy
from mpi4py import MPI
import scipy

from common.program_options  import Configuration
from solvers                 import fgmres, MatvecOpRat, SolverInfo
from .integrator             import Integrator
from scripts.eigenvalue_util import gen_matrix
from solvers                 import fgmres, matvec_rat, SolverInfo

class Ros2(Integrator):
   def __init__(self, param: Configuration, rhs_handle: Callable, preconditioner=None) -> None:
      super().__init__(param, preconditioner)
      self.rhs_handle     = rhs_handle
      self.tol            = param.tolerance
      self.gmres_restart  = param.gmres_restart
      self.assembled_A = None
      self.lu = None

   def __step__(self, Q: numpy.ndarray, dt: float):

      rhs    = self.rhs_handle(Q)
      Q_flat = Q.flatten()

      A = MatvecOpRat(dt, Q, rhs, self.rhs_handle)

      b = A(Q_flat) + rhs.flatten() * dt

      maxiter = 20000 // self.gmres_restart
      if self.preconditioner is not None:
         # maxiter = 200 // self.gmres_restart
      if self.assembled_A is None:
         self.assembled_A = gen_matrix(A)
         self.lu = scipy.sparse.linalg.splu(self.assembled_A)

      def custom_precond(v):
         return self.lu.solve(v)

      t0 = time()
      Qnew, norm_r, norm_b, num_iter, flag, residuals = fgmres(
         A, b, x0=Q_flat, tol=self.tol, restart=self.gmres_restart, maxiter=maxiter, preconditioner=custom_precond,
         verbose=self.verbose_solver)
      t1 = time()

      self.solver_info = SolverInfo(flag, t1 - t0, num_iter, residuals)

      if MPI.COMM_WORLD.rank == 0:
         result_type = 'convergence' if flag == 0 else 'stagnation/interruption'
         print(f'FGMRES {result_type} at iteration {num_iter} in {t1 - t0:4.3f} s to a solution with'
               f' relative residual {norm_r/norm_b : .2e}')

      self.failure_flag = flag

      return numpy.reshape(Qnew, Q.shape)
