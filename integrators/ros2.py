import numpy
from time import time
from typing import Callable

from mpi4py import MPI

from common.program_options import Configuration
from .integrator            import Integrator
from solvers                import fgmres, matvec_rat, SolverInfo

class Ros2(Integrator):
   def __init__(self, param: Configuration, rhs_handle: Callable, preconditioner=None) -> None:
      super().__init__(param, preconditioner)
      self.rhs_handle     = rhs_handle
      self.tol            = param.tolerance
      self.gmres_restart  = param.gmres_restart

   def __step__(self, Q: numpy.ndarray, dt: float):

      rhs    = self.rhs_handle(Q)
      Q_flat = Q.flatten()

      def A(v):
         return matvec_rat(v, dt, Q, rhs, self.rhs_handle)

      b = A(Q_flat) + rhs.flatten() * dt

      maxiter = 20000 // self.gmres_restart
      if self.preconditioner is not None:
         # maxiter = 200 // self.gmres_restart
         maxiter = 420 // self.gmres_restart
         maxiter = min(2, maxiter)

      t0 = time()
#      Qnew, local_error, num_iter, flag, residuals = fgmres(
#         A, b, x0=Q_flat, tol=self.tol, restart=100, maxiter=None, preconditioner=self.preconditioner, verbose=False)

      from solvers.gcrot import gcrot
      Qnew, local_error, num_iter, flag, residuals = gcrot(A, b, x0=Q_flat, tol=self.tol)
      t1 = time()
      write_solver_stats(num_iter, t1 - t0, flag, residuals)
      local_error = numpy.linalg.norm(b-A@Qnew)/numpy.linalg.norm(b)

      if flag == 0:
         print(f'GCROT converged at iteration {num_iter} in {t1 - t0:4.1f} s to a solution with'
               f' relative residual norm {local_error : .2e}')
      else:
         print(f'GCROT stagnation/interruption at iteration {num_iter} in {t1 - t0:4.1f} s, returning a solution with'
               f' relative local error {local_error: .2e}')

      return numpy.reshape(Qnew, Q.shape)
