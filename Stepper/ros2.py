import numpy
from scipy.sparse.linalg import LinearOperator
from time import time

from typing import Callable

from Solver.linsol       import fgmres
from Solver.matvec       import matvec_rat
from Stepper.stepper     import Stepper
from Output.solver_stats import write_solver_stats

class Ros2(Stepper):
   def __init__(self, rhs_handle: Callable, tol: float, preconditioner=None) -> None:
      super().__init__(preconditioner)
      self.rhs_handle     = rhs_handle
      self.tol            = tol

   def __step__(self, Q: numpy.ndarray, dt: float):

      rhs    = self.rhs_handle(Q)
      Q_flat = Q.flatten()
      n      = Q_flat.shape[0]

      A = LinearOperator((n,n), matvec=lambda v: matvec_rat(v, dt, Q, rhs, self.rhs_handle))
      b = A(Q_flat) + rhs.flatten() * dt

      t0 = time()
#      Qnew, local_error, num_iter, flag, residuals = fgmres(
#         A, b, x0=Q_flat, tol=self.tol, restart=100, maxiter=None, preconditioner=self.preconditioner, verbose=False)

      from Solver.gcrot import gcrot
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
