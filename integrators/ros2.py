import numpy
from time import time
from typing import Callable

from mpi4py import MPI

from solvers.linsol      import fgmres
from solvers.matvec      import matvec_rat
from integrators.stepper     import Stepper
from output.solver_stats import write_solver_stats

class Ros2(Stepper):
   def __init__(self, rhs_handle: Callable, tol: float, preconditioner=None) -> None:
      super().__init__(preconditioner)
      self.rhs_handle     = rhs_handle
      self.tol            = tol

   def __step__(self, Q: numpy.ndarray, dt: float):

      rhs    = self.rhs_handle(Q)
      Q_flat = Q.flatten()

      A =lambda v: matvec_rat(v, dt, Q, rhs, self.rhs_handle)
      b = A(Q_flat) + rhs.flatten() * dt

      t0 = time()
      Qnew, norm_r, norm_b, num_iter, flag, residuals = fgmres(
         A, b, x0=Q_flat, tol=self.tol, restart=100, maxiter=None, preconditioner=self.preconditioner, verbose=False)
      t1 = time()

      write_solver_stats(num_iter, t1 - t0, flag, residuals)

      if MPI.COMM_WORLD.rank == 0:
         if flag == 0:
            print(f'FGMRES converged at iteration {num_iter} in {t1 - t0:4.1f} s to a solution with'
                  f' relative residual {norm_r/norm_b : .2e}')
         else:
            print(f'FGMRES stagnation/interruption at iteration {num_iter} in {t1 - t0:4.1f} s, returning a solution with'
                  f' relative residual {norm_r/norm_b : .2e}')

      return numpy.reshape(Qnew, Q.shape)
