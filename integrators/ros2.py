import numpy
import scipy
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
      self.X = None
      self.B = None
      self.nstep = 0
      self.bsize = 8 # TODO : config
      self.pointer = 0

   def __step__(self, Q: numpy.ndarray, dt: float):

      rhs    = self.rhs_handle(Q)
      Q_flat = Q.flatten()
      n      = Q_flat.shape[0]

      def A(v):
         return matvec_rat(v, dt, Q, rhs, self.rhs_handle)

      b = A(Q_flat) + rhs.flatten() * dt

      maxiter = 20000 // self.gmres_restart
      if self.preconditioner is not None:
         maxiter = 200 // self.gmres_restart

      print('Residual when first guess is previous sol:', numpy.linalg.norm(b - A(Q_flat)))

      if self.nstep+1 >= 4 and self.X is not None:
         BB = self.B.T
         XX = self.X.T

         pinvBB = numpy.linalg.pinv(BB)
         approxInvA = scipy.sparse.linalg.LinearOperator((n,n), matvec=lambda vv: XX @ (pinvBB @ vv))

         Q_x0 = approxInvA(b)
         extrap = numpy.linalg.norm(b - A(Q_x0))
         print('Residual after ML:', extrap)

      else:
         Q_x0 = Q_flat.copy()


      t0 = time()
      Qnew, norm_r, norm_b, num_iter, flag, residuals = fgmres(
         A, b, x0=Q_x0, tol=self.tol, restart=self.gmres_restart, maxiter=maxiter, preconditioner=self.preconditioner,
         verbose=self.verbose_solver)
      t1 = time()

      self.solver_info = SolverInfo(flag, t1 - t0, num_iter, residuals)

      if MPI.COMM_WORLD.rank == 0:
         result_type = 'convergence' if flag == 0 else 'stagnation/interruption'
         print(f'FGMRES {result_type} at iteration {num_iter} in {t1 - t0:4.3f} s to a solution with'
               f' relative residual {norm_r/norm_b : .2e}')


      if self.X is None:
         self.B = numpy.zeros((self.bsize, n))
         self.X = numpy.zeros((self.bsize, n))

      self.X[self.pointer, :] = Qnew
      self.B[self.pointer, :] = b

      self.nstep += 1
      self.pointer = self.nstep % self.bsize

      return numpy.reshape(Qnew, Q.shape)
