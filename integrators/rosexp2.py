from time    import time
from typing  import Callable

from mpi4py  import MPI
import numpy

from common.program_options import Configuration
from .integrator            import Integrator, SolverInfo
from solvers                import fgmres, matvec_fun, matvec_rat, pmex

class RosExp2(Integrator):
   def __init__(self, param: Configuration, rhs_full: Callable, rhs_imp: Callable, preconditioner):
      super().__init__(param, preconditioner)

      if MPI.COMM_WORLD.size > 1:
         raise ValueError(f'RosExp2 has only been tested with 1 PE. Gotta make sure it works with more than that.')

      self.rhs_full = rhs_full
      self.rhs_imp = rhs_imp
      self.tol = param.tolerance
      self.gmres_restart = param.gmres_restart

   def __step__(self, Q, dt):
      rhs_full = self.rhs_full(Q)
      rhs_imp = self.rhs_imp(Q)

      Q_flat = Q.flatten()
      n = len(Q_flat)

      def J_exp(v):
         return matvec_fun(v, dt, Q, rhs_full, self.rhs_full) \
                - matvec_fun(v, dt, Q, rhs_imp, self.rhs_imp)

      vec = numpy.zeros((2, n))
      vec[1,:] = rhs_full.flatten()

      tic = time()
      phiv, stats = pmex([1.], J_exp, vec, tol=self.tol,task1=False)
      time_exp = time() - tic
      if MPI.COMM_WORLD.rank == 0:
         print(f'PMEX convergence at iteration {stats[2]} (using {stats[0]} internal substeps and'
               f' {stats[1]} rejected expm)')

      tic = time()
      def A(v):
         return matvec_rat(v, dt, Q, rhs_imp, self.rhs_imp)
      b = ( A(Q_flat) + phiv * dt ).flatten()
      Q_x0 = Q_flat.copy()
      Qnew, norm_r, norm_b, num_iter, flag, residuals = fgmres(
         A, b, x0=Q_x0, tol=self.tol, restart=self.gmres_restart, maxiter=None, preconditioner=self.preconditioner,
         verbose=self.verbose_solver)
      time_imp = time() - tic

      self.solver_info = SolverInfo(flag, time_imp, num_iter, residuals)

      if MPI.COMM_WORLD.rank == 0:
         result_type = 'convergence' if flag == 0 else 'stagnation/interruption'
         print(f'FGMRES {result_type} at iteration {num_iter} in {time_imp:4.1f} s to a solution with'
               f' relative residual {norm_r/norm_b: .2e}')

         print(f'Elapsed time: exponential {time_exp:.3f} secs ; implicit {time_imp:.3f} secs')

      return numpy.reshape(Qnew, Q.shape)
