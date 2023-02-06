from time import time

from mpi4py import MPI
import numpy
from scipy.sparse.linalg import LinearOperator

from Output.solver_stats import write_solver_stats
from solvers.linsol      import fgmres
from solvers.matvec      import matvec_fun
from solvers.pmex        import pmex
from integrators.stepper     import Stepper

class PartRosExp2(Stepper):
   def __init__(self, rhs_full, rhs_imp, tol, preconditioner):
      super().__init__(preconditioner)

      if MPI.COMM_WORLD.size > 1:
         raise ValueError(f'RosExp2 has only been tested with 1 PE. Gotta make sure it works with more than that.')

      self.rhs_full = rhs_full
      self.rhs_imp = rhs_imp
      self.tol = tol

   def __step__(self, Q: numpy.ndarray, dt: float):

      rhs_full = self.rhs_full(Q)
      rhs_imp = self.rhs_imp(Q)
      f_imp = rhs_imp.flatten()
      f_exp = (rhs_full - rhs_imp).flatten()

      def J_full(v): return matvec_fun(v, dt, Q, rhs_full, self.rhs_full)
      def J_imp(v):  return matvec_fun(v, dt, Q, rhs_imp, self.rhs_imp)
      def J_exp(v):  return J_full(v) - J_imp(v)

      Q_flat = Q.flatten()
      n = len(Q_flat)

      vec = numpy.zeros((2, n))
      vec[0,:] = 0.5 * f_imp
      vec[1,:] = f_exp.copy()

      tic = time()
      phiv, stats = pmex([1.], J_exp, vec, tol=self.tol,task1=False)
      time_exp = time() - tic
      print(f'PMEX converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)')

      tic = time()
      A = LinearOperator((n,n), matvec = lambda v: v - J_imp(v) / 2)
      b = ( A(Q_flat) + (phiv + 0.5 * f_imp) * dt ).flatten()
      Q_x0 = Q_flat.copy()
      Qnew, local_error, num_iter, flag, residuals = fgmres(
         A, b, x0=Q_x0, tol=self.tol, restart=100, maxiter=None, preconditioner=self.preconditioner, verbose=False)
      time_imp = time() - tic

      write_solver_stats(num_iter, time_imp, flag, residuals)

      if flag == 0:
         print(f'FGMRES converged at iteration {num_iter} in {time_imp:4.1f} s to a solution with'
               f' relative local error {local_error : .2e}')
      else:
         print(f'FGMRES stagnation/interruption at iteration {num_iter} in {time_imp:4.1f} s, returning a solution with'
               f' relative local error {local_error: .2e}')

      print(f'Elapsed time: exponential {time_exp:.3f} secs ; implicit {time_imp:.3f} secs')

      return numpy.reshape(Qnew, Q.shape)
