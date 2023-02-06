import numpy
from time import time

from Output.solver_stats import write_solver_stats
from solvers.nonlin      import newton_krylov
from integrators.stepper     import Stepper

class CrankNicolson(Stepper):
   def __init__(self, rhs, tol, preconditioner=None):
      super().__init__(preconditioner)
      self.rhs = rhs
      self.tol = tol

   def CN_system(self, Q_plus, Q, dt, rhs):
      return (Q_plus - Q) / dt - 0.5 * ( rhs(Q_plus) + rhs(Q) )

   def __step__(self, Q, dt):
      def CN_fun(Q_plus): return self.CN_system(Q_plus, Q, dt, self.rhs)

      maxiter = None
      if self.preconditioner is not None:
         self.preconditioner.prepare(dt, Q)
         maxiter = 800

      # Update solution
      t0 = time()
      newQ, nb_iter, residuals = newton_krylov(CN_fun, Q, f_tol=self.tol, fgmres_restart=30,
         fgmres_precond=self.preconditioner, verbose=False, maxiter=maxiter)
      t1 = time()

      write_solver_stats(nb_iter, t1 - t0, 0, residuals)

      return numpy.reshape(newQ, Q.shape)
