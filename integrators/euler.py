from common.program_options import Configuration
from .integrator            import Integrator

class ForwardEuler(Integrator):
   def __init__(self, param: Configuration, rhs):
      super().__init__(param, preconditioner=None)
      self.rhs = rhs

   def __step__(self, Q, dt):
      Q = Q + self.rhs(Q) * dt
      return Q


class BackwardEuler(Integrator):
   def __init__(self, rhs, tol, preconditioner=None):
      super().__init__(preconditioner)
      self.rhs = rhs
      self.tol = tol

   def CN_system(self, Q_plus, Q, dt, rhs):
      return (Q_plus - Q) / dt -(rhs(Q_plus))

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

      self.solver_info = SolverInfo(0, t1 - t0, nb_iter, residuals)

      return numpy.reshape(newQ, Q.shape)