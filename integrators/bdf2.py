import numpy
from time import time

from solvers import newton_krylov
from .integrator import Integrator, SolverInfo


class Bdf2(Integrator):
    def __init__(self, param, rhs, preconditioner=None, init_substeps=1, **kwargs):
        super().__init__(param, **kwargs)
        self.rhs = rhs
        self.tol = param.tolerance
        self.init_substeps = init_substeps
        self.Qprev = None

    def __step__(self, Q, dt):
        t0 = time()
        if self.Qprev is None:
            # Initialize with the backward Euler method
            newQ = Q.copy()
            for _ in range(self.init_substeps):
                init_dt = dt / self.init_substeps
                nonlin_fun = lambda Q_plus: (Q_plus - newQ) / init_dt - 0.5 * self.rhs(Q_plus)

                newQ, num_iter, residuals = newton_krylov(nonlin_fun, newQ, f_tol=self.tol)
        else:
            maxiter = None

            def nonlin_fun(Q_plus):
                return (Q_plus - 4.0 / 3.0 * Q + 1.0 / 3.0 * self.Qprev) / dt - 2.0 / 3.0 * self.rhs(Q_plus)

            if self.preconditioner is not None:
                self.preconditioner.prepare(dt, Q, self.Qprev)
                maxiter = 800
            newQ, num_iter, residuals = newton_krylov(
                nonlin_fun, Q, f_tol=self.tol, fgmres_precond=self.preconditioner, verbose=False, maxiter=maxiter
            )
        t1 = time()

        self.solver_info = SolverInfo(0, t1 - t0, num_iter, residuals)

        self.Qprev = Q.copy()

        return numpy.reshape(newQ, Q.shape)
