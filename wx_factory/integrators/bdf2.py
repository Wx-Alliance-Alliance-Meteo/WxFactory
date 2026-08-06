import numpy

from ..solvers import newton_krylov
from .integrator import Integrator


class Bdf2(Integrator):
    def __init__(self, param, rhs, *, preconditioner=None, init_substeps=1, context=None):
        super().__init__(param, context=context, preconditioner=preconditioner)
        self.rhs = rhs
        self.tol = param.tolerance
        self.init_substeps = init_substeps
        self.Qprev = None

    def __step__(self, Q, dt):
        if self.Qprev is None:
            # Initialize with the backward Euler method
            newQ = Q.copy()
            init_dt = dt / self.init_substeps
            for _ in range(self.init_substeps):

                def nonlin_fun(Q_plus, previous=newQ):
                    return (Q_plus - previous) / init_dt - 0.5 * self.rhs(Q_plus)

                newQ, _, _ = newton_krylov(nonlin_fun, newQ, f_tol=self.tol)
        else:
            maxiter = None

            def nonlin_fun(Q_plus):
                return (Q_plus - 4.0 / 3.0 * Q + 1.0 / 3.0 * self.Qprev) / dt - 2.0 / 3.0 * self.rhs(Q_plus)

            if self.preconditioner is not None:
                self.preconditioner.prepare(dt, Q, self.Qprev)
                maxiter = 800
            newQ, _, _ = newton_krylov(
                nonlin_fun,
                Q,
                f_tol=self.tol,
                preconditioner=self.preconditioner,
                verbose=False,
                maxiter=maxiter,
            )
        self.Qprev = Q.copy()

        return numpy.reshape(newQ, Q.shape)


REGISTRY = {
    "bdf2": lambda cfg, rhs, prec, ctx: Bdf2(cfg, rhs.full, preconditioner=prec, context=ctx),
}
