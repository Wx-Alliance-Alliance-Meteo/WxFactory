import numpy
from time import time

from .integrator import Integrator, SolverInfo
from ..solvers import newton_krylov


class CrankNicolson(Integrator):
    def __init__(self, param, rhs, *, device=None, preconditioner=None):
        super().__init__(param, device=device, preconditioner=preconditioner)
        self.rhs = rhs
        self.tol = param.tolerance

    def CN_system(self, Q_plus, Q, dt, rhs):
        return (Q_plus - Q) / dt - 0.5 * (rhs(Q_plus) + rhs(Q))

    def __step__(self, Q, dt):
        def CN_fun(Q_plus):
            return self.CN_system(Q_plus, Q, dt, self.rhs)

        maxiter = None
        if self.preconditioner is not None:
            self.preconditioner.prepare(dt, Q)
            maxiter = 800

        # Update solution
        t0 = time()
        newQ, num_iter, residuals = newton_krylov(
            CN_fun,
            Q,
            f_tol=self.tol,
            restart=30,
            preconditioner=self.preconditioner,
            verbose=False,
            maxiter=maxiter,
        )
        t1 = time()

        self.solver_info = SolverInfo(0, t1 - t0, num_iter, residuals)

        return numpy.reshape(newQ, Q.shape)


REGISTRY = {
    "crank_nicolson": lambda cfg, rhs, prec, dev: CrankNicolson(cfg, rhs.full, preconditioner=prec, device=dev),
}
