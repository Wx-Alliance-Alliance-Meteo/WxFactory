from collections.abc import Callable
from time import time

import numpy

from ..common.configuration import Configuration
from ..solvers import SolverInfo, newton_krylov
from .integrator import Integrator


class BackwardEuler(Integrator):
    def __init__(self, param: Configuration, rhs_handle: Callable, *, device=None, preconditioner=None) -> None:
        super().__init__(param, device=device, preconditioner=preconditioner)
        self.rhs = rhs_handle
        self.tol = param.tolerance

    def __step__(self, Q, dt):
        def BE_fun(Q_plus):
            return (Q_plus - Q) / dt - self.evaluate_rhs(self.rhs, Q_plus)

        maxiter = None
        if self.preconditioner is not None:
            self.preconditioner.prepare(dt, Q)
            maxiter = 800

        # Update solution
        t0 = time()
        newQ, num_iter, residuals = newton_krylov(
            BE_fun,
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
    "backward_euler": lambda cfg, rhs, prec, dev: BackwardEuler(cfg, rhs.full, preconditioner=prec, device=dev),
}
