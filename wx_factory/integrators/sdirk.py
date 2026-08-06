import math
from collections.abc import Callable

from ..common.configuration import Configuration
from ..solvers import newton_krylov
from .integrator import Integrator


class SDIRKLstable(Integrator):
    def __init__(self, param: Configuration, rhs_handle: Callable, *, context=None, preconditioner=None) -> None:
        super().__init__(param, context=context, preconditioner=preconditioner)
        self.rhs = rhs_handle
        self.tol = param.tolerance
        self.sdirkparam = 1.0 + 1.0 / math.sqrt(2.0)

    def SDIRKLstable_system1(self, Q1, Q, dt, rhs):
        return (Q1 - Q) / dt - self.sdirkparam * rhs(Q1)

    def SDIRKLstable_system2(self, Q2, Q, Q1, dt, rhs):
        return (Q2 - Q) / dt - (1.0 - 2.0 * self.sdirkparam) * rhs(Q1) - self.sdirkparam * rhs(Q2)

    def __step__(self, Q, dt):
        def SDIRK_fun1(Q1):
            return self.SDIRKLstable_system1(Q1, Q, dt, self.rhs)

        def SDIRK_fun2(Q2):
            return self.SDIRKLstable_system2(Q2, Q, Q1, dt, self.rhs)

        maxiter = None
        if self.preconditioner is not None:
            self.preconditioner.prepare(dt, Q)
            maxiter = 800

        # Update solution
        Q1, _, _ = newton_krylov(
            SDIRK_fun1,
            Q,
            f_tol=self.tol,
            restart=30,
            preconditioner=self.preconditioner,
            verbose=False,
            maxiter=maxiter,
        )
        Q2, _, _ = newton_krylov(
            SDIRK_fun2,
            Q,
            f_tol=self.tol,
            restart=30,
            preconditioner=self.preconditioner,
            verbose=False,
            maxiter=maxiter,
        )
        newQ = Q + dt * (0.5 * self.rhs(Q1) + 0.5 * self.rhs(Q2))

        return newQ


REGISTRY = {
    "sdirk": lambda cfg, rhs, prec, context: SDIRKLstable(cfg, rhs.full, preconditioner=prec, context=context),
}
