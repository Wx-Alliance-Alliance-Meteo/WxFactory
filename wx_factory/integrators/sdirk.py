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
        self.gmres_restart = param.gmres_restart
        self.sdirkparam = 1.0 + 1.0 / math.sqrt(2.0)

    def SDIRKLstable_system1(self, Q1, Q, dt, rhs):
        return (Q1 - Q) / dt - self.sdirkparam * rhs(Q1)

    def __step__(self, Q, dt):
        working, rhs = self._implicit_state_and_rhs(Q, self.rhs)

        def SDIRK_fun1(Q1):
            return self.SDIRKLstable_system1(Q1, working, dt, rhs)

        maxiter = 100
        if self.preconditioner is not None:
            self.preconditioner.prepare(dt, Q)
            maxiter = 800

        Q1, _, _ = newton_krylov(
            SDIRK_fun1,
            working,
            f_tol=self.tol,
            restart=self.gmres_restart,
            krylov_dtype=Q.dtype,
            maxiter_linear=10,
            preconditioner=self.preconditioner,
            verbose=False,
            maxiter=maxiter,
            context=self.context,
            jacobian=self._newton_jacobian(rhs, dt, self.sdirkparam),
        )
        first_rhs = rhs(Q1)

        def SDIRK_fun2(Q2):
            return (Q2 - working) / dt - (1 - 2 * self.sdirkparam) * first_rhs - self.sdirkparam * rhs(Q2)

        Q2, _, _ = newton_krylov(
            SDIRK_fun2,
            working,
            f_tol=self.tol,
            restart=self.gmres_restart,
            krylov_dtype=Q.dtype,
            maxiter_linear=10,
            preconditioner=self.preconditioner,
            verbose=False,
            maxiter=maxiter,
            context=self.context,
            jacobian=self._newton_jacobian(rhs, dt, self.sdirkparam),
        )
        newQ = working + dt * (0.5 * first_rhs + 0.5 * rhs(Q2))

        return newQ.to(Q.dtype)


REGISTRY = {
    "sdirk": lambda cfg, rhs, prec, context: SDIRKLstable(cfg, rhs.full, preconditioner=prec, context=context),
}
