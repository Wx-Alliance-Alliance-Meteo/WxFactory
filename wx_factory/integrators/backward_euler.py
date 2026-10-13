from collections.abc import Callable

from ..common.configuration import Configuration
from ..solvers import newton_krylov
from .integrator import Integrator


class BackwardEuler(Integrator):
    def __init__(self, param: Configuration, rhs_handle: Callable, *, context=None, preconditioner=None) -> None:
        super().__init__(param, context=context, preconditioner=preconditioner)
        self.rhs = rhs_handle
        self.tol = param.tolerance
        self.gmres_restart = param.gmres_restart

    def __step__(self, Q, dt):
        working, rhs = self._implicit_state_and_rhs(Q, self.rhs)

        def BE_fun(Q_plus):
            return (Q_plus - working) / dt - rhs(Q_plus)

        maxiter = 100
        if self.preconditioner is not None:
            self.preconditioner.prepare(dt, Q)
            maxiter = 800

        newQ, _, _ = newton_krylov(
            BE_fun,
            working,
            f_tol=self.tol,
            restart=self.gmres_restart,
            krylov_dtype=Q.dtype,
            maxiter_linear=10,
            preconditioner=self.preconditioner,
            verbose=False,
            maxiter=maxiter,
            context=self.context,
            jacobian=self._newton_jacobian(rhs, dt, 1),
        )
        return newQ.reshape(Q.shape).to(Q.dtype)


REGISTRY = {
    "backward_euler": lambda cfg, rhs, prec, context: BackwardEuler(
        cfg, rhs.full, preconditioner=prec, context=context
    ),
}
