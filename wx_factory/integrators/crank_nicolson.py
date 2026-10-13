from ..solvers import newton_krylov
from .integrator import Integrator


class CrankNicolson(Integrator):
    def __init__(self, param, rhs, *, context=None, preconditioner=None):
        super().__init__(param, context=context, preconditioner=preconditioner)
        self.rhs = rhs
        self.tol = param.tolerance
        self.gmres_restart = param.gmres_restart

    def __step__(self, Q, dt):
        working, rhs = self._implicit_state_and_rhs(Q, self.rhs)
        previous_rhs = rhs(working)

        def CN_fun(Q_plus):
            return (Q_plus - working) / dt - 0.5 * (rhs(Q_plus) + previous_rhs)

        maxiter = 100
        if self.preconditioner is not None:
            self.preconditioner.prepare(dt, Q)
            maxiter = 800

        newQ, _, _ = newton_krylov(
            CN_fun,
            working,
            f_tol=self.tol,
            restart=self.gmres_restart,
            krylov_dtype=Q.dtype,
            maxiter_linear=10,
            preconditioner=self.preconditioner,
            verbose=False,
            maxiter=maxiter,
            context=self.context,
            jacobian=self._newton_jacobian(rhs, dt, 0.5),
        )
        return newQ.reshape(Q.shape).to(Q.dtype)


REGISTRY = {
    "crank_nicolson": lambda cfg, rhs, prec, context: CrankNicolson(
        cfg, rhs.full, preconditioner=prec, context=context
    ),
}
