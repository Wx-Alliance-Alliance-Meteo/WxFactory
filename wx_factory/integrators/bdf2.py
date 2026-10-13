from ..solvers import newton_krylov
from .integrator import Integrator


class Bdf2(Integrator):
    def __init__(self, param, rhs, *, preconditioner=None, init_substeps=1, context=None):
        super().__init__(param, context=context, preconditioner=preconditioner)
        self.rhs = rhs
        self.tol = param.tolerance
        self.gmres_restart = param.gmres_restart
        self.init_substeps = init_substeps
        self.Qprev = None

    def __step__(self, Q, dt):
        working, rhs = self._implicit_state_and_rhs(Q, self.rhs)
        if self.Qprev is None:
            newQ = working.clone()
            init_dt = dt / self.init_substeps
            for _ in range(self.init_substeps):

                def nonlin_fun(Q_plus, previous=newQ):
                    return (Q_plus - previous) / init_dt - rhs(Q_plus)

                newQ, _, _ = newton_krylov(
                    nonlin_fun,
                    newQ,
                    f_tol=self.tol,
                    context=self.context,
                    maxiter=100,
                    restart=self.gmres_restart,
                    krylov_dtype=Q.dtype,
                    maxiter_linear=10,
                    jacobian=self._newton_jacobian(rhs, init_dt, 1),
                )
        else:
            maxiter = 100

            def nonlin_fun(Q_plus):
                return (Q_plus - 4.0 / 3.0 * working + 1.0 / 3.0 * self.Qprev) / dt - 2.0 / 3.0 * rhs(Q_plus)

            if self.preconditioner is not None:
                self.preconditioner.prepare(dt, Q, self.Qprev)
                maxiter = 800
            newQ, _, _ = newton_krylov(
                nonlin_fun,
                working,
                restart=self.gmres_restart,
                krylov_dtype=Q.dtype,
                maxiter_linear=10,
                f_tol=self.tol,
                preconditioner=self.preconditioner,
                verbose=False,
                maxiter=maxiter,
                context=self.context,
                jacobian=self._newton_jacobian(rhs, dt, 2 / 3),
            )
        self.Qprev = working.clone()

        return newQ.reshape(Q.shape).to(Q.dtype)


REGISTRY = {
    "bdf2": lambda cfg, rhs, prec, ctx: Bdf2(cfg, rhs.full, preconditioner=prec, context=ctx),
}
