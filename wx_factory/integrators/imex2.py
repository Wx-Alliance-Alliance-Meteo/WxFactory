from typing import Callable


from ..common.configuration import Configuration
from .integrator import Integrator
from ..solvers import newton_krylov


class Imex2(Integrator):
    def __init__(self, param: Configuration, rhs_exp: Callable, rhs_imp: Callable, *, device=None):
        super().__init__(param, device=device)

        self.rhs_exp = rhs_exp
        self.rhs_imp = rhs_imp
        self.tol = param.tolerance

    def __step__(self, Q, dt):
        rhs = Q + dt / 2 * self.rhs_exp(Q)

        def g(v):
            return v - dt / 2 * self.rhs_imp(v) - rhs

        Y1, _, _ = newton_krylov(g, Q)

        # Update solution
        return Q + dt * (self.rhs_imp(Y1) + self.rhs_exp(Y1))

REGISTRY = {
    "imex2": lambda cfg, rhs, prec, dev: Imex2(cfg, rhs.explicit, rhs.implicit, device=dev),
}
