from ..common.configuration import Configuration
from .integrator import Integrator


class Euler1(Integrator):
    def __init__(self, param: Configuration, rhs, *, context=None):
        super().__init__(param, context=context)
        if self.context.comm.rank == 0:
            print("WARNING: Running with first-order explicit Euler timestepping.")
            print("         This is UNSTABLE and should be used only for debugging.")
        self.rhs = rhs

    def __step__(self, Q, dt):
        Q = Q + self.rhs(Q) * dt
        return Q


REGISTRY = {
    "euler1": lambda cfg, rhs, prec, ctx: Euler1(cfg, rhs.full, context=ctx),
}
