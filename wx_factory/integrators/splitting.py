import numpy

from ..common.configuration import Configuration
from .integrator import Integrator


class LieSplitting(Integrator):
    def __init__(self, param: Configuration, scheme1: Integrator, scheme2: Integrator):
        super().__init__(param, preconditioner=None)
        self.scheme1 = scheme1
        self.scheme2 = scheme2

    def __step__(self, Q, dt):
        Q1 = self.scheme1.step(Q, dt)
        Q2 = self.scheme2.step(Q1, dt)
        return Q2


class StrangSplitting(Integrator):
    def __init__(self, param: Configuration, scheme1: Integrator, scheme2: Integrator):
        super().__init__(param, preconditioner=None)
        self.scheme1 = scheme1
        self.scheme2 = scheme2

    def __step__(self, Q, dt):
        Q1 = self.scheme1.step(Q, dt / 2)
        Q2 = self.scheme2.step(Q1, dt)
        Q3 = self.scheme1.step(Q2, dt / 2)
        return Q3


class OS22Splitting(Integrator):
    def __init__(self, param: Configuration, scheme1: Integrator, scheme2: Integrator, os_param):
        super().__init__(param, preconditioner=None)
        self.scheme1 = scheme1
        self.scheme2 = scheme2
        self.os_param = os_param
        self.alpha = numpy.array(
            [
                [(2 * self.os_param - 1) / (2 * self.os_param - 2), 1 - self.os_param],
                [-1 / (2 * self.os_param - 2), self.os_param],
            ]
        )

    def __step__(self, Q, dt):
        for numofstage in range(0, self.alpha.shape[0]):
            if self.alpha[numofstage, 0] != 0:
                Q = self.scheme1.step(Q, self.alpha[numofstage, 0] * dt)
            if self.alpha[numofstage, 1] != 0:
                Q = self.scheme2.step(Q, self.alpha[numofstage, 1] * dt)
        return Q


def _make_generic_splitting_factory(cls):
    def factory(cfg, rhs, prec, dev):
        from . import resolve
        sub1 = resolve(cfg.splitting_integrator_1, cfg, rhs, prec, dev)
        sub2 = resolve(cfg.splitting_integrator_2, cfg, rhs, prec, dev)
        return cls(cfg, sub1, sub2)
    return factory


def _strang_epi2_ros2(cfg, rhs, prec, dev):
    from .epi import Epi
    from .ros2 import Ros2
    return StrangSplitting(cfg, Epi(cfg, 2, rhs.explicit, device=dev), Ros2(cfg, rhs.implicit, preconditioner=prec, device=dev))


def _strang_ros2_epi2(cfg, rhs, prec, dev):
    from .epi import Epi
    from .ros2 import Ros2
    return StrangSplitting(cfg, Ros2(cfg, rhs.implicit, preconditioner=prec, device=dev), Epi(cfg, 2, rhs.explicit, device=dev))


REGISTRY = {
    "lie": _make_generic_splitting_factory(LieSplitting),
    "strang": _make_generic_splitting_factory(StrangSplitting),
    "strang_epi2_ros2": _strang_epi2_ros2,
    "strang_ros2_epi2": _strang_ros2_epi2,
}
