import math

import numpy

from ..common.configuration import Configuration
from .integrator import Integrator


class Splitting(Integrator):
    def __init__(self, param: Configuration, scheme1: Integrator, scheme2: Integrator, *, context=None):
        super().__init__(param, context=context, preconditioner=None)
        self.scheme1 = scheme1
        self.scheme2 = scheme2

    def __prestep__(self, Q, dt):
        self.failure_flag = 0
        self.substep_number = 0

    def _substep(self, scheme, Q, dt):
        scheme.failure_flag = 0
        try:
            result = scheme.step(Q, dt)
        except Exception:
            self.failure_flag = scheme.failure_flag or 1
            raise
        if scheme.failure_flag:
            self.failure_flag = scheme.failure_flag
            raise RuntimeError("Splitting subintegrator did not converge")
        self.substep_number += 1
        if self.context.comm.rank == 0:
            name = getattr(scheme, "name", type(scheme).__name__)
            print(
                f"Elapsed time for substep {self.substep_number} ({name}): {scheme.latest_time:.3f} secs",
                flush=True,
            )
        return result


class LieSplitting(Splitting):
    def __step__(self, Q, dt):
        Q1 = self._substep(self.scheme1, Q, dt)
        return self._substep(self.scheme2, Q1, dt)


class StrangSplitting(Splitting):
    def __step__(self, Q, dt):
        Q1 = self._substep(self.scheme1, Q, dt / 2)
        Q2 = self._substep(self.scheme2, Q1, dt)
        return self._substep(self.scheme1, Q2, dt / 2)


class OS22Splitting(Splitting):
    def __init__(self, param: Configuration, scheme1: Integrator, scheme2: Integrator, os_param, *, context=None):
        super().__init__(param, scheme1, scheme2, context=context)
        if os_param is None or not math.isfinite(os_param) or not 0 <= os_param <= 0.5:
            raise ValueError("OS22 requires 0 <= os22_parameter <= 0.5 for nonnegative substeps")
        self.os_param = os_param
        self.alpha = numpy.array(
            [
                [(2 * self.os_param - 1) / (2 * self.os_param - 2), 1 - self.os_param],
                [-1 / (2 * self.os_param - 2), self.os_param],
            ]
        )

    def __step__(self, Q, dt):
        for stage in self.alpha:
            if stage[0] != 0:
                Q = self._substep(self.scheme1, Q, stage[0] * dt)
            if stage[1] != 0:
                Q = self._substep(self.scheme2, Q, stage[1] * dt)
        return Q


def _subintegrator(name, cfg, operator, shape, prec, ctx):
    from ..rhs.rhs_selector import RhsBundle
    from . import PARTITIONED_REGISTRY, REGISTRY, resolve

    if name in PARTITIONED_REGISTRY:
        raise ValueError("Nested partitioned integrators are unsupported in splitting substeps")
    if name not in REGISTRY:
        raise ValueError(f"Unknown splitting subintegrator '{name}'")
    method = resolve(name, cfg, RhsBundle(full=operator, shape=shape), prec, ctx)
    if name == "bdf2" or getattr(method, "n_prev", 0) > 0:
        raise ValueError("Splitting requires one-step subintegrators; multistep histories are unsupported")
    method.name = name.upper()
    return method


def _schemes(cfg, rhs, prec, ctx):
    scheme1 = _subintegrator(cfg.splitting_integrator_1, cfg, rhs.explicit, rhs.shape, None, ctx)
    scheme2 = _subintegrator(cfg.splitting_integrator_2, cfg, rhs.implicit, rhs.shape, prec, ctx)
    return scheme1, scheme2


def _make_generic_splitting_factory(cls):
    def factory(cfg, rhs, prec, ctx):
        scheme1, scheme2 = _schemes(cfg, rhs, prec, ctx)
        return cls(cfg, scheme1, scheme2, context=ctx)

    return factory


def _os22(cfg, rhs, prec, ctx):
    scheme1, scheme2 = _schemes(cfg, rhs, prec, ctx)
    return OS22Splitting(cfg, scheme1, scheme2, cfg.os22_parameter, context=ctx)


def _strang_epi2_ros2(cfg, rhs, prec, ctx):
    explicit = _subintegrator("epi2", cfg, rhs.explicit, rhs.shape, None, ctx)
    implicit = _subintegrator("ros2", cfg, rhs.implicit, rhs.shape, prec, ctx)
    return StrangSplitting(cfg, explicit, implicit, context=ctx)


def _strang_ros2_epi2(cfg, rhs, prec, ctx):
    explicit = _subintegrator("epi2", cfg, rhs.explicit, rhs.shape, None, ctx)
    implicit = _subintegrator("ros2", cfg, rhs.implicit, rhs.shape, prec, ctx)
    return StrangSplitting(cfg, implicit, explicit, context=ctx)


REGISTRY = {}

PARTITIONED_REGISTRY = {
    "lie": _make_generic_splitting_factory(LieSplitting),
    "strang": _make_generic_splitting_factory(StrangSplitting),
    "os22": _os22,
    "strang_epi2_ros2": _strang_epi2_ros2,
    "strang_ros2_epi2": _strang_ros2_epi2,
}
