"""Advance tracer densities and an internal clock, restoring analytic meteorology."""

import torch

from .integrator import Integrator


class PrescribedTransportIntegrator(Integrator):
    def __init__(self, config, method, rhs, *, context):
        super().__init__(config, context=context)
        self.method = method
        self.rhs = rhs

    def __step__(self, state, dt):
        self.method.sim_time = self.sim_time
        clock = torch.full_like(state[:1], self.sim_time / self.rhs.period)
        augmented = torch.cat((state[5:], clock))
        evolved = self.method.step(augmented, dt)
        self.failure_flag = self.method.failure_flag
        result = self.rhs.restore(state, self.sim_time + dt)
        result[5:] = evolved[:-1]
        self.rhs.time = self.sim_time + dt
        return result
