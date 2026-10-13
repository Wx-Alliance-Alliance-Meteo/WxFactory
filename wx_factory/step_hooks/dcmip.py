import numpy

from . import step_hook


class ExponentialFilterHook(step_hook.StepHook):

    def __init__(self, geom, metric, operators, config):
        self.metric = metric
        self.operators = operators
        self.prescribed_transport = config.advection_only
        self.filter_matrix = operators.make_filter_3d(
            strength=config.expfilter_strength,
            order=config.expfilter_order,
            cutoff=config.expfilter_cutoff,
            geom=geom,
        )

    def process(self, Q: numpy.ndarray, t: float) -> numpy.ndarray:
        if self.prescribed_transport:
            jacobian = self.metric.volume.volume_jacobian
            filtered = Q.clone()
            filtered[5:] = ((jacobian * Q[5:]) @ self.filter_matrix) / jacobian
            return filtered
        return self.operators.apply_filter_3d(Q, self.metric, self.filter_matrix)
