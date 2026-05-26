import numpy

from . import step_hook
from ..common.definitions import idx_rho, idx_rho_u1, idx_rho_u2, idx_rho_w
from ..init.dcmip import dcmip_T11_update_winds, dcmip_T12_update_winds


class DcmipT11WindHook(step_hook.StepHook):
    """Overwrite momentum components with DCMIP test 11 prescribed winds after each step."""

    def __init__(self, geom, metric, operators, config):
        self.geom = geom
        self.metric = metric
        self.operators = operators
        self.config = config

    def process(self, Q: numpy.ndarray, t: float) -> numpy.ndarray:
        u1_contra, u2_contra, w_wind = dcmip_T11_update_winds(
            self.geom, self.metric, self.operators, self.config, time=t
        )
        Q[idx_rho_u1, :, :, :] = Q[idx_rho, :, :, :] * u1_contra
        Q[idx_rho_u2, :, :, :] = Q[idx_rho, :, :, :] * u2_contra
        Q[idx_rho_w, :, :, :] = Q[idx_rho, :, :, :] * w_wind
        return Q


class DcmipT12WindHook(step_hook.StepHook):
    """Overwrite momentum components with DCMIP test 12 prescribed winds after each step."""

    def __init__(self, geom, metric, operators, config):
        self.geom = geom
        self.metric = metric
        self.operators = operators
        self.config = config

    def process(self, Q: numpy.ndarray, t: float) -> numpy.ndarray:
        u1_contra, u2_contra, w_wind = dcmip_T12_update_winds(
            self.geom, self.metric, self.operators, self.config, time=t
        )
        Q[idx_rho_u1, :, :, :] = Q[idx_rho, :, :, :] * u1_contra
        Q[idx_rho_u2, :, :, :] = Q[idx_rho, :, :, :] * u2_contra
        Q[idx_rho_w, :, :, :] = Q[idx_rho, :, :, :] * w_wind
        return Q
