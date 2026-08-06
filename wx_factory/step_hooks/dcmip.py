import numpy

from ..common.definitions import idx_rho, idx_rho_theta, idx_rho_u1, idx_rho_u2, idx_rho_u3
from ..init.dcmip import dcmip_prescribed_rho_theta, dcmip_T11_update_winds, dcmip_T12_update_winds
from . import step_hook


class ExponentialFilterHook(step_hook.StepHook):
    """Apply a configured exponential modal filter after each time step."""

    def __init__(self, geom, metric, operators, config):
        self.metric = metric
        self.operators = operators
        self.filter_matrix = operators.make_filter_3d(
            strength=config.expfilter_strength,
            order=config.expfilter_order,
            cutoff=config.expfilter_cutoff,
            geom=geom,
        )

    def process(self, Q: numpy.ndarray, t: float) -> numpy.ndarray:
        return self.operators.apply_filter_3d(Q, self.metric, self.filter_matrix)


class _DcmipAdvectionHook(step_hook.StepHook):
    """Restore the prescribed meteorological state after each step of a DCMIP advection test.

    The DCMIP tracer-transport tests (section 1 of the test case document) are pure advection: the
    dynamic updates of the velocity, temperature, pressure and density must be disabled, and the
    time-dependent winds are prescribed analytically. The dynamical core still integrates the full
    Euler system -- which is what transports the tracers, each of them obeying the continuity
    equation -- so after every step we put the five meteorological variables back to their
    prescribed values. Only the tracers are left to evolve.
    """

    def __init__(self, geom, metric, operators, config):
        self.geom = geom
        self.metric = metric
        self.operators = operators
        self.config = config

        # Density and potential temperature are analytic and time independent for these tests, but
        # they are only built on first use, so that constructing a hook stays free of the geometry.
        self._prescribed = None

    def update_winds(self, time: float):
        raise NotImplementedError

    def process(self, Q: numpy.ndarray, t: float) -> numpy.ndarray:
        if self._prescribed is None:
            self._prescribed = dcmip_prescribed_rho_theta(self.geom)
        rho, rho_theta = self._prescribed

        u1_contra, u2_contra, w_wind = self.update_winds(t)

        Q[idx_rho] = rho
        Q[idx_rho_theta] = rho_theta
        Q[idx_rho_u1] = rho * u1_contra
        Q[idx_rho_u2] = rho * u2_contra
        Q[idx_rho_u3] = rho * w_wind

        return Q


class DcmipT11WindHook(_DcmipAdvectionHook):
    """DCMIP test 1-1: 3D deformational flow."""

    def update_winds(self, time: float):
        return dcmip_T11_update_winds(self.geom, self.metric, self.operators, self.config, time=time)


class DcmipT12WindHook(_DcmipAdvectionHook):
    """DCMIP test 1-2: Hadley-like meridional circulation."""

    def update_winds(self, time: float):
        return dcmip_T12_update_winds(self.geom, self.metric, self.operators, self.config, time=time)
