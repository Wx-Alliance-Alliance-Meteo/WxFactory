from ..common.configuration import Configuration
from ..geometry import (
    Cartesian3D,
    CartesianMetric5D,
    CubedSphere2D,
    CubedSphere3D,
    DFROperators,
    Geometry,
    Metric2D,
    Metric5D,
)
from ..init.initialize import initialize_cartesian3d, initialize_euler, initialize_sw
from ..simulation.initial_state import InitialState
from ..step_hooks import ScharMountainHook, StepHook


def init_state_vars(
    geom: Geometry, operators: DFROperators, param: Configuration, step_hooks: dict[type, StepHook]
) -> InitialState:
    """Get intial value for state variables as well at topography information, based on the test case."""

    topo = None
    metric = None
    dataset = None

    if param.equations == "euler" and isinstance(geom, Cartesian3D):
        metric = CartesianMetric5D(geom, param)
        metric.build_metric()
        Q = initialize_cartesian3d(geom, metric, param)

    elif param.equations == "euler" and isinstance(geom, CubedSphere3D):
        metric = Metric5D(geom, param)
        if param.enable_schar_mountain:
            step_hooks[ScharMountainHook].metric = metric
            step_hooks[ScharMountainHook].apply(1 if param.schar_mountain_step == 0 else 0)
        Q, topo = initialize_euler(geom, metric, operators, param)

    elif param.equations == "shallow_water" and isinstance(geom, CubedSphere2D):
        metric = Metric2D(geom)
        Q, topo, dataset = initialize_sw(geom, metric, param)

    else:
        raise ValueError(f"Unrecognized combination of equations ({param.equations} and geometry ({geom}))")

    return InitialState(Q, topo, metric, dataset)
