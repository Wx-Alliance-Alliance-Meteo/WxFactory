import numpy
from numpy.typing import NDArray


from ..common.configuration import Configuration
from ..geometry import (
    DFROperators,
    Geometry,
    Metric2D,
    Metric3DTopo,
    Cartesian3D,
    CubedSphere2D,
    CubedSphere3D,
)
from ..init.initialize import initialize_cartesian3d, initialize_euler, initialize_sw, Topo
from typing import Dict, Type
from ..step_hooks import StepHook, ScharMountainHook
from ..simulation.initial_state import InitialState


def init_state_vars(
    geom: Geometry, operators: DFROperators, param: Configuration, step_hooks: Dict[Type, StepHook]
) -> InitialState:
    """Get intial value for state variables as well at topography information, based on the test case."""

    topo = None
    metric = None
    dataset = None

    # Cartesian3D is a CubedSphere3D subclass, so it must be matched first (flat slab, its own IC).
    if param.equations == "euler" and isinstance(geom, Cartesian3D):
        metric = Metric3DTopo(geom, operators)
        metric.build_metric()
        Q = initialize_cartesian3d(geom, param)

    elif param.equations == "euler" and isinstance(geom, CubedSphere3D):
        metric = Metric3DTopo(geom, operators)
        if param.enable_schar_mountain:
            step_hooks[ScharMountainHook].metric = metric
            step_hooks[ScharMountainHook].apply(1 if param.schar_mountain_step == 0 else 0)
        Q, topo = initialize_euler(geom, metric, operators, param)
        # Q: dimensions [5,nk,nj,ni], order ρ, u, v, w, θ

    elif param.equations == "shallow_water" and isinstance(geom, CubedSphere2D):
        metric = Metric2D(geom)
        Q, topo, dataset = initialize_sw(geom, metric, operators, param)

    else:
        raise ValueError(f"Unrecognized combination of equations ({param.equations} and geometry ({geom}))")

    return InitialState(Q, topo, metric, dataset)
