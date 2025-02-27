import numpy
from numpy.typing import NDArray


from common.configuration import Configuration
from geometry import DFROperators, Geometry, Metric2D, Metric3DTopo, Cartesian2D, CubedSphere2D, CubedSphere3D
from init.initialize import initialize_cartesian2d, initialize_euler, initialize_sw, Topo


def init_state_vars(
    geom: Geometry, operators: DFROperators, param: Configuration
) -> tuple[NDArray[numpy.float64], Topo | None, Metric2D | Metric3DTopo | None]:
    """Get intial value for state variables as well at topography information, based on the test case."""

    topo = None
    metric = None

    if param.equations == "euler" and isinstance(geom, CubedSphere3D):
        metric = Metric3DTopo(geom, operators)
        Q, topo = initialize_euler(geom, metric, operators, param)
        # Q: dimensions [5,nk,nj,ni], order ρ, u, v, w, θ

    elif param.equations == "euler" and isinstance(geom, Cartesian2D):
        Q = initialize_cartesian2d(geom, param)

    elif param.equations == "shallow_water" and isinstance(geom, CubedSphere2D):
        metric = Metric2D(geom)
        Q, topo = initialize_sw(geom, metric, operators, param)

    else:
        raise ValueError(f"Unrecognized combination of equations ({param.equations} and geometry ({geom}))")

    return Q, topo, metric
