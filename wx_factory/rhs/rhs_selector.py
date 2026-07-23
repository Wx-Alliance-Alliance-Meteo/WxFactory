"""Select the right-hand-side (RHS) functions for a given set of equations and geometry.

The RHS to use depends on a pair: the equation set (``euler``, ``shallow_water``, ...) and the
geometry it runs on (``CubedSphere3D``, ``CubedSphere2D``, ``Cartesian2D``, ...). Rather than a
chain of ``if``/``isinstance`` tests, each supported combination is registered in ``RHS_REGISTRY``
and looked up by :func:`resolve_rhs`. Adding a new equation set therefore means adding one factory
and one ``@register_rhs(...)`` line, in one place.

Some time integrators are *partitioned*: IMEX, Rosenbrock-exponential and operator-splitting
schemes step an explicit and an implicit RHS separately. The bundle returned here always exposes
``full`` plus the partitioned handles ``explicit`` and ``implicit``; a combination that does not
provide a partition leaves that handle pointing at a placeholder that raises a clear error if a
partitioned integrator tries to use it.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from ..common import Configuration
from ..geometry import Cartesian2D, CubedSphere2D, CubedSphere3D, DFROperators, Geometry, Metric2D, Metric3DTopo
from ..init.initialize import Topo
from ..pde import PDEEulerCartesian, PDEEulerCubesphere
from ..process_topology import ProcessTopology

from .rhs_sw import RhsShallowWater
from .rhs_advection2d import RhsAdvection2d
from .rhs_dfr import RHSDirecFluxReconstruction, RHSDirecFluxReconstruction_mpi_v2


@dataclass
class RhsContext:
    """Everything an RHS factory may need to build its functions.

    A single context object is passed to every factory so they all share one signature, the same
    way the time-integrator factories all take ``(config, rhs, preconditioner, device)``."""

    geom: Geometry
    operators_real: DFROperators
    operators_complex: DFROperators
    metric: Metric2D | Metric3DTopo | None
    topo: Optional[Topo]
    ptopo: Optional[ProcessTopology]
    param: Configuration
    fields_shape: Tuple[int, ...]
    debug: bool = False


def _unavailable_partition(_):
    raise NotImplementedError(
        "This combination of equations and geometry does not provide a partitioned (explicit / "
        "implicit) RHS. Use a non-partitioned time integrator, or implement the partitioned RHS "
        "for this problem."
    )


class RhsBundle:
    """The set of RHS functions associated with a certain geometry and set of equations.

    Attributes:
        full       -- The full RHS. Every time integrator uses this.
        explicit   -- Explicit part, for partitioned (IMEX / splitting) integrators.
        implicit   -- Implicit part, for partitioned integrators.
        shape      -- Shape of the state vector this bundle operates on.

    A partition that is not provided for a given problem raises ``NotImplementedError`` when called,
    rather than being absent, so that selecting an unsupported integrator fails with a clear message.
    """

    def __init__(
        self,
        *,
        full: Callable,
        shape: Tuple[int, ...],
        explicit: Optional[Callable] = None,
        implicit: Optional[Callable] = None,
    ) -> None:
        self.full = full
        self.shape = shape
        self.explicit = explicit if explicit is not None else _unavailable_partition
        self.implicit = implicit if implicit is not None else _unavailable_partition


RhsFactory = Callable[[RhsContext], RhsBundle]

# Maps (equations, geometry class) -> factory that builds the RhsBundle for that combination.
RHS_REGISTRY: dict[tuple[str, type], RhsFactory] = {}


def register_rhs(equations: str, geometry_class: type) -> Callable[[RhsFactory], RhsFactory]:
    """Register a factory for a given (equations, geometry class) combination."""

    def decorator(factory: RhsFactory) -> RhsFactory:
        key = (equations, geometry_class)
        if key in RHS_REGISTRY:
            raise ValueError(f"An RHS factory is already registered for {key}")
        RHS_REGISTRY[key] = factory
        return factory

    return decorator


def resolve_rhs(ctx: RhsContext) -> RhsBundle:
    """Build the RhsBundle for the equations and geometry described by ``ctx``."""
    if ctx.param.discretization not in ("dfr", "dg"):
        raise ValueError(f"Unknown discretization {ctx.param.discretization}")

    key = (ctx.param.equations, type(ctx.geom))
    try:
        factory = RHS_REGISTRY[key]
    except KeyError:
        raise ValueError(
            f"No RHS registered for equations '{ctx.param.equations}' on geometry "
            f"{type(ctx.geom).__name__}. Registered combinations: "
            f"{sorted((eq, geom.__name__) for eq, geom in RHS_REGISTRY)}"
        )
    return factory(ctx)


@register_rhs("euler", CubedSphere3D)
def _euler_cubesphere(ctx: RhsContext) -> RhsBundle:
    # The state is the 5 Euler variables followed by any number of passively advected quantities.
    pde = PDEEulerCubesphere(ctx.geom, ctx.param, ctx.metric, num_var=ctx.fields_shape[0])
    full = RHSDirecFluxReconstruction_mpi_v2(
        pde,
        ctx.geom,
        ctx.operators_real,
        ctx.operators_complex,
        ctx.metric,
        ctx.topo,
        ctx.ptopo,
        ctx.param,
        ctx.fields_shape,
        debug=ctx.debug,
    )
    return RhsBundle(full=full, shape=ctx.fields_shape, implicit=full.implicit, explicit=full.explicit)


@register_rhs("shallow_water", CubedSphere2D)
def _shallow_water_cubesphere(ctx: RhsContext) -> RhsBundle:
    if ctx.param.case_number <= 1:
        # Advection-only test cases
        full = RhsAdvection2d(
            ctx.fields_shape,
            ctx.geom,
            ctx.operators_real,
            ctx.metric,
            ctx.ptopo,
            ctx.geom.num_solpts,
            ctx.geom.num_elements_horizontal,
        )
    else:
        # Full shallow-water equations
        full = RhsShallowWater(
            ctx.fields_shape,
            ctx.geom,
            ctx.operators_real,
            ctx.operators_complex,
            ctx.metric,
            ctx.topo,
            ctx.ptopo,
        )
    return RhsBundle(full=full, shape=ctx.fields_shape)


@register_rhs("euler", Cartesian2D)
def _euler_cartesian(ctx: RhsContext) -> RhsBundle:
    pde = PDEEulerCartesian(ctx.geom, ctx.param, ctx.metric)
    full = RHSDirecFluxReconstruction(
        pde,
        ctx.geom,
        ctx.operators_real,
        ctx.operators_complex,
        ctx.metric,
        ctx.topo,
        ctx.ptopo,
        ctx.param,
        ctx.fields_shape,
    )
    return RhsBundle(full=full, shape=ctx.fields_shape)
