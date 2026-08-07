"""Select RHS functions by equation set and geometry."""

from collections.abc import Callable
from dataclasses import dataclass

from ..common import Configuration
from ..geometry import (
    Cartesian3D,
    CubedSphere2D,
    CubedSphere3D,
    DFROperators,
    Geometry,
    Metric2D,
    Metric3DTopo,
)
from ..init.initialize import Topo
from ..pde import PDEEuler3D
from ..process_topology import ProcessTopology
from .rhs_advection2d import RhsAdvection2d
from .rhs_dfr import RHSDirecFluxReconstruction_mpi
from .rhs_sw import RhsShallowWater


@dataclass
class RhsContext:
    """Inputs shared by RHS factories."""

    geom: Geometry
    operators_real: DFROperators
    metric: Metric2D | Metric3DTopo | None
    topo: Topo | None
    ptopo: ProcessTopology | None
    param: Configuration
    fields_shape: tuple[int, ...]
    debug: bool = False


_GENERIC_NO_PARTITION = (
    "the explicit / implicit partition is not implemented for this combination of equations and geometry"
)


def _unavailable_partition(reason: str) -> Callable:
    """Create an unavailable partition that reports its reason."""

    def unavailable(_):
        raise NotImplementedError(f"No partitioned right-hand side: {reason}.")

    return unavailable


class RhsBundle:
    """Full RHS and optional explicit and implicit partitions."""

    def __init__(
        self,
        *,
        full: Callable,
        shape: tuple[int, ...],
        explicit: Callable | None = None,
        implicit: Callable | None = None,
        partition_reason: str | None = None,
    ) -> None:
        if (explicit is None) != (implicit is None):
            raise ValueError("An RhsBundle defines both the explicit and implicit partitions, or neither")

        self.full = full
        self.shape = shape
        self.partition_reason = None if explicit is not None else (partition_reason or _GENERIC_NO_PARTITION)
        self.explicit = explicit if explicit is not None else _unavailable_partition(self.partition_reason)
        self.implicit = implicit if implicit is not None else _unavailable_partition(self.partition_reason)

    @property
    def has_partition(self) -> bool:
        """Return whether both RHS partitions are available."""
        return self.partition_reason is None


RhsFactory = Callable[[RhsContext], RhsBundle]

# Map each equation and geometry pair to its factory.
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


def _euler_bundle(ctx: RhsContext, pde, full) -> RhsBundle:
    if pde.advection_only:
        return RhsBundle(
            full=full,
            shape=ctx.fields_shape,
            partition_reason=(
                "advection_only is on, which freezes the Euler dynamics and transports the tracers "
                "with a prescribed wind, so there is no vertically-stiff partition to separate"
            ),
        )
    return RhsBundle(full=full, shape=ctx.fields_shape, implicit=full.implicit, explicit=full.explicit)


@register_rhs("euler", CubedSphere3D)
def _euler_cubesphere(ctx: RhsContext) -> RhsBundle:
    # Additional state variables are passive tracers.
    pde = PDEEuler3D(ctx.geom, ctx.param, ctx.metric, num_var=ctx.fields_shape[0])
    full = RHSDirecFluxReconstruction_mpi(
        pde,
        ctx.geom,
        ctx.operators_real,
        ctx.metric,
        ctx.topo,
        ctx.ptopo,
        ctx.param,
        debug=ctx.debug,
    )
    return _euler_bundle(ctx, pde, full)


# Cartesian slabs use the 3D Euler RHS with an identity metric.
@register_rhs("euler", Cartesian3D)
def _euler_cartesian3d(ctx: RhsContext) -> RhsBundle:
    pde = PDEEuler3D(ctx.geom, ctx.param, ctx.metric, num_var=ctx.fields_shape[0])
    # Cartesian case numbers do not follow the DCMIP advection convention.
    if getattr(ctx.param, "advection_only", "auto") == "auto":
        pde.advection_only = False
    full = RHSDirecFluxReconstruction_mpi(
        pde,
        ctx.geom,
        ctx.operators_real,
        ctx.metric,
        ctx.topo,
        ctx.ptopo,
        ctx.param,
        debug=ctx.debug,
    )
    return _euler_bundle(ctx, pde, full)


@register_rhs("shallow_water", CubedSphere2D)
def _shallow_water_cubesphere(ctx: RhsContext) -> RhsBundle:
    reason: str
    if ctx.param.case_number <= 1 and ctx.param.case_number != -2:
        # Advection tests.
        reason = "the partitioned right-hand side is not implemented for 2D advection"
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
        # Shallow-water dynamics.
        reason = "the partitioned right-hand side is not implemented for the shallow-water equations"
        full = RhsShallowWater(
            ctx.geom,
            ctx.operators_real,
            ctx.metric,
            ctx.topo,
            ctx.ptopo,
        )
    return RhsBundle(full=full, shape=ctx.fields_shape, partition_reason=reason)
