"""Select the geometry (grid) for a given problem.

Which grid to build depends on a pair: the grid type (``cubed_sphere``, ``cartesian2d``, ...) and
the equation set that runs on it (``euler`` needs a 3D cubed sphere, ``shallow_water`` a 2D one).
Each supported combination registers a factory in ``GEOMETRY_REGISTRY`` and is looked up by
:func:`resolve_geometry`, so adding a grid means adding one factory and one ``@register_geometry``
line -- no change to the simulation setup code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from mpi4py import MPI

from ..process_topology import ProcessTopology

from .cartesian_2d_mesh import Cartesian2D
from .cubed_sphere_2d import CubedSphere2D
from .cubed_sphere_3d import CubedSphere3D
from .geometry import Geometry

if TYPE_CHECKING:
    from ..common import Configuration
    from ..device import Device
    from ..simulation.simulation import Simulation


@dataclass
class GeometryContext:
    """Everything a geometry factory may need to build its grid.

    A single context object is passed to every factory so they all share one signature, like the
    RHS and time-integrator factories. The cubed-sphere angles are optional because a Cartesian
    grid does not use them."""

    config: "Configuration"
    device: "Device"
    comm: MPI.Comm
    num_elements_horizontal: int
    num_solpts: int
    total_num_elements_horizontal: int
    lambda0: Optional[float] = None
    phi0: Optional[float] = None
    alpha0: Optional[float] = None

    @classmethod
    def from_simulation(cls, sim: "Simulation") -> "GeometryContext":
        return cls(
            config=sim.config,
            device=sim.device,
            comm=sim.comm,
            num_elements_horizontal=sim.num_elements_horizontal,
            num_solpts=sim.num_solpts,
            total_num_elements_horizontal=sim.total_num_elements_horizontal,
            lambda0=getattr(sim, "lambda0", None),
            phi0=getattr(sim, "phi0", None),
            alpha0=getattr(sim, "alpha0", None),
        )


GeometryFactory = Callable[["GeometryContext"], Geometry]

# Maps (grid_type, equations) -> factory that builds the geometry for that combination.
GEOMETRY_REGISTRY: dict[tuple[str, str], GeometryFactory] = {}

# Key used for a supplied grid file, which always describes a 2D cubed-sphere grid.
_GRID_FILE_KEY = ("cubed_sphere", "shallow_water")


def register_geometry(grid_type: str, equations: str) -> Callable[[GeometryFactory], GeometryFactory]:
    """Register a factory for a given (grid_type, equations) combination."""

    def decorator(factory: GeometryFactory) -> GeometryFactory:
        key = (grid_type, equations)
        if key in GEOMETRY_REGISTRY:
            raise ValueError(f"A geometry factory is already registered for {key}")
        GEOMETRY_REGISTRY[key] = factory
        return factory

    return decorator


def resolve_geometry(ctx: "GeometryContext") -> Geometry:
    """Build the geometry for the grid type and equations described by ``ctx``."""
    if ctx.config.grid_file != "":
        # A grid file always describes a 2D cubed-sphere grid, whatever the other options say.
        return GEOMETRY_REGISTRY[_GRID_FILE_KEY](ctx)

    key = (ctx.config.grid_type, ctx.config.equations)
    try:
        factory = GEOMETRY_REGISTRY[key]
    except KeyError:
        raise ValueError(
            f"No geometry registered for grid_type / equations = {key}. "
            f"Registered combinations: {sorted(GEOMETRY_REGISTRY)}"
        )
    return factory(ctx)


@register_geometry("cubed_sphere", "shallow_water")
def _cubed_sphere_2d(ctx: "GeometryContext") -> Geometry:
    ptopo = ProcessTopology(ctx.device, comm_in=ctx.comm)
    return CubedSphere2D(
        ctx.num_elements_horizontal,
        ctx.num_solpts,
        ctx.total_num_elements_horizontal,
        ctx.lambda0,
        ctx.phi0,
        ctx.alpha0,
        ptopo,
    )


@register_geometry("cubed_sphere", "euler")
def _cubed_sphere_3d(ctx: "GeometryContext") -> Geometry:
    ptopo = ProcessTopology(ctx.device, comm_in=ctx.comm)
    return CubedSphere3D(
        ctx.num_elements_horizontal,
        ctx.config.num_elements_vertical,
        ctx.num_solpts,
        ctx.total_num_elements_horizontal,
        ctx.lambda0,
        ctx.phi0,
        ctx.alpha0,
        ctx.config.ztop,
        ptopo,
        ctx.config,
    )


@register_geometry("cartesian2d", "euler")
def _cartesian_2d(ctx: "GeometryContext") -> Geometry:
    return Cartesian2D(
        (ctx.config.x0, ctx.config.x1),
        (ctx.config.z0, ctx.config.z1),
        ctx.num_elements_horizontal,
        ctx.config.num_elements_vertical,
        ctx.num_solpts,
        ctx.total_num_elements_horizontal,
        ctx.device,
    )
