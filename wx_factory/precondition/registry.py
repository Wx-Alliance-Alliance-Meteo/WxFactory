"""Select the preconditioner for a linear-solver-based time integrator.

Which preconditioner to build is chosen by the ``preconditioner`` configuration option. Each
available preconditioner registers a factory in ``PRECONDITIONER_REGISTRY`` and is looked up by
:func:`resolve_preconditioner`, so adding one means adding a class and a ``@register_preconditioner``
line -- no change to the simulation setup code.

There are currently no built-in preconditioners: the historical ones were removed because they were
broken. ``preconditioner = none`` (the default) means "no preconditioning" and resolves to ``None``.
See ``doc/contribute.md`` for how to add one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple

if TYPE_CHECKING:
    from ..common import Configuration
    from ..device import Device
    from ..geometry import DFROperators, Geometry
    from ..rhs import RhsBundle
    from .preconditioner import Preconditioner


@dataclass
class PreconditionerContext:
    """Everything a preconditioner factory may need to build its preconditioner.

    A single context object is passed to every factory so they all share one signature, the same
    way the RHS and time-integrator factories do."""

    config: "Configuration"
    device: "Device"
    geometry: "Geometry"
    operators: "DFROperators"
    rhs: "RhsBundle"
    metric: object
    topography: object
    ptopo: object
    fields_shape: Tuple[int, ...]


PreconditionerFactory = Callable[["PreconditionerContext"], "Preconditioner"]

# Maps the `preconditioner` config value to a factory that builds it. "none" is handled separately.
PRECONDITIONER_REGISTRY: dict[str, PreconditionerFactory] = {}


def register_preconditioner(name: str) -> Callable[[PreconditionerFactory], PreconditionerFactory]:
    """Register a factory for the given ``preconditioner`` config value."""

    def decorator(factory: PreconditionerFactory) -> PreconditionerFactory:
        if name in PRECONDITIONER_REGISTRY:
            raise ValueError(f"A preconditioner is already registered under '{name}'")
        PRECONDITIONER_REGISTRY[name] = factory
        return factory

    return decorator


def resolve_preconditioner(ctx: "PreconditionerContext") -> Optional["Preconditioner"]:
    """Build the preconditioner selected by ``ctx.config.preconditioner`` (``None`` if 'none')."""
    name = ctx.config.preconditioner
    if name == "none":
        return None

    try:
        factory = PRECONDITIONER_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown preconditioner '{name}'. Registered preconditioners: "
            f"{sorted(PRECONDITIONER_REGISTRY)} (plus 'none' for no preconditioning)."
        )
    return factory(ctx)
