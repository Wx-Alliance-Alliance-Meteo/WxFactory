"""Select the output manager for a given geometry and output format.

Which output manager to build depends on a pair: the geometry's output family (its ``output_family``
class attribute, e.g. ``cartesian`` or ``cubesphere``) and the ``output_format`` config option
(``netcdf``, ``fst``, ...). Each supported combination registers a factory in ``OUTPUT_REGISTRY``
and is looked up by :func:`resolve_output`.

A family whose output does not depend on the format (like the Cartesian one) registers with
``output_format=None``; :func:`resolve_output` falls back to that entry when there is no exact match
for the requested format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from .output_manager import OutputManager
from .output_cubesphere_netcdf import OutputCubesphereNetcdf
from .output_cartesian import OutputCartesian
from .output_cubesphere_fst import OutputCubesphereFst

if TYPE_CHECKING:
    from ..common import Configuration
    from ..device import Device
    from ..geometry import DFROperators, Geometry


@dataclass
class OutputContext:
    """Everything an output factory may need to build its output manager.

    A single context object is passed to every factory so they all share one signature, like the
    other registries. Some fields (metric, topography, dataset, process topology) are only used by
    the cubed-sphere output managers."""

    config: "Configuration"
    device: "Device"
    geometry: "Geometry"
    operators: "DFROperators"
    metric: object = None
    topography: object = None
    dataset: object = None
    ptopo: object = None


OutputFactory = Callable[["OutputContext"], OutputManager]

# Maps (output family, output format) -> factory. A None format is a family-wide default, used when
# the family's output does not depend on the format.
OUTPUT_REGISTRY: dict[tuple[str, Optional[str]], OutputFactory] = {}


def register_output(output_family: str, output_format: Optional[str] = None):
    """Register a factory for a geometry output family and (optionally) a specific output format."""

    def decorator(factory: OutputFactory) -> OutputFactory:
        key = (output_family, output_format)
        if key in OUTPUT_REGISTRY:
            raise ValueError(f"An output factory is already registered for {key}")
        OUTPUT_REGISTRY[key] = factory
        return factory

    return decorator


def resolve_output(ctx: "OutputContext") -> OutputManager:
    """Build the output manager for the geometry and output format described by ``ctx``."""
    family = ctx.geometry.output_family
    fmt = ctx.config.output_format

    factory = OUTPUT_REGISTRY.get((family, fmt)) or OUTPUT_REGISTRY.get((family, None))
    if factory is None:
        raise ValueError(
            f"No output registered for geometry family '{family}' with format '{fmt}'. "
            f"Registered combinations: {sorted((f, o) for f, o in OUTPUT_REGISTRY)}"
        )
    return factory(ctx)


@register_output("cubesphere", "netcdf")
def _cubesphere_netcdf(ctx: "OutputContext") -> OutputManager:
    return OutputCubesphereNetcdf(
        ctx.config,
        ctx.geometry,
        ctx.operators,
        ctx.device,
        ctx.metric,
        ctx.topography,
        ctx.dataset,
        ctx.ptopo,
    )


@register_output("cubesphere", "fst")
def _cubesphere_fst(ctx: "OutputContext") -> OutputManager:
    return OutputCubesphereFst(
        ctx.config,
        ctx.geometry,
        ctx.operators,
        ctx.device,
        ctx.metric,
        ctx.topography,
        ctx.ptopo,
    )


@register_output("cartesian")
def _cartesian_images(ctx: "OutputContext") -> OutputManager:
    """Cartesian slabs are visualised as x-z images regardless of the requested output_format."""
    return OutputCartesian(ctx.config, ctx.geometry, ctx.operators, ctx.device)
