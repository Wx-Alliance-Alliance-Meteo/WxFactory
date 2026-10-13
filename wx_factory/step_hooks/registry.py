"""Geometry hooks precede initial-state construction; state hooks need the metric and operators.

Multiple providers may apply to the same configuration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..geometry import CubedSphere3D
from .dcmip import ExponentialFilterHook
from .schar_mountain import ScharMountainHook
from .step_hook import StepHook

if TYPE_CHECKING:
    from ..common import Configuration
    from ..geometry import DFROperators, Geometry

PHASE_GEOMETRY = "geometry"
PHASE_STATE = "state"
_PHASES = (PHASE_GEOMETRY, PHASE_STATE)


@dataclass
class StepHookContext:
    """The metric and operators are available only in the state phase."""

    config: Configuration
    geometry: Geometry
    operators: DFROperators | None = None
    metric: object = None


StepHookProvider = Callable[["StepHookContext"], StepHook | None]

STEP_HOOK_REGISTRY: dict[str, tuple[str, StepHookProvider]] = {}


def register_step_hook(name: str, phase: str) -> Callable[[StepHookProvider], StepHookProvider]:
    if phase not in _PHASES:
        raise ValueError(f"Unknown step-hook phase '{phase}'. Expected one of {_PHASES}")

    def decorator(provider: StepHookProvider) -> StepHookProvider:
        if name in STEP_HOOK_REGISTRY:
            raise ValueError(f"A step hook is already registered under '{name}'")
        STEP_HOOK_REGISTRY[name] = (phase, provider)
        return provider

    return decorator


def resolve_step_hooks(ctx: StepHookContext, phase: str) -> dict[type[StepHook], StepHook]:
    hooks: dict[type[StepHook], StepHook] = {}
    for hook_phase, provider in STEP_HOOK_REGISTRY.values():
        if hook_phase != phase:
            continue
        hook = provider(ctx)
        if hook is not None:
            hooks[type(hook)] = hook
    return hooks


@register_step_hook("schar_mountain", phase=PHASE_GEOMETRY)
def _schar_mountain(ctx: StepHookContext) -> StepHook | None:
    if ctx.config.enable_schar_mountain and isinstance(ctx.geometry, CubedSphere3D):
        return ScharMountainHook(ctx.config, ctx.geometry)
    return None


@register_step_hook("exponential_filter", phase=PHASE_STATE)
def _exponential_filter(ctx: StepHookContext) -> StepHook | None:
    if ctx.config.expfilter_apply:
        return ExponentialFilterHook(ctx.geometry, ctx.metric, ctx.operators, ctx.config)
    return None
