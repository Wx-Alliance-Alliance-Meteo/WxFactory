"""Select which step hooks a simulation should run.

A step hook post-processes the state after every time step (for example, imposing a prescribed wind
field or a mountain topography). Unlike the geometry / RHS / output choices, hooks are not a single
mutually-exclusive selection: zero or more may apply to a given configuration. Each hook registers a
*provider* that inspects the configuration and returns a hook instance when it applies, or ``None``.

Hooks are resolved in two phases, because they become buildable at different points of the setup:

* ``geometry`` -- right after the geometry is built (only the geometry and config are available).
  ``init_state_vars`` reads these hooks while building the initial state, so they must exist first.
* ``state`` -- after the initial state exists (the metric and operators are then available).

Add a hook by writing the hook class and registering a provider here; nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Optional, Type

from ..geometry import CubedSphere3D

from .step_hook import StepHook
from .schar_mountain import ScharMountainHook
from .dcmip import DcmipT11WindHook, DcmipT12WindHook, ExponentialFilterHook

if TYPE_CHECKING:
    from ..common import Configuration
    from ..geometry import DFROperators, Geometry

# The two points at which hooks can be resolved (see the module docstring).
PHASE_GEOMETRY = "geometry"
PHASE_STATE = "state"
_PHASES = (PHASE_GEOMETRY, PHASE_STATE)


@dataclass
class StepHookContext:
    """Everything a step-hook provider may need. ``operators`` and ``metric`` are only available in
    the ``state`` phase (they are built from the initial state)."""

    config: "Configuration"
    geometry: "Geometry"
    operators: Optional["DFROperators"] = None
    metric: object = None


# A provider returns a hook instance if it applies to this configuration, otherwise None.
StepHookProvider = Callable[["StepHookContext"], Optional[StepHook]]

# Maps a hook name to the (phase, provider) that builds it.
STEP_HOOK_REGISTRY: Dict[str, tuple[str, StepHookProvider]] = {}


def register_step_hook(name: str, phase: str) -> Callable[[StepHookProvider], StepHookProvider]:
    """Register a provider for a step hook, resolved during the given phase."""
    if phase not in _PHASES:
        raise ValueError(f"Unknown step-hook phase '{phase}'. Expected one of {_PHASES}")

    def decorator(provider: StepHookProvider) -> StepHookProvider:
        if name in STEP_HOOK_REGISTRY:
            raise ValueError(f"A step hook is already registered under '{name}'")
        STEP_HOOK_REGISTRY[name] = (phase, provider)
        return provider

    return decorator


def resolve_step_hooks(ctx: "StepHookContext", phase: str) -> Dict[Type[StepHook], StepHook]:
    """Return the hooks (keyed by type) that apply to ``ctx`` for the given phase."""
    hooks: Dict[Type[StepHook], StepHook] = {}
    for _name, (hook_phase, provider) in STEP_HOOK_REGISTRY.items():
        if hook_phase != phase:
            continue
        hook = provider(ctx)
        if hook is not None:
            hooks[type(hook)] = hook
    return hooks


@register_step_hook("schar_mountain", phase=PHASE_GEOMETRY)
def _schar_mountain(ctx: "StepHookContext") -> Optional[StepHook]:
    if ctx.config.enable_schar_mountain and isinstance(ctx.geometry, CubedSphere3D):
        return ScharMountainHook(ctx.config, ctx.geometry)
    return None


@register_step_hook("dcmip_t11_wind", phase=PHASE_STATE)
def _dcmip_t11(ctx: "StepHookContext") -> Optional[StepHook]:
    if ctx.config.case_number == 11:
        return DcmipT11WindHook(ctx.geometry, ctx.metric, ctx.operators, ctx.config)
    return None


@register_step_hook("dcmip_t12_wind", phase=PHASE_STATE)
def _dcmip_t12(ctx: "StepHookContext") -> Optional[StepHook]:
    if ctx.config.case_number == 12:
        return DcmipT12WindHook(ctx.geometry, ctx.metric, ctx.operators, ctx.config)
    return None


@register_step_hook("exponential_filter", phase=PHASE_STATE)
def _exponential_filter(ctx: "StepHookContext") -> Optional[StepHook]:
    if ctx.config.expfilter_apply:
        return ExponentialFilterHook(ctx.geometry, ctx.metric, ctx.operators, ctx.config)
    return None
