"""Different methods to step forward in time."""

from .integrator import Integrator

from . import (
    backward_euler as _backward_euler,
    bdf2 as _bdf2,
    crank_nicolson as _crank_nicolson,
    epi as _epi,
    epi_stiff as _epi_stiff,
    euler1 as _euler1,
    imex2 as _imex2,
    neural as _neural,
    partrosexp2 as _partrosexp2,
    ros2 as _ros2,
    rosexp2 as _rosexp2,
    splitting as _splitting,
    srerk as _srerk,
    tvdrk3 as _tvdrk3,
)

from .backward_euler import BackwardEuler
from .bdf2 import Bdf2
from .crank_nicolson import CrankNicolson
from .epi import Epi
from .epi_stiff import EpiStiff
from .euler1 import Euler1
from .imex2 import Imex2
from .partrosexp2 import PartRosExp2
from .ros2 import Ros2
from .rosexp2 import RosExp2
from .splitting import StrangSplitting, LieSplitting
from .srerk import Srerk
from .tvdrk3 import Tvdrk3
from .neural import Neural

REGISTRY: dict = {}
for _mod in [
    _backward_euler, _bdf2, _crank_nicolson, _epi, _epi_stiff, _euler1, _imex2,
    _neural, _partrosexp2, _ros2, _rosexp2, _splitting, _srerk, _tvdrk3,
]:
    REGISTRY.update(_mod.REGISTRY)


def resolve(name: str, config, rhs, preconditioner, device) -> Integrator:
    """Create the integrator identified by `name`."""
    if name not in REGISTRY:
        raise ValueError(f"Time integration method '{name}' not supported")
    return REGISTRY[name](config, rhs, preconditioner, device)


__all__ = [
    "Epi", "EpiStiff", "Euler1", "Imex2", "Integrator", "PartRosExp2",
    "Ros2", "RosExp2", "StrangSplitting", "LieSplitting", "Srerk", "Tvdrk3",
    "BackwardEuler", "CrankNicolson", "Bdf2", "Neural",
    "REGISTRY", "resolve",
]
