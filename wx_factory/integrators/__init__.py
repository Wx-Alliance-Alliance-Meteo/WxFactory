"""Different methods to step forward in time."""

from ..context import Context
from . import (
    backward_euler as _backward_euler,
    bdf2 as _bdf2,
    crank_nicolson as _crank_nicolson,
    epi as _epi,
    epi_stiff as _epi_stiff,
    euler1 as _euler1,
    imex2 as _imex2,
    partrosexp2 as _partrosexp2,
    ros2 as _ros2,
    rosexp2 as _rosexp2,
    sdirk as _sdirk,
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
from .integrator import Integrator
from .partrosexp2 import PartRosExp2
from .ros2 import Ros2
from .rosexp2 import RosExp2
from .sdirk import SDIRKLstable
from .splitting import LieSplitting, OS22Splitting, StrangSplitting
from .srerk import Srerk
from .tvdrk3 import Tvdrk3

REGISTRY: dict = {}
for _mod in [
    _backward_euler,
    _bdf2,
    _crank_nicolson,
    _epi,
    _epi_stiff,
    _euler1,
    _imex2,
    _partrosexp2,
    _ros2,
    _rosexp2,
    _sdirk,
    _splitting,
    _srerk,
    _tvdrk3,
]:
    REGISTRY.update(_mod.REGISTRY)


def resolve(name: str, config, rhs, preconditioner, context: Context) -> Integrator:
    """Create the integrator identified by `name`."""
    if name not in REGISTRY:
        raise ValueError(f"Time integration method '{name}' not supported")
    return REGISTRY[name](config, rhs, preconditioner, context)


__all__ = [
    "REGISTRY",
    "BackwardEuler",
    "Bdf2",
    "CrankNicolson",
    "Epi",
    "EpiStiff",
    "Euler1",
    "Imex2",
    "Integrator",
    "LieSplitting",
    "OS22Splitting",
    "PartRosExp2",
    "Ros2",
    "RosExp2",
    "SDIRKLstable",
    "Srerk",
    "StrangSplitting",
    "Tvdrk3",
    "resolve",
]
