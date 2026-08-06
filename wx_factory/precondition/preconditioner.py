from abc import ABC, abstractmethod

from mpi4py import MPI
from torch import Tensor

from ..common.configuration import Configuration
from ..solvers import MatvecOp


class Preconditioner(MatvecOp, ABC):
    """Describes a matrix-like object that can be used to precondition a linear system.

    A concrete preconditioner implements ``__apply__`` (how it acts on a vector) and, if it needs
    to update internal state at the start of each time step, overrides ``prepare``. Integrators call
    ``prepare`` uniformly, so no per-type dispatch is needed. See ``doc/contribute.md`` for the
    recipe to add one via the preconditioner registry."""

    def __init__(self, dtype, shape: tuple, param: Configuration) -> None:
        super().__init__(self.apply, dtype, shape)
        self.verbose = param.verbose_precond if MPI.COMM_WORLD.rank == 0 else 0

    def prepare(self, dt: float, Q: Tensor) -> None:
        """Update per-time-step internal state before the step's linear solves. Default: no-op."""

    def __call__(self, vec: Tensor, x0: Tensor | None = None, verbose: int | None = None) -> Tensor:
        return self.apply(vec, x0, verbose)

    def apply(self, vec: Tensor, x0: Tensor | None = None, verbose: int | None = None) -> Tensor:
        if verbose is None:
            verbose = self.verbose
        return self.__apply__(vec, x0, verbose)

    @abstractmethod
    def __apply__(self, vec: Tensor, x0: Tensor | None = None, verbose: int | None = None) -> Tensor:
        pass
