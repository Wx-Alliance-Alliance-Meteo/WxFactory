from abc import ABC, abstractmethod
from time import time

from torch import Tensor

from ..common import Configuration
from ..context import Context
from ..output.output_manager import OutputManager
from ..precondition import Preconditioner
from ..solvers import SolverInfo, fgmres


class Integrator(ABC):
    """Describes the time-stepping mechanism of the simulation.

    Attributes:

       output_manager -- OutputManager object that an Integrator can use. When it is present, the Integrator
                         can output some of its intermediary data that can be useful for analysing performance.
       solver_info    -- At each timestep, the content of solver_info is outputted (if output_manager is present)
                         If a certain (derived type) Integrator wants to log information about its convergence,
                         performance and other internal data, it should create a SolverInfo object and assign it
                         to self.solver_info
       preconditioner -- Optional object that can be used to precondition a problem. It must provide a "prepare"
                         and a "__call__" method.
       context         -- Object that describes the execution context (including hardware information)

    """

    latest_time: float
    output_manager: OutputManager | None
    context: Context
    preconditioner: Preconditioner | None
    solver_info: SolverInfo | None

    def __init__(
        self,
        param: Configuration,
        *,
        output_manager: OutputManager | None = None,
        context: Context | None = None,
        preconditioner=None,
    ) -> None:
        self.output_manager = output_manager
        self.preconditioner = preconditioner
        self.context = context if context is not None else Context.get_default()
        self.param = param
        self.verbose_solver = param.verbose_solver
        self.solver_info = None
        self.sim_time = -1.0
        self.failure_flag = 0
        self.num_completed_steps = 0

    def _solve_linear(self, A, b, x0=None, tol=1e-8, restart=20, maxiter=None):
        return fgmres(
            A,
            b,
            x0=x0,
            tol=tol,
            restart=restart,
            maxiter=maxiter,
            preconditioner=self.preconditioner,
            verbose=self.verbose_solver,
            context=self.context,
        )

    @abstractmethod
    def __step__(self, Q: Tensor, dt: float) -> Tensor:
        pass

    def __prestep__(self, Q: Tensor, dt: float) -> None:
        pass

    def step(self, Q: Tensor, dt: float) -> Tensor:
        """Advance the system forward in time"""
        t0 = time()

        self.__prestep__(Q, dt)

        if self.preconditioner is not None:
            self.preconditioner.prepare(dt, Q)

        # The stepping itself
        result = self.__step__(Q, dt)

        t1 = time()
        self.latest_time = t1 - t0

        self.solver_info = None

        self.sim_time += dt
        self.num_completed_steps += 1

        return result
