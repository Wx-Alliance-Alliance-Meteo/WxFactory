from abc import ABC, abstractmethod
from time import time

import torch
from torch import Tensor

from ..common import Configuration
from ..context import Context
from ..output.output_manager import OutputManager
from ..precondition import Preconditioner
from ..solvers import fgmres


class Integrator(ABC):
    """Describes the time-stepping mechanism of the simulation.

    Attributes:

       output_manager -- OutputManager object that an Integrator can use. When it is present, the Integrator
                         can output some of its intermediary data that can be useful for analysing performance.
       preconditioner -- Optional object that can be used to precondition a problem. It must provide a "prepare"
                         and a "__call__" method.
       context         -- Object that describes the execution context (including hardware information)

    """

    latest_time: float
    output_manager: OutputManager | None
    context: Context
    preconditioner: Preconditioner | None

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
        self.sim_time = -1.0
        self.failure_flag = 0
        self.num_completed_steps = 0

    def _initial_exode_step(self, rhs, state, dt):
        """Convert the physical wave rate to a first trial in auxiliary time."""
        if not hasattr(rhs, "wave_rate"):
            return 1.0
        rate = rhs.wave_rate(state)
        # Use RK4's stability radius for EXLRK4(3), and a unit startup scale otherwise.
        radius = 2.785293563405282 if self.param.exode_method.upper() == "EXLRK4(3)" else 1.0
        return min(1.0, 0.8 * radius / (dt * rate)) if rate > 0 else 1.0

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

    def _implicit_state_and_rhs(self, state, rhs):
        """Evaluate implicit residuals in float64 to reduce cancellation."""
        if hasattr(rhs, "double_precision_rhs"):
            if state.is_cuda and hasattr(rhs.metric, "offload_derivatives"):
                rhs.metric.offload_derivatives()
            rhs = rhs.double_precision_rhs()
            state = state.to(torch.float64)
        return state, rhs

    def _newton_jacobian(self, rhs, dt, rhs_weight):
        """Build I/dt - rhs_weight F' at each Newton iterate when available."""
        if getattr(self.param, "jacobian", None) != "analytic" or not hasattr(rhs, "linearize"):
            return None

        def linearize(state):
            action = rhs.linearize(state)

            def matvec(vector):
                vector = vector.to(state.dtype)
                return vector / dt - rhs_weight * action(vector).reshape(vector.shape)

            return matvec

        return linearize

    @abstractmethod
    def __step__(self, Q: Tensor, dt: float) -> Tensor:
        pass

    def __prestep__(self, Q: Tensor, dt: float) -> None:
        pass

    def step(self, Q: Tensor, dt: float) -> Tensor:
        t0 = time()

        self.__prestep__(Q, dt)

        if self.preconditioner is not None:
            self.preconditioner.prepare(dt, Q)

        result = self.__step__(Q, dt)

        t1 = time()
        self.latest_time = t1 - t0

        self.sim_time += dt
        self.num_completed_steps += 1

        return result
