from abc import ABC, abstractmethod
from collections.abc import Callable
from time import time

import numpy
import torch
from numpy.typing import NDArray

from ..common import Configuration
from ..device import Device
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
       device         -- Object that describes on what hardware (CPU/GPU) the code will run

    """

    latest_time: float
    output_manager: OutputManager | None
    device: Device
    preconditioner: Preconditioner | None
    solver_info: SolverInfo | None

    def __init__(
        self,
        param: Configuration,
        *,
        output_manager: OutputManager | None = None,
        device: Device | None = None,
        preconditioner=None,
    ) -> None:
        self.output_manager = output_manager
        self.preconditioner = preconditioner
        self.device = device if device is not None else Device.get_default()
        self.param = param
        self.verbose_solver = param.verbose_solver
        self.solver_info = None
        self.sim_time = -1.0
        self.failure_flag = 0
        self.num_completed_steps = 0
        self.phi_rhs_double = bool(getattr(param, "phi_rhs_double", 1))

    def evaluate_rhs(self, rhs: Callable, Q: NDArray) -> NDArray:
        """Evaluate an update-driving tendency at its configured precision."""
        if not self.phi_rhs_double or Q.dtype != torch.float32:
            return rhs(Q)
        return rhs(Q.to(torch.float64)).to(Q.dtype)

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
            device=self.device,
        )

    @abstractmethod
    def __step__(self, Q: numpy.ndarray, dt: float) -> numpy.ndarray:
        pass

    def __prestep__(self, Q: numpy.ndarray, dt: float) -> None:
        pass

    def step(self, Q: numpy.ndarray, dt: float):
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
