from abc import ABC, abstractmethod
from time import time
from typing import Optional

import numpy

from ..common import Configuration
from ..device import Device
from ..precondition.factorization import Factorization
from ..precondition.multigrid import Multigrid
from ..output.output_manager import OutputManager
from ..solvers import SolverInfo, fgmres, global_norm
from ..rhs.rhs import RHS


class Integrator(ABC):
    """Describes the time-stepping mechanism of the simulation.

    Attributes:

       output_manager -- OutputManager object that an Integrator can use. When it is present, the Integrator
                         can output some of its intermediary data that can be useful for analysing performance.
                         For now, it must be assigned *after* the Integrator has been initialized.
       solver_info    -- At each timestep, the content of solver_info is outputted (if output_manager is present)
                         If a certain (derived type) Integrator wants to log information about its convergence,
                         performance and other internal data, it should create a SolverInfo object and assign it
                         to self.solver_info
       preconditioner -- Optional object that can be used to precondition a problem. It must provide a "prepare"
                         and a "__call__" method.
       device         -- Object that describes on what hardware (CPU/GPU) the code will run

    """

    latest_time: float
    output_manager: Optional[OutputManager]
    device: Device
    preconditioner: Optional[Multigrid]
    solver_info: Optional[SolverInfo]

    def __init__(
        self,
        param: Configuration,
        *,
        output_manager: Optional[OutputManager] = None,
        device: Optional[Device] = None,
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

    def _solve_linear(self, A, b, x0=None, tol=1e-8, restart=20, maxiter=None):
        return fgmres(
            A, b, x0=x0, tol=tol,
            restart=restart, maxiter=maxiter,
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
            if isinstance(self.preconditioner, Multigrid):
                self.preconditioner.prepare(dt, Q)
            elif isinstance(self.preconditioner, Factorization):
                if hasattr(self, "A"):
                    self.preconditioner.prepare(self.A)
                else:
                    print(
                        f"Trying to use a factorization-based preconditioner, but you didn't provide a matrix"
                        f"(must define it in the __prestep__ method of your integrator)"
                    )

        # The stepping itself
        result = self.__step__(Q, dt)

        t1 = time()
        self.latest_time = t1 - t0

        # Output info from completed step (if possible)
        if self.output_manager is not None:

            solver_info = self.solver_info if self.solver_info is not None else SolverInfo()

            rhs_times = None
            if hasattr(self, "rhs") and isinstance(self.rhs, RHS):
                self.rhs.retrieve_last_times()
                rhs_times = self.rhs.timings

            self.output_manager.store_solver_stats(
                t1 - t0, self.sim_time, dt, solver_info, self.preconditioner, rhs_times
            )

            if hasattr(self, "rhs") and isinstance(self.rhs, RHS):
                self.rhs.clear_timings()

        self.solver_info = None

        self.sim_time += dt
        self.num_completed_steps += 1

        return result
