from abc import ABC, abstractmethod
from itertools import combinations
import math
from time import time
from typing import Optional

import numpy

from common import Configuration
from device import Device
from precondition.factorization import Factorization
from precondition.multigrid import Multigrid
from output.output_manager import OutputManager
from solvers import SolverInfo
from rhs.rhs import RHS


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

    def __init__(self, param: Configuration, **kwargs) -> None:
        self.output_manager = None
        self.preconditioner = None

        if "output_manager" in kwargs:
            self.output_manager = kwargs["output_manager"]

        if "device" in kwargs:
            self.device = kwargs["device"]
        else:
            self.device = Device.get_default()

        if "preconditioner" in kwargs:
            self.preconditioner = kwargs["preconditioner"]

        self.verbose_solver = param.verbose_solver
        self.solver_info = None
        self.sim_time = -1.0
        self.failure_flag = 0
        self.num_completed_steps = 0

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

        self.solver_info = None

        self.sim_time += dt
        self.num_completed_steps += 1

        return result


class scipy_counter:  # TODO : tempo
    """Callback object for linear solvers (from Scipy and others)."""

    def __init__(self, disp=False):
        self._disp = disp
        self.niter = 0
        self.res = 0.0

    def __call__(self, rk=None):
        self.niter += 1
        if rk is not None:
            self.res = float(rk)
        if self._disp:
            print(f"iter {self.niter:3d}\trk = {str(rk)}", flush=True)

    def num_iter(self):
        return self.niter


def alpha_coeff(c):
    """Compute the coefficients for stiffness resilient exponential methods based on node values c."""
    m = len(c)
    alpha = numpy.zeros((m, m))
    for i in range(m):
        c_no_i = [cc for (j, cc) in enumerate(c) if j != i]
        denom = c[i] ** 2 * math.prod([c[i] - cl for cl in c_no_i])
        for k in range(m):
            sp = sum([math.prod(v) for v in combinations(c_no_i, m - k - 1)])
            alpha[k, i] = (-1) ** (m - k + 1) * math.factorial(k + 2) * sp / denom

    return alpha
