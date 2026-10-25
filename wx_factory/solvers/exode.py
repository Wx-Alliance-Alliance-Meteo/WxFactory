"""Evaluate exponential and phi-function actions through an auxiliary ODE."""

import math
from collections.abc import Callable
from dataclasses import dataclass

from mpi4py import MPI

from ..context import Context


@dataclass
class ExodeStartup:
    """First accepted auxiliary step from the preceding successful solve."""

    first_accepted: float | None = None


def exode(
    tau_out,
    A,
    u,
    method="BS3(2)",
    controller="deadbeat",
    rtol=1e-3,
    atol=1e-6,
    context: Context | None = None,
    startup: ExodeStartup | None = None,
    initial_step: Callable[[], float] | None = None,
):
    """Solve y' = A(y) + sum_j t**j/j! u[j+1], starting from u[0]."""
    # The integrator package also imports the exponential solvers.
    from ..integrators.butcher import METHODS

    tau_out = float(tau_out)
    if not math.isfinite(tau_out) or tau_out <= 0:
        raise ValueError("EXODE requires a positive, finite output time")
    method_name = method.upper()
    if method_name not in METHODS:
        raise ValueError(f"Unknown EXODE method {method!r}; choose from {', '.join(METHODS)}")

    def rhs(time, state):
        tendency = A(state)
        for order, vector in enumerate(u[1:]):
            tendency += time**order / math.factorial(order) * vector
        return tendency

    comm = context.comm if context is not None else MPI.COMM_WORLD
    first_step = startup.first_accepted if startup is not None else None
    if first_step is None:
        first_step = initial_step() if initial_step is not None else tau_out
    if not math.isfinite(first_step) or first_step <= 0:
        raise ValueError("EXODE initial step must be positive and finite")
    first_step = comm.allreduce(min(first_step, tau_out), op=MPI.MIN)
    solver = METHODS[method_name](
        rhs,
        0.0,
        u[0].clone(),
        tau_out,
        controller=controller or None,
        first_step=first_step,
        rtol=rtol,
        atol=atol,
        context=context,
    )
    first_accepted = None
    while solver.status == "running":
        previous_time = solver.t
        solver.step()
        if solver.status == "failed":
            raise RuntimeError(f"EXODE integration failed at t={solver.t}")
        if first_accepted is None:
            first_accepted = solver.t - previous_time

    if startup is not None:
        startup.first_accepted = first_accepted
    stats = (
        solver.nfev,
        solver.failed_steps,
        solver.nfev,
        solver.estimate_error(),
        solver.error_norm_old,
        solver.h_previous,
        solver.h,
    )
    return solver.y, stats
