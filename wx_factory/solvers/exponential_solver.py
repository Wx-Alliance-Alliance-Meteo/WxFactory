"""Registry and common interface for exponential-system solvers."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

from ..context import Context
from .exode import ExodeStartup, exode
from .kiops import kiops
from .pmex import PmexStartup, pmex


@dataclass(frozen=True)
class ExponentialSolverRequest:
    """Inputs shared by the exponential solvers used by time integrators."""

    tau_out: float | Sequence[float]
    operator: Callable
    vectors: Any
    tolerance: float
    krylov_mmax: int
    context: Context
    krylov_minit: int | None = None
    krylov_mmin: int | None = None
    exode_method: str = "BS3(2)"
    exode_controller: str = "deadbeat"
    announce: bool = True
    initial_step: Callable[[], float] | None = None


@dataclass(frozen=True)
class ExponentialSolverResult:
    """Normalized result returned by every registered exponential solver."""

    value: Any
    iterations: int
    rejected_steps: int
    local_error: float | None
    final_krylov_size: int | None


ExponentialSolver = Callable[[ExponentialSolverRequest], ExponentialSolverResult]
EXPONENTIAL_SOLVER_REGISTRY: dict[str, ExponentialSolver] = {}


def register_exponential_solver(name: str) -> Callable[[ExponentialSolver], ExponentialSolver]:
    """Register an exponential solver adapter under a configuration name."""

    def decorator(solver: ExponentialSolver) -> ExponentialSolver:
        if name in EXPONENTIAL_SOLVER_REGISTRY:
            raise ValueError(f"An exponential solver is already registered as '{name}'")
        EXPONENTIAL_SOLVER_REGISTRY[name] = solver
        return solver

    return decorator


def resolve_exponential_solver(name: str) -> ExponentialSolver:
    """Return the adapter registered as ``name``."""

    try:
        solver = EXPONENTIAL_SOLVER_REGISTRY[name]
        if name == "exode":
            return partial(solver, startup=ExodeStartup())
        if name in ("pmex", "pmex_ne"):
            return partial(solver, startup=PmexStartup())
        return solver
    except KeyError as exc:
        raise ValueError(
            f"Exponential solver '{name}' is not registered. Registered solvers: {sorted(EXPONENTIAL_SOLVER_REGISTRY)}"
        ) from exc


def _krylov_result(value, stats, request: ExponentialSolverRequest, label: str) -> ExponentialSolverResult:
    result = ExponentialSolverResult(value, stats[2], stats[1], stats[4], stats[5])
    if request.announce and request.context.comm.rank == 0:
        print(
            f"{label} converged at iteration {result.iterations} "
            f"(using {stats[0]} internal substeps and {result.rejected_steps} rejected expm) "
            f"to a solution with local error {result.local_error:.2e}",
            flush=True,
        )
    return result


def _krylov_kwargs(request: ExponentialSolverRequest) -> dict[str, Any]:
    kwargs = {
        "tol": request.tolerance,
        "mmax": request.krylov_mmax,
        "task1": False,
        "context": request.context,
    }
    if request.krylov_minit is not None:
        kwargs["m_init"] = request.krylov_minit
    if request.krylov_mmin is not None:
        kwargs["mmin"] = request.krylov_mmin
    return kwargs


@register_exponential_solver("pmex")
def _solve_pmex(request: ExponentialSolverRequest, *, startup: PmexStartup | None = None) -> ExponentialSolverResult:
    value, stats = pmex(request.tau_out, request.operator, request.vectors, startup=startup, **_krylov_kwargs(request))
    return _krylov_result(value, stats, request, "PMEX")


@register_exponential_solver("pmex_ne")
def _solve_pmex_ne(request: ExponentialSolverRequest, *, startup: PmexStartup | None = None) -> ExponentialSolverResult:
    kwargs = _krylov_kwargs(request)
    kwargs.setdefault("mmin", 16)
    value, stats = pmex(request.tau_out, request.operator, request.vectors, startup=startup, **kwargs)
    return _krylov_result(value, stats, request, "PMEX NE")


@register_exponential_solver("kiops")
def _solve_kiops(request: ExponentialSolverRequest) -> ExponentialSolverResult:
    value, stats = kiops(request.tau_out, request.operator, request.vectors, **_krylov_kwargs(request))
    return _krylov_result(value, stats, request, "KIOPS")


@register_exponential_solver("exode")
def _solve_exode(request: ExponentialSolverRequest, *, startup: ExodeStartup | None = None) -> ExponentialSolverResult:
    tau_out = request.tau_out
    if not isinstance(tau_out, (int, float)):
        if len(tau_out) != 1:
            raise ValueError("EXODE supports exactly one output time")
        tau_out = tau_out[0]

    value, stats = exode(
        tau_out,
        request.operator,
        request.vectors,
        method=request.exode_method,
        controller=request.exode_controller,
        # The configured tolerance controls both terms of the local RK error test.
        rtol=request.tolerance,
        atol=request.tolerance,
        context=request.context,
        startup=startup,
        initial_step=request.initial_step,
    )
    result = ExponentialSolverResult(value, stats[0], stats[1], stats[3], None)
    if request.announce and request.context.comm.rank == 0:
        print(
            f"EXODE converged after {result.iterations} matrix-vector products "
            f"with {result.rejected_steps} rejected steps (max local error {result.local_error:.2e})",
            flush=True,
        )
    return result


__all__ = [
    "EXPONENTIAL_SOLVER_REGISTRY",
    "ExponentialSolver",
    "ExponentialSolverRequest",
    "ExponentialSolverResult",
    "register_exponential_solver",
    "resolve_exponential_solver",
]
