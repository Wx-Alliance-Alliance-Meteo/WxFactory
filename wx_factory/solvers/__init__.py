"""Different methods to solve a system of equations."""

from .exode import exode
from .exponential_solver import (
    EXPONENTIAL_SOLVER_REGISTRY,
    ExponentialSolverRequest,
    ExponentialSolverResult,
    register_exponential_solver,
    resolve_exponential_solver,
)
from .fgmres import fgmres
from .global_operations import global_dotprod, global_inf_norm, global_norm
from .kiops import kiops
from .matvec import MatvecOp, MatvecOpBasic, MatvecOpRat, matvec_fun, matvec_rat
from .nonlin import KrylovJacobian, newton_krylov
from .pmex import pmex
from .solver_info import SolverInfo

__all__ = [
    "EXPONENTIAL_SOLVER_REGISTRY",
    "ExponentialSolverRequest",
    "ExponentialSolverResult",
    "KrylovJacobian",
    "MatvecOp",
    "MatvecOpBasic",
    "MatvecOpRat",
    "SolverInfo",
    "exode",
    "fgmres",
    "global_dotprod",
    "global_inf_norm",
    "global_norm",
    "kiops",
    "matvec_fun",
    "matvec_rat",
    "newton_krylov",
    "pmex",
    "register_exponential_solver",
    "resolve_exponential_solver",
]
