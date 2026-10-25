import math
from collections.abc import Callable
from time import time

import numpy
import torch

from ..common.configuration import Configuration
from ..solvers import ExponentialSolverRequest, resolve_exponential_solver
from .integrator import Integrator


class PartRosExp2(Integrator):
    """Second-order partitioned Rosenbrock-exponential integrator."""

    def __init__(
        self,
        param: Configuration,
        rhs_full: Callable,
        rhs_imp: Callable,
        rhs_exp: Callable,
        *,
        context=None,
        preconditioner=None,
    ):
        super().__init__(param, context=context, preconditioner=preconditioner)
        self.rhs_full = rhs_full
        self.rhs_imp = rhs_imp
        self.rhs_exp = rhs_exp
        self.tol = param.tolerance
        self.krylov_mmax = param.krylov_mmax
        self.krylov_m = None
        self.exponential_solver = param.exponential_solver
        self.solve_exponential = resolve_exponential_solver(self.exponential_solver)
        self.krylov_size = param.krylov_size
        self.exode_method = param.exode_method
        self.exode_controller = param.exode_controller

    def _apply_phi(self, J_exp: Callable, vec, initial_step=None):
        """Evaluate phi_1(J_exp) @ vec with the configured exponential solver."""
        solver = self.exponential_solver

        pmex_family = solver in ("pmex", "pmex_ne")
        result = self.solve_exponential(
            ExponentialSolverRequest(
                [1.0],
                J_exp,
                vec,
                self.tol,
                self.krylov_mmax,
                self.context,
                krylov_minit=(self.krylov_m or 10) if pmex_family else self.krylov_size,
                krylov_mmin=16 if solver in ("pmex_ne", "kiops") else 10,
                exode_method=self.exode_method,
                exode_controller=self.exode_controller,
                initial_step=initial_step,
            )
        )

        if result.final_krylov_size is not None:
            if pmex_family:
                self.krylov_m = result.final_krylov_size
            else:
                self.krylov_size = math.floor(0.7 * result.final_krylov_size + 0.3 * self.krylov_size)

        return result.value

    def __step__(self, Q: numpy.ndarray, dt: float):
        rhsobj = self.rhs_full

        f1 = self.rhs_imp(Q)
        f2 = self.rhs_exp(Q)

        linearization = rhsobj.linearize_partitions(Q)

        def J_exp(v):
            return dt * linearization.explicit(v)

        def solve_columns(vector):
            return linearization.solve(vector, dt)

        f_imp = f1.flatten()
        f_exp = f2.flatten()

        n = f_imp.shape[0]
        vec = torch.zeros((2, n), dtype=Q.dtype)
        vec[0, :] = 0.5 * f_imp
        vec[1, :] = f_exp

        tic = time()
        phiv = self._apply_phi(J_exp, vec, lambda: self._initial_exode_step(rhsobj, Q, dt))
        time_exp = time() - tic

        tic = time()
        rhs_delta = ((phiv.reshape(-1) + 0.5 * f_imp) * dt).reshape(Q.shape)
        if hasattr(linearization, "release_explicit"):
            linearization.release_explicit()
        del vec, phiv, f2, f_exp
        delta = solve_columns(rhs_delta)
        time_imp = time() - tic

        if self.context.comm.rank == 0:
            print(
                f"PartRosExp2 direct column solve {time_imp:.3f} s ; exponential {time_exp:.3f} s",
                flush=True,
            )

        return Q + delta


REGISTRY: dict = {}

PARTITIONED_REGISTRY = {
    "partrosexp2": lambda cfg, rhs, prec, ctx: PartRosExp2(
        cfg, rhs.full, rhs.implicit, rhs.explicit, preconditioner=prec, context=ctx
    ),
}
