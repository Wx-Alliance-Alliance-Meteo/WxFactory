import math
from collections.abc import Callable
from time import time

import numpy
import torch

from ..common.configuration import Configuration
from ..jacobian import (
    assemble_vertical_blocks,
    columns_to_state,
    forcing_jac_prepare,
    forcing_jvp,
    j2_flux_matvec,
    j2_prepare,
    solve_stiff_columns,
    split_vertical_blocks,
    state_to_columns,
)
from ..solvers import ExponentialSolverRequest, resolve_exponential_solver
from .integrator import Integrator


class PartRosExp2(Integrator):
    """Second-order partitioned Rosenbrock-exponential integrator.

    Splits the right-hand side as f = f1 + f2. The column-local stiff partition f1 contains vertical
    mass, vertical-momentum and thermodynamic fluxes plus gravity. The complementary f2 partition
    contains the terrain-balanced horizontal-momentum operator and non-stiff forcing. It advances

        (I - h/2 J1) delta = 1/2 (e^{h J2} + I) h f1 + phi1(h J2) h f2,   y_{n+1} = y_n + delta,

    J1 is assembled and solved by columns; J2 is applied matrix-free.
    """

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
        self.rhs_imp = rhs_imp  # Vertically stiff partition.
        self.rhs_exp = rhs_exp  # Complementary partition.
        self.tol = param.tolerance
        self.krylov_mmax = param.krylov_mmax
        self.krylov_m = None  # Recycled Krylov size.
        self.exponential_solver = param.exponential_solver
        self.solve_exponential = resolve_exponential_solver(self.exponential_solver)
        self.krylov_size = param.krylov_size
        self.exode_method = param.exode_method
        self.exode_controller = param.exode_controller

    def _apply_phi(self, J_exp: Callable, vec):
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

        # Split one vertical-block assembly between J1 and J2.
        lower, diag, upper = assemble_vertical_blocks(rhsobj, Q)
        momentum_blocks, stiff_blocks = split_vertical_blocks(rhsobj, lower, diag, upper)
        del lower, diag, upper

        j2_base = j2_prepare(self.rhs_full, Q, momentum_blocks)
        forcing_base = forcing_jac_prepare(self.rhs_full, Q)
        f_imp = f1.flatten()
        f_exp = f2.flatten()

        # Apply J2 analytically inside the exponential solver.
        def J_exp(v):
            vv = v.reshape(Q.shape)
            jflux = j2_flux_matvec(self.rhs_full, Q, vv, j2_base)
            jforcing = forcing_jvp(self.rhs_full, Q, vv, forcing_base)
            return (dt * (jflux + jforcing)).flatten()

        # phi0 acts on f1/2 and phi1 on f2, so the solver returns e^{hJ2} f1/2 + phi1(hJ2) f2; adding
        # f1/2 completes 1/2 (e^{hJ2} + I) f1.
        #
        # The equivalent rewrite h f1 + h phi1(hJ2) [f2 + h/2 J2 f1] was tried and is not used. It
        # sends h/2 J2 f1 through the Krylov space instead of f1/2, which is smaller only while
        # |h J2| < 1; past that the Krylov space receives a larger vector and the accuracy is worse,
        # by a factor growing linearly in |h J2|. Since the point of an exponential integrator is to
        # allow large steps, the form used here is the one that does not degrade with h.
        n = f_imp.shape[0]
        vec = torch.zeros((2, n), dtype=Q.dtype)
        vec[0, :] = 0.5 * f_imp
        vec[1, :] = f_exp

        tic = time()
        phiv = self._apply_phi(J_exp, vec)
        time_exp = time() - tic

        tic = time()
        rhs_delta = ((phiv.reshape(-1) + 0.5 * f_imp) * dt).reshape(Q.shape)
        delta_col = solve_stiff_columns(rhsobj, stiff_blocks, state_to_columns(rhsobj, rhs_delta), dt)
        delta = columns_to_state(rhsobj, delta_col, rhs_delta)
        time_imp = time() - tic

        if self.context.comm.rank == 0:
            print(
                f"PartRosExp2 direct column solve {time_imp:.3f} s ; exponential {time_exp:.3f} s",
                flush=True,
            )

        return Q + delta


REGISTRY = {
    "partrosexp2": lambda cfg, rhs, prec, ctx: PartRosExp2(
        cfg, rhs.full, rhs.implicit, rhs.explicit, preconditioner=prec, context=ctx
    ),
}
