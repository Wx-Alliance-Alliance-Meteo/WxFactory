import math
from time import time
from typing import Callable

import numpy

from ..common.configuration import Configuration
from .integrator import Integrator, SolverInfo
from ..solvers import pmex, kiops, exode
from ..rhs.vertical_jacobian import (
    assemble_j1_blocks_analytic,
    block_thomas_solve,
    state_to_col,
    col_to_state,
    j2_prepare,
    j2_flux_matvec,
    forcing_jac_prepare,
    forcing_jvp,
)


class PartRosExp2(Integrator):
    """Partitioned Rosenbrock-exponential (PartRosExp2, Dallerit et al. 2024).

    Splits the right-hand side f = f1 + f2 with f1 the vertically-stiff part (vertical flux divergence
    + gravity) and f2 the horizontal remainder, and advances

        (I - h/2 J1) delta = h f1 + h phi1(h J2) [ f2 + h/2 J2 f1 ],   y_{n+1} = y_n + delta,

    with J1 = df1/dy assembled analytically (exact derivative of the discrete f1) and solved directly
    per column by block-Thomas, and J2 = df2/dy applied matrix-free inside a single PMEX phi1
    evaluation. J1 is reassembled and refactored every step.
    """

    def __init__(
        self,
        param: Configuration,
        rhs_full: Callable,
        rhs_imp: Callable,
        rhs_exp: Callable,
        *,
        device=None,
        preconditioner=None,
    ):
        super().__init__(param, device=device, preconditioner=preconditioner)
        self.rhs_full = rhs_full  # the RHS object: callable (full) and carrier of .implicit / geometry
        self.rhs_imp = rhs_imp  # f1 = the vertically-stiff partition
        self.rhs_exp = rhs_exp  # f2 = the horizontal partition (computed directly, not full - f1)
        self.tol = param.tolerance
        self.jacobian_method = param.jacobian_method  # kept for config compatibility; J_exp is now fully analytic
        self.krylov_mmax = param.krylov_mmax  # cap the exponential Krylov space (memory)
        self.krylov_m = None  # previous step's final Krylov size, recycled as the next m_init (see __step__)
        # Which solver evaluates phi_1(J_exp) @ vec (pmex / kiops / exode), honoured like in epi.py.
        self.exponential_solver = param.exponential_solver
        self.krylov_size = param.krylov_size  # kiops / pmex_ne restart size
        self.exode_method = param.exode_method
        self.exode_controller = param.exode_controller

    def _apply_phi(self, J_exp: Callable, vec):
        """Evaluate phi_1(J_exp) @ vec with the configured exponential solver (pmex / kiops / exode).

        Mirrors epi.py's dispatch and stats conventions; returns the phi vector. pmex/kiops carry a
        recycled Krylov size across steps; exode is an adaptive RK and carries none."""
        rank0 = self.device.comm.rank == 0
        solver = self.exponential_solver

        if solver in ("pmex", "pmex_ne"):
            m_init = self.krylov_m if self.krylov_m is not None else 10
            mmin = 16 if solver == "pmex_ne" else 10
            phiv, stats = pmex(
                [1.0], J_exp, vec, tol=self.tol, m_init=m_init, mmin=mmin,
                mmax=self.krylov_mmax, task1=False, device=self.device,
            )
            self.krylov_m = stats[5]  # final Krylov size, reused as next step's m_init
            if rank0:
                print(
                    f"PMEX convergence at iteration {stats[2]} (using {stats[0]} internal substeps"
                    f" and {stats[1]} rejected expm)",
                    flush=True,
                )
        elif solver == "kiops":
            phiv, stats = kiops(
                [1], J_exp, vec, tol=self.tol, m_init=self.krylov_size, mmin=16,
                mmax=self.krylov_mmax, task1=False, device=self.device,
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)
            if rank0:
                print(
                    f"KIOPS convergence at iteration {stats[2]} (using {stats[0]} internal substeps"
                    f" and {stats[1]} rejected expm)",
                    flush=True,
                )
        elif solver == "exode":
            phiv, stats = exode(
                1.0, J_exp, vec, method=self.exode_method, controller=self.exode_controller,
                atol=self.tol, task1=False, verbose=False, device=self.device,
            )
            if rank0:
                print(
                    f"EXODE converged at iteration {stats[0]} with {stats[1]} rejected steps"
                    f" (local error {stats[3]:.2e})",
                    flush=True,
                )
        else:
            raise ValueError(f"Unrecognized exponential solver {solver!r}")

        return phiv

    def __step__(self, Q: numpy.ndarray, dt: float):
        xp = self.device.xp
        rhsobj = self.rhs_full

        f1 = self.rhs_imp(Q)
        f2 = self.rhs_exp(Q)  # horizontal partition, computed directly (no full - f1 cancellation)
        j2_base = j2_prepare(self.rhs_full, Q)  # frozen base of the analytic J2, computed once per step
        forcing_base = forcing_jac_prepare(self.rhs_full, Q)  # frozen base of the analytic forcing Jacobian
        f_imp = f1.flatten()
        f_exp = f2.flatten()

        # Exponential part: both the horizontal flux Jacobian and the non-stiff forcing Jacobian are
        # applied analytically, so J_exp is finite-difference-free (no float32 FD noise, no per-matvec
        # global reduction for the step scale).
        def J_exp(v):
            vv = v.reshape(Q.shape)
            jflux = j2_flux_matvec(self.rhs_full, Q, vv, j2_base)
            jforcing = forcing_jvp(self.rhs_full, Q, vv, forcing_base)
            return (dt * (jflux + jforcing)).flatten()

        n = f_imp.shape[0]
        vec = xp.zeros((2, n), dtype=Q.dtype)
        vec[0, :] = 0.5 * f_imp
        vec[1, :] = f_exp

        tic = time()
        phiv = self._apply_phi(J_exp, vec)
        time_exp = time() - tic

        # Implicit part
        tic = time()
        rhs_delta = ((phiv.reshape(-1) + 0.5 * f_imp) * dt).reshape(Q.shape)
        L, A, U = assemble_j1_blocks_analytic(rhsobj, Q)
        bc = state_to_col(rhsobj, rhs_delta)
        dc = block_thomas_solve(rhsobj, L, A, U, bc, dt)
        delta = col_to_state(rhsobj, dc, rhs_delta)
        time_imp = time() - tic

        self.solver_info = SolverInfo(0, time_imp, 1, [])
        if self.device.comm.rank == 0:
            print(
                f"PartRosExp2 direct column solve {time_imp:.3f} s ; exponential {time_exp:.3f} s",
                flush=True,
            )

        return Q + delta


REGISTRY = {
    "partrosexp2": lambda cfg, rhs, prec, dev: PartRosExp2(
        cfg, rhs.full, rhs.implicit, rhs.explicit, preconditioner=prec, device=dev
    ),
}
