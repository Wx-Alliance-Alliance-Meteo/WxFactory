from time import time
from typing import Callable

import numpy

from ..common.configuration import Configuration
from .integrator import Integrator, SolverInfo
from ..solvers import matvec_fun, pmex
from ..solvers.global_operations import global_inf_norm
from ..rhs.vertical_jacobian import (
    assemble_j1_blocks_analytic,
    block_thomas_solve,
    state_to_col,
    col_to_state,
    j2_prepare,
    j2_flux_matvec,
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
        self.jacobian_method = param.jacobian_method  # for the matrix-free J2 products (use 'fd' in single)
        self.krylov_mmax = param.krylov_mmax  # cap the exponential Krylov space (memory)
        self.krylov_m = None  # previous step's final Krylov size, recycled as the next m_init (see __step__)

    def __step__(self, Q: numpy.ndarray, dt: float):
        xp = self.device.xp
        rhsobj = self.rhs_full

        f1 = self.rhs_imp(Q)
        f2 = self.rhs_exp(Q)  # horizontal partition, computed directly (no full - f1 cancellation)
        forcing_base = self.rhs_full.forcing_only(Q)  # base for the FD of the non-stiff forcing
        j2_base = j2_prepare(self.rhs_full, Q)  # frozen base of the analytic J2, computed once per step
        f_imp = f1.flatten()
        f_exp = f2.flatten()

        # Q is fixed across every matvec of the PMEX solve, so its single-precision FD step scale is
        # too: compute it once here instead of a global reduction inside each matvec_fun call.
        q_scale = max(1.0, float(global_inf_norm(Q)))

        # Exponential part
        def J_exp(v):
            jflux = j2_flux_matvec(self.rhs_full, Q, v.reshape(Q.shape), j2_base).flatten()
            jforcing = matvec_fun(
                v, dt, Q, forcing_base, self.rhs_full.forcing_only, self.jacobian_method, q_scale=q_scale
            )
            return dt * jflux + jforcing

        n = f_imp.shape[0]
        vec = xp.zeros((2, n), dtype=Q.dtype)
        vec[0, :] = 0.5 * f_imp
        vec[1, :] = f_exp

        # Recycle the Krylov size across steps. 
        m_init = self.krylov_m if self.krylov_m is not None else 10

        tic = time()
        phiv, stats = pmex(
            [1.0], J_exp, vec, tol=self.tol, m_init=m_init, mmax=self.krylov_mmax, task1=False, device=self.device
        )
        time_exp = time() - tic
        self.krylov_m = stats[5]  # final Krylov size of this step, reused as next step's m_init
        if self.device.comm.rank == 0:
            print(
                f"PMEX convergence at iteration {stats[2]} (using {stats[0]} internal substeps"
                f" and {stats[1]} rejected expm)",
                flush=True,
            )

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
