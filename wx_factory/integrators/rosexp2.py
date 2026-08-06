from collections.abc import Callable
from time import time

import numpy
import torch

from ..common.configuration import Configuration
from ..jacobian import fd_jacobian_matvec, fd_rosenbrock_matvec
from ..solvers import ExponentialSolverRequest, resolve_exponential_solver
from .integrator import Integrator, SolverInfo


class RosExp2(Integrator):
    def __init__(
        self, param: Configuration, rhs_full: Callable, rhs_imp: Callable, *, context=None, preconditioner=None
    ):
        super().__init__(param, context=context, preconditioner=preconditioner)

        self.rhs_full = rhs_full
        self.rhs_imp = rhs_imp
        self.tol = param.tolerance
        self.gmres_restart = param.gmres_restart
        self.krylov_mmax = param.krylov_mmax
        self.solve_exponential = resolve_exponential_solver(param.exponential_solver)
        self.exode_method = param.exode_method
        self.exode_controller = param.exode_controller

    def __step__(self, Q, dt):
        rhs_full = self.rhs_full(Q)
        rhs_imp = self.rhs_imp(Q)

        Q_flat = Q.flatten()
        n = len(Q_flat)

        def J_exp(v):
            return fd_jacobian_matvec(v, dt, Q, rhs_full, self.rhs_full) - fd_jacobian_matvec(
                v, dt, Q, rhs_imp, self.rhs_imp
            )

        vec = torch.zeros((2, n), dtype=Q.dtype)
        vec[1, :] = rhs_full.flatten()

        tic = time()
        exponential_result = self.solve_exponential(
            ExponentialSolverRequest(
                [1.0],
                J_exp,
                vec,
                self.tol,
                self.krylov_mmax,
                self.context,
                exode_method=self.exode_method,
                exode_controller=self.exode_controller,
            )
        )
        phiv = exponential_result.value
        time_exp = time() - tic

        tic = time()

        def A(v):
            return fd_rosenbrock_matvec(v, dt, Q, rhs_imp, self.rhs_imp)

        b = (A(Q_flat) + phiv * dt).flatten()
        Q_x0 = Q_flat.copy()
        Qnew, norm_r, norm_b, num_iter, flag, residuals = self._solve_linear(
            A,
            b,
            x0=Q_x0,
            tol=self.tol,
            restart=self.gmres_restart,
        )
        time_imp = time() - tic

        self.solver_info = SolverInfo(flag, time_imp, num_iter, residuals)

        if self.context.comm.rank == 0:
            result_type = "convergence" if flag == 0 else "stagnation/interruption"
            print(
                f"FGMRES {result_type} at iteration {num_iter} in {time_imp:4.1f} s to a solution with"
                f" relative residual {norm_r / norm_b: .2e}"
            )

            print(f"Elapsed time: exponential {time_exp:.3f} secs ; implicit {time_imp:.3f} secs")

        return numpy.reshape(Qnew, Q.shape)


REGISTRY = {
    "rosexp2": lambda cfg, rhs, prec, ctx: RosExp2(cfg, rhs.full, rhs.full, preconditioner=prec, context=ctx),
}
