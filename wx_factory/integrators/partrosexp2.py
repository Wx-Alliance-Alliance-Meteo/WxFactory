from time import time
from typing import Callable

from mpi4py import MPI
import numpy
from scipy.sparse.linalg import LinearOperator

from common.configuration import Configuration
from .integrator import Integrator, SolverInfo
from solvers import fgmres, matvec_fun, pmex


class PartRosExp2(Integrator):
    def __init__(self, param: Configuration, rhs_full: Callable, rhs_imp: Callable, **kwargs):
        super().__init__(param, **kwargs)

        self.rhs_full = rhs_full
        self.rhs_imp = rhs_imp
        self.tol = param.tolerance
        self.gmres_restart = param.gmres_restart

    def __step__(self, Q: numpy.ndarray, dt: float):

        rhs_full = self.rhs_full(Q)
        rhs_imp = self.rhs_imp(Q)
        f_imp = rhs_imp.flatten()
        f_exp = (rhs_full - rhs_imp).flatten()

        def J_full(v):
            return matvec_fun(v, dt, Q, rhs_full, self.rhs_full)

        def J_imp(v):
            return matvec_fun(v, dt, Q, rhs_imp, self.rhs_imp)

        def J_exp(v):
            return J_full(v) - J_imp(v)

        Q_flat = Q.flatten()
        n = len(Q_flat)

        vec = numpy.zeros((2, n))
        vec[0, :] = 0.5 * f_imp
        vec[1, :] = f_exp.copy()

        tic = time()
        phiv, stats = pmex([1.0], J_exp, vec, tol=self.tol, task1=False, device=self.device)
        time_exp = time() - tic
        if self.device.comm.rank == 0:
            print(
                f"PMEX convergence at iteration {stats[2]} (using {stats[0]} internal substeps"
                f" and {stats[1]} rejected expm)"
            )

        tic = time()

        def A(v):
            return v - J_imp(v) / 2

        b = (A(Q_flat) + (phiv + 0.5 * f_imp) * dt).flatten()
        Q_x0 = Q_flat.copy()
        Qnew, norm_r, norm_b, num_iter, flag, residuals = fgmres(
            A,
            b,
            x0=Q_x0,
            tol=self.tol,
            restart=self.gmres_restart,
            maxiter=None,
            preconditioner=self.preconditioner,
            verbose=self.verbose_solver,
            device=self.device,
        )
        time_imp = time() - tic

        self.solver_info = SolverInfo(flag, time_imp, num_iter, residuals)

        if self.device.comm.rank == 0:
            result_type = "convergence" if flag == 0 else "stagnation/interruption"
            print(
                f"FGMRES {result_type} at iteration {num_iter} in {time_imp:4.1f} s to a solution with"
                f" relative residual {norm_r/norm_b: .2e}"
            )

            print(f"Elapsed time: exponential {time_exp:.3f} secs ; implicit {time_imp:.3f} secs")

        return numpy.reshape(Qnew, Q.shape)
