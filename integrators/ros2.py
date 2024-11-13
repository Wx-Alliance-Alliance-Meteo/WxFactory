from time import time
from typing import Callable

import numpy
from mpi4py import MPI

from common.configuration import Configuration
from solvers import fgmres, MatvecOpRat, SolverInfo
from .integrator import Integrator
from solvers import fgmres, gcrot, matvec_rat, SolverInfo


class Ros2(Integrator):
    Q_flat: numpy.ndarray
    A: MatvecOpRat
    b: numpy.ndarray

    def __init__(self, param: Configuration, rhs_handle: Callable, rhs_handle_complex=None, **kwargs) -> None:
        super().__init__(param, **kwargs)
        self.rhs_handle = rhs_handle
        self.rhs_handle_complex = rhs_handle_complex
        self.tol = param.tolerance
        self.gmres_restart = param.gmres_restart
        self.linear_solver = param.linear_solver

    def __prestep__(self, Q: numpy.ndarray, dt: float) -> None:
        xp = self.device.xp

        rhs = self.rhs_handle(Q)
        self.Q_flat = xp.ravel(Q)
        if self.rhs_handle_complex is not None:
            self.A  = MatvecOpRat(dt, Q, rhs, self.rhs_handle_complex, self.device)
        else:
            self.A = MatvecOpRat(dt, Q, rhs, self.rhs_handle, self.device)
        self.b = self.A(self.Q_flat) + xp.ravel(rhs) * dt

    def __step__(self, Q: numpy.ndarray, dt: float):
        xp = self.device.xp

        maxiter = 20000 // self.gmres_restart
        if self.preconditioner is not None:
            maxiter = 400 // self.gmres_restart

        if self.linear_solver == "fgmres":
            t0 = time()
            Qnew, norm_r, norm_b, num_iter, flag, residuals = fgmres(
                self.A,
                self.b,
                x0=self.Q_flat,
                tol=self.tol,
                restart=self.gmres_restart,
                maxiter=maxiter,
                preconditioner=self.preconditioner,
                verbose=self.verbose_solver,
                device=self.device,
            )
            t1 = time()

            self.solver_info = SolverInfo(flag, t1 - t0, num_iter, residuals)

            if MPI.COMM_WORLD.rank == 0:
                result_type = "convergence" if flag == 0 else "stagnation/interruption"
                print(
                    f"FGMRES {result_type} at iteration {num_iter} in {t1 - t0:4.3f} s to a solution with"
                    f" relative residual {norm_r/norm_b : .2e}"
                )
        else:
            t0 = time()
            Qnew, local_error, num_iter, flag, residuals = gcrot(self.A, self.b, x0=self.Q_flat, tol=self.tol)
            t1 = time()
            local_error = xp.linalg.norm(self.b - self.A(Qnew)) / xp.linalg.norm(self.b)

            if flag == 0:
                print(
                    f"GCROT converged at iteration {num_iter} in {t1 - t0:4.1f} s to a solution with"
                    f" relative residual norm {local_error : .2e}"
                )
            else:
                print(
                    f"GCROT stagnation/interruption at iteration {num_iter} in {t1 - t0:4.1f} s, returning a solution with"
                    f" relative local error {local_error: .2e}"
                )

        self.failure_flag = flag

        return xp.reshape(Qnew, Q.shape)
