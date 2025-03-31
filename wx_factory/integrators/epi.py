from collections import deque
import math
from typing import Callable

from mpi4py import MPI
import numpy

from common.configuration import Configuration
from solvers import (
    kiops,
    matvec_fun,
    pmex,
    pmex_1s,
    pmex_ne1s,
    cwy_ne,
    cwy_1s,
    cwy_ne1s,
    icwy_1s,
    icwy_ne,
    icwy_ne1s,
    icwy_neiop,
    dcgs2,
    kiops_nest,
    exode,
)

from .integrator import Integrator, SolverInfo


class Epi(Integrator):
    def __init__(
        self, param: Configuration, order: int, rhs: Callable, init_method=None, init_substeps: int = 1, **kwargs
    ):
        super().__init__(param, preconditioner=None, **kwargs)
        self.rhs = rhs
        self.tol = param.tolerance
        self.krylov_size = 1
        self.jacobian_method = param.jacobian_method
        self.exponential_solver = param.exponential_solver
        self.case_number = param.case_number
        self.int = param.time_integrator
        self.exode_method = param.exode_method
        self.exode_controller = param.exode_controller

        if order == 2:
            self.A = self.device.xp.array([[]])
        elif order == 3:
            self.A = self.device.xp.array([[2 / 3]])
        elif order == 4:
            self.A = self.device.xp.array([[-3 / 10, 3 / 40], [32 / 5, -11 / 10]])
        elif order == 5:
            self.A = self.device.xp.array([[-4 / 5, 2 / 5, -4 / 45], [12, -9 / 2, 8 / 9], [3, 0, -1 / 3]])
        elif order == 6:
            self.A = self.device.xp.array(
                [
                    [-49 / 60, 351 / 560, -359 / 1260, 367 / 6720],
                    [92 / 7, -99 / 14, 176 / 63, -1 / 2],
                    [485 / 21, -151 / 14, 23 / 9, -31 / 168],
                ]
            )
        else:
            raise ValueError(f"Unsupported order {order} for EPI method")

        k, self.n_prev = self.A.shape
        # Limit max phi to 1 for EPI 2
        if order == 2:
            k -= 1
        self.max_phi = k + 1
        self.previous_Q = deque()
        self.previous_rhs = deque()
        self.dt = 0.0

        if init_method or self.n_prev == 0:
            self.init_method = init_method
        else:
            self.init_method = Epi(param, 2, rhs, **kwargs)

        self.init_substeps = init_substeps

    def __step__(self, Q: numpy.ndarray, dt: float):

        # If dt changes, discard saved value and redo initialization
        mpirank = self.device.comm.rank
        if self.dt and abs(self.dt - dt) > 1e-10:
            self.previous_Q = deque()
            self.previous_rhs = deque()
        self.dt = dt

        # Initialize saved values using init_step method
        if len(self.previous_Q) < self.n_prev:
            self.previous_Q.appendleft(Q)
            self.previous_rhs.appendleft(self.rhs(Q))
            dt /= self.init_substeps
            for i in range(self.init_substeps):
                Q = self.init_method.step(Q, dt)

            return Q

        # Regular EPI step
        rhs = self.rhs(Q)

        def matvec_handle(v):
            return matvec_fun(v, dt, Q, rhs, self.rhs, self.jacobian_method)

        vec = self.device.xp.zeros((self.max_phi + 1, rhs.size))
        vec[1, :] = rhs.flatten()
        for i in range(self.n_prev):
            J_deltaQ = matvec_fun(self.previous_Q[i] - Q, 1.0, Q, rhs, self.rhs, self.jacobian_method)

            # R(y_{n-i})
            r = (self.previous_rhs[i] - rhs) - self.device.xp.reshape(J_deltaQ, Q.shape)

            for k, alpha in enumerate(self.A[:, i], start=2):
                # v_k = Sum_{i=1}^{n_prev} A_{k,i} R(y_{n-i})
                vec[k, :] += alpha * r.flatten()

        # ----pmex with norm estimate-----
        if self.exponential_solver == "pmex_ne":
            phiv, stats = pmex(
                [1.0],
                matvec_handle,
                vec,
                tol=self.tol,
                m_init=self.krylov_size,
                mmin=16,
                mmax=64,
                task1=False,
                device=self.device,
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"PMEX NE converged at iteration {stats[2]} (using {stats[0]} internal substeps "
                    f" and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----pmex with 1-sync-----
        elif self.exponential_solver == "pmex_1s":
            phiv, stats = pmex_1s(
                [1.0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"PMEX 1s converged at iteration {stats[2]} (using {stats[0]} internal substeps "
                    f"and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----pmex with norm estimate+1s-----
        elif self.exponential_solver == "pmex_ne1s":
            phiv, stats = pmex_ne1s(
                [1.0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"PMEX NE+1s converged at iteration {stats[2]} (using {stats[0]} internal substeps "
                    f"and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----- icwy norm estimate + 1sync-----
        elif self.exponential_solver == "icwy_ne1s":
            phiv, stats = icwy_ne1s(
                [1.0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"ICWY NE+1S converged at iteration {stats[2]} (using {stats[0]} internal substeps "
                    f"and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----- icwy norm estimate -----
        elif self.exponential_solver == "icwy_ne":
            phiv, stats = icwy_ne(
                [1.0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"ICWY NE converged at iteration {stats[2]} (using {stats[0]} internal substeps "
                    f"and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----- icwy 1-sync-----
        elif self.exponential_solver == "icwy_1s":
            phiv, stats = icwy_1s(
                [1.0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"ICWY 1S converged at iteration {stats[2]} (using {stats[0]} internal substeps "
                    f"and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----- icwy iop+norm estimate-----
        elif self.exponential_solver == "icwy_neiop":
            phiv, stats = icwy_neiop(
                [1.0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"ICWY NE+IOP converged at iteration {stats[2]} (using {stats[0]} internal substeps "
                    f"and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----- cwy norm estimate + 1sync-----
        elif self.exponential_solver == "cwy_ne1s":
            phiv, stats = cwy_ne1s(
                [1.0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"CWY NE+1S converged at iteration {stats[2]} (using {stats[0]} internal substeps "
                    f"and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----- cwy norm estimate -----
        elif self.exponential_solver == "cwy_ne":
            phiv, stats = cwy_ne(
                [1.0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"CWY NE converged at iteration {stats[2]} (using {stats[0]} internal substeps "
                    f"and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----- cwy 1-sync-----
        elif self.exponential_solver == "cwy_1s":
            phiv, stats = cwy_1s(
                [1.0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"CWY 1S converged at iteration {stats[2]} (using {stats[0]} internal substeps "
                    f"and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----- dcgs2 -----
        elif self.exponential_solver == "dcgs2":
            phiv, stats = dcgs2(
                [1.0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"DCGS2 converged at iteration {stats[2]} (using {stats[0]} internal substeps "
                    f"and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----- kiops + norm estimate-----
        elif self.exponential_solver == "kiops_ne":
            phiv, stats = kiops_nest(
                [1.0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"KIOPS NE converged at iteration {stats[2]} (using {stats[0]} internal substeps "
                    f"and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )
        # ----- EXODE ------
        elif self.exponential_solver == "exode":
            phiv, stats = exode(
                1.0,
                matvec_handle,
                vec,
                method=self.exode_method,
                controller=self.exode_controller,
                atol=self.tol,
                task1=False,
                verbose=False,
            )

            # comment out for scaling test
            if mpirank == 0:
                print(
                    f"EXODE converged at iteration {stats[0]}, with {stats[1]} rejected steps "
                    f"with local error {stats[3]}"
                )

        # ----- Regular PMEX ------
        elif self.exponential_solver == "pmex":
            phiv, stats = pmex([1.0], matvec_handle, vec, tol=self.tol, mmax=64, task1=False, device=self.device)

            if mpirank == 0:
                print(
                    f"PMEX converged at iteration {stats[2]} (using {stats[0]} internal substeps and"
                    f" {stats[1]} rejected expm) to a solution with local error {stats[4]:.2e}",
                    flush=True,
                )

        # ----- Regular KIOPS ------
        elif self.exponential_solver == "kiops":
            phiv, stats = kiops(
                [1],
                matvec_handle,
                vec,
                tol=self.tol,
                m_init=self.krylov_size,
                mmin=16,
                mmax=64,
                task1=False,
                device=self.device,
            )

            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"KIOPS converged at iteration {stats[2]} (using {stats[0]} internal substeps and"
                    f" {stats[1]} rejected expm) to a solution with local error {stats[4]:.2e}",
                    flush=True,
                )

        else:
            raise ValueError(f"Unrecognized exponential solver {self.exponential_solver}")

        self.solver_info = SolverInfo(total_num_it=stats[2])

        # Save values for the next timestep
        if self.n_prev > 0:
            self.previous_Q.pop()
            self.previous_Q.appendleft(Q)
            self.previous_rhs.pop()
            self.previous_rhs.appendleft(rhs)

        # Update solution
        return Q + self.device.xp.reshape(phiv, Q.shape) * dt
