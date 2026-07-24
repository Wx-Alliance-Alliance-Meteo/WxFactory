from collections import deque
import math
from typing import Callable

import numpy
from numpy.typing import NDArray

from ..common.configuration import Configuration
from ..solvers import (
    kiops,
    matvec_fun,
    MatvecOpBasic,
    pmex,
    exode,
)

from .integrator import Integrator, SolverInfo

_COEFF_TABLES = {
    2: [[]],
    3: [[2 / 3]],
    4: [[-3 / 10, 3 / 40], [32 / 5, -11 / 10]],
    5: [[-4 / 5, 2 / 5, -4 / 45], [12, -9 / 2, 8 / 9], [3, 0, -1 / 3]],
    6: [
        [-49 / 60, 351 / 560, -359 / 1260, 367 / 6720],
        [92 / 7, -99 / 14, 176 / 63, -1 / 2],
        [485 / 21, -151 / 14, 23 / 9, -31 / 168],
    ],
}


class Epi(Integrator):
    def __init__(
        self,
        param: Configuration,
        order: int,
        rhs: Callable,
        jac: Callable = None,
        init_method=None,
        init_substeps: int = 1,
        *,
        device=None,
    ):
        super().__init__(param, device=device)
        self.rhs = rhs
        self.jac = jac
        self.tol = param.tolerance
        self.krylov_size = 1
        self.krylov_mmax = param.krylov_mmax
        self.jacobian_method = param.jacobian_method
        self.exponential_solver = param.exponential_solver
        self.exode_method = param.exode_method
        self.exode_controller = param.exode_controller

        if order not in _COEFF_TABLES:
            raise ValueError(f"Unsupported order {order} for EPI method. Supported orders: {sorted(_COEFF_TABLES)}")
        self.A = self.device.xp.array(_COEFF_TABLES[order])

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
            self.init_method = Epi(param, 2, rhs, device=device)

        self.init_substeps = init_substeps

    def __step__(self, Q: NDArray, dt: float):

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

        if self.jac is not None:
            matvec_handle = lambda v: self.jac(v, Q, dt)
        else:
            matvec_handle = MatvecOpBasic(dt, Q, self.rhs, self.param)

        vec = self.device.xp.zeros((self.max_phi + 1, math.prod(rhs.shape)), dtype=Q.dtype)
        vec[1, :] = rhs.flatten()
        for i in range(self.n_prev):
            if self.jac is not None:
                J_deltaQ = self.jac(self.previous_Q[i] - Q, Q, 1.0)
            else:
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
                mmax=self.krylov_mmax,
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
                device=self.device,
            )

            # comment out for scaling test
            if mpirank == 0:
                print(
                    f"EXODE converged at iteration {stats[0]}, with {stats[1]} rejected steps "
                    f"with local error {stats[3]}"
                )

        # ----- Regular PMEX ------
        elif self.exponential_solver == "pmex":
            phiv, stats = pmex(
                [1.0], matvec_handle, vec, tol=self.tol, mmax=self.krylov_mmax, task1=False, device=self.device
            )

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
                mmax=self.krylov_mmax,
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


def _make_epi_factory(order):
    return lambda cfg, rhs, prec, dev: Epi(cfg, order, rhs.full, init_substeps=10, device=dev)


REGISTRY = {f"epi{o}": _make_epi_factory(o) for o in range(2, 7)}
