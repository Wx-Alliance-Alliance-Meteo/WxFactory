import math
from collections import deque

import numpy
import torch

from ..common.configuration import Configuration
from ..jacobian import fd_jacobian_matvec
from ..solvers import ExponentialSolverRequest, resolve_exponential_solver
from .epi import Epi
from .integrator import Integrator
from .srerk import alpha_coeff


class EpiStiff(Integrator):
    def __init__(
        self, param: Configuration, order: int, rhs, init_method=None, init_substeps: int = 1, *, context=None
    ):
        super().__init__(param, context=context)
        self.rhs = rhs
        self.tol = param.tolerance
        self.krylov_size = 1
        self.krylov_mmax = param.krylov_mmax
        self.exponential_solver = param.exponential_solver
        self.solve_exponential = resolve_exponential_solver(self.exponential_solver)
        self.exode_method = param.exode_method
        self.exode_controller = param.exode_controller

        if order < 2:
            raise ValueError("Unsupported order for EPI method")
        self.A = alpha_coeff([-i for i in range(-1, 1 - order, -1)])

        _, self.n_prev = self.A.shape

        self.max_phi = order if order > 2 else 1
        self.previous_Q = deque()
        self.previous_rhs = deque()
        self.dt = 0.0

        if init_method or self.n_prev == 0:
            self.init_method = init_method
        else:
            self.init_method = Epi(param, 2, rhs, context=self.context)

        self.init_substeps = init_substeps

    def __step__(self, Q: numpy.ndarray, dt: float):
        # If dt changes, discard saved value and redo initialization
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
            return fd_jacobian_matvec(v, dt, Q, rhs, self.rhs)

        vec = torch.zeros((self.max_phi + 1, math.prod(rhs.shape)), dtype=Q.dtype)
        vec[1, :] = rhs.flatten()
        for i in range(self.n_prev):
            J_deltaQ = fd_jacobian_matvec(self.previous_Q[i] - Q, 1.0, Q, rhs, self.rhs)

            # R(y_{n-i})
            r = (self.previous_rhs[i] - rhs) - J_deltaQ.reshape(Q.shape)

            for k, alpha in enumerate(self.A[:, i]):
                # v_k = Sum_{i=1}^{n_prev} A_{k,i} R(y_{n-i})
                vec[k + 3, :] += alpha * r.flatten()

        use_recycled_size = self.exponential_solver in ("pmex_ne", "kiops")
        result = self.solve_exponential(
            ExponentialSolverRequest(
                [1.0],
                matvec_handle,
                vec,
                self.tol,
                self.krylov_mmax,
                self.context,
                krylov_minit=self.krylov_size if use_recycled_size else None,
                krylov_mmin=16 if use_recycled_size else None,
                exode_method=self.exode_method,
                exode_controller=self.exode_controller,
            )
        )
        phiv = result.value
        if use_recycled_size and result.final_krylov_size is not None:
            self.krylov_size = math.floor(0.7 * result.final_krylov_size + 0.3 * self.krylov_size)

        # Save values for the next timestep
        if self.n_prev > 0:
            self.previous_Q.pop()
            self.previous_Q.appendleft(Q)
            self.previous_rhs.pop()
            self.previous_rhs.appendleft(rhs)

        # Update solution
        return Q + numpy.reshape(phiv, Q.shape) * dt


def _make_epi_stiff_factory(order):
    return lambda cfg, rhs, prec, context: EpiStiff(cfg, order, rhs.full, init_substeps=10, context=context)


REGISTRY = {f"epi_stiff{o}": _make_epi_stiff_factory(o) for o in range(2, 10)}
