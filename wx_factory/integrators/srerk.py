import math
from collections.abc import Callable
from itertools import combinations

import numpy
import torch

from ..common.configuration import Configuration
from ..jacobian import fd_jacobian_matvec
from ..solvers import (
    ExponentialSolverRequest,
    resolve_exponential_solver,
)
from .integrator import Integrator


def alpha_coeff(c):
    """Compute the coefficients for stiffness resilient exponential methods based on node values c."""
    m = len(c)
    alpha = numpy.zeros((m, m))
    for i in range(m):
        c_no_i = [cc for (j, cc) in enumerate(c) if j != i]
        denom = c[i] ** 2 * math.prod([c[i] - cl for cl in c_no_i])
        for k in range(m):
            sp = sum([math.prod(v) for v in combinations(c_no_i, m - k - 1)])
            alpha[k, i] = (-1) ** (m - k + 1) * math.factorial(k + 2) * sp / denom
    return alpha


def opt_nodes(order: int):
    if order < 3:
        raise ValueError("Order should be at least 3")

    coeff = lambda p, q: (
        (-1) ** (p + q)
        * math.factorial(p + q + 2)
        / (math.factorial(q) * math.factorial(q + 2) * math.factorial(p - q))
    )

    c = []
    for o in list(range(2, order - 2, 2)) + [order - 2]:
        p = numpy.polynomial.Polynomial([coeff(o, q) for q in range(o + 1)])
        c.append(p.roots())

    c.append(numpy.ones(1))
    return c


class Srerk(Integrator):
    """Stiffness resilient exponential Runge-Kutta methods"""

    def __init__(
        self,
        param: Configuration,
        order: int,
        rhs: Callable,
        jac: Callable | None = None,
        nodes: list | None = None,
        *,
        context=None,
    ):
        """
        If the nodes are NOT specified, return the SRERK method of the specified order with min error terms
        If the nodes are specified, return the SRERK method with these nodes and ignore the 'order' parameter
        """

        super().__init__(param, context=context)
        self.rhs = rhs
        self.jac = jac
        self.tol = param.tolerance
        self.krylov_size = 1
        self.krylov_mmax = param.krylov_mmax
        self.exponential_solver = param.exponential_solver
        self.solve_exponential = resolve_exponential_solver(self.exponential_solver)

        if nodes:
            self.c = nodes
        else:
            self.c = opt_nodes(order)
        self.n_proj = len(self.c)

        self.alpha = []
        for i in range(self.n_proj - 1):
            self.alpha.append(
                torch.as_tensor(alpha_coeff(self.c[i]), dtype=self.context.real_dtype, device=self.context.torch_device)
            )

    def _solve_projection(self, tau_out, matvec_handle, vec):
        result = self.solve_exponential(
            ExponentialSolverRequest(
                tau_out,
                matvec_handle,
                vec,
                self.tol,
                self.krylov_mmax,
                self.context,
                krylov_minit=self.krylov_size,
                krylov_mmin=16,
            )
        )
        if result.final_krylov_size is not None:
            self.krylov_size = math.floor(0.7 * result.final_krylov_size + 0.3 * self.krylov_size)
        return result.value

    def __step__(self, Q: torch.Tensor, dt: float):
        rhs = self.rhs(Q)
        if self.jac is not None:
            matvec_handle = lambda v: self.jac(v, Q, dt)
            jacobian = lambda v: matvec_handle(v) / dt
        elif getattr(self.param, "jacobian", None) == "analytic" and hasattr(self.rhs, "linearize"):
            jacobian = self.rhs.linearize(Q)
            matvec_handle = lambda v: dt * jacobian(v)
        else:
            matvec_handle = lambda v: fd_jacobian_matvec(v, dt, Q, rhs, self.rhs)
            jacobian = lambda v: matvec_handle(v) / dt

        vec = Q.new_zeros((2, rhs.numel()))
        vec[1, :] = rhs.flatten()
        z = self._solve_projection(self.c[0], matvec_handle, vec)

        for i_proj in range(1, self.n_proj):
            stages = z
            for i in range(stages.shape[0]):
                stages[i] = Q.flatten() + dt * stages[i]
            remainders = torch.empty_like(stages)
            for i in range(stages.shape[0]):
                stage = stages[i].reshape(Q.shape)
                remainders[i] = (self.rhs(stage) - rhs).flatten() - jacobian(stage - Q)

            vec = Q.new_zeros((stages.shape[0] + 3, rhs.numel()))
            vec[1, :] = rhs.flatten()
            vec[3:, :] = self.alpha[i_proj - 1].to(Q) @ remainders
            z = self._solve_projection(self.c[i_proj], matvec_handle, vec)

        return Q + dt * z.reshape(Q.shape)


def _make_srerk_factory(order):
    return lambda cfg, rhs, prec, ctx: Srerk(cfg, order, rhs.full, context=ctx)


REGISTRY = {f"srerk{o}": _make_srerk_factory(o) for o in range(3, 10)}
