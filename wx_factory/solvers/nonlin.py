import math
import sys
from time import time

import numpy
import scipy.optimize
import torch

from ..context import Context
from .fgmres import fgmres
from .global_operations import global_inf_norm, global_norm


def _machine_eps(dtype) -> float:
    """Machine epsilon for the working precision, inferred from the array dtype (numpy or torch)."""
    return float(numpy.finfo(numpy.float32).eps) if "32" in str(dtype) else float(numpy.finfo(numpy.float64).eps)


def newton_krylov(
    F,
    x0,
    restart=30,
    maxiter_linear=1,
    preconditioner=None,
    verbose=False,
    maxiter=None,
    f_tol=None,
    f_rtol=None,
    x_tol=None,
    x_rtol=None,
    line_search="armijo",
    context: Context | None = None,
    eta0=1e-3,
    jacobian=None,
    krylov_dtype=None,
):
    """Solve F(x)=0 using Newton iterations and a matrix-free Krylov solve.

    jacobian(state) may return an analytic action at the current iterate.
    krylov_dtype sets basis storage precision independently of the residual.
    """
    if context is None:
        context = Context.get_default()
    t_start = time()
    iteration = 0

    gamma = 0.9
    eta_max = 0.9999
    eta_treshold = 0.1
    eta = eta0

    if f_tol is None:
        f_tol = _machine_eps(x0.dtype) ** (1.0 / 3)
    if f_rtol is None:
        f_rtol = numpy.inf
    if x_tol is None:
        x_tol = numpy.inf
    if x_rtol is None:
        x_rtol = numpy.inf

    f0_norm = None

    func = lambda z: F(z.reshape(x0.shape)).flatten()
    x = x0.flatten()

    dx = torch.full_like(x, float("inf"))
    Fx = func(x)
    Fx_norm = global_norm(Fx, context=context)
    if not math.isfinite(float(Fx_norm)):
        raise ValueError("Non-finite initial Newton residual")

    jacobian_factory = None if jacobian is None else lambda state: jacobian(state.reshape(x0.shape))
    linearization = KrylovJacobian(
        x.clone(),
        Fx,
        func,
        restart=restart,
        maxiter_linear=maxiter_linear,
        preconditioner=preconditioner,
        context=context,
        jacobian_factory=jacobian_factory,
        krylov_dtype=krylov_dtype,
    )

    if maxiter is None:
        maxiter = 100 * (x.shape[0] + 1)

    if line_search not in (None, "armijo", "wolfe"):
        raise ValueError("Invalid line search")

    residuals = []
    terminated = False

    for n in range(maxiter):
        iteration += 1
        f_norm = global_inf_norm(Fx, comm=context.comm)
        x_norm = global_inf_norm(x, comm=context.comm)

        dx_norm = global_inf_norm(dx, comm=context.comm)

        residuals.append((f_norm, time() - t_start, 0.0))

        if f0_norm is None:
            f0_norm = f_norm

        terminated = (
            f_norm == 0
            or (f_norm <= f_tol and f_norm / f_rtol <= f0_norm)
            and (dx_norm <= x_tol and dx_norm / x_rtol <= x_norm)
        )

        if terminated:
            break

        tol = min(eta, eta * float(Fx_norm))
        dx = -linearization.solve(Fx, tol=tol)

        if line_search:
            s, x, Fx, Fx_norm_new = _nonlin_line_search(func, x, Fx, dx, context, line_search)
        else:
            s = 1.0
            x = x + dx
            Fx = func(x)
            Fx_norm_new = global_norm(Fx, context=context)

        if not math.isfinite(float(Fx_norm_new)):
            raise ValueError("Non-finite Newton residual")
        linearization.update(x.clone(), Fx)

        eta_A = gamma * float(Fx_norm_new) ** 2 / float(Fx_norm) ** 2
        if gamma * eta**2 < eta_treshold:
            eta = min(eta_max, eta_A)
        else:
            eta = min(eta_max, max(eta_A, gamma * eta**2))

        Fx_norm = Fx_norm_new

        if verbose:
            sys.stdout.write(f"{n:3d}:  |F(x)| = {global_inf_norm(Fx, comm=context.comm):.3e}; step {s}\n")
            sys.stdout.flush()
    else:
        raise RuntimeError("Newton–Krylov did not converge within the maximum number of iterations")

    if terminated:
        print(f"A solution was found after {iteration - 1} steps of the JFNK method.")

    return x.reshape(x0.shape), iteration - 1, residuals


def _nonlin_line_search(func, x, Fx, dx, context: Context, search_type="armijo", rdiff=1e-8, smin=1e-2):
    tmp_s = [0]
    tmp_Fx = [Fx]
    tmp_phi = [float(global_norm(Fx, context=context)) ** 2]
    s_norm = float(global_norm(x, context=context)) / float(global_norm(dx, context=context))

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]
        xt = x + s * dx
        v = func(xt)
        p = float(global_norm(v, context=context)) ** 2
        if store:
            tmp_s[0] = s
            tmp_phi[0] = p
            tmp_Fx[0] = v
        return p

    def derphi(s):
        ds = (abs(s) + s_norm + 1) * rdiff
        return (phi(s + ds, store=False) - phi(s)) / ds

    # linesearch has moved in Scipy 1.14.0
    from packaging import version

    if version.parse(scipy.__version__) >= version.parse("1.14.0"):
        linesearch_module = scipy.optimize._linesearch
    else:
        linesearch_module = scipy.optimize.linesearch

    if search_type == "wolfe":
        s, _, _ = linesearch_module.scalar_search_wolfe1(phi, derphi, tmp_phi[0], xtol=1e-2, amin=smin)
    elif search_type == "armijo":
        s, _ = linesearch_module.scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0], amin=smin)

    if s is None:
        # No suitable step length found. Take the full Newton step, and hope for the best.
        s = 1.0

    x = x + s * dx
    if s == tmp_s[0]:
        Fx = tmp_Fx[0]
    else:
        Fx = func(x)
    Fx_norm = global_norm(Fx, context=context)

    return s, x, Fx, Fx_norm


class KrylovJacobian:
    def __init__(
        self,
        x,
        f,
        func,
        restart,
        maxiter_linear,
        preconditioner,
        context: Context,
        jacobian_factory=None,
        krylov_dtype=None,
    ):
        self.func = func
        self.jacobian_factory = jacobian_factory
        self.krylov_dtype = krylov_dtype
        self.action = None
        self.shape = (f.shape[0], x.shape[0])
        self.dtype = f.dtype
        self.context = context

        self.restart = restart
        self.maxiter_linear = maxiter_linear
        self.preconditioner = preconditioner

        self.x0 = x
        self.f0 = f
        self.rdiff = math.sqrt(_machine_eps(x.dtype))
        self._update_diff_step()

    def _update_diff_step(self):
        if self.jacobian_factory is not None:
            return
        mx = global_inf_norm(self.x0, comm=self.context.comm)
        mf = global_inf_norm(self.f0, comm=self.context.comm)
        self.omega = self.rdiff * max(1, mx) / max(1, mf)

    def matvec(self, v):
        if self.jacobian_factory is not None:
            if self.action is None:
                self.action = self.jacobian_factory(self.x0)
            return self.action(v).flatten()
        nv = global_norm(v, context=self.context)
        if float(nv) == 0:
            return 0 * v
        sc = self.omega / nv
        return (self.func(self.x0 + sc * v) - self.f0) / sc

    def __call__(self, v):
        return self.matvec(v)

    def solve(self, rhs, tol=0):
        sol, *_ = fgmres(
            self,
            rhs,
            tol=tol,
            restart=self.restart,
            maxiter=self.maxiter_linear,
            preconditioner=self.preconditioner,
            context=self.context,
            krylov_dtype=self.krylov_dtype,
        )
        return sol

    def update(self, x, f):
        self.action = None
        self.x0 = x
        self.f0 = f
        self._update_diff_step()
