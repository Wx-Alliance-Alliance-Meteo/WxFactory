import numpy
from math import sqrt, copysign
from warnings import warn
import logging
from scipy.integrate._ivp.common import validate_max_step, validate_first_step, warn_extraneous
from scipy.integrate._ivp.base import OdeSolver, DenseOutput

import math
import mpi4py.MPI
import linsol

def limiter(u, kappa):
   # Step-size ratio limiter. Applies an arctangent limiter parametrized by KAPPA to U.
   return  1 + kappa * math.atan((u - 1) / kappa)

def norm(x):
   """Compute RMS norm across all PEs."""
   local_sum = x @ x
   global_sum = numpy.array([0.0])
   mpi4py.MPI.COMM_WORLD.Allreduce(numpy.array([local_sum]), global_sum)
   return math.sqrt(global_sum[0] / global_size(x))

def global_size(x):
   global_sum = numpy.array([0])
   mpi4py.MPI.COMM_WORLD.Allreduce(numpy.array([x.size]), global_sum)
   return global_sum

def validate_tol(rtol, atol, y):
    """Validate tolerance values. atol can be scalar or array-like, rtol a
    scalar. The bound values are from RKSuite. These differ from those in
    scipy. Bounds are applied without warning.
    """
    atol = numpy.asarray(atol)
    if atol.ndim > 0 and atol.shape != (y.size, ):
        raise ValueError("`atol` has wrong shape.")
    if numpy.any(atol < 0):
        raise ValueError("`atol` must be positive.")
    if not isinstance(rtol, float):
        raise ValueError("`rtol` must be a float.")
    if rtol < 0:
        raise ValueError("`rtol` must be positive.")

    # atol cannot be exactly zero.
    # For double precision float: sqrt(tiny) ~ 1.5e-154
    tiny = numpy.finfo(y.dtype).tiny
    atol = numpy.maximum(atol, sqrt(tiny))

    # rtol is bounded from both sides.
    # The lower bound is lower than in scipy.
    epsneg = numpy.finfo(y.dtype).epsneg
    rtol = numpy.minimum(numpy.maximum(rtol, 10 * epsneg), 0.01)
    return rtol, atol

def select_initial_step(fun, t0, y0, f0, order, rtol, atol):
   """Empirically select a good initial step."""
   if y0.size == 0:
      return numpy.inf

   scale = atol + numpy.abs(y0) * rtol
   d0 = norm(y0 / scale)
   d1 = norm(f0 / scale)
   if d0 < 1e-5 or d1 < 1e-5:
      h0 = 1e-6
   else:
      h0 = 0.01 * d0 / d1

   y1 = y0 + h0 * f0
   f1 = fun(t0 + h0, y1)
   d2 = norm((f1 - f0) / scale) / h0

   if d1 <= 1e-15 and d2 <= 1e-15:
      h1 = max(1e-6, h0 * 1e-3)
   else:
      h1 = (0.01 / max(d1, d2))**(1 / (order + 1))

   return min(100 * h0, h1)

class RungeKutta(OdeSolver):
    """
    Base class for explicit runge kutta methods.
    """

    # effective number of stages
    n_stages: int = NotImplemented

    # order of the main method
    order: int = NotImplemented

    # order of the secondary embedded method
    error_estimator_order: int = NotImplemented

    # runge kutta coefficient matrix
    A: numpy.ndarray = NotImplemented              # shape: [n_stages, n_stages]

    # output coefficients (weights)
    B: numpy.ndarray = NotImplemented              # shape: [n_stages]

    # time fraction coefficients (nodes)
    C: numpy.ndarray = NotImplemented              # shape: [n_stages]

    # error coefficients (weights Bh - B)
    E: numpy.ndarray = NotImplemented              # shape: [n_stages + 1]

    # Parameters for stepsize control, optional
    sc_params = NotImplemented                     # tuple, or str

    def _init_min_step_parameters(self):
        """Define the parameters h_min_a and h_min_b for the min_step rule:
            min_step = max(h_min_a * abs(t), h_min_b)
        from RKSuite.
        """

        # minimum difference between distinct C-values
        cdiff = 1.
        for c1 in self.C:
            for c2 in self.C:
                diff = abs(c1 - c2)
                if diff:
                    cdiff = min(cdiff, diff)
        if cdiff < 1e-3:
            cdiff = 1e-3
            logging.warning(
                'Some C-values of this Runge Kutta method are nearly the '
                'same but not identical. This limits the minimum stepsize'
                'You may want to check the implementation of this method.')

        # determine min_step parameters
        epsneg = numpy.finfo(self.y.dtype).epsneg
        tiny = numpy.finfo(self.y.dtype).tiny
        h_min_a = 10 * epsneg / cdiff
        h_min_b = sqrt(tiny)
        return h_min_a, h_min_b

    def _init_sc_control(self, sc_params):
        coefs = {
            "deadbeat": (1, 0, 0, 0.9),    # elementary controller (I)
            "PI3040": (0.7, -0.4, 0, 0.8), # PI controller (Gustafsson)
            "PI4020": (0.6, -0.2, 0, 0.8), # PI controller for nonstiff methods
            "H211PI": (1/6, 1/6, 0, 0.8),  # LP filter of PI structure
            "H110": (1/3, 0, 0, 0.8),      # I controller (convolution filter)
            "H211D": (1/2, 1/2, 1/2, 0.8), # LP filter with gain = 1/2
            "H211b": (1/4, 1/4, 1/4, 0.8)  # general purpose LP filter
        }
        if self.sc_params == NotImplemented:
            # use standard controller if not specified otherwise
            sc_params = sc_params or "standard"
        else:
            # use default controller of method if not specified otherwise
            sc_params = sc_params or self.sc_params
        if (isinstance(sc_params, str) and sc_params in coefs):
            kb1, kb2, a, g = coefs[sc_params]
        elif isinstance(sc_params, tuple) and len(sc_params) == 4:
            kb1, kb2, a, g = sc_params
        else:
            raise ValueError('invalid sc_params')

        # set all parameters
        self.minbeta1 = kb1 * self.error_exponent
        self.minbeta2 = kb2 * self.error_exponent
        self.minalpha = -a
        self.safety = g
        self.safety_sc = g ** (kb1 + kb2)
        self.standard_sc = True                                # for first step

    def __init__(self, fun, y0, t0, t_bound, max_step=numpy.inf, rtol=1e-3,
                 atol=1e-6, vectorized=False, first_step=None,
                 sc_params="H211b", verbose=False, **extraneous):
        warn_extraneous(extraneous)
        super(RungeKutta, self).__init__(fun, t0, y0, t_bound, vectorized,
                                         support_complex=True)
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.y)
        self.f = self.fun(self.t, self.y)
        if self.f.dtype != self.y.dtype:
            raise TypeError('dtypes of solution and derivative do not match')
        self.error_exponent = -1 / (self.error_estimator_order + 1)
        self.h_min_a, self.h_min_b = self._init_min_step_parameters()
        self._init_sc_control(sc_params)

        # size of first step:
        if first_step is None:
            b = self.t + self.direction * min(
                abs(self.t_bound - self.t), self.max_step)
            self.h_abs = select_initial_step(self.fun, self.t, self.y, self.f, self.error_estimator_order, self.rtol, self.atol)
        else:
            self.h_abs = validate_first_step(first_step, t0, t_bound)

        self.K = numpy.empty((self.n_stages + 1, self.n), self.y.dtype)
        self.FSAL = 1 if self.E[self.n_stages] else 0
        self.h_previous = None
        self.y_old = None
        self.nb_rejected = 0

    def run(self):
       step=0
       while self.status == 'running':
          message = self.step()
          step += 1
       return self.y

    def _step_impl(self):
        # mostly follows the scipy implementation of scipy's RungeKutta
        t = self.t
        y = self.y

        h_abs, min_step = self._reassess_stepsize(t, y)
        if h_abs is None:
            # linear extrapolation for last step
            return True, None

        # loop until the step is accepted
        step_accepted = False
        step_rejected = False
        while not step_accepted:

            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP
            h = h_abs * self.direction
            t_new = t + h

            # calculate stages needed for output
            self.K[0] = self.f
            for i in range(1, self.n_stages):
                self._rk_stage(h, i)

            # calculate error norm and solution
            y_new, error_norm = self._comp_sol_err(y, h)

            # TODO : option
            delta = 1.2

            # evaluate error
            if error_norm < delta:
                step_accepted = True

                if self.standard_sc:
                    factor = self.safety * error_norm ** self.error_exponent
                    self.standard_sc = False

                else:
                    # use second order SC controller
                    h_ratio = h / self.h_previous
                    factor = self.safety_sc * ( error_norm**self.minbeta1 * self.error_norm_old**self.minbeta2 * h_ratio**self.minalpha)

                if step_rejected:
                    factor = min(1, factor)

                h_abs *= limiter(factor,2)

            else:
                h_abs *= limiter(self.safety * error_norm ** self.error_exponent, 2)
                step_rejected = True
                self.nb_rejected += 1

        if not self.FSAL:
            # evaluate ouput point for interpolation and next step
            self.K[self.n_stages] = self.fun(t + h, y_new)

        # store for next step, interpolation and stepsize control
        self.h_previous = h
        self.y_old = y
        self.h_abs = h_abs
        self.f = self.K[self.n_stages].copy()
        self.error_norm_old = error_norm

        # output
        self.t = t_new
        self.y = y_new

        return True, None

    def _reassess_stepsize(self, t, y):
        # limit step size
        h_abs = self.h_abs
        min_step = max(self.h_min_a * (abs(t) + h_abs), self.h_min_b)
        if h_abs < min_step or h_abs > self.max_step:
            h_abs = min(self.max_step, max(min_step, h_abs))
            self.standard_sc = True

        # handle final integration steps
        d = abs(self.t_bound - t)                     # remaining interval
        if d < 2 * h_abs:
            if d > h_abs:
                # h_abs < d < 2 * h_abs: "look ahead".
                # split d over last two steps. This reduces the chance of a
                # very small last step.
                h_abs = max(0.5 * d, min_step)
                self.standard_sc = True
            elif d >= min_step:
                # d <= h_abs: Don't step over t_bound
                h_abs = d
            else:
                # d < min_step: use linear extrapolation in this rare case
                h = self.t_bound - t
                y_new = y + h * self.f
                self.h_previous = h
                self.y_old = y
                self.t = self.t_bound
                self.y = y_new
                self.f = None                      # signals _dense_output_impl
                logging.warning(
                    'Linear extrapolation was used in the final step.')
                return None, min_step

        return h_abs, min_step

    def _estimate_error(self, K, h):
        # exclude K[-1] if not FSAL. It could contain nan or inf
        _, n = K[:self.n_stages + self.FSAL].shape
        err = numpy.empty(n)
        for i in range(n): # TODO : this is slow
           err[i] = linsol.global_dotprod(K[:self.n_stages + self.FSAL,i], self.E[:self.n_stages + self.FSAL])
        return err * h

    def _estimate_error_norm(self, K, h, scale):
        return norm(self._estimate_error(K, h) / scale)

    def _comp_sol_err(self, y, h):
        """Compute solution and error.
        The calculation of `scale` differs from scipy: The average instead of
        the maximum of abs(y) of the current and previous steps is used.
        """
        _, n = self.K[:self.n_stages].shape
        y_new = y.copy()
        for i in range(n): # TODO : this is slow
           y_new[i] = h * linsol.global_dotprod(self.K[:self.n_stages,i], self.B)
        scale = self.atol + self.rtol * 0.5*(numpy.abs(y) + numpy.abs(y_new))

        if self.FSAL:
            # do FSAL evaluation if needed for error estimate
            self.K[self.n_stages, :] = self.fun(self.t + h, y_new)

        error_norm = self._estimate_error_norm(self.K, h, scale)
        return y_new, error_norm

    def _rk_stage(self, h, i):
        """compute a single RK stage"""
        dy = h * linsol.global_dotprod(self.K[:i, :].T,  self.A[i, :i])
        self.K[i] = self.fun(self.t + self.C[i] * h, self.y + dy)


class RK23(RungeKutta):
    """Explicit Runge-Kutta method of order 3(2)."""
    order = 3
    error_estimator_order = 2
    n_stages = 3
    C = numpy.array([0, 1/2, 3/4])
    A = numpy.array([
        [0, 0, 0],
        [1/2, 0, 0],
        [0, 3/4, 0]
    ])
    B = numpy.array([2/9, 1/3, 4/9])
    E = numpy.array([5/72, -1/12, -1/9, 1/8])

class RK45(RungeKutta):
    """Explicit Runge-Kutta method of order 5(4)."""
    order = 5
    error_estimator_order = 4
    n_stages = 6
    C = numpy.array([0, 1/5, 3/10, 4/5, 8/9, 1])
    A = numpy.array([
        [0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
    ])
    B = numpy.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    E = numpy.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40])
