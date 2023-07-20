import numpy
from math import sqrt, copysign
from warnings import warn
import logging
from scipy.integrate._ivp.base import DenseOutput
from solvers.global_operations import global_norm

MIN_FACTOR = 0.2
MAX_FACTOR = 4.0
MAX_FACTOR0 = 10
NFS = numpy.array(0)                                         # failed step counter

class RungeKutta:
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

   # error coefficients (weights Bh - B); for non-FSAL methods E[-1] == 0.
   E: numpy.ndarray = NotImplemented              # shape: [n_stages + 1]

   # dense output interpolation coefficients, optional
   P: numpy.ndarray = NotImplemented              # shape: [n_stages + 1,
   #                                                     order_polynomial]

   # Parameters for stepsize control, optional
   sc_params = "standard"                      # tuple, or str

   max_factor = MAX_FACTOR0                    # initially
   min_factor = MIN_FACTOR

   def __init__(self, fun, t0, y0, t_bound, max_step=numpy.inf, rtol=1e-3,
                atol=1e-6, vectorized=False, first_step=None,
                sc_params=None, **extraneous):

      self.t_old = None
      self.t = t0
      self._fun, self.y = fun, y0
      self.t_bound = t_bound
      self.vectorized = vectorized

      if vectorized:
         def fun_single(t, y):
            return self._fun(t, y[:, None]).ravel()
         fun_vectorized = self._fun
      else:
         fun_single = self._fun

         def fun_vectorized(t, y):
            f = numpy.empty_like(y)
            for i, yi in enumerate(y.T):
               f[:, i] = self._fun(t, yi)
            return f

      def fun(t, y):
         self.nfev += 1
         return self.fun_single(t, y)

      self.fun = fun
      self.fun_single = fun_single
      self.fun_vectorized = fun_vectorized

      self.direction = numpy.sign(t_bound - t0) if t_bound != t0 else 1
      self.n = self.y.size
      self.status = 'running'

      self.nfev = 0
      self.njev = 0
      self.nlu = 0

      self.max_step = max_step
      self.rtol, self.atol = rtol, atol
      self.f = self.fun(self.t, self.y)
      if self.f.dtype != self.y.dtype:
         raise TypeError('dtypes of solution and derivative do not match')
      self.error_exponent = -1 / (self.error_estimator_order + 1)
      self.h_min_a, self.h_min_b = self._init_min_step_parameters()
      self.tiny_err = self.h_min_b
      self._init_sc_control(sc_params)

      # size of first step:
      if first_step is None:
         raise RuntimeError("`first_step` not defined.")
      else:
         if first_step <= 0:
            raise ValueError("`first_step` must be positive.")
         if first_step > numpy.abs(t_bound - t0):
            raise ValueError("`first_step` exceeds bounds.")
         self.h_abs = first_step

      self.K = numpy.empty((self.n_stages + 1, self.n), self.y.dtype)
      self.FSAL = 1 if self.E[self.n_stages] else 0
      self.h_previous = None
      self.y_old = None
      NFS[()] = 0                                # global failed step counter

   def step(self):
      """Perform one integration step.

      Returns
      -------
      message : string or None
          Report from the solver. Typically a reason for a failure if
          `self.status` is 'failed' after the step was taken or None
          otherwise.
      """
      if self.status != 'running':
         raise RuntimeError("Attempt to step on a failed or finished "
                             "solver.")

      if self.n == 0 or self.t == self.t_bound:
         # Handle corner cases of empty solver or no integration.
         self.t_old = self.t
         self.t = self.t_bound
         message = None
         self.status = 'finished'
      else:
         t = self.t
         success, message = self._step_impl()

         if not success:
            self.status = 'failed'
         else:
            self.t_old = t
            if self.direction * (self.t - self.t_bound) >= 0:
               self.status = 'finished'

      return message

   def dense_output(self):
      """Compute a local interpolant over the last successful step.

      Returns
      -------
      sol : `DenseOutput`
          Local interpolant over the last successful step.
      """
      if self.t_old is None:
         raise RuntimeError("Dense output is available after a successful "
                            "step was made.")

      if self.n == 0 or self.t == self.t_old:
         # Handle corner cases of empty solver and no integration.
         return ConstantDenseOutput(self.t_old, self.t, self.y)
      else:
         return self._dense_output_impl()

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
      coefs = {"G": (0.7, -0.4, 0, 0.9),
               "S": (0.6, -0.2, 0, 0.9),
               "W": (2, -1, -1, 0.8),
               "standard": (1, 0, 0, 0.9)}
      # use default controller of method if not specified otherwise
      sc_params = sc_params or self.sc_params
      if (isinstance(sc_params, str) and sc_params in coefs):
         kb1, kb2, a, g = coefs[sc_params]
      elif isinstance(sc_params, tuple) and len(sc_params) == 4:
         kb1, kb2, a, g = sc_params
      else:
         raise ValueError('sc_params should be a tuple of length 3 or one '
                          'of the strings "G", "S", "W" or "standard"')
      # set all parameters
      self.minbeta1 = kb1 * self.error_exponent
      self.minbeta2 = kb2 * self.error_exponent
      self.minalpha = -a
      self.safety = g
      self.safety_sc = g ** (kb1 + kb2)
      self.standard_sc = True                                # for first step

   def _step_impl(self):
      # mostly follows the scipy implementation of scipy's RungeKutta
      t = self.t
      y = self.y

      h_abs, min_step = self._reassess_stepsize(t, y)

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

         # evaluate error
         if error_norm < 1:
            step_accepted = True

            if error_norm < self.tiny_err:
               factor = self.max_factor
               self.standard_sc = True

            elif self.standard_sc:
               factor = self.safety * error_norm ** self.error_exponent
               self.standard_sc = False

            else:
               # use second order SC controller
               h_ratio = h / self.h_previous
               factor = self.safety_sc * (
                   error_norm ** self.minbeta1 *
                   self.error_norm_old ** self.minbeta2 *
                   h_ratio ** self.minalpha)
               factor = min(self.max_factor, max(self.min_factor, factor))

            if step_rejected:
               factor = min(1, factor)

            h_abs *= factor

            if factor < MAX_FACTOR:
               # reduce max_factor when on scale.
               self.max_factor = MAX_FACTOR

         else:
            step_rejected = True
            h_abs *= max(self.min_factor,
                          self.safety * error_norm ** self.error_exponent)

            NFS[()] += 1

            if numpy.isnan(error_norm) or numpy.isinf(error_norm):
               return False, "Overflow or underflow encountered."

      if not self.FSAL:
         # evaluate ouput point for interpolation and next step
         self.K[self.n_stages] = self.fun(t + h, y_new)

      # store for next step, interpolation and stepsize control
      self.h_previous = h
      self.y_old = y
      self.h_abs = h_abs
      self.f_old = self.f
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
         else:
            # d <= h_abs: Don't step over t_bound
            h_abs = d

      return h_abs, min_step

   def _estimate_error(self, K, h):
      # exclude K[-1] if not FSAL. It could contain nan or inf
      return h * (K[:self.n_stages + self.FSAL].T @
                  self.E[:self.n_stages + self.FSAL])

   def _estimate_error_global_norm(self, K, h, scale):
      return global_norm(self._estimate_error(K, h) / scale)

   def _comp_sol_err(self, y, h):
      """Compute solution and error.
      The calculation of `scale` differs from scipy: The average instead of
      the maximum of abs(y) of the current and previous steps is used.
      """
      y_new = y + h * (self.K[:self.n_stages].T @ self.B)
      scale = self.atol + self.rtol * numpy.maximum(numpy.abs(y), numpy.abs(y_new))

      if self.FSAL:
         # do FSAL evaluation if needed for error estimate
         self.K[self.n_stages, :] = self.fun(self.t + h, y_new)

      error_norm = self._estimate_error_global_norm(self.K, h, scale)
      return y_new, error_norm

   def _rk_stage(self, h, i):
      """compute a single RK stage"""
      dy = h * (self.K[:i, :].T @ self.A[i, :i])
      self.K[i] = self.fun(self.t + self.C[i] * h, self.y + dy)

   def _dense_output_impl(self):
      """return denseOutput, detect if step was extrapolated linearly"""

      if isinstance(self.P, numpy.ndarray):
         # normal output
         Q = self.K.T @ self.P
         return HornerDenseOutput(self.t_old, self.t, self.y_old, Q)

      # if no interpolant is implemented
      return CubicDenseOutput(self.t_old, self.t, self.y_old, self.y,
                                self.f_old, self.f)

class HornerDenseOutput(DenseOutput):
   """use Horner's rule for the evaluation of the dense output polynomials.
   """
   def __init__(self, t_old, t, y_old, Q):
      super(HornerDenseOutput, self).__init__(t_old, t)
      self.h = t - t_old
      self.Q = Q * self.h
      self.y_old = y_old

   def _call_impl(self, t):
      # scaled time
      x = (t - self.t_old) / self.h

      # Horner's rule:
      y = self.Q.T[-1, :, numpy.newaxis] * x
      for q in reversed(self.Q.T[:-1]):
         y += q[:, numpy.newaxis]
         y *= x
      y += self.y_old[:, numpy.newaxis]

      if t.shape:
         return y
      else:
         return y[:, 0]


class CubicDenseOutput(DenseOutput):
   """Cubic, C1 continuous interpolator
   """
   def __init__(self, t_old, t, y_old, y, f_old, f):
      super(CubicDenseOutput, self).__init__(t_old, t)
      self.h = t - t_old
      self.y_old = y_old
      self.f_old = f_old
      self.y = y
      self.f = f

   def _call_impl(self, t):
      # scaled time
      x = (t - self.t_old) / self.h

      # qubic hermite spline:
      h00 = (1.0 + 2.0*x) * (1.0 - x)**2
      h10 = x * (1.0 - x)**2 * self.h
      h01 = x**2 * (3.0 - 2.0*x)
      h11 = x**2 * (x - 1.0) * self.h

      # output
      y = (h00 * self.y_old[:, numpy.newaxis] + h10 * self.f_old[:, numpy.newaxis]
           + h01 * self.y[:, numpy.newaxis] + h11 * self.f[:, numpy.newaxis])

      if t.shape:
         return y
      else:
         return y[:, 0]
