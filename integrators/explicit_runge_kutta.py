from math import sqrt
import logging
import numpy
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

   # Parameters for stepsize control, optional
   sc_params = "standard"                      # tuple, or str

   max_factor = MAX_FACTOR0                    # initially
   min_factor = MIN_FACTOR

   def __init__(self, fun, t0, y0, t_bound, max_step=numpy.inf, rtol=1e-3,
                atol=1e-6, first_step=None, sc_params=None):

      self.t_old = None
      self.t = t0
      self._fun, self.y = fun, y0
      self.t_bound = t_bound

      def self_fun(t, y):
         self.nfev += 1
         return self._fun(t, y)

      self.fun = self_fun

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
      self.error_norm_old = None
      self.h_min_a, self.h_min_b = self._init_min_step_parameters()
      self.tiny_err = self.h_min_b

      # Initialize controler
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

      # size of first step:
      if first_step is None:
         raise RuntimeError("`first_step` not defined.")
      
      if first_step <= 0:
         raise ValueError("`first_step` must be positive.")
      if first_step > numpy.abs(t_bound - t0):
         raise ValueError("`first_step` exceeds bounds.")
      self.h = first_step

      self.K = numpy.empty((self.n_stages + 1, self.n), self.y.dtype)
      self.FSAL = 1 if self.E[self.n_stages] else 0
      self.h_previous = None
      self.y_old = None
      NFS[()] = 0                                # global failed step counter

   def step(self):
      if self.status != 'running':
         raise RuntimeError("Attempt to step on a failed or finished "
                             "solver.")

      if self.n == 0 or self.t == self.t_bound:
         # Handle corner cases of empty solver or no integration.
         self.t_old = self.t
         self.t = self.t_bound
         self.status = 'finished'
      else:
         t = self.t
         success = self._step_impl()

         if not success:
            self.status = 'failed'
         else:
            self.t_old = t
            if self.t - self.t_bound >= 0:
               self.status = 'finished'

   def _step_impl(self):
      # mostly follows the scipy implementation of scipy's RungeKutta
      t = self.t
      y = self.y

      h, min_step = self._reassess_stepsize(t)

      # loop until the step is accepted
      step_accepted = False
      step_rejected = False
      while not step_accepted:
         if h < min_step:
            return False
         t_new = t + h

         # calculate RK stages
         self.K[0] = self.f
         for i in range(1, self.n_stages):
            dy = h * (self.K[:i, :].T @ self.A[i, :i])
            self.K[i] = self.fun(self.t + self.C[i] * h, self.y + dy)

         # Update solution
         y_new = y + h * (self.K[:self.n_stages].T @ self.B)

         # calculate error norm
         if self.FSAL:
            # do FSAL evaluation if needed for error estimate
            self.K[self.n_stages, :] = self.fun(self.t + h, y_new)

         scale = self.atol + self.rtol * numpy.maximum(numpy.abs(y), numpy.abs(y_new))
         # exclude K[-1] if not FSAL. It could contain nan or inf
         err_estimate =  h * (self.K[:self.n_stages + self.FSAL].T @ self.E[:self.n_stages + self.FSAL])
         error_norm = global_norm(err_estimate / scale)

         # evaluate error
         if error_norm < 1:
            step_accepted = True

            if error_norm < self.tiny_err:
               factor = self.max_factor
               self.standard_sc = True

            elif self.standard_sc:
               factor = self.safety * error_norm**self.error_exponent
               self.standard_sc = False

            else:
               # use second order SC controller
               h_ratio = h / self.h_previous
               factor = self.safety_sc * ( error_norm**self.minbeta1 *
                   self.error_norm_old**self.minbeta2 * h_ratio ** self.minalpha)
               factor = min(self.max_factor, max(self.min_factor, factor))

            if step_rejected:
               factor = min(1, factor)

            h *= factor

            if factor < MAX_FACTOR:
               # reduce max_factor when on scale.
               self.max_factor = MAX_FACTOR

         else:
            step_rejected = True
            h *= max(self.min_factor, self.safety * error_norm ** self.error_exponent)

            NFS[()] += 1

            if numpy.isnan(error_norm) or numpy.isinf(error_norm):
               return False, "Overflow or underflow encountered."

      if not self.FSAL:
         # evaluate ouput point for the next step
         self.K[self.n_stages] = self.fun(t + h, y_new)

      # store for next step, interpolation and stepsize control
      self.h_previous = h
      self.y_old = y
      self.h = h
      self.f = self.K[self.n_stages].copy()
      self.error_norm_old = error_norm

      # output
      self.t = t_new
      self.y = y_new

      return True

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

   def _reassess_stepsize(self, t):
      # limit step size
      h = self.h
      min_step = max(self.h_min_a * (abs(t) + h), self.h_min_b)
      if h < min_step or h > self.max_step:
         h = min(self.max_step, max(min_step, h))
         self.standard_sc = True

      # handle final integration steps
      d = abs(self.t_bound - t)                     # remaining interval
      if d < 2 * h:
         if d > h:
            # h < d < 2 * h: "look ahead".
            # split d over last two steps. This reduces the chance of a
            # very small last step.
            h = max(0.5 * d, min_step)
            self.standard_sc = True
         else:
            # d <= h: Don't step over t_bound
            h = d

      return h, min_step
