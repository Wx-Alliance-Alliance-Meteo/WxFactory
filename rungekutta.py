import math
import numpy
import mpi4py.MPI
import sys

MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.

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

def validate_tol(rtol, atol, n):
   """Validate tolerance values."""
   EPS = numpy.finfo(float).eps
   if rtol < 100 * EPS:
      warn("`rtol` is too low, setting to {}".format(100 * EPS))
      rtol = 100 * EPS

   atol = numpy.asarray(atol)
   if atol.ndim > 0 and atol.shape != (n,):
      raise ValueError("`atol` has wrong shape.")

   if numpy.any(atol < 0):
      raise ValueError("`atol` must be positive.")

   return rtol, atol

def validate_first_step(first_step, t0, t_bound):
   """Assert that first_step is valid and return it."""
   if first_step <= 0:
      raise ValueError("`first_step` must be positive.")
   if first_step > numpy.abs(t_bound - t0):
      raise ValueError("`first_step` exceeds bounds.")
   return first_step

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

def limiter(u, kappa):
   # Step-size ratio limiter. Applies an arctangent limiter parametrized by KAPPA to U.
   return  1 + kappa * math.atan((u - 1) / kappa)

class RungeKutta:
   """Base class for explicit Runge-Kutta methods."""
   C: numpy.ndarray = NotImplemented
   A: numpy.ndarray = NotImplemented
   B: numpy.ndarray = NotImplemented
   E: numpy.ndarray = NotImplemented
   order: int = NotImplemented
   error_estimator_order: int = NotImplemented
   n_stages: int = NotImplemented

   def __init__(self, fun, y0, t0:float = 0, t_bound=math.inf, rtol:float = 1e-3, atol:float = 1e-6, first_step=None, nb_step=None, sc_params="H211b", verbose=False):
      self.nfev = 0
      def fun_handle(t, y):
         self.nfev += 1
         return fun(t, y)
      self.fun = fun_handle

      self.nb_step = nb_step

      self.t = t0
      self.t_bound = t_bound

      self.y = y0
      self.n = y0.size
      self.f = self.fun(self.t, self.y)

      self.nb_rejected = 0

      self.status = 'running'

      self.rtol, self.atol = validate_tol(rtol, atol, self.n)

      if first_step is None:
         self.h_abs = select_initial_step(self.fun, self.t, self.y, self.f, self.error_estimator_order, self.rtol, self.atol)
      else:
         self.h_abs = validate_first_step(first_step, t0, self.t_bound)

      self.error_exponent = -1 / (self.error_estimator_order + 1)
      self._init_sc_control(sc_params)

      self.K = numpy.empty((self.n_stages + 1, self.n), self.y.dtype)
      self.FSAL = 1 if self.E[self.n_stages] else 0
      self.h_previous = None
      self.y_old = None

      self.verbose = verbose

   def _init_sc_control(self, sc_params):
      coefs = {
            "deadbeat": (1, 0, 0, 0.8),    # elementary controller (I)
            "PI3040": (0.7, -0.4, 0, 0.8), # PI controller (Gustafsson)
            "PI4020": (0.6, -0.2, 0, 0.8), # PI controller for nonstiff methods
            "H211PI": (1/6, 1/6, 0, 0.8),  # LP filter of PI structure
            "H110": (1/3, 0, 0, 0.8),      # I controller (convolution filter)
            "H211D": (1/2, 1/2, 1/2, 0.8), # LP filter with gain = 1/2
            "H211b": (1/4, 1/4, 1/4, 0.8)  # general purpose LP filter
      }

      if (isinstance(sc_params, str) and sc_params in coefs):
         kb1, kb2, a, g = coefs[sc_params]
      elif isinstance(sc_params, tuple) and len(sc_params) == 4:
         kb1, kb2, a, g = sc_params
      else:
         raise ValueError('Invalid sc_params')

      # set all parameters
      self.minbeta1 = kb1 * self.error_exponent
      self.minbeta2 = kb2 * self.error_exponent
      self.minalpha = -a
      self.safety = g
      self.safety_sc = g**(kb1 + kb2)
      self.elementary = True     # for first step

   def _estimate_error(self, K, h):
       # exclude K[-1] if not FSAL.
       return numpy.dot(K[:self.n_stages + self.FSAL].T, self.E[:self.n_stages + self.FSAL]) * h

   def _estimate_error_norm(self, K, h, scale):
       return norm(self._estimate_error(K, h) / scale)

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
               if self.t - self.t_bound >= 0:
                   self.status = 'finished'
       return message

   def run(self):
      step=0
      while self.status == 'running':
         message = self.step()
         step += 1
         if self.nb_step is not None:
            if step >= self.nb_step:
               return self.y
      return self.y

   def _step_impl(self):

      t = self.t
      y = self.y

      min_step = 10 * numpy.abs(numpy.nextafter(t, numpy.inf) - t)

      if self.h_abs < min_step:
          h_abs = min_step
      else:
          h_abs = self.h_abs

      step_accepted = False
      step_rejected = False

      while not step_accepted:
         if h_abs < min_step:
            return False, self.TOO_SMALL_STEP

         h = h_abs
         t_new = t + h

         if t_new - self.t_bound > 0:
             t_new = self.t_bound

         h = t_new - t
         h_abs = numpy.abs(h)

         self.K[0] = self.f
         for i in range(1, self.n_stages):
            self._rk_stage(h, i)
         y_new, error_norm = self._comp_sol_err(y, h)

         if error_norm < 1:
            step_accepted = True

            if self.elementary:
               factor = self.safety * error_norm**self.error_exponent
               self.elementary = False

            else:
               # Second order controller
               h_ratio = h / self.h_previous
               factor = self.safety_sc * ( error_norm**self.minbeta1 * self.error_norm_old**self.minbeta2 * h_ratio**self.minalpha )
#               factor = min(MAX_FACTOR, max(MIN_FACTOR, factor))
               factor = limiter(factor, 2)

            if step_rejected:
               factor = min(1, factor)

            h_abs *= factor

         else:
#            h_abs *= max(MIN_FACTOR, self.safety * error_norm**self.error_exponent)
            h_abs *= limiter(self.safety * error_norm**self.error_exponent, 2)
            step_rejected = True
            self.nb_rejected += 1
            if self.verbose: print(f"dt={h} rejected")

      if not self.FSAL:
          # evaluate ouput point for interpolation and next step
          self.K[self.n_stages] = self.fun(t + h, y_new)

      if self.verbose: print(f"dt={h} accepted")
      self.h_previous = h
      self.y_old = y
      self.h_abs = h_abs
      self.f = self.K[self.n_stages].copy()
      self.error_norm_old = error_norm

      # output
      self.t = t_new
      self.y = y_new

      return True, None

   def _rk_stage(self, h, i):
       """compute a single RK stage"""
       dy = h * (self.K[:i, :].T @ self.A[i, :i])
       self.K[i] = self.fun(self.t + self.C[i] * h, self.y + dy)

   def _comp_sol_err(self, y, h):
       """Compute solution and error.
       The calculation of `scale` differs from scipy: The average instead of
       the maximum of abs(y) of the current and previous steps is used.
       """
       y_new = y + h * (self.K[:self.n_stages].T @ self.B)
       scale = self.atol + self.rtol * 0.5*(numpy.abs(y) + numpy.abs(y_new))

       if self.FSAL:
           # do FSAL evaluation if needed for error estimate
           self.K[self.n_stages, :] = self.fun(self.t + h, y_new)

       error_norm = self._estimate_error_norm(self.K, h, scale)
       return y_new, error_norm

class Heun_Euler(RungeKutta):
    """Explicit Runge-Kutta method of order 2(1)."""
    order = 2
    error_estimator_order = 1
    n_stages = 2
    C = numpy.array([0, 1])
    A = numpy.array([
        [0, 0],
        [1, 0]
    ])
    B = numpy.array([1/2, 1/2])
    E = numpy.array([1, 0, 0])

class RK12(RungeKutta):
    """Explicit Runge-Kutta method of order 1(2)."""
    order = 2
    error_estimator_order = 1
    n_stages = 3
    C = numpy.array([0, 1/2, 1])
    A = numpy.array([
        [0, 0, 0],
        [1/2, 0, 0],
        [1/256, 255/256, 0]
    ])
    B = numpy.array([1/512, 255/256, 1/512])
    E = numpy.array( [1/256, 255/256, 0, 0])

class RK23(RungeKutta):
    """Explicit Runge-Kutta method of order 3(2)."""
    order = 3
    error_estimator_order = 2
    n_stages = 3
    C = numpy.array([0, 1/2, 3/4])
    A = numpy.array([
        [0,   0, 0],
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
