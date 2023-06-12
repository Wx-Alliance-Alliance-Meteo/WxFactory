import math
import numpy
import mpi4py.MPI
import scipy.linalg
import scipy.integrate

from integrators.butcher import *

# TODO: enlever
from scipy.integrate._ivp.common import OdeSolution
from scipy.integrate._ivp.base import OdeSolver

def exode(τ_out, A, u, method='ARK3(2)4L[2]SA-ERK', rtol=1e-3, atol = 1e-6, task1 = False, verbose=False):

   if not hasattr(exode, "first_step"):
      exode.first_step = None

   # TODO : implement dense output for output at intermediate values of τ_out

   ppo, n = u.shape
   p = ppo - 1

   if p == 0:
      p = 1
      # Add extra column of zeros
      u = numpy.row_stack((u, numpy.zeros(len(u))))

   y0 = u[0].copy()

   def fun(t, x):
      ret = A(x)
      for j in range(p):
         ret += t**j/math.factorial(j) * u[j+1]
      return ret

   MESSAGES = {0: "The solver successfully reached the end of the integration interval.",
               1: "A termination event occurred."}

   t_span = [0, τ_out]
   t_eval=None 
   dense_output=False
   events=None
   vectorized=False
   args=None
   
   method = method.upper()
   
   if method not in METHODS and not (inspect.isclass(method) and issubclass(method, OdeSolver)):
      raise ValueError("`method` must be one of {} or OdeSolver class." .format(METHODS))

   t0, tf = map(float, t_span)

   if t_eval is not None:
       t_eval = numpy.asarray(t_eval)
       if t_eval.ndim != 1:
           raise ValueError("`t_eval` must be 1-dimensional.")

       if numpy.any(t_eval < min(t0, tf)) or numpy.any(t_eval > max(t0, tf)):
           raise ValueError("Values in `t_eval` are not within `t_span`.")

       d = numpy.diff(t_eval)
       if tf > t0 and numpy.any(d <= 0) or tf < t0 and numpy.any(d >= 0):
           raise ValueError("Values in `t_eval` are not properly sorted.")

       if tf > t0:
           t_eval_i = 0
       else:
           # Make order of t_eval decreasing to use numpy.searchsorted.
           t_eval = t_eval[::-1]
           # This will be an upper bound for slices.
           t_eval_i = t_eval.shape[0]

   if method in METHODS:
      method = METHODS[method]

   solver = method(fun, t0, y0, tf, vectorized=vectorized, first_step=exode.first_step, rtol=rtol, atol=atol)

   if t_eval is None:
       ts = [t0]
       ys = [y0]
   elif t_eval is not None and dense_output:
       ts = []
       ti = [t0]
       ys = []
   else:
       ts = []
       ys = []

   interpolants = []

   status = None
   while status is None:
       message = solver.step()

       if solver.status == 'finished':
           status = 0
       elif solver.status == 'failed':
           status = -1
           break

       t_old = solver.t_old
       t = solver.t
       y = solver.y

       if dense_output:
           sol = solver.dense_output()
           interpolants.append(sol)
       else:
           sol = None

       if t_eval is None:
           ts.append(t)
           ys.append(y)
       else:
           # The value in t_eval equal to t will be included.
           if solver.direction > 0:
               t_eval_i_new = numpy.searchsorted(t_eval, t, side='right')
               t_eval_step = t_eval[t_eval_i:t_eval_i_new]
           else:
               t_eval_i_new = numpy.searchsorted(t_eval, t, side='left')
               # It has to be done with two slice operations, because
               # you can't slice to 0th element inclusive using backward
               # slicing.
               t_eval_step = t_eval[t_eval_i_new:t_eval_i][::-1]

           if t_eval_step.size > 0:
               if sol is None:
                   sol = solver.dense_output()
               ts.append(t_eval_step)
               ys.append(sol(t_eval_step))
               t_eval_i = t_eval_i_new

       if t_eval is not None and dense_output:
           ti.append(t)

   message = MESSAGES.get(status, message)

   if t_eval is None:
       ts = numpy.array(ts)
       ys = numpy.vstack(ys).T
   elif ts:
       ts = numpy.hstack(ts)
       ys = numpy.hstack(ys)

   if dense_output:
       if t_eval is None:
           sol = OdeSolution(ts, interpolants)
       else:
           sol = OdeSolution(ti, interpolants)
   else:
       sol = None

   # TODO : assemble solution with dense output
   
   solution = ys[:,-1]
   stats = (solver.nfev, 0) # TODO

   exode.first_step = numpy.median(numpy.diff(ts))

   return solution, stats
