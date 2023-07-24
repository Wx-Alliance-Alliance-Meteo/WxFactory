import math
import numpy

from integrators.butcher import *

def exode(τ_out, A, u, method='ARK3(2)4L[2]SA-ERK', rtol=1e-3, atol = 1e-6, task1 = False, verbose=False):

   if not hasattr(exode, "first_step"):
      exode.first_step = τ_out # TODO : use CFL condition ?

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

   method = method.upper()
   
   if method not in METHODS:
      raise ValueError("`method` must be one of {}." .format(METHODS))
   else:
      method = METHODS[method]

   t0, tf = map(float, [0, τ_out])

   solver = method(fun, t0, y0, tf, first_step=exode.first_step, rtol=rtol, atol=atol)

   ts = [t0]

   status = None
   while status is None:
      message = solver.step() # TODO : enlever message

      if solver.status == 'finished':
         status = 0
      elif solver.status == 'failed':
         status = -1
         break

      t_old = solver.t_old
      t = solver.t
      y = solver.y

      ts.append(t)

   ts = numpy.array(ts)

   solution = solver.y
   stats = (solver.nfev, 0) # TODO

   exode.first_step = numpy.median(numpy.diff(ts))

   return solution, stats
