import math
import numpy
import mpi4py.MPI
import scipy.linalg

import rungekutta
from linsol        import fgmres

# TODO : prendre en argument un pas de temps explicite (ex: CFL)
def exode(τ_out, A, u, method='RK23', rtol=1e-3, atol = 1e-6, task1 = False, verbose=False):

   # TODO : implement dense output for output at multiple τ_out

   ppo, n = u.shape
   p = ppo - 1

   if p == 0:
      p = 1
      # Add extra column of zeros
      u = numpy.row_stack((u, numpy.zeros(len(u))))

   y = u[0].copy()

   def fun(t, x):
      ret = A(x)
      for j in range(p):
         ret += t**j/math.factorial(j) * u[j+1]
      return ret

   if method == 'RK23':
      rk = rungekutta.RK23(fun, y, 0, 1, rtol=rtol, atol=atol, verbose=verbose)
      y = rk.run()
      stats = (rk.nfev, rk.nb_rejected) # TODO
   elif method == 'RK45':
      rk = rungekutta.RK45(fun, y, 0, 1, rtol=rtol, atol=atol, verbose=verbose)
      y = rk.run()
      stats = (rk.nfev, rk.nb_rejected) # TODO
   elif method == 'ROS2':
      dt = 1
      τ_now = 0

      Op = scipy.sparse.linalg.LinearOperator((n,n), matvec=lambda x: x - 0.5 * dt * A(x))

      b = Op(y) + fun(τ_now, y) * dt

      # TODO : gcro
      y, local_error, num_iter, flag = fgmres(Op, b, x0=y, tol=atol, preconditioner=None)
     
      stats = (num_iter, 0) # TODO
   else:
      print('Error : unknown method')
      exit(1)

   return y, stats
