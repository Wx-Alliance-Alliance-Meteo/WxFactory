import sys
import numpy
import math
import scipy.sparse.linalg
import scipy.optimize
from time import time

from .fgmres import fgmres
from .global_operations import global_norm, global_inf_norm

def newton_krylov(F, x0, fgmres_restart=30, fgmres_maxiter=1, fgmres_precond=None, verbose=False, maxiter=None, f_tol=None, f_rtol=None, x_tol=None, x_rtol=None, line_search='armijo'):

   t_start = time()
   iteration = 0

   gamma = 0.9
   eta_max = 0.9999
   eta_treshold = 0.1
   eta = 1e-3

   if f_tol is None:
      f_tol = numpy.finfo(numpy.float_).eps ** (1./3)
   if f_rtol is None:
      f_rtol = numpy.inf
   if x_tol is None:
      x_tol = numpy.inf
   if x_rtol is None:
      x_rtol = numpy.inf

   f0_norm = None

   func = lambda z: F(numpy.reshape(z, x0.shape)).flatten()
   x = x0.flatten()

   dx = numpy.full_like(x, numpy.inf)
   Fx = func(x)
   Fx_norm = global_norm(Fx)

   jacobian = KrylovJacobian(x.copy(), Fx, func, fgmres_restart=fgmres_restart, fgmres_maxiter=fgmres_maxiter, fgmres_precond=fgmres_precond)

   if maxiter is None:
      maxiter = 100*(x.size+1)

   if line_search not in (None, 'armijo', 'wolfe'):
      raise ValueError("Invalid line search")

   residuals = []

   for n in range(maxiter):

      iteration += 1
      f_norm = global_inf_norm(Fx)
      x_norm = global_inf_norm(x)


      dx_norm = global_inf_norm(dx)

      residuals.append((f_norm, time() - t_start, 0.0))

      if f0_norm is None:
         f0_norm = f_norm

      if f_norm == 0:
         terminated = True

      terminated = (f_norm <= f_tol and f_norm / f_rtol <= f0_norm) and (dx_norm <= x_tol and dx_norm / x_rtol <= x_norm)

      if terminated:
         break

      tol = min(eta, eta*Fx_norm)
      dx = -jacobian.solve(Fx, tol=tol)

      # Line search, or Newton step
      if line_search:
         s, x, Fx, Fx_norm_new = _nonlin_line_search(func, x, Fx, dx, line_search)
      else:
         s = 1.0
         x = x + dx
         Fx = func(x)
         Fx_norm_new = global_norm(Fx)

      jacobian.update(x.copy(), Fx)

      # Adjust forcing parameters for inexact methods
      eta_A = gamma * Fx_norm_new**2 / Fx_norm**2
      if gamma * eta**2 < eta_treshold:
         eta = min(eta_max, eta_A)
      else:
         eta = min(eta_max, max(eta_A, gamma*eta**2))

      Fx_norm = Fx_norm_new

      # Print status
      if verbose:
         sys.stdout.write(f'{n:3d}:  |F(x)| = {global_inf_norm(Fx):.3e}; step {s}\n')
         sys.stdout.flush()
   else:
      print('The maximum number of iterations allowed by the JFNK method has been reached.')

   if terminated == 1:
      print(f'A solution was found after {iteration-1} steps of the JFNK method.')

   return numpy.reshape(x, x0.shape), iteration - 1, residuals


def _nonlin_line_search(func, x, Fx, dx, search_type='armijo', rdiff=1e-8,
                  smin=1e-2):
   tmp_s = [0]
   tmp_Fx = [Fx]
   tmp_phi = [global_norm(Fx)**2]
   s_norm = global_norm(x) / global_norm(dx)

   def phi(s, store=True):
      if s == tmp_s[0]:
         return tmp_phi[0]
      xt = x + s*dx
      v = func(xt)
      p = global_norm(v)**2
      if store:
         tmp_s[0] = s
         tmp_phi[0] = p
         tmp_Fx[0] = v
      return p

   def derphi(s):
      ds = (abs(s) + s_norm + 1) * rdiff
      return (phi(s+ds, store=False) - phi(s)) / ds

   if search_type == 'wolfe': # TODO : parallel ?
      s, phi1, phi0 = scipy.optimize.linesearch.scalar_search_wolfe1(phi, derphi, tmp_phi[0], xtol=1e-2, amin=smin)
   elif search_type == 'armijo':
      s, phi1 = scipy.optimize.linesearch.scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0], amin=smin)

   if s is None:
      # No suitable step length found. Take the full Newton step, and hope for the best.
      s = 1.0

   x = x + s*dx
   if s == tmp_s[0]:
      Fx = tmp_Fx[0]
   else:
      Fx = func(x)
   Fx_norm = global_norm(Fx)

   return s, x, Fx, Fx_norm

class KrylovJacobian:

   def __init__(self, x, f, func, fgmres_restart, fgmres_maxiter, fgmres_precond):
      self.func = func
      self.shape = (f.size, x.size)
      self.dtype = f.dtype

      self.fgmres_restart = fgmres_restart
      self.fgmres_maxiter = fgmres_maxiter
      self.fgmres_precond = fgmres_precond

      self.x0 = x
      self.f0 = f
      self.rdiff = math.sqrt( numpy.finfo(x.dtype).eps )
      self._update_diff_step()

      self.op = scipy.sparse.linalg.aslinearoperator(self)

   def _update_diff_step(self):
      mx = global_inf_norm(self.x0)
      mf = global_inf_norm(self.f0)
      self.omega = self.rdiff * max(1, mx) / max(1, mf)

   def matvec(self, v):
      nv = global_norm(v)
      if nv == 0:
         return 0*v
      sc = self.omega / nv
      return (self.func(self.x0 + sc*v) - self.f0) / sc

   def solve(self, rhs, tol=0):
      sol, res, norm_b, nb_iter, info, residuals = fgmres(self.op, rhs, tol=tol, restart=self.fgmres_restart, maxiter=self.fgmres_maxiter, preconditioner=self.fgmres_precond)
      # print(f'reached residual {res:.3e} after {nb_iter:3d} iterations')
      return sol

   def update(self, x, f):
      self.x0 = x
      self.f0 = f
      self._update_diff_step()
