import math
import mpi4py
import numpy

from copy import copy
from time import time

from cubed_sphere  import cubed_sphere
from initialize    import initialize_sw
from interpolation import interpolator
from linsol        import fgmres, global_norm
from matrices      import DFR_operators
from matvec        import matvec_rat
from metric        import Metric
from rhs_sw        import rhs_sw

def func_that_returns_its_input(x):
   return x

def explicit_euler_smoothe(x, A, b, h, num_iter=1):
   for i in range(num_iter):
      x = x + h * (b - A(x))
   return x

def runge_kutta_smoothe(x, A, b, h, num_iter=1):
   for i in range(num_iter):
      x1 = x + h * (b - A(x))
      x2 = 0.75 * x + 0.25 * x1 + 0.25 * h * (b - A(x))
      x = 1.0/3.0 * x + 2.0/3.0 * x2 + 2.0/3.0 * h * (b - A(x))
   return x

def make_restrict(interp, shape):
   def restrict(vec):
      return interp(vec.reshape(shape))
   return restrict

def make_prolong(interp, shape):
   def prolong(vec):
      return interp(vec.reshape(shape), reverse=True)
   return prolong

def make_matrix_op(field, rhs, dt, matvec):
   def op(vec):
      return matvec(numpy.ravel(vec), dt, field, rhs)
   return op

def make_rhs_op(rhs, geom, operators, metric, topo, ptopo, num_points, num_elem_horiz, case_number, apply_filter):
   def op(vec):
      return rhs(vec, geom, operators, metric, topo, ptopo, num_points, num_elem_horiz, case_number, apply_filter)
   return op


class MG_params:

   def __init__(self, param, ptopo, num_levels=None) -> None:
      if num_levels is None:
         num_levels = int(math.log2(param.initial_nbsolpts))
         if 2**num_levels != param.initial_nbsolpts:
            raise ValueError('Cannot do multigrid stuff if the order of the problem is not a power of 2')

      if param.equations != 'shallow_water':
         raise ValueError('Cannot use the multigrid solver with anything other than shallow water')

      self.max_level = num_levels

      self.rhs = rhs_sw
      self.matvec = matvec_rat

      self.smoothers = {}
      self.interpolators = {}
      self.restrict_operators = {}
      self.prolong_operators = {}
      self.matrix_operators = {}
      self.rhs_operators = {}
      self.params = {}

      order = param.initial_nbsolpts
      for level in range(self.max_level, -1, -1):
         p = copy(param)
         p.nb_elements_horizontal = p.nb_elements_horizontal * p.nbsolpts
         p.nbsolpts = 1
         p.discretization = 'fv'
         if level < self.max_level:
            p.nb_elements_horizontal = self.params[level+1].nb_elements_horizontal // 2
         
         self.params[level] = p

         # print(f'level: {level}. Params: {self.params[level]}')
         
         geom        = cubed_sphere(p.nb_elements_horizontal, p.nb_elements_vertical, p.nbsolpts, p.λ0, p.ϕ0, p.α0, p.ztop, ptopo, p)
         operators   = DFR_operators(geom, p)
         metric      = Metric(geom)
         field, topo = initialize_sw(geom, metric, operators, p)

         self.rhs_operators[level] = make_rhs_op(self.rhs, geom, operators, metric, topo, ptopo, p.nbsolpts, p.nb_elements_horizontal, p.case_number, False)

         self.smoothers[level] = explicit_euler_smoothe

         if level > 0:
            self.interpolators[level] = interpolator('fv', order, 'fv', order//2, 'bilinear')
            self.restrict_operators[level] = make_restrict(self.interpolators[level], field.shape)

         if level < self.max_level:
            self.prolong_operators[level] = make_prolong(self.interpolators[level + 1], field.shape)

         order = order // 2

   def compute_matrix_operators(self, field, dt):

      next_field = field
      for level in range(self.max_level, -1, -1):
         # print(f'Level: {level}, next_field.shape: {next_field.shape}')
         # self.matrix_operators[level] = lambda v: self.matvec(v, dt, next_field, self.rhs_operators[level])
         self.matrix_operators[level] = make_matrix_op(next_field, self.rhs_operators[level], dt, self.matvec)
         if level > 0:
            next_field = self.restrict_operators[level](next_field)

   def get_smoother(self, level):
      return self.smoothers[level]

   def get_matrix_op(self, level):
      return self.matrix_operators[level]

   def get_interpolator(self, level):
      return self.interpolators[level]

   def get_restrict_op(self, level):
      return self.restrict_operators[level]

   def get_prolong_op(self, level):
      return self.prolong_operators[level]


def mg(b, x0, level, mg_params, gamma=1, dt=0.0):

   A        = mg_params.get_matrix_op(level)
   smoothe  = mg_params.get_smoother(level)
   restrict = mg_params.get_restrict_op(level)    if level > 0 else None
   prolong  = mg_params.get_prolong_op(level - 1) if level > 0 else None

   if level == 0:
      return x0
      # Just solve the problem (approximately)
      x, _, num_iter, _ = fgmres(A, b, x0=x0, tol=1e-1)
      return x

   x = smoothe(x0, A, b, dt)

   if level > 1:
      # Go down a level and solve that
      residual = numpy.ravel(restrict(A(x) - b))
      v = numpy.zeros_like(residual)
      for i in range(gamma):
         v = mg(residual, v, level - 1, mg_params, gamma=gamma)

      x = x - numpy.ravel(prolong(v))  # Correction
   x = smoothe(x, A, b, dt)

   return x


def mg_solve(b, dt, mg_params, field=None, tolerance=1e-7, gamma=1, max_num_it=100):

   t0 = time()

   x = numpy.zeros_like(b)

   norm_b = global_norm(b)
   tol_relative = tolerance * norm_b

   if norm_b < 1e-15:
      return x, 0, time() - t0

   if field is not None:
      mg_params.compute_matrix_operators(field, dt)

   level = mg_params.max_level
   A = mg_params.get_matrix_op(level)
   num_it = 0
   for it in range(max_num_it):
      x = mg(b, x, level, mg_params, gamma=gamma, dt=500)
      num_it += 1

      # Check tolerance exit condition
      if it < max_num_it - 1:
         residual = A(x) - b
         norm_r = global_norm(residual)
         # print(f'norm_r = {norm_r:.2e} (rel tol {tol_relative:.2e})')
         if norm_r < tol_relative:
            return x, num_it, time() - t0

   return x, num_it, time() - t0


def global_norm(vec):
   """Compute vector norm across all PEs"""
   local_sum = vec @ vec
   return math.sqrt( mpi4py.MPI.COMM_WORLD.allreduce(local_sum) )
