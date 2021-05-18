from copy import copy
from time import time

import numpy

from cubed_sphere    import cubed_sphere
from initialize      import initialize_sw
from interpolation   import LagrangeSimpleInterpolator, interpolator
from linsol          import fgmres
from matvec          import matvec_rat
from matrices        import DFR_operators
from metric          import Metric
from rhs_sw          import rhs_sw

class FV_preconditioner:

   def __init__(self, param, initial_geom, ptopo):
      self.max_iter = 1000

      self.param = copy(param)
      self.param.discretization         = 'fv'
      self.param.nb_elements_horizontal = self.param.nb_elements_horizontal * self.param.nbsolpts
      self.param.nbsolpts               = 1

      if self.param.equations != 'shallow water':
         raise ValueError('Preconditioner is only implemented for the shallow water equations. '
                          'Need to make it a bit more flexible')

      self.ptopo        = ptopo
      self.fv_geom      = cubed_sphere(self.param.nb_elements_horizontal, self.param.nb_elements_vertical,
                                       self.param.nbsolpts, self.param.λ0, self.param.ϕ0, self.param.α0, self.param.ztop,
                                       self.ptopo, self.param)
      self.fv_operators = DFR_operators(self.fv_geom, self.param)
      self.fv_metric    = Metric(self.fv_geom)
      field, self.fv_topo   = initialize_sw(self.fv_geom, self.fv_metric, self.fv_operators, self.param)
      self.field_shape  = field.shape
      self.rhs_function = rhs_sw

      self.fv_matrix    = None
      self.fv_rhs       = None

      self.dg_geom       = initial_geom
      self.dg_nb_sol_pts = param.nbsolpts
      self.fv_nb_sol_pts = self.dg_nb_sol_pts # Cannot be different from dg_nb_sol_pts for now. TODO implement
      self.interpolator  = LagrangeSimpleInterpolator(initial_geom)

      self.interpolate = interpolator('dg', self.dg_nb_sol_pts, 'fv', self.fv_nb_sol_pts, 'lagrange')

   def dg_to_fv(self, vec):
      # return self.interpolator.eval_grid_fast(vec.reshape(self.field_shape), self.dg_nb_sol_pts - 3, self.dg_nb_sol_pts, equidistant=False)
      return self.interpolate(vec.reshape(self.field_shape))

   def fv_to_dg(self, vec):
      return self.interpolate(vec.reshape(self.field_shape), reverse=True)

   def apply(self, vec):

      t0 = time()

      input_vec = numpy.ravel(self.dg_to_fv(vec))
      output_vec, _, num_iter, _ = fgmres(
         self.fv_matrix, input_vec, preconditioner=None, tol=1e-2, maxiter=self.max_iter)
      output_vec = numpy.ravel(self.fv_to_dg(output_vec))

      t1 = time()
      precond_time = t1 - t0
      print(f'Preconditioned in {num_iter} iterations and {precond_time:.2f} s')

      return output_vec

   def init_time_step(self, matvec_func, dt, field, matvec_handle):
      fv_field = self.dg_to_fv(field)
      self.fv_rhs = lambda vec: self.rhs_function(
         vec, self.fv_geom, self.fv_operators, self.fv_metric, self.fv_topo, self.ptopo, self.param.nbsolpts,
         self.param.nb_elements_horizontal, self.param.case_number, False)
      self.fv_matrix = lambda vec: matvec_rat(vec, dt, fv_field, self.fv_rhs)

      self.remaining_uses = 1

   def __call__(self, vec):
      return self.apply(vec)
