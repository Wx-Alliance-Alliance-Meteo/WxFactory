from time import time

from linsol import fgmres
from matvec import matvec_rat
from rhs_fv import rhs_sw_fv
from rhs_sw import rhs_sw

class FV_preconditioner:

   def __init__(self, param, geometry, operators, metric, topo, ptopo):
      self.max_iter = 1000

      self.geometry     = geometry
      self.operators    = operators
      self.metric       = metric
      self.topo         = topo
      self.ptopo        = ptopo
      self.rhs_function = rhs_sw_fv
      # self.rhs_function = rhs_sw

      self.fv_matrix    = None
      self.fv_rhs       = None
      self.nb_sol_pts   = param.nbsolpts
      self.nb_elements  = param.nb_elements_horizontal
      self.case_number  = param.case_number

   def apply(self, vec):

      t0 = time()

      input_vec = vec  # TODO interpolate to FV grid
      output_vec, _, num_iter, _ = fgmres(
         self.fv_matrix, input_vec, preconditioner=None, tol=1e-4, maxiter=self.max_iter)

      t1 = time()
      precond_time = t1 - t0

      print(f'Preconditioned in {num_iter} iterations and {precond_time:.2f} s')

      return output_vec

   def init_time_step(self, matvec_func, dt, field, matvec_handle):
      fv_field = field  # TODO interpolate to FV grid
      self.fv_rhs = lambda vec: self.rhs_function(
         vec, self.geometry, self.operators, self.metric, self.topo, self.ptopo, self.nb_sol_pts, self.nb_elements,
         self.case_number, False)
      self.fv_matrix = lambda vec: matvec_rat(vec, dt, fv_field, self.fv_rhs)

   def __call__(self, vec):
      return self.apply(vec)
