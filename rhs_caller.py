from copy import copy
import numpy

from cubed_sphere  import cubed_sphere
from matrices      import DFR_operators
from metric        import Metric
from initialize    import initialize_sw
from interpolation import LagrangeSimpleInterpolator


class RhsCaller:
   def __init__(self, function, geometry, operators, metric, topo, ptopo, nb_sol_pts, nb_elements, case_num,
                use_filter, timers):
      self.function   = function
      self.geometry   = geometry
      self.operators  = operators
      self.metric     = metric
      self.topo       = topo
      self.ptopo      = ptopo
      self.nb_sol_pts = nb_sol_pts
      self.nb_elem    = nb_elements
      self.case_num   = case_num
      self.use_filter = use_filter
      self.timers     = timers

   def __call__(self, field):
      return self.function(field, self.geometry, self.operators, self.metric, self.topo, self.ptopo,
                           self.nb_sol_pts, self.nb_elem, self.case_num, self.use_filter, self.timers)



class RhsCallerLowRes(RhsCaller):
   def __init__(self, function, geometry, operators, metric, topo, ptopo, nb_sol_pts, nb_elements, case_num,
                use_filter, timers, param):

      RhsCaller.__init__(self, function, geometry, operators, metric, topo, ptopo, nb_sol_pts, nb_elements,
                         case_num, use_filter, timers)

      self.rank = self.ptopo.rank
      self.low_order = max(self.nb_sol_pts - 2, 3)

      param_small = copy(param)
      param_small.nbsolpts = self.low_order

      self.low_geometry  = cubed_sphere(self.nb_elem, self.low_order, param.λ0, param.ϕ0, param.α0, self.ptopo)
      self.low_operators = DFR_operators(self.low_geometry, param_small)
      self.low_metric    = Metric(self.low_geometry)
      _, self.low_topo   = initialize_sw(self.low_geometry, self.low_operators, self.low_metric, param_small)

      self.interpolator  = LagrangeSimpleInterpolator(self.geometry)

      print('large order = {}, small order = {}'.format(self.nb_sol_pts, self.low_order))


   def __call__(self, field):

      low_res_field = self.restrict(field)
      rhs = self.function(low_res_field, self.low_geometry, self.low_operators, self.low_metric, self.low_topo, self.ptopo,
                          self.low_order, self.nb_elem, self.case_num, self.use_filter)
      result = self.prolong(rhs)

      return result
      # return self.function(field, self.low_geometry, self.low_operators, self.low_metric, self.low_topo, self.ptopo,
      #                      self.low_order, self.nb_elem, self.case_num, self.use_filter)


   def restrict(self, field):
      return self.interpolator.eval_grid_fast(field, self.low_order, self.nb_sol_pts)

   def prolong(self, field):
      return self.interpolator.eval_grid_fast(field, self.nb_sol_pts, self.low_order)

