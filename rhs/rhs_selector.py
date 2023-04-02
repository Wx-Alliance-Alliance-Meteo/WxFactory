from typing import Callable, Optional, Tuple

import numpy

from geometry                 import Cartesian2D, CubedSphere
from init.initialize          import Topo
from rhs.rhs_bubble           import rhs_bubble
from rhs.rhs_bubble_fv        import rhs_bubble_fv
from rhs.rhs_bubble_implicit  import rhs_bubble_implicit
from rhs.rhs_euler            import rhs_euler
from rhs.rhs_euler_fv         import rhs_euler_fv
from rhs.rhs_sw               import rhs_sw

# For type hints
from common.parallel        import DistributedWorld
from common.program_options import Configuration
from geometry               import DFROperators, Geometry, Metric

_RHSType = Callable[[numpy.array], numpy.array]

def rhs_selector(geom: Geometry,
                 operators: DFROperators,
                 metric: Metric,
                 topo: Topo,
                 ptopo: Optional[DistributedWorld],
                 param: Configuration, ) \
      -> Tuple[_RHSType, Optional[_RHSType], Optional[_RHSType]]:
   '''Determine handles to appropriate RHS functions.'''
   rhs_handle   = None
   rhs_implicit = None
   rhs_explicit = None

   if param.equations == "euler" and isinstance(geom, CubedSphere):
      if param.discretization == 'dg':
         rhs_handle = lambda q: rhs_euler(
            q, geom, operators, metric, ptopo, param.nbsolpts, param.nb_elements_horizontal,
            param.nb_elements_vertical, param.case_number)
      elif param.discretization == 'fv':
         # If we implement a FV-specific version of rhs_euler, we should select it here
         rhs_handle = lambda q: rhs_euler(
            q, geom, operators, metric, ptopo, param.nbsolpts, param.nb_elements_horizontal,
            param.nb_elements_vertical, param.case_number)

   elif param.equations == 'euler' and isinstance(geom, Cartesian2D):
      if param.discretization == 'dg':
         rhs_handle = lambda q: rhs_bubble(
            q, geom, operators, param.nbsolpts, param.nb_elements_horizontal, param.nb_elements_vertical)
      elif param.discretization == 'fv':
         # If we implement a FV-specific version of rhs_bubble, we should select it here
         rhs_handle = lambda q: rhs_bubble(
            q, geom, operators, param.nbsolpts, param.nb_elements_horizontal, param.nb_elements_vertical)

      rhs_implicit = lambda q: rhs_bubble_implicit(
         q, geom, operators, param.nbsolpts, param.nb_elements_horizontal, param.nb_elements_vertical)
      rhs_explicit = lambda q: rhs_handle(q) - rhs_implicit(q)

   elif param.equations == "shallow_water":
      rhs_handle = lambda q: rhs_sw(
         q, geom, operators, metric, topo, ptopo, param.nbsolpts, param.nb_elements_horizontal)

   if rhs_handle is None:
      raise ValueError('Could not determine appropriate RHS function. Check your parameters.')

   return rhs_handle, rhs_implicit, rhs_explicit
