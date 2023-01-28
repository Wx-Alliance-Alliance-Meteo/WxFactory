from typing import Callable, Union, Tuple

from init.initialize          import Topo
from rhs.rhs_bubble           import rhs_bubble
from rhs.rhs_bubble_fv        import rhs_bubble_fv
from rhs.rhs_bubble_implicit  import rhs_bubble_implicit
from rhs.rhs_euler            import rhs_euler
from rhs.rhs_sw               import rhs_sw

# For type hints
from common.parallel             import Distributed_World
from common.program_options      import Configuration
from geometry.geometry           import Geometry
from geometry.metric             import Metric
from geometry.matrices           import DFR_operators

def rhs_selector(geom: Geometry, operators: DFR_operators, metric: Metric, topo: Topo, ptopo: Union[Distributed_World, None], param: Configuration, ) \
      -> Tuple[Callable, Callable, Callable]:

   rhs_handle   = None
   rhs_implicit = None
   rhs_explicit = None

   if param.equations == "euler" and param.grid_type == 'cubed_sphere':
      rhs_handle = lambda q: rhs_euler(q, geom, operators, metric, ptopo, param.nbsolpts, param.nb_elements_horizontal,
            param.nb_elements_vertical, param.case_number)

   elif param.equations == 'euler' and param.grid_type == 'cartesian2d':
      if param.discretization == 'dg':
         rhs_handle = lambda q: rhs_bubble(q, geom, operators, param.nbsolpts, param.nb_elements_horizontal, param.nb_elements_vertical)
      elif param.discretization == 'fv':
         rhs_handle = lambda q: rhs_bubble_fv(q, geom, operators, param.nbsolpts, param.nb_elements_horizontal, param.nb_elements_vertical)
      rhs_implicit = lambda q: rhs_bubble_implicit(q, geom, operators, param.nbsolpts, param.nb_elements_horizontal, param.nb_elements_vertical)
      rhs_explicit = lambda q: rhs_handle(q) - rhs_implicit(q)

   elif param.equations == "shallow_water":
      rhs_handle = lambda q: rhs_sw(q, geom, operators, metric, topo, ptopo, param.nbsolpts, param.nb_elements_horizontal)

   return rhs_handle, rhs_implicit, rhs_explicit
