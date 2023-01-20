
from Init.initialize          import initialize_cartesian2d, initialize_euler, initialize_sw, Topo
from Rhs.rhs_bubble           import rhs_bubble
from Rhs.rhs_bubble_implicit  import rhs_bubble_implicit
from Rhs.rhs_euler            import rhs_euler
from Rhs.rhs_sw               import rhs_sw

# For type hints
import numpy
from typing import Callable, Union, Tuple
from Common.parallel             import Distributed_World
from Common.program_options      import Configuration
from Geometry.geometry           import Geometry
from Geometry.metric             import Metric, Metric_3d_topo
from Geometry.matrices           import DFR_operators

def init_state_vars(geom: Geometry, operators: DFR_operators, ptopo: Union[Distributed_World, None], param: Configuration) \
      -> Tuple[numpy.ndarray, Topo, Union[Metric, Metric_3d_topo, None], Callable, Callable, Callable]:
   
   Q            = None
   topo         = None
   metric       = None
   rhs_handle   = None
   rhs_implicit = None
   rhs_explicit = None

   if param.equations == "euler" and param.grid_type == 'cubed_sphere':
      metric = Metric_3d_topo(geom, operators)
      Q, topo = initialize_euler(geom, metric, mtrx, param)
      # Q: dimensions [5,nk,nj,ni], order ρ, u, v, w, θ
      rhs_handle = lambda q: rhs_euler(q, geom, operators, metric, ptopo, param.nbsolpts, param.nb_elements_horizontal,
            param.nb_elements_vertical, param.case_number)

   elif param.equations == 'euler' and param.grid_type == 'cartesian2d':
      Q = initialize_cartesian2d(geom, param)
      rhs_handle = lambda q: rhs_bubble(q, geom, operators, param.nbsolpts, param.nb_elements_horizontal, param.nb_elements_vertical)
      rhs_implicit = lambda q: rhs_bubble_implicit(q, geom, operators, param.nbsolpts, param.nb_elements_horizontal, param.nb_elements_vertical)
      rhs_explicit = lambda q: rhs_handle(q) - rhs_implicit(q)

   elif param.equations == "shallow_water":
      metric = Metric(geom)
      Q, topo = initialize_sw(geom, metric, operators, param)
      rhs_handle = lambda q: rhs_sw(q, geom, operators, metric, topo, ptopo, param.nbsolpts, param.nb_elements_horizontal)

   return Q, topo, metric, rhs_handle, rhs_implicit, rhs_explicit
