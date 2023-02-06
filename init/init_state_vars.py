import numpy

from init.initialize          import initialize_cartesian2d, initialize_euler, initialize_sw, Topo

# For type hints
from typing import Union, Tuple
from common.program_options      import Configuration
from geometry.geometry           import Geometry
from geometry.metric             import Metric, Metric_3d_topo
from geometry.matrices           import DFR_operators

def init_state_vars(geom: Geometry, operators: DFR_operators, param: Configuration) \
      -> Tuple[numpy.ndarray, Topo, Union[Metric, Metric_3d_topo, None]]:
   
   Q            = None
   topo         = None
   metric       = None

   if param.equations == "euler" and param.grid_type == 'cubed_sphere':
      metric = Metric_3d_topo(geom, operators)
      Q, topo = initialize_euler(geom, metric, operators, param)
      # Q: dimensions [5,nk,nj,ni], order ρ, u, v, w, θ

   elif param.equations == 'euler' and param.grid_type == 'cartesian2d':
      Q = initialize_cartesian2d(geom, param)

   elif param.equations == "shallow_water":
      metric = Metric(geom)
      Q, topo = initialize_sw(geom, metric, operators, param)

   return Q, topo, metric
