import numpy

from init.initialize          import initialize_cartesian2d, initialize_euler, initialize_sw, Topo

# For type hints
from typing import Union, Tuple
from common.program_options      import Configuration
from geometry                    import DFROperators, Geometry, Metric, Metric3DTopo, CubedSphere, Cartesian2D

def init_state_vars(geom: Geometry, operators: DFROperators, param: Configuration) \
      -> Tuple[numpy.ndarray, Topo, Union[Metric, Metric3DTopo, None]]:

   Q            = None
   topo         = None
   metric       = None

   if param.equations == "euler" and isinstance(geom, CubedSphere):
      metric = Metric3DTopo(geom, operators)
      Q, topo = initialize_euler(geom, metric, operators, param)
      # Q: dimensions [5,nk,nj,ni], order ρ, u, v, w, θ

   elif param.equations == 'euler' and isinstance(geom, Cartesian2D):
      Q = initialize_cartesian2d(geom, param)

   elif param.equations == "shallow_water" and isinstance(geom, CubedSphere):
      metric = Metric(geom)
      Q, topo = initialize_sw(geom, metric, operators, param)

   return Q, topo, metric
