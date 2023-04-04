from typing import Optional

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

class RhsBundle:
   def __init__(self,
                geom: Geometry,
                operators: DFROperators,
                metric: Metric,
                topo: Topo,
                ptopo: Optional[DistributedWorld],
                param: Configuration) -> None:
      '''Determine handles to appropriate RHS functions.'''

      if param.equations == "euler" and isinstance(geom, CubedSphere):
         if param.discretization == 'dg':
            self.full = lambda q: rhs_euler(
               q, geom, operators, metric, ptopo, param.nbsolpts, param.nb_elements_horizontal,
               param.nb_elements_vertical, param.case_number)
         elif param.discretization == 'fv':
            # If we implement a FV-specific version of rhs_euler, we should select it here
            self.full = lambda q: rhs_euler(
               q, geom, operators, metric, ptopo, param.nbsolpts, param.nb_elements_horizontal,
               param.nb_elements_vertical, param.case_number)

      elif param.equations == 'euler' and isinstance(geom, Cartesian2D):
         if param.discretization == 'dg':
            self.full = lambda q: rhs_bubble(
               q, geom, operators, param.nbsolpts, param.nb_elements_horizontal, param.nb_elements_vertical)
         elif param.discretization == 'fv':
            # If we implement a FV-specific version of rhs_bubble, we should select it here
            self.full = lambda q: rhs_bubble(
               q, geom, operators, param.nbsolpts, param.nb_elements_horizontal, param.nb_elements_vertical)

         self.implicit = lambda q: rhs_bubble_implicit(
            q, geom, operators, param.nbsolpts, param.nb_elements_horizontal, param.nb_elements_vertical)
         self.explicit = lambda q: self.full(q) - self.implicit(q)

      elif param.equations == "shallow_water":
         self.full = lambda q: rhs_sw(
            q, geom, operators, metric, topo, ptopo, param.nbsolpts, param.nb_elements_horizontal)
