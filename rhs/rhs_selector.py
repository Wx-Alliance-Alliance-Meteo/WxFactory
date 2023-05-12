from typing import Optional

from geometry                  import Cartesian2D, CubedSphere
from init.initialize           import Topo
from rhs.rhs_bubble            import rhs_bubble
from rhs.rhs_bubble_cuda       import rhs_bubble_cuda
from rhs.rhs_bubble_convective import rhs_bubble as rhs_bubble_convective
from rhs.rhs_bubble_fv         import rhs_bubble_fv
from rhs.rhs_bubble_implicit   import rhs_bubble_implicit
from rhs.rhs_euler             import rhs_euler
from rhs.rhs_euler_convective  import rhs_euler_convective
from rhs.rhs_euler_fv          import rhs_euler_fv
from rhs.rhs_sw                import rhs_sw
from rhs.rhs_advection2d       import rhs_advection2d

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

         self.convective = lambda q: rhs_euler_convective(
            q, geom, operators, metric, ptopo, param.nbsolpts, param.nb_elements_horizontal,
            param.nb_elements_vertical, param.case_number)
         self.viscous = lambda q: self.full(q) - self.convective(q)

      elif param.equations == 'euler' and isinstance(geom, Cartesian2D):
         if param.discretization == 'dg':
            table = {"cpu": rhs_bubble, "cuda": rhs_bubble_cuda}
            self.full = lambda q, f=table[param.device]: f(
               q, geom, operators, param.nbsolpts, param.nb_elements_horizontal, param.nb_elements_vertical)
         elif param.discretization == 'fv':
            # If we implement a FV-specific version of rhs_bubble, we should select it here
            table = {"cpu": rhs_bubble, "cuda": rhs_bubble_cuda}
            self.full = lambda q, f=table[param.device]: f(
               q, geom, operators, param.nbsolpts, param.nb_elements_horizontal, param.nb_elements_vertical)

         self.implicit = lambda q: rhs_bubble_implicit(
            q, geom, operators, param.nbsolpts, param.nb_elements_horizontal, param.nb_elements_vertical)
         self.explicit = lambda q: self.full(q) - self.implicit(q)

         self.convective = lambda q: rhs_bubble_convective(
            q, geom, operators, param.nbsolpts, param.nb_elements_horizontal, param.nb_elements_vertical)
         self.viscous = lambda q: self.full(q) - self.convective(q)

      elif param.equations == "shallow_water":
         if param.case_number <= 1: # Pure advection
            self.full = lambda q: rhs_advection2d(
               q, geom, operators, metric, ptopo, param.nbsolpts, param.nb_elements_horizontal)
         else:
            self.full = lambda q: rhs_sw(
               q, geom, operators, metric, topo, ptopo, param.nbsolpts, param.nb_elements_horizontal)
