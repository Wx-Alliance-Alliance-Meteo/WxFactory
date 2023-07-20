from typing import Callable, Optional, Tuple

import numpy

from geometry                  import Cartesian2D, CubedSphere
from init.initialize           import Topo
from rhs.rhs_bubble            import rhs_bubble
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
                param: Configuration,
                fields_shape: Tuple[int, ...]) -> None:
      '''Determine handles to appropriate RHS functions.'''

      self.shape = fields_shape

      def generate_rhs(rhs_func: Callable, *args, **kwargs) -> Callable[[numpy.ndarray], numpy.ndarray]:
         '''Generate a function that calls the given (RHS) function on a vector. The generated function will
         first reshape the vector, then return a result with the original input vector shape.'''
         def actual_rhs(vec: numpy.ndarray):
            old_shape = vec.shape
            result = rhs_func(vec.reshape(self.shape), *args, **kwargs)
            return result.reshape(old_shape)

         return actual_rhs

      if param.equations == "euler" and isinstance(geom, CubedSphere):
         rhs_fn = rhs_euler
         if param.discretization == 'fv': rhs_fn = rhs_euler       # Fix rhs_euler_fv to be able to use it here

         self.full = generate_rhs(rhs_fn, geom, operators, metric, ptopo, param.nbsolpts, param.nb_elements_horizontal,
                                  param.nb_elements_vertical, param.case_number)
         self.convective = generate_rhs(rhs_euler_convective, geom, operators, metric, ptopo, param.nbsolpts,
                                        param.nb_elements_horizontal, param.nb_elements_vertical, param.case_number)
         self.viscous = lambda q: self.full(q) - self.convective(q)

      elif param.equations == 'euler' and isinstance(geom, Cartesian2D):
         rhs_fn = rhs_bubble
         if param.discretization == 'fv': rhs_fn = rhs_bubble      # Fix rhs_bubble_fv to be able to use it here

         self.full = generate_rhs(
            rhs_fn, geom, operators, param.nbsolpts, param.nb_elements_horizontal, param.nb_elements_vertical)
         self.implicit = generate_rhs(
            rhs_bubble_implicit, geom, operators, param.nbsolpts, param.nb_elements_horizontal,
            param.nb_elements_vertical)
         self.explicit = lambda q: self.full(q) - self.implicit(q)
         self.convective = generate_rhs(
            rhs_bubble_convective, geom, operators, param.nbsolpts, param.nb_elements_horizontal,
            param.nb_elements_vertical)
         self.viscous = lambda q: self.full(q) - self.convective(q)

      elif param.equations == "shallow_water":
         if param.case_number <= 1: # Pure advection
            self.full = generate_rhs(
               rhs_advection2d, geom, operators, metric, ptopo, param.nbsolpts, param.nb_elements_horizontal)
         else:
            self.full = generate_rhs(
               rhs_sw, geom, operators, metric, topo, ptopo, param.nbsolpts, param.nb_elements_horizontal)
