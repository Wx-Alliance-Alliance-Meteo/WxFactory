from typing import Callable, Optional, Tuple

from mpi4py   import MPI
import numpy
from numpy.typing import NDArray

from common.device             import Device
from common.configuration      import Configuration
from common.process_topology   import ProcessTopology
from geometry                  import Cartesian2D, CubedSphere, DFROperators, Geometry, Metric, \
                                      Metric3DTopo
from init.initialize           import Topo
from rhs.fluxes                import ausm_2d_fv, upwind_2d_fv, rusanov_2d_fv
from rhs.rhs_bubble            import rhs_bubble
from rhs.rhs_bubble_convective import rhs_bubble as rhs_bubble_convective
from rhs.rhs_bubble_fv         import rhs_bubble_fv
from rhs.rhs_bubble_implicit   import rhs_bubble_implicit
from rhs.rhs_euler             import RhsEuler
from rhs.rhs_euler_convective  import rhs_euler_convective
from rhs.rhs_euler_fv          import rhs_euler_fv
from rhs.rhs_sw                import RhsShallowWater
from rhs.rhs_sw_stiff          import rhs_sw_stiff
from rhs.rhs_sw_nonstiff       import rhs_sw_nonstiff
from rhs.rhs_advection2d       import rhs_advection2d

class RhsBundle:
   '''Set of RHS functions that are associated with a certain geometry and equations
   '''
   def __init__(self,
                geom: Geometry,
                operators: DFROperators,
                metric: Metric | Metric3DTopo | None,
                topo: Optional[Topo],
                ptopo: Optional[ProcessTopology],
                param: Configuration,
                fields_shape: Tuple[int, ...],
                device: Device) -> None:
      '''Determine handles to appropriate RHS functions.'''

      self.shape = fields_shape

      def generate_rhs(rhs_func: Callable, *args, **kwargs) -> Callable[[numpy.ndarray], numpy.ndarray]:
         '''Generate a function that calls the given (RHS) function on a vector. The generated function will
         first reshape the vector, then return a result with the original input vector shape.'''
         # if MPI.COMM_WORLD.rank == 0: print(f'Generating {rhs_func} with shape {self.shape}')
         def actual_rhs(vec: numpy.ndarray):
            old_shape = vec.shape
            result = rhs_func(vec.reshape(self.shape), *args, **kwargs)
            return result.reshape(old_shape)

         return actual_rhs

      if param.equations == "euler" and isinstance(geom, CubedSphere):
         # rhs_functions = {'dg': rhs_euler,
         #                  'fv': rhs_euler}

         # self.full = generate_rhs(rhs_functions[param.discretization],
         #                          geom, operators, metric, ptopo, param.nbsolpts, param.nb_elements_horizontal,
         #                          param.nb_elements_vertical, param.case_number, device=device)
         self.full = RhsEuler(fields_shape, geom, operators, metric, ptopo, param.nbsolpts, param.nb_elements_horizontal,
                              param.nb_elements_vertical, param.case_number, device=device)
         self.convective = generate_rhs(rhs_euler_convective, geom, operators, metric, ptopo, param.nbsolpts,
                                        param.nb_elements_horizontal, param.nb_elements_vertical, param.case_number)
         self.viscous = lambda q: self.full(q) - self.convective(q)

      elif param.equations == 'euler' and isinstance(geom, Cartesian2D):
         self.full = generate_rhs(
               rhs_bubble, geom, operators, param.nbsolpts, param.nb_elements_horizontal,
               param.nb_elements_vertical)
         
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
            # self.full = generate_rhs(
            #    rhs_sw, geom, operators, metric, topo, ptopo, param.nbsolpts, param.nb_elements_horizontal)
            self.full = RhsShallowWater(fields_shape,
                                        geom, operators, metric, topo, ptopo,
                                        param.nbsolpts, param.nb_elements_horizontal,
                                        device)
            

            self.implicit = generate_rhs(rhs_sw_stiff, geom, operators, metric, topo, ptopo, param.nbsolpts,
                                         param.nb_elements_horizontal)
            self.explicit = generate_rhs(rhs_sw_nonstiff, geom, operators, metric, topo, ptopo, param.nbsolpts,
                                         param.nb_elements_horizontal)
