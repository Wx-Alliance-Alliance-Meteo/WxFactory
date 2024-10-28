from typing import Callable, Optional, Tuple

from mpi4py import MPI
from numpy import ndarray
from numpy.typing import NDArray

from common.device import Device
from common.configuration import Configuration
from common.process_topology import ProcessTopology
from geometry import Cartesian2D, CubedSphere3D, DFROperators, Geometry, Metric2D, Metric3DTopo
from init.initialize import Topo
from rhs.rhs_euler import RhsEuler
from rhs.rhs import get_rhs


class RhsBundle:
    """Set of RHS functions that are associated with a certain geometry and equations"""

    def __init__(
        self,
        geom: Geometry,
        operators: DFROperators,
        metric: Metric2D | Metric3DTopo | None,
        topo: Optional[Topo],
        ptopo: Optional[ProcessTopology],
        param: Configuration,
        fields_shape: Tuple[int, ...],
        device: Device,
    ) -> None:
        """Determine handles to appropriate RHS functions."""

        self.shape = fields_shape

        def generate_rhs(rhs_func: Callable, *args, **kwargs) -> Callable[[ndarray], ndarray]:
            """Generate a function that calls the given (RHS) function on a vector. The generated function will
            first reshape the vector, then return a result with the original input vector shape."""

            # if MPI.COMM_WORLD.rank == 0: print(f'Generating {rhs_func} with shape {self.shape}')
            def actual_rhs(vec: ndarray) -> ndarray:
                old_shape = vec.shape
                # print(vec.reshape(self.shape))
                result = rhs_func(vec.reshape(self.shape))
                return result.reshape(old_shape)

            return actual_rhs

        if param.equations == "euler" and isinstance(geom, CubedSphere3D):
            # rhs_functions = {'dg': rhs_euler,
            #                  'fv': rhs_euler}

            # self.full = generate_rhs(rhs_functions[param.discretization],
            #                          geom, operators, metric, ptopo, param.nbsolpts, param.nb_elements_horizontal,
            #                          param.nb_elements_vertical, param.case_number, device=device)
            self.full = RhsEuler(
                fields_shape,
                geom,
                operators,
                metric,
                ptopo,
                param.nbsolpts,
                param.nb_elements_horizontal,
                param.nb_elements_vertical,
                param.case_number,
                device=device,
            )

        else:
            # Determine whether the RHS will be of DFR or FV type
            rhs_class = get_rhs(param.discretization)

            rhs_obj = rhs_class(
                param.equations + "-" + "cartesian", geom, operators, metric, topo, ptopo, param, device
            )

            self.full = generate_rhs(rhs_obj)

        # self.explicit = generate_rhs(
        #    rhs_class, param.equations + '-' + 'cartesian', geom,
        #    operators, metric, topo, ptopo, param, device)

        # self.implicit = generate_rhs(
        #    rhs_class, param.equations + '-' + 'cartesian', geom,
        #    operators, metric, topo, ptopo, param, device)

        #    self.convective = generate_rhs(rhs_euler_convective, geom, operators, metric, ptopo, param.nbsolpts,
        #                                   param.nb_elements_horizontal, param.nb_elements_vertical, param.case_number)
        #    self.viscous = lambda q: self.full(q) - self.convective(q)

        # elif param.equations == 'euler' and isinstance(geom, Cartesian2D):
        #    self.full = RhsBubble(fields_shape, geom, operators, param.nbsolpts, param.nb_elements_horizontal,
        #          param.nb_elements_vertical, device)

        #    self.implicit = generate_rhs(
        #       rhs_bubble_implicit, geom, operators, param.nbsolpts, param.nb_elements_horizontal,
        #       param.nb_elements_vertical)
        #    self.explicit = lambda q: self.full(q) - self.implicit(q)

        #    self.convective = generate_rhs(
        #       rhs_bubble_convective, geom, operators, param.nbsolpts, param.nb_elements_horizontal,
        #       param.nb_elements_vertical)
        #    self.viscous = lambda q: self.full(q) - self.convective(q)

        # elif param.equations == "shallow_water":
        #    if param.case_number <= 1: # Pure advection
        #       self.full = generate_rhs(
        #          rhs_advection2d, geom, operators, metric, ptopo, param.nbsolpts, param.nb_elements_horizontal)
        #    else:
        #       # self.full = generate_rhs(
        #       #    rhs_sw, geom, operators, metric, topo, ptopo, param.nbsolpts, param.nb_elements_horizontal)
        #       self.full = RhsShallowWater(fields_shape,
        #                                   geom, operators, metric, topo, ptopo,
        #                                   param.nbsolpts, param.nb_elements_horizontal)

        #       self.implicit = generate_rhs(rhs_sw_stiff, geom, operators, metric, topo, ptopo, param.nbsolpts,
        #                                    param.nb_elements_horizontal)
        #       self.explicit = generate_rhs(rhs_sw_nonstiff, geom, operators, metric, topo, ptopo, param.nbsolpts,
        #                                    param.nb_elements_horizontal)
