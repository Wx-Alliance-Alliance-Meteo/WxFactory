from typing import Callable, Optional, Tuple

from mpi4py import MPI
from numpy import ndarray
from numpy.typing import NDArray

from common.device import Device
from common.configuration import Configuration
from common.process_topology import ProcessTopology
from geometry import Cartesian2D, CubedSphere2D, CubedSphere3D, DFROperators, Geometry, Metric2D, Metric3DTopo
from init.initialize import Topo
from .rhs_euler import RhsEuler
from .rhs_sw import RhsShallowWater
from .rhs_dfr import RHSDirecFluxReconstruction, RHSDirecFluxReconstruction_mpi
from .rhs_fv import RHSFiniteVolume


from pde import PDEEulerCartesian, PDEEulerCubesphere


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
        debug: bool = False,
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

        if param.discretization == "dfr" or param.discretization == "dg":
            rhs_class = RHSDirecFluxReconstruction
        elif param.discretization == "fv":
            rhs_class = RHSFiniteVolume
        else:
            raise ValueError(f"Unknown discretization {param.discretization}")

        if param.equations == "euler" and isinstance(geom, CubedSphere3D):
            pde = PDEEulerCubesphere(geom, param, metric)
            self.full = RHSDirecFluxReconstruction_mpi(
                pde, geom, operators, metric, topo, ptopo, param, fields_shape, debug=debug
            )
            # rhs_functions = {'dg': rhs_euler,
            #                  'fv': rhs_euler}

            # self.full = generate_rhs(rhs_functions[param.discretization],
            #                          geom, operators, metric, ptopo, param.num_solpts, param.num_elements_horizontal,
            #                          param.num_elements_vertical, param.case_number, device=device)
            # self.full.extra = RhsEuler(
            #     fields_shape,
            #     geom,
            #     operators,
            #     metric,
            #     ptopo,
            #     param.num_solpts,
            #     param.num_elements_horizontal,
            #     param.num_elements_vertical,
            #     param.case_number,
            #     device=geom.device,
            # )

        elif param.equations == "shallow_water" and isinstance(geom, CubedSphere2D):
            self.full = RhsShallowWater(
                fields_shape, geom, operators, metric, topo, ptopo, param.num_solpts, param.num_elements_horizontal
            )

        else:
            pde = PDEEulerCartesian(geom, param, metric)
            self.full = rhs_class(pde, geom, operators, metric, topo, ptopo, param, fields_shape)
