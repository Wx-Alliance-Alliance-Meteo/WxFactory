from typing import Callable, Optional, Tuple

from numpy.typing import NDArray

from common import Configuration
from geometry import Cartesian2D, CubedSphere2D, CubedSphere3D, DFROperators, Geometry, Metric2D, Metric3DTopo
from init.initialize import Topo
from pde import PDEEulerCartesian, PDEEulerCubesphere
from process_topology import ProcessTopology

from .rhs_bubble_convective import rhs_bubble as rhs_bubble_convective
from .rhs_bubble_implicit import rhs_bubble_implicit
from .rhs_sw import RhsShallowWater
from .rhs_dfr import RHSDirecFluxReconstruction, RHSDirecFluxReconstruction_mpi
from .rhs_fv import RHSFiniteVolume


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

        def not_implemented(_):
            raise ValueError(f"Partitioned integrators have not been ported to the new layout yet")

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
                fields_shape, geom, operators, metric, topo, ptopo, geom.num_solpts, geom.num_elements_horizontal
            )

        elif param.equations == "euler" and isinstance(geom, Cartesian2D):
            pde = PDEEulerCartesian(geom, param, metric)
            self.full = rhs_class(pde, geom, operators, metric, topo, ptopo, param, fields_shape)

            self.implicit = not_implemented
            self.explicit = not_implemented
            self.convective = not_implemented
            self.viscous = not_implemented
        else:
            raise ValueError(f"Unrecognized combination of equations ({param.equations}) and geometry ({type(geom)})")
