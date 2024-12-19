from abc import ABC, abstractmethod
import signal
import time

from mpi4py import MPI
from numpy.typing import NDArray

from common.configuration import Configuration
from common.device import Device
from common.process_topology import ProcessTopology
from geometry import DFROperators, Geometry, Metric2D, Metric3DTopo
from pde import PDE


class RHS(ABC):
    def __init__(
        self,
        pde: PDE,
        geometry: Geometry,
        operators: DFROperators,
        metric: Metric2D | Metric3DTopo,
        topography,
        process_topo: ProcessTopology,
        config: Configuration,
        expected_shape: tuple[int, ...],
        debug: bool = False,
    ) -> None:
        self.pde = pde
        self.geom = geometry
        self.ops = operators
        self.metric = metric
        self.topo = topography
        self.ptopo = process_topo
        self.config = config
        self.device = geometry.device
        self.expected_shape = expected_shape
        self.debug = debug

        self.num_dim = self.pde.num_dim
        self.num_var = self.pde.num_var

        # print(f"{MPI.COMM_WORLD.rank} debug = {self.debug}", flush=True)

        # Initially set all arrays to None, these will be allocated later
        self.f_x1 = None
        self.f_x2 = None
        self.f_x3 = None

        self.q_itf_x1 = None
        self.q_itf_x2 = None
        self.q_itf_x3 = None

        self.f_itf_x1 = None
        self.f_itf_x2 = None
        self.f_itf_x3 = None

        self.df1_dx1 = None
        self.df2_dx2 = None
        self.df3_dx3 = None

        self.q_itf_s = None
        self.q_itf_n = None
        self.q_itf_w = None
        self.q_itf_e = None

        # Initialize rhs matrix
        self.rhs = None

    def __call__(self, q: NDArray) -> NDArray:

        # 0. Preserve array shape
        given_shape = q.shape

        self.allocate_arrays(q)

        # 1. Extrapolate the solution to the boundaries of the element
        self.solution_extrapolation(q)

        self.start_communication()

        # 2. Compute the pointwise fluxes
        self.pointwise_fluxes(q)

        # 3. Compute the derivatives of the discontinuous fluxes
        self.flux_divergence_partial()

        self.end_communication()

        # 4. Compute the Riemann fluxes
        self.riemann_fluxes()

        # 5. Complete the divergence operation
        self.flux_divergence()

        # 6. Add forcing terms
        self.forcing_terms(q)

        # At this moment, a deep copy needs to be returned
        # otherwise issues are encountered after. This needs to be fixed
        return self.rhs.reshape(given_shape).copy()

    def full(self, q: NDArray) -> NDArray:
        return self.__call__(q)

    def allocate_arrays(self, q: NDArray):
        xp = self.device.xp

        if self.f_x1 is None or self.f_x1.dtype != q.dtype:
            self.f_x1 = xp.zeros_like(q)
            self.f_x2 = xp.zeros_like(q)
            self.f_x3 = xp.zeros_like(q)
            self.rhs = xp.empty_like(q)

            self.pressure = xp.zeros_like(q[0])
            self.log_p = xp.zeros_like(q[0])

            self.wflux_adv_x1 = xp.zeros_like(q[0])
            self.wflux_pres_x1 = xp.zeros_like(q[0])

            self.wflux_adv_x2 = xp.zeros_like(q[0])
            self.wflux_pres_x2 = xp.zeros_like(q[0])

            self.wflux_adv_x3 = xp.zeros_like(q[0])
            self.wflux_pres_x3 = xp.zeros_like(q[0])

            self.w_df1_dx1 = xp.zeros_like(q[0])
            self.w_df2_dx2 = xp.zeros_like(q[0])
            self.w_df3_dx3 = xp.zeros_like(q[0])

            self.forcing = xp.zeros_like(q)

    @abstractmethod
    def solution_extrapolation(self, q: NDArray) -> None:
        pass

    @abstractmethod
    def pointwise_fluxes(self, q: NDArray) -> None:
        pass

    def riemann_fluxes(self) -> None:
        xp = self.device.xp
        if self.f_itf_x1 is None or self.f_itf_x1.dtype != self.q_itf_x1.dtype:
            self.f_itf_x1 = xp.zeros_like(self.q_itf_x1)
            self.f_itf_x2 = xp.zeros_like(self.q_itf_x2)
            self.f_itf_x3 = xp.zeros_like(self.q_itf_x3)

        self.pde.riemann_fluxes(
            self.q_itf_x1, self.q_itf_x2, self.q_itf_x3, self.f_itf_x1, self.f_itf_x2, self.f_itf_x3
        )

    @abstractmethod
    def flux_divergence_partial(self) -> None:
        pass

    @abstractmethod
    def flux_divergence(self) -> None:
        pass

    def forcing_terms(self, q: NDArray) -> None:
        self.pde.forcing_terms(self.rhs, q)

    def start_communication(self) -> None:
        pass

    def end_communication(self) -> None:
        pass
