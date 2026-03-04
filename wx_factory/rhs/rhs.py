from abc import ABC, abstractmethod
import time

import numpy
from numpy.typing import NDArray

from common import Configuration
from geometry import DFROperators, Geometry, Metric2D, Metric3DTopo
from pde import PDE
from process_topology import ProcessTopology, ExchangeRequest


class RHS(ABC):
    req_r: ExchangeRequest
    req_u: ExchangeRequest
    req_t: ExchangeRequest

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

        self.timestamps = []
        self.timings = []

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
        
        # ESAV variables
        self.v_itf_x1 = None
        self.v_itf_x3 = None
        
        self.dv_dx1 = None
        self.dv_dx3 = None
        
        self.v_jump_x1 = None
        self.v_jump_x3 = None
        
        self.sigma = None

        # Initialize rhs matrix
        self.rhs = None

    def clear_timings(self):
        self.timestamps = []
        self.timings = []

    def retrieve_last_times(self):
        self.timings.append(self.device.elapsed(self.timestamps))

    def __call__(self, q: NDArray) -> NDArray:

        # 0.a Process timing
        if len(self.timestamps) > 0:  # Process timing from previous steps
            self.retrieve_last_times()
        else:
            self.timestamps = [None for _ in range(9)]

        # 0.b Preserve array shape
        given_shape = q.shape

        self.allocate_arrays(q)

        self.timestamps[0] = self.device.timestamp(name="extrap")

        # 1. Extrapolate the solution to the boundaries of the element
        self.solution_extrapolation(q)
        self.timestamps[1] = self.device.timestamp(name="start comm")

        self.start_communication()
        self.timestamps[2] = self.device.timestamp(name="pointwise flux")

        # 2. Compute the pointwise fluxes
        self.pointwise_fluxes(q)
        self.timestamps[3] = self.device.timestamp(name="flux div 1")

        # 3. Compute the derivatives of the discontinuous fluxes - the volume integral
        self.flux_divergence_partial()
        self.timestamps[4] = self.device.timestamp(name="end comm")

        self.end_communication()
        self.timestamps[5] = self.device.timestamp(name="riemann")

        # 4. Compute the Riemann fluxes
        self.riemann_fluxes()
        self.timestamps[6] = self.device.timestamp(name="flux div 2")

        # 5. Complete the divergence operation
        self.flux_divergence()
        self.timestamps[7] = self.device.timestamp(name="forcing")

        # 6. Add forcing terms
        self.forcing_terms(q)
        self.timestamps[8] = self.device.timestamp(name="artificial viscosity")
        
        #
        #
        # 7. Add artificial viscosity for entropy stability
        #
        #
        
        # 7.1 Extrapolate the entropy variables to the boundaries of the element
        # TODO: Precompute v from q
        self.solution_extrapolation_entropy(v)
        
        #
        # 7.2. Compute gradient of v via BR1 (Theta_i, w)
        #
        # 7.2.1 Compute the derivatives of the entropy variables 
        self.entropy_gradient_partial(v: NDArray)
        # 7.2.2 Compute the jump across the interfaces
        self.entropy_jump()
        # 7.2.3 Complete the gradient operation by ading the boundary terms
        self.entropy_gradient()
        
        #
        # 7.3 Compute the viscosity coeff
        #
        # 7.3.1 Compute the entropy residual
        # Compute volume term
        
        
        
        
        # 7.3 Compute the viscosity coefficients epsilon_k
        
        # 7.4 Compute the jacobian du/dv
        
        # 7.5 Compute viscous flux (-sigma_i, w)
        
        # 7.5 Extrapolate the entropy fluxes
        
        # 7.6 Apply central flux (boundary terms  for viscous flux sigma_i

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

            itf_shape = q.shape[:4] + (2 * self.geom.num_solpts**2,)

            self.q_itf_x1 = xp.empty(itf_shape, dtype=q.dtype)
            self.q_itf_x2 = xp.empty_like(self.q_itf_x1)
            self.q_itf_x3 = xp.empty_like(self.q_itf_x1)

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

    def print_times(self) -> None:
        timings = numpy.array(self.timings)
        extrapolation = timings[:, 0].mean() * 1000.0
        start_comm = timings[:, 1].mean() * 1000.0
        pw_flux = timings[:, 2].mean() * 1000.0
        flux_div_1 = timings[:, 3].mean() * 1000.0
        end_comm = timings[:, 4].mean() * 1000.0
        riemann = timings[:, 5].mean() * 1000.0
        flux_div_2 = timings[:, 6].mean() * 1000.0
        forcing = timings[:, 7].mean() * 1000.0
        total = timings[:, -1].mean() * 1000.0
        print(
            f"RHS times:\n"
            f"  Extrapolation:  {extrapolation:5.1f} ms\n"
            f"  Start comm:     {start_comm:5.1f} ms\n"
            f"  Pointwise flux: {pw_flux:5.1f} ms\n"
            f"  Flux div 1:     {flux_div_1:5.1f} ms\n"
            f"  End comm:       {end_comm:5.1f} ms\n"
            f"  Riemann:        {riemann:5.1f} ms\n"
            f"  Flux div 2:     {flux_div_2:5.1f} ms\n"
            f"  Forcing:        {forcing:5.1f} ms\n"
            f"  -------------------------\n"
            f"  Total:          {total:5.1f}"
        )
