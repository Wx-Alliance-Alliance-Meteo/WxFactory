from abc import ABC, abstractmethod

import numpy
import torch
from numpy.typing import NDArray
from torch import Tensor

from ..common import Configuration
from ..common.definitions import idx_rho_u2
from ..geometry import DFROperators, Geometry, Metric2D, Metric3DTopo
from ..pde import PDE
from ..process_topology import ExchangeRequest, ProcessTopology


class RHS(ABC):
    req_r: ExchangeRequest
    req_u: ExchangeRequest
    req_t: ExchangeRequest

    def __init__(
        self,
        pde: PDE | None,
        geometry: Geometry,
        operators_real: DFROperators,
        metric: Metric2D | Metric3DTopo,
        topography,
        process_topo: ProcessTopology,
        config: Configuration | None,
        debug: bool = False,
    ) -> None:
        self.pde = pde
        self.geom = geometry
        self.ops_real = operators_real
        self.metric = metric
        self.topo = topography
        self.ptopo = process_topo
        self.config = config
        self.context = geometry.context
        self.debug = debug

        # Cache whether this geometry represents a y-invariant x-z slab.
        self.y_invariant_slab = getattr(geometry, "is_y_invariant_slab", False)

        if pde is not None:
            self.num_dim = self.pde.num_dim
            self.num_var = self.pde.num_var

        # Default value, when __call__ is not called
        self.ops = self.ops_real

        self.timestamps = []
        self.timings_real = []
        self.timings_complex = []

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

        self.latest_time_complex = False

    def clear_timings(self):
        self.timestamps: list[float | torch.cuda.Event | None] = []
        self.timings_real = []
        self.timings_complex = []

    def retrieve_last_times(self, is_complex: bool):
        if is_complex:
            self.timings_complex.append(self.context.elapsed(self.timestamps))
        else:
            self.timings_real.append(self.context.elapsed(self.timestamps))

    def __call__(self, q: Tensor) -> Tensor:

        # Process timing
        if len(self.timestamps) > 0:  # Process timing from previous steps
            self.retrieve_last_times(self.latest_time_complex)
        else:
            self.timestamps = [None for _ in range(9)]

        # Preserve array shape
        given_shape = q.shape

        self.ops = self.operators_for(q)
        self.latest_time_complex = torch.is_complex(q)

        self.allocate_arrays(q)

        self.timestamps[0] = self.context.timestamp(name="extrap")

        # Extrapolate the solution to the boundaries of the element
        self.solution_extrapolation(q)
        self.timestamps[1] = self.context.timestamp(name="start comm")

        self.start_communication()
        self.timestamps[2] = self.context.timestamp(name="pointwise flux")

        # Compute the pointwise fluxes
        self.pointwise_fluxes(q)
        self.timestamps[3] = self.context.timestamp(name="flux div 1")

        # Compute the derivatives of the discontinuous fluxes
        self.flux_divergence_partial()
        self.timestamps[4] = self.context.timestamp(name="end comm")

        self.end_communication()
        self.timestamps[5] = self.context.timestamp(name="riemann")

        # Compute the Riemann fluxes
        self.riemann_fluxes()
        self.timestamps[6] = self.context.timestamp(name="flux div 2")

        # Complete the divergence operation
        self.flux_divergence()
        self.timestamps[7] = self.context.timestamp(name="forcing")

        # Add forcing terms
        self.forcing_terms(q)
        self.pin_y_momentum()
        self.timestamps[8] = self.context.timestamp()

        # At this moment, a deep copy needs to be returned
        # otherwise issues are encountered after. This needs to be fixed
        return self.rhs.reshape(given_shape).clone()

    def pin_y_momentum(self) -> None:
        """Set the y-momentum tendency to zero for a y-invariant x-z slab.

        Apply this to every partition so their sum remains the full right-hand side.
        """
        if self.y_invariant_slab:
            self.rhs[idx_rho_u2] = 0.0

    def full(self, q: NDArray) -> NDArray:
        return self.__call__(q)

    def operators_for(self, q: NDArray) -> DFROperators:
        """Return the real operator set."""
        return self.ops_real

    def allocate_arrays(self, q: Tensor):
        if self.f_x1 is None or self.f_x1.dtype != q.dtype:
            self.f_x1 = torch.zeros_like(q)
            self.f_x2 = torch.zeros_like(q)
            self.f_x3 = torch.zeros_like(q)
            self.rhs = torch.empty_like(q)

    @abstractmethod
    def solution_extrapolation(self, q: NDArray) -> None:
        pass

    @abstractmethod
    def pointwise_fluxes(self, q: NDArray) -> None:
        pass

    def riemann_fluxes(self) -> None:
        if self.f_itf_x1 is None or self.f_itf_x1.dtype != self.q_itf_x1.dtype:
            self.f_itf_x1 = torch.zeros_like(self.q_itf_x1)
            self.f_itf_x2 = torch.zeros_like(self.q_itf_x2)
            self.f_itf_x3 = torch.zeros_like(self.q_itf_x3)

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
        for timings, is_complex in zip([self.timings_real, self.timings_complex], [False, True]):
            if len(timings) <= 1:
                continue
            timings = numpy.array(timings)
            extrapolation = timings[1:, 0].sum()
            start_comm = timings[1:, 1].sum()
            pw_flux = timings[1:, 2].sum()
            flux_div_1 = timings[1:, 3].sum()
            end_comm = timings[1:, 4].sum()
            riemann = timings[1:, 5].sum()
            flux_div_2 = timings[1:, 6].sum()
            forcing = timings[1:, 7].sum()
            total = timings[1:, -1].sum()
            num_calls = len(timings) - 1
            print(
                f"RHS times ({'real' if not is_complex else 'complex'}, {num_calls} calls):\n"
                f"                   Total | per call  (ms)\n"
                f"  -------------------------------\n"
                f"  Extrapolation:  {extrapolation:-6.1f} | {extrapolation / num_calls:-6.2f}\n"
                f"  Start comm:     {start_comm:-6.1f} | {start_comm / num_calls:-6.2f}\n"
                f"  Pointwise flux: {pw_flux:-6.1f} | {pw_flux / num_calls:-6.2f}\n"
                f"  Flux div 1:     {flux_div_1:-6.1f} | {flux_div_1 / num_calls:-6.2f}\n"
                f"  End comm:       {end_comm:-6.1f} | {end_comm / num_calls:-6.2f}\n"
                f"  Riemann:        {riemann:-6.1f} | {riemann / num_calls:-6.2f}\n"
                f"  Flux div 2:     {flux_div_2:-6.1f} | {flux_div_2 / num_calls:-6.2f}\n"
                f"  Forcing:        {forcing:-6.1f} | {forcing / num_calls:-6.2f}\n"
                f"  -------------------------------\n"
                f"  Total:        {total:8.1f} | {total / num_calls:6.1f}\n",
                flush=True,
            )
