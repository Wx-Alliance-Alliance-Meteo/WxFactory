from abc import ABC, abstractmethod

import numpy
import torch
from numpy.typing import NDArray

from ..common import Configuration
from ..common.definitions import idx_rho_u2
from ..device import differentiable_mode
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
        operators_complex: DFROperators,
        metric: Metric2D | Metric3DTopo,
        topography,
        process_topo: ProcessTopology,
        config: Configuration | None,
        expected_shape: tuple[int, ...],
        debug: bool = False,
    ) -> None:
        self.pde = pde
        self.geom = geometry
        self.ops_real = operators_real
        self.ops_complex = operators_complex
        self.metric = metric
        self.topo = topography
        self.ptopo = process_topo
        self.config = config
        self.device = geometry.device
        self.expected_shape = expected_shape
        self.debug = debug

        # Cache whether this geometry represents a y-invariant x-z slab.
        self.y_invariant_slab = getattr(geometry, "is_y_invariant_slab", False)

        if pde is not None:
            self.num_dim = self.pde.num_dim
            self.num_var = self.pde.num_var

        # Default value, when __call__ is not called
        self.ops = self.ops_real

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

        # Initialize rhs matrix
        self.rhs = None
        self._workspace_invalidated = False

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

        self.ops = self.ops_complex if torch.is_complex(q) else self.ops_real

        # Forward AD cannot write into scratch tensors captured from an earlier RHS call.
        if differentiable_mode():
            self.invalidate_workspace()

        self.allocate_arrays(q)
        self._workspace_invalidated = False

        self.timestamps[0] = self.device.timestamp(name="extrap")

        # 1. Extrapolate the solution to the boundaries of the element
        self.solution_extrapolation(q)
        self.timestamps[1] = self.device.timestamp(name="start comm")

        self.start_communication()
        self.timestamps[2] = self.device.timestamp(name="pointwise flux")

        # 2. Compute the pointwise fluxes
        self.pointwise_fluxes(q)
        self.timestamps[3] = self.device.timestamp(name="flux div 1")

        # 3. Compute the derivatives of the discontinuous fluxes
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
        self.pin_y_momentum()
        self.timestamps[8] = self.device.timestamp()

        # At this moment, a deep copy needs to be returned
        # otherwise issues are encountered after. This needs to be fixed
        return self.rhs.reshape(given_shape).copy()

    def pin_y_momentum(self) -> None:
        """Set the y-momentum tendency to zero for a y-invariant x-z slab.

        Apply this to every partition so their sum remains the full right-hand side.
        """
        if self.y_invariant_slab:
            self.rhs[idx_rho_u2] = 0.0

    def full(self, q: NDArray) -> NDArray:
        return self.__call__(q)

    def allocate_arrays(self, q: NDArray):
        if self.workspace_needs_allocation(self.f_x1, q.dtype):
            self.f_x1 = torch.zeros_like(q)
            self.f_x2 = torch.zeros_like(q)
            self.f_x3 = torch.zeros_like(q)
            self.rhs = torch.empty_like(q)

    def invalidate_workspace(self) -> None:
        """Require fresh scratch arrays on the next evaluation."""
        self._workspace_invalidated = True

    def workspace_needs_allocation(self, array, dtype) -> bool:
        """Return whether a scratch array must be allocated."""
        return self._workspace_invalidated or array is None or array.dtype != dtype

    @abstractmethod
    def solution_extrapolation(self, q: NDArray) -> None:
        pass

    @abstractmethod
    def pointwise_fluxes(self, q: NDArray) -> None:
        pass

    def riemann_fluxes(self) -> None:
        if self.workspace_needs_allocation(self.f_itf_x1, self.q_itf_x1.dtype):
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
