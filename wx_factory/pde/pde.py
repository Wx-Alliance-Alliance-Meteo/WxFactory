from abc import ABC, abstractmethod

from numpy.typing import NDArray

from ..common import Configuration
from ..geometry import Geometry, Metric2D, Metric3DTopo


class PDE(ABC):
    """PDE groups a set of parameters and function used for the computation of a right-hand side (RHS)."""

    def __init__(
        self,
        geometry: Geometry,
        config: Configuration,
        metric: Metric2D | Metric3DTopo,
        num_dim: int,
        num_var: int,
        num_elem: int,
    ):
        self.geometry = geometry
        self.config = config
        self.context = geometry.context
        self.metric = metric

        self.num_dim = num_dim
        self.num_var = num_var
        self.num_elem = num_elem

    @abstractmethod
    def pointwise_fluxes(self, q: NDArray, flux_x1: NDArray, flux_x2: NDArray, flux_x3: NDArray):
        pass

    @abstractmethod
    def riemann_fluxes(
        self,
        q_itf_x1: NDArray,
        q_itf_x2: NDArray,
        q_itf_x3: NDArray,
        flux_itf_x1: NDArray,
        flux_itf_x2: NDArray,
        flux_itf_x3: NDArray,
    ):
        pass
