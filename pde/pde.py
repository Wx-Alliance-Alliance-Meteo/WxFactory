from abc import ABC, abstractmethod
from typing import Callable

from numpy.typing import NDArray

from common.configuration import Configuration
from common.device import Device
from geometry import Geometry, Metric2D, Metric3DTopo


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
        pointwise_func: Callable,
        riemann_func: Callable,
    ):
        self.geom = geometry
        self.config = config
        self.device = geometry.device
        self.metric = metric

        self.num_dim = num_dim
        self.num_var = num_var
        self.num_elem = num_elem

        if pointwise_func is None or riemann_func is None:
            raise ValueError(f"Must provide a pointwise and a Riemann flux function")

        self.pointwise_func = pointwise_func
        self.riemann_func = riemann_func

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
