from abc import ABC, abstractmethod

from torch import Tensor

from ..common import Configuration
from ..geometry import Geometry, Metric2D


class PDE(ABC):
    def __init__(
        self,
        geometry: Geometry,
        config: Configuration,
        metric: Metric2D,
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
    def pointwise_fluxes(self, q: Tensor, flux_x1: Tensor, flux_x2: Tensor, flux_x3: Tensor):
        pass

    @abstractmethod
    def riemann_fluxes(
        self,
        q_itf_x1: Tensor,
        q_itf_x2: Tensor,
        q_itf_x3: Tensor,
        flux_itf_x1: Tensor,
        flux_itf_x2: Tensor,
        flux_itf_x3: Tensor,
    ):
        pass
