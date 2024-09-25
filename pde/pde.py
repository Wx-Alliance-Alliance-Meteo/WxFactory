from abc import ABC, abstractmethod
from numpy.typing import NDArray


def get_pde(name):
    if name == "euler-cartesian":
        from pde.pde_euler_cartesian import PDEEulerCartesian
        return PDEEulerCartesian

    elif name == "euler-cubesphere":
        from pde.pde_euler_cubedsphere import PDEEulerCubedSphere
        return PDEEulerCubeSphere


class PDE(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def pointwise_fluxes(self, q, fluxes):
        pass

    @abstractmethod
    def riemann_fluxes(self, q_itf_x1: NDArray, q_itf_x2: NDArray, q_itf_x3: NDArray,
                       fluxes_itf_x1: NDArray, flux_itf_x2: NDArray, fluxes_itf_x3: NDArray) -> NDArray:
        pass

    @abstractmethod
    def forcing_terms(self, rhs, q):
        pass
