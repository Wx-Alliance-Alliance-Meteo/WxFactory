from abc import ABC, abstractmethod
from numpy import ndarray

from common.array_module import get_array_module
from pde.pde import get_pde

# class RHS_OLD(ABC):
#     def __init__(self, shape: tuple[int, ...],  *params, **kwparams) -> None:
#         self.shape = shape
#         self.params = params
#         self.kwparams = kwparams

#     def __call__(self, vec: ndarray) -> ndarray:
#         """Compute the value of the right-hand side based on the input state.

#         :param vec: Vector containing the input state. It can have any shape, as long as its size is the same as the
#                     one used to create this RHS object
#         :return: Value of the right-hand side, in the same shape as the input
#         """
#         old_shape = vec.shape
#         result = self.__compute_rhs__(vec.reshape(
#             self.shape), *self.params, **self.kwparams)
#         return result.reshape(old_shape)

#     @abstractmethod
#     def __compute_rhs__(self, vec: ndarray, *params, **kwparams) -> ndarray:
#         pass


def get_rhs(name):
    if name == "dfr" or name == "dg":
        from rhs.rhs_dfr import RHS_DFR
        return RHS_DFR
    if name == "fv":
        from rhs.rhs_fv import RHS_FV
        return RHS_FV


class RHS(ABC):
    def __init__(self, pde_name, geometry, operators, metric,
                 topology, process_topo, config, device) -> None:
        self.pde_name = pde_name
        self.geom = geometry
        self.ops = operators
        self.metric = metric
        self.topo = topology
        self.ptopo = process_topo
        self.config = config
        self.device = device
        self.num_dim = config.num_dim

        # Instantiate appropriate PDE object
        self.pde = get_pde(pde_name)(config, device)

        # Must be allocated at every child class
        self.rhs = None

    def __call__(self, q: ndarray) -> ndarray:

        # 1. Extrapolate the solution to the boundaries of the element
        self.solution_extrapolation(q)

        # 2. Compute the pointwise fluxes
        self.pointwise_fluxes(q)

        # 3. Compute the derivatives of the discontinuous fluxes
        self.flux_divergence_partial()

        # 4. Compute the Riemann fluxes
        self.riemann_fluxes()

        # 5. Complete the divergence operation
        self.flux_divergence()

        # 6. Add forcing terms
        self.forcing_terms(q)

        # At this moment, a deep copy needs to be returned
        # otherwise issues are encountered after. This needs to be fixed
        return 1.0*self.rhs

    def full(self, q: ndarray) -> ndarray:
        return self.__call__(q)

    @abstractmethod
    def solution_extrapolation(self, q: ndarray) -> None:
        pass

    @abstractmethod
    def pointwise_fluxes(self, q: ndarray) -> None:
        pass

    def riemann_fluxes(self) -> None:
        self.pde.riemann_fluxes(self.q_itf_x1, self.q_itf_x2, self.q_itf_x3,
                                self.f_itf_x1, self.f_itf_x2, self.f_itf_x3)

    @abstractmethod
    def flux_divergence_partial(self) -> None:
        pass

    @abstractmethod
    def flux_divergence(self) -> None:
        pass

    def forcing_terms(self, q: ndarray) -> None:
        self.pde.forcing_terms(self.rhs, q)
