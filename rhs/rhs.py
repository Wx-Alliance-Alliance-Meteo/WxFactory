from abc import ABC, abstractmethod
from numpy.typing import NDArray

from common.array_module import get_array_module
from pde.pde import get_pde

# class RHS_OLD(ABC):
#     def __init__(self, shape: tuple[int, ...],  *params, **kwparams) -> None:
#         self.shape = shape
#         self.params = params
#         self.kwparams = kwparams

#     def __call__(self, vec: NDArray) -> NDArray:
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
#     def __compute_rhs__(self, vec: NDArray, *params, **kwparams) -> NDArray:
#         pass


def get_rhs(name):
    if name == "DFR":
        from rhs.rhs_dfr import RHS_DFR
        return RHS_DFR


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
        self.num_dim = num_dim = config.num_dim

        # Extract the appropriate array module
        xp = device.xp

        # Make sure the right CPU/GPU module is used
        self.pde = get_pde(pde_name)(config)

        # Initialize arrays that will be used in this RHS
        nb_solpts = operators.extrap_x.shape[0]
        nb_var = config.nb_var
        nb_elems = config.nb_elements_horizontal * config.nb_elements_vertical

        nb_itf_solpts_x1 = operators.extrap_x.shape[1]
        nb_itf_solpts_x3 = operators.extrap_z.shape[1]

        if num_dim == 3:
            nb_solpts *= operators.extrap_y.shape[0]
            nb_elems *= config.nb_elements_relief_layer
            nb_itf_solpts_x2 = operators.extrap_y.shape[1]

        # Assume two-dimensions first
        self.f_x1 = xp.empty((nb_var, nb_elems, nb_solpts))
        self.f_x2 = None
        self.f_x3 = xp.empty((nb_var, nb_elems, nb_solpts))

        self.q_itf_x1 = xp.empty((nb_var, nb_elems, nb_itf_solpts_x1))
        self.q_itf_x2 = None
        self.q_itf_x3 = xp.empty((nb_var, nb_elems, nb_itf_solpts_x3))

        self.f_itf_x1 = xp.empty_like(self.q_itf_x1)
        self.f_itf_x2 = None
        self.f_itf_x3 = xp.empty_like(self.q_itf_x3)

        self.df1_dx1 = xp.empty_like(self.f_x1)
        self.df2_dx2 = None
        self.df3_dx3 = xp.empty_like(self.f_x1)

        # Add third-dimension arrays if needed
        if num_dim == 3:
            self.f_x2 = xp.empty((nb_var, nb_elems, nb_solpts))
            self.q_itf_x2 = xp.empty((nb_var, nb_elems, nb_itf_solpts_x2))
            self.f_itf_x2 = xp.empty_like(self.q_itf_x2)
            self.df2_dx2 = xp.empty_like(self.f_x2)

        # Initialize rhs matrix
        self.rhs = xp.empty_like(self.f_x1)

    def __call__(self, q: NDArray) -> NDArray:
        # 1. Extrapolate the solution to the boundaries of the element
        self.solution_extrapolation(q)

        # 2. Compute the pointwise fluxes
        self.pointwise_fluxes(q)

        # 3. Compute the derivatives of the discontinuous fluxes
        self.flux_divergence_partial()

        # 4. Compute the Riemann fluxes
        self.riemann_fluxes()

        # 5. Add the correction term to the divergence
        self.flux_correction()

        # 6. Add forcing terms
        self.forcing_terms(q)

        return self.rhs

    @abstractmethod
    def solution_extrapolation(self, q: NDArray) -> None:
        pass

    @abstractmethod
    def pointwise_fluxes(self, q: NDArray) -> None:
        pass

    def riemann_fluxes(self) -> None:
        self.pde.riemann_fluxes(self.q_itf_x1, self.q_itf_x2, self.q_itf_x3,
                                self.f_itf_x1, self.f_itf_x2, self.f_itf_x3)

    @abstractmethod
    def flux_divergence_partial(self) -> None:
        pass

    @abstractmethod
    def flux_correction(self) -> None:
        pass

    def forcing_terms(self, q: NDArray) -> None:
        self.pde.forcing_terms(self.rhs, q)
