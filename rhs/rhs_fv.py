from numpy.typing import NDArray

from rhs.rhs import RHS

class RHS_FV(RHS):

    def __call__(self, Q: NDArray) -> NDArray:

    def solution_extrapolation(self, Q: NDArray) -> NDArray:
        pass

    def pointwise_fluxes(self, Q: NDArray, fx1: NDArray, fx2: NDArray, fx3: NDArray) -> None:
        # Not applicable in FV
        pass

    def flux_divergence(self, flux: NDArray, flux_riem: NDArray) -> NDArray:        
        return - (df1_dx1 + df3_dx3)
    