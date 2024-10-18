from typing import TypeVar

import cupy as cp
from   numpy.typing import NDArray

from common.definitions import idx_rho_u1, idx_rho_u2, idx_rho_w, idx_rho, heat_capacity_ratio
from .cuda_module import CudaModule, DimSpec, Dim, Requires, TemplateSpec, cuda_kernel

class Rusanov(metaclass=CudaModule,
               path="rhs_euler.cu",
               defines=(("heat_capacity_ratio", heat_capacity_ratio),
                        ("idx_rho", idx_rho),
                        ("idx_rho_u1", idx_rho_u1),
                        ("idx_rho_u2", idx_rho_u2),
                        ("idx_rho_w", idx_rho_w))):

   @cuda_kernel[Number := TypeVar("Number", bound=cp.generic), Requires.DoubleOrComplex(Number),
                Real := TypeVar("Real", bound=cp.generic), Requires.ModulusOf(Real, Number)] \
   (DimSpec.groupby_first_x(lambda s: Dim(s[3], s[1], s[2] - 1)), TemplateSpec.array_dtype(0, 6))
   def compute_flux_i(
      flux_x1_itf_i: NDArray[Number], wflux_adv_x1_itf_i: NDArray[Number], wflux_pres_x1_itf_i: NDArray[Number],
      variables_itf_i: NDArray[Number], pressure_itf_i: NDArray[Number], u1_itf_i: NDArray[Number],
      sqrtG_itf_i: NDArray[Real], H_contra_1x_itf_i: NDArray[Real],
      nh: int, nk_nh: int, nk_nv: int, ne: int, advection_only: bool): ...

   @cuda_kernel[Number := TypeVar("Number", bound=cp.generic), Requires.DoubleOrComplex(Number),
                Real := TypeVar("Real", bound=cp.generic), Requires.ModulusOf(Real, Number)] \
   (DimSpec.groupby_first_x(lambda s: Dim(s[4], s[1], s[2] - 1)), TemplateSpec.array_dtype(0, 6))
   def compute_flux_j(
      flux_x2_itf_j: NDArray[Number], wflux_adv_x2_itf_j: NDArray[Number], wflux_pres_x2_itf_j: NDArray[Number],
      variables_itf_j: NDArray[Number], pressure_itf_j: NDArray[Number], u2_itf_j: NDArray[Number],
      sqrtG_itf_j: NDArray[Real], H_contra_2x_itf_j: NDArray[Real],
      nh: int, nk_nh: int, nk_nv: int, ne: int, advection_only: bool): ...

   @cuda_kernel[Number := TypeVar("Number", bound=cp.generic), Requires.DoubleOrComplex(Number),
                Real := TypeVar("Real", bound=cp.generic), Requires.ModulusOf(Real, Number)] \
   (DimSpec.groupby_first_x(lambda s: Dim(s[4], s[1], s[2] - 1)), TemplateSpec.array_dtype(0, 6))
   def compute_flux_k(
      flux_x3_itf_k: NDArray[Number], wflux_adv_x3_itf_k: NDArray[Number], wflux_pres_x3_itf_k: NDArray[Number],
      variables_itf_k: NDArray[Number], pressure_itf_k: NDArray[Number], w_itf_k: NDArray[Number],
      sqrtG_itf_k: NDArray[Real], H_contra_3x_itf_k: NDArray[Real],
      nv: int, nk_nh: int, ne: int, advection_only: bool): ...
