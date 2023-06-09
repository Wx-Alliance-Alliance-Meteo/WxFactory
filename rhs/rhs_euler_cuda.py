import numpy as np
import cupy as cp

from common.definitions import idx_rho_u1, idx_rho_u2, idx_rho_w, idx_rho, idx_rho_theta, gravity, p0, Rd, cpd, cvd, heat_capacity_ratio
from common.cuda_module import CudaModule, DimSpec, Dim, cuda_kernel
from init.dcmip import dcmip_schar_damping

# For type hints
from common.parallel import DistributedWorld
from geometry import CubedSphere, DFROperators, Metric3DTopo

from numpy.typing import NDArray


class RHSEuler(CudaModule,
               path="rhs_euler.cu",
               defines=(("heat_capacity_ratio", heat_capacity_ratio),
                        ("idx_rho", idx_rho),
                        ("idx_rho_u1", idx_rho_u1),
                        ("idx_rho_u2", idx_rho_u2),
                        ("idx_rho_w", idx_rho_w))):
    
    @cuda_kernel(DimSpec.groupby_first_x(lambda s: Dim(s[3], s[2], s[1])))
    def compute_flux_k(flux_x3_itf_k: NDArray[cp.float64], wflux_adv_x3_itf_k: NDArray[cp.float64], wflux_pres_x3_itf_k: NDArray[cp.float64],
                       variables_itf_k: NDArray[cp.float64], pressure_itf_k: NDArray[cp.float64], w_itf_k: NDArray[cp.float64],
                       sqrtG_itf_k: NDArray[cp.float64], H_contra_3x_itf_k: NDArray[cp.float64],
                       nv: int, nk_nh: int, ne: int, advection_only: bool): ...
