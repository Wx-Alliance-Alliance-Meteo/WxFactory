import cupy as cp

from .cubed_sphere import CubedSphere
from .cuda_matrices import CudaDFROperators
from .metric import Metric3DTopo

# typing
from typing import Self
from numpy.typing import NDArray


class CudaMetric3DTopo(Metric3DTopo):

    def __init__(self: Self, geom: CubedSphere, matrix: CudaDFROperators):
        super().__init__(geom, matrix)
    
    def build_metric(self: Self):
        super().build_metric()

        # move results to gpu
        # TODO: perform computations on GPU

        self.christoffel_1_01: NDArray[cp.float64] = cp.asarray(self.christoffel_1_01)
        self.christoffel_1_02: NDArray[cp.float64] = cp.asarray(self.christoffel_1_02)
        self.christoffel_1_03: NDArray[cp.float64] = cp.asarray(self.christoffel_1_03)
        self.christoffel_1_11: NDArray[cp.float64] = cp.asarray(self.christoffel_1_11)
        self.christoffel_1_12: NDArray[cp.float64] = cp.asarray(self.christoffel_1_12)
        self.christoffel_1_13: NDArray[cp.float64] = cp.asarray(self.christoffel_1_13)
        self.christoffel_1_22: NDArray[cp.float64] = cp.asarray(self.christoffel_1_22)
        self.christoffel_1_23: NDArray[cp.float64] = cp.asarray(self.christoffel_1_23)
        self.christoffel_1_33: NDArray[cp.float64] = cp.asarray(self.christoffel_1_33)

        self.christoffel_2_01: NDArray[cp.float64] = cp.asarray(self.christoffel_2_01)
        self.christoffel_2_02: NDArray[cp.float64] = cp.asarray(self.christoffel_2_02)
        self.christoffel_2_03: NDArray[cp.float64] = cp.asarray(self.christoffel_2_03)
        self.christoffel_2_11: NDArray[cp.float64] = cp.asarray(self.christoffel_2_11)
        self.christoffel_2_12: NDArray[cp.float64] = cp.asarray(self.christoffel_2_12)
        self.christoffel_2_13: NDArray[cp.float64] = cp.asarray(self.christoffel_2_13)
        self.christoffel_2_22: NDArray[cp.float64] = cp.asarray(self.christoffel_2_22)
        self.christoffel_2_23: NDArray[cp.float64] = cp.asarray(self.christoffel_2_23)
        self.christoffel_2_33: NDArray[cp.float64] = cp.asarray(self.christoffel_2_33)

        self.christoffel_3_01: NDArray[cp.float64] = cp.asarray(self.christoffel_3_01)
        self.christoffel_3_02: NDArray[cp.float64] = cp.asarray(self.christoffel_3_02)
        self.christoffel_3_03: NDArray[cp.float64] = cp.asarray(self.christoffel_3_03)
        self.christoffel_3_11: NDArray[cp.float64] = cp.asarray(self.christoffel_3_11)
        self.christoffel_3_12: NDArray[cp.float64] = cp.asarray(self.christoffel_3_12)
        self.christoffel_3_13: NDArray[cp.float64] = cp.asarray(self.christoffel_3_13)
        self.christoffel_3_22: NDArray[cp.float64] = cp.asarray(self.christoffel_3_22)
        self.christoffel_3_23: NDArray[cp.float64] = cp.asarray(self.christoffel_3_23)
        self.christoffel_3_33: NDArray[cp.float64] = cp.asarray(self.christoffel_3_33)

        self.H_cov: NDArray[cp.float64] = cp.asarray(self.H_cov)
        self.H_cov_11: NDArray[cp.float64] = self.H_cov[0,0,:,:,:]
        self.H_cov_12: NDArray[cp.float64] = self.H_cov[0,1,:,:,:]
        self.H_cov_13: NDArray[cp.float64] = self.H_cov[0,2,:,:,:]
        self.H_cov_21: NDArray[cp.float64] = self.H_cov[1,0,:,:,:]
        self.H_cov_22: NDArray[cp.float64] = self.H_cov[1,1,:,:,:]
        self.H_cov_23: NDArray[cp.float64] = self.H_cov[1,2,:,:,:]
        self.H_cov_31: NDArray[cp.float64] = self.H_cov[2,0,:,:,:]
        self.H_cov_32: NDArray[cp.float64] = self.H_cov[2,1,:,:,:]
        self.H_cov_33: NDArray[cp.float64] = self.H_cov[2,2,:,:,:]

        self.H_cov_itf_i: NDArray[cp.float64] = cp.asarray(self.H_cov_itf_i)
        self.H_cov_11_itf_i: NDArray[cp.float64] = self.H_cov_itf_i[0,0,:,:,:]
        self.H_cov_12_itf_i: NDArray[cp.float64] = self.H_cov_itf_i[0,1,:,:,:]
        self.H_cov_13_itf_i: NDArray[cp.float64] = self.H_cov_itf_i[0,2,:,:,:]
        self.H_cov_21_itf_i: NDArray[cp.float64] = self.H_cov_itf_i[1,0,:,:,:]
        self.H_cov_22_itf_i: NDArray[cp.float64] = self.H_cov_itf_i[1,1,:,:,:]
        self.H_cov_23_itf_i: NDArray[cp.float64] = self.H_cov_itf_i[1,2,:,:,:]
        self.H_cov_31_itf_i: NDArray[cp.float64] = self.H_cov_itf_i[2,0,:,:,:]
        self.H_cov_32_itf_i: NDArray[cp.float64] = self.H_cov_itf_i[2,1,:,:,:]
        self.H_cov_33_itf_i: NDArray[cp.float64] = self.H_cov_itf_i[2,2,:,:,:]

        self.H_cov_itf_j: NDArray[cp.float64] = cp.asarray(self.H_cov_itf_j)
        self.H_cov_11_itf_j: NDArray[cp.float64] = self.H_cov_itf_j[0,0,:,:,:]
        self.H_cov_12_itf_j: NDArray[cp.float64] = self.H_cov_itf_j[0,1,:,:,:]
        self.H_cov_13_itf_j: NDArray[cp.float64] = self.H_cov_itf_j[0,2,:,:,:]
        self.H_cov_21_itf_j: NDArray[cp.float64] = self.H_cov_itf_j[1,0,:,:,:]
        self.H_cov_22_itf_j: NDArray[cp.float64] = self.H_cov_itf_j[1,1,:,:,:]
        self.H_cov_23_itf_j: NDArray[cp.float64] = self.H_cov_itf_j[1,2,:,:,:]
        self.H_cov_31_itf_j: NDArray[cp.float64] = self.H_cov_itf_j[2,0,:,:,:]
        self.H_cov_32_itf_j: NDArray[cp.float64] = self.H_cov_itf_j[2,1,:,:,:]
        self.H_cov_33_itf_j: NDArray[cp.float64] = self.H_cov_itf_j[2,2,:,:,:]

        self.H_cov_itf_k: NDArray[cp.float64] = cp.asarray(self.H_cov_itf_k)
        self.H_cov_11_itf_k: NDArray[cp.float64] = self.H_cov_itf_k[0,0,:,:,:]
        self.H_cov_12_itf_k: NDArray[cp.float64] = self.H_cov_itf_k[0,1,:,:,:]
        self.H_cov_13_itf_k: NDArray[cp.float64] = self.H_cov_itf_k[0,2,:,:,:]
        self.H_cov_21_itf_k: NDArray[cp.float64] = self.H_cov_itf_k[1,0,:,:,:]
        self.H_cov_22_itf_k: NDArray[cp.float64] = self.H_cov_itf_k[1,1,:,:,:]
        self.H_cov_23_itf_k: NDArray[cp.float64] = self.H_cov_itf_k[1,2,:,:,:]
        self.H_cov_31_itf_k: NDArray[cp.float64] = self.H_cov_itf_k[2,0,:,:,:]
        self.H_cov_32_itf_k: NDArray[cp.float64] = self.H_cov_itf_k[2,1,:,:,:]
        self.H_cov_33_itf_k: NDArray[cp.float64] = self.H_cov_itf_k[2,2,:,:,:]

        self.H_contra: NDArray[cp.float64] = cp.asarray(self.H_contra)
        self.H_contra_11: NDArray[cp.float64] = self.H_contra[0,0,:,:,:]
        self.H_contra_12: NDArray[cp.float64] = self.H_contra[0,1,:,:,:]
        self.H_contra_13: NDArray[cp.float64] = self.H_contra[0,2,:,:,:]
        self.H_contra_21: NDArray[cp.float64] = self.H_contra[1,0,:,:,:]
        self.H_contra_22: NDArray[cp.float64] = self.H_contra[1,1,:,:,:]
        self.H_contra_23: NDArray[cp.float64] = self.H_contra[1,2,:,:,:]
        self.H_contra_31: NDArray[cp.float64] = self.H_contra[2,0,:,:,:]
        self.H_contra_32: NDArray[cp.float64] = self.H_contra[2,1,:,:,:]
        self.H_contra_33: NDArray[cp.float64] = self.H_contra[2,2,:,:,:]

        self.H_contra_itf_i: NDArray[cp.float64] = cp.asarray(self.H_contra_itf_i)
        self.H_contra_11_itf_i: NDArray[cp.float64] = self.H_contra_itf_i[0,0,:,:,:]
        self.H_contra_12_itf_i: NDArray[cp.float64] = self.H_contra_itf_i[0,1,:,:,:]
        self.H_contra_13_itf_i: NDArray[cp.float64] = self.H_contra_itf_i[0,2,:,:,:]
        self.H_contra_21_itf_i: NDArray[cp.float64] = self.H_contra_itf_i[1,0,:,:,:]
        self.H_contra_22_itf_i: NDArray[cp.float64] = self.H_contra_itf_i[1,1,:,:,:]
        self.H_contra_23_itf_i: NDArray[cp.float64] = self.H_contra_itf_i[1,2,:,:,:]
        self.H_contra_31_itf_i: NDArray[cp.float64] = self.H_contra_itf_i[2,0,:,:,:]
        self.H_contra_32_itf_i: NDArray[cp.float64] = self.H_contra_itf_i[2,1,:,:,:]
        self.H_contra_33_itf_i: NDArray[cp.float64] = self.H_contra_itf_i[2,2,:,:,:]

        self.H_contra_itf_j: NDArray[cp.float64] = cp.asarray(self.H_contra_itf_j)
        self.H_contra_11_itf_j: NDArray[cp.float64] = self.H_contra_itf_j[0,0,:,:,:]
        self.H_contra_12_itf_j: NDArray[cp.float64] = self.H_contra_itf_j[0,1,:,:,:]
        self.H_contra_13_itf_j: NDArray[cp.float64] = self.H_contra_itf_j[0,2,:,:,:]
        self.H_contra_21_itf_j: NDArray[cp.float64] = self.H_contra_itf_j[1,0,:,:,:]
        self.H_contra_22_itf_j: NDArray[cp.float64] = self.H_contra_itf_j[1,1,:,:,:]
        self.H_contra_23_itf_j: NDArray[cp.float64] = self.H_contra_itf_j[1,2,:,:,:]
        self.H_contra_31_itf_j: NDArray[cp.float64] = self.H_contra_itf_j[2,0,:,:,:]
        self.H_contra_32_itf_j: NDArray[cp.float64] = self.H_contra_itf_j[2,1,:,:,:]
        self.H_contra_33_itf_j: NDArray[cp.float64] = self.H_contra_itf_j[2,2,:,:,:]

        self.H_contra_itf_k: NDArray[cp.float64] = cp.asarray(self.H_contra_itf_k)
        self.H_contra_11_itf_k: NDArray[cp.float64] = self.H_contra_itf_k[0,0,:,:,:]
        self.H_contra_12_itf_k: NDArray[cp.float64] = self.H_contra_itf_k[0,1,:,:,:]
        self.H_contra_13_itf_k: NDArray[cp.float64] = self.H_contra_itf_k[0,2,:,:,:]
        self.H_contra_21_itf_k: NDArray[cp.float64] = self.H_contra_itf_k[1,0,:,:,:]
        self.H_contra_22_itf_k: NDArray[cp.float64] = self.H_contra_itf_k[1,1,:,:,:]
        self.H_contra_23_itf_k: NDArray[cp.float64] = self.H_contra_itf_k[1,2,:,:,:]
        self.H_contra_31_itf_k: NDArray[cp.float64] = self.H_contra_itf_k[2,0,:,:,:]
        self.H_contra_32_itf_k: NDArray[cp.float64] = self.H_contra_itf_k[2,1,:,:,:]
        self.H_contra_33_itf_k: NDArray[cp.float64] = self.H_contra_itf_k[2,2,:,:,:]

        self.sqrtG: NDArray[cp.float64] = cp.asarray(self.sqrtG)
        self.sqrtG_itf_i: NDArray[cp.float64] = cp.asarray(self.sqrtG_itf_i)
        self.sqrtG_itf_j: NDArray[cp.float64] = cp.asarray(self.sqrtG_itf_j)
        self.sqrtG_itf_k: NDArray[cp.float64] = cp.asarray(self.sqrtG_itf_k)
        self.inv_sqrtG: NDArray[cp.float64] = cp.asarray(self.inv_sqrtG)

        self.coriolis_f: NDArray[cp.float64] = cp.asarray(self.coriolis_f)

        self.inv_dzdeta: NDArray[cp.float64] = cp.asarray(self.inv_dzdeta)
        