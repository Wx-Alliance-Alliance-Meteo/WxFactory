"""Spatial contractions with leading component axes.

Array indices 0, 1, 2 correspond to 5D spatial indices 1, 2, 3.
Explicit formulas avoid pointwise batched matrix-vector products on the GPU.
"""

import torch
from torch import Tensor


def spatial_matrix_vector(matrix: Tensor, vector: Tensor) -> Tensor:
    """Return M_ij x_j for a 3x3 matrix at each trailing grid point."""
    return torch.stack(
        (
            matrix[0, 0] * vector[0] + matrix[0, 1] * vector[1] + matrix[0, 2] * vector[2],
            matrix[1, 0] * vector[0] + matrix[1, 1] * vector[1] + matrix[1, 2] * vector[2],
            matrix[2, 0] * vector[0] + matrix[2, 1] * vector[1] + matrix[2, 2] * vector[2],
        )
    )


def antisymmetric_velocity_contraction(tensor: Tensor, velocity: Tensor) -> Tensor:
    """Return u^j A_ij, assuming A_ji=-A_ij and A_ii=0."""
    a12, a13, a23 = tensor[0, 1], tensor[0, 2], tensor[1, 2]
    return torch.stack(
        (
            velocity[1] * a12 + velocity[2] * a13,
            -velocity[0] * a12 + velocity[2] * a23,
            -velocity[0] * a13 - velocity[1] * a23,
        )
    )


def relative_covariant_curl(derivatives: tuple[Tensor, ...]) -> Tensor:
    """Return v_{j,i} - v_{i,j} from the six off-diagonal velocity derivatives."""
    dv2_dx1, dv3_dx1, dv1_dx2, dv3_dx2, dv1_dx3, dv2_dx3 = derivatives
    curl_12 = dv2_dx1 - dv1_dx2
    curl_13 = dv3_dx1 - dv1_dx3
    curl_23 = dv3_dx2 - dv2_dx3
    zero = torch.zeros_like(curl_12)
    return torch.stack((zero, curl_12, curl_13, -curl_12, zero, curl_23, -curl_13, -curl_23, zero)).reshape(
        (3, 3) + curl_12.shape
    )
