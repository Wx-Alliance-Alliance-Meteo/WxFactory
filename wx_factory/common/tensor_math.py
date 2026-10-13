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
