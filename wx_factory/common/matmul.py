"""Small matrix-multiplication helpers built directly on torch.

``apply_op`` applies a (small, dense) operator matrix to the last dimension of an array, which is
the workhorse of the DFR spatial discretisation (extrapolation, derivative and boundary-correction
operators). It supports the ``alpha``/``beta``/``out`` accumulation pattern so callers can fuse a
sequence of operator applications into a single output buffer.
"""

import torch


def kron(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Kronecker product. ``torch.kron`` requires contiguous operands."""
    return torch.kron(a.contiguous(), b.contiguous())


def _matmul(a, b, alpha=1.0, beta=0.0, out=None):
    """Compute ``out = alpha * (a @ b) + beta * out`` (or just ``alpha * (a @ b)`` when out is None)."""
    if out is None:
        result = torch.matmul(a, b)
        if alpha != 1.0:
            result *= alpha
        return result

    if beta == 0.0:
        torch.matmul(a, b, out=out)
        if alpha != 1.0:
            out *= alpha
    elif alpha == 1.0 and beta == 1.0:
        out += torch.matmul(a, b)
    else:
        out *= beta
        out += alpha * torch.matmul(a, b)
    return out


def apply_op(a, b, alpha: float = 1.0, beta: float = 0.0, out=None):
    """Apply operator ``b`` to the last dimension of ``a``: ``out = alpha * (a @ b) + beta * out``.

    ``a`` is flattened to 2D over its leading dimensions, multiplied by ``b`` and reshaped back. When
    ``out`` is given the product is accumulated into it in place (its reshape is a view of the same
    storage), which is what lets the RHS reuse a single buffer across successive operator applications.
    """
    sh = a.shape
    a = a.reshape(-1, sh[-1])

    if out is not None:
        out_reshaped = out.reshape(-1, b.shape[-1])
        _matmul(a, b, alpha=alpha, beta=beta, out=out_reshaped)
        return out

    result = _matmul(a, b, alpha=alpha, beta=beta)
    return result.reshape(*sh[:-1], -1)


def _column_stencil(operator):
    """Retain the exact nonzero coefficients and source indices for each output point."""
    matrix = operator.cpu()
    columns = [torch.nonzero(column, as_tuple=True)[0] for column in matrix.T]
    width = max(len(rows) for rows in columns)
    indices = torch.zeros((matrix.shape[1], width), dtype=torch.long, device="cpu")
    weights = torch.zeros((matrix.shape[1], width), dtype=matrix.dtype, device="cpu")
    for point, rows in enumerate(columns):
        indices[point, : len(rows)] = rows
        weights[point, : len(rows)] = matrix[rows, point]
    return indices.to(operator.device), weights.to(operator.device)


def sparse_dfr_derivative(volume, faces, indices, weights, face_indices, face_weights):
    volume_derivative = (volume[..., indices] * weights).sum(dim=-1)
    face_correction = (faces[..., face_indices] * face_weights).sum(dim=-1)
    return volume_derivative + face_correction


class DFRDerivative:
    """Apply a volume derivative and its common-face correction."""

    def __init__(self, volume_operator, face_operator, *, action):
        self.volume_operator = volume_operator
        self.face_operator = face_operator
        self.action = action
        self.indices, self.weights = _column_stencil(volume_operator)
        self.face_indices, self.face_weights = _column_stencil(face_operator)

    def dense(self, volume, faces):
        return apply_op(volume, self.volume_operator) + apply_op(faces, self.face_operator)

    def __call__(self, volume, faces):
        return self.action(self, volume, faces)
