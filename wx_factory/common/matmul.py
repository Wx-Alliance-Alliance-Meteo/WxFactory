"""Small matrix-multiplication helpers built directly on torch.

``apply_op`` applies a (small, dense) operator matrix to the last dimension of an array, which is
the workhorse of the DFR spatial discretisation (extrapolation, derivative and boundary-correction
operators). It supports the ``alpha``/``beta``/``out`` accumulation pattern so callers can fuse a
sequence of operator applications into a single output buffer.
"""

import torch

from ..device import differentiable_mode

# PyTorch has no forward-AD rule for matmul with ``out=``; differentiable mode uses a temporary.
_MATMUL_NEEDS_OUT_FALLBACK = differentiable_mode()


def kron(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Kronecker product. ``torch.kron`` requires contiguous operands."""
    return torch.kron(a.contiguous(), b.contiguous())


def maximum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise maximum that also works for complex tensors.

    ``torch.maximum`` is undefined for complex tensors, but the complex-step Jacobian
    (``jacobian_method = complex``) pushes complex perturbations through the flux eigenvalue
    estimate. Complex values are ordered lexicographically (real part, then imaginary part), which
    matches how the differentiated wave-speed bound must behave. Real tensors take the fast path.
    """
    if not torch.is_complex(a):
        return torch.maximum(a, b)

    mask = (a.real > b.real) | ((a.real == b.real) & (a.imag > b.imag))
    return torch.where(mask, a, b)


def _matmul(a, b, alpha=1.0, beta=0.0, out=None):
    """Compute ``out = alpha * (a @ b) + beta * out`` (or just ``alpha * (a @ b)`` when out is None)."""
    if out is None:
        result = torch.matmul(a, b)
        if alpha != 1.0:
            result *= alpha
        return result

    if _MATMUL_NEEDS_OUT_FALLBACK:
        prod = torch.matmul(a, b)
        if beta == 0.0:
            out.copy_(prod if alpha == 1.0 else prod * alpha)
        elif alpha == 1.0 and beta == 1.0:
            out.add_(prod)
        else:
            out.mul_(beta).add_(alpha * prod)
        return out

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
