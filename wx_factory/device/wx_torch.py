import torch
import numpy
import scipy
import typing

torch.Tensor.copy = torch.Tensor.clone
torch.Tensor.astype = torch.Tensor.to


def _to_host(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else numpy.asarray(value)


class _TorchLinalg:
    """The dense linear algebra the exponential solvers need, for torch tensors.

    Both operations act on the Krylov Hessenberg matrix, which is at most (mmax + 1) square, so
    running them on the host with scipy costs nothing measurable even when the tensors live on a
    GPU. The results come back as tensors, on whichever device torch is currently using.
    """

    def solve_triangular(self, a, b, **kwargs) -> torch.Tensor:
        return torch.as_tensor(scipy.linalg.solve_triangular(_to_host(a), _to_host(b), **kwargs))

    def expm(self, a) -> torch.Tensor:
        return torch.as_tensor(scipy.linalg.expm(_to_host(a)))


class TorchAlg:
    """Stand-in for scipy, for tensors that may live on a GPU."""

    def __init__(self):
        self.linalg = _TorchLinalg()


class TorchXp:
    s_ = numpy.s_

    def __init__(self, device=None):
        self.device = device

    def array(self, value, **kwargs) -> torch.Tensor:
        return torch.tensor(value, **kwargs, device=self.device)

    def append(self, *args, **kwargs) -> torch.Tensor:
        return torch.cat(args, **kwargs)

    def repeat(self, *args, **kwargs) -> torch.Tensor:
        return torch.repeat_interleave(*args, **kwargs)

    def kron(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return torch.kron(
            first if first.is_contiguous() else first.contiguous(),
            second if second.is_contiguous() else second.contiguous(),
        )

    def transpose(
        self, array: torch.Tensor, axis: typing.Optional[typing.List[int] | typing.Tuple[int, ...]] = None
    ) -> torch.Tensor:
        if axis is None or len(axis) == 0:
            axis = list(reversed(range(len(array.shape))))
        return array.permute(axis)

    def identity(self, dim: int, **kwargs) -> torch.Tensor:
        return torch.eye(dim, dim, **kwargs)

    def flip(self, array: torch.Tensor, axis: typing.Tuple[int, ...]):
        return torch.flip(array, axis)

    def iscomplexobj(self, array: torch.Tensor) -> bool:
        return array.is_complex()

    def _promote(self, value, like: torch.Tensor) -> torch.Tensor:
        """torch.minimum/maximum only take tensors, where numpy happily takes a scalar."""
        if isinstance(value, torch.Tensor):
            return value
        return torch.as_tensor(value, dtype=like.dtype, device=like.device)

    def minimum(self, a, b) -> torch.Tensor:
        reference = a if isinstance(a, torch.Tensor) else b
        return torch.minimum(self._promote(a, reference), self._promote(b, reference))

    def maximum(self, a, b) -> torch.Tensor:
        reference = a if isinstance(a, torch.Tensor) else b
        a = self._promote(a, reference)
        b = self._promote(b, reference)

        if not self.iscomplexobj(a):
            return torch.maximum(a, b)

        real_a = a.real
        imag_a = a.imag
        real_b = b.real
        imag_b = b.imag

        real_mask = real_a > real_b
        equals_mask = real_a == real_b
        imag_mask = imag_a > imag_b

        mask = real_mask | (equals_mask & imag_mask)

        return torch.where(mask, a, b)

    def __getattr__(self, name):
        return getattr(torch, name)
