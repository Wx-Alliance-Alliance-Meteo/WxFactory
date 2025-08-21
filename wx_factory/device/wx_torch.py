import torch
import numpy
import typing

torch.Tensor.copy = torch.Tensor.clone
torch.Tensor.astype = torch.Tensor.to

class TorchXp:
    s_ = numpy.s_

    def __init__(self):
        pass

    def array(self, value, **kwargs) -> torch.Tensor:
        return torch.tensor(value, **kwargs, device=None)
    
    def append(self, *args, **kwargs) -> torch.Tensor:
        return torch.cat(args, **kwargs)
    
    def repeat(self, *args, **kwargs) -> torch.Tensor:
        return torch.repeat_interleave(*args, **kwargs)
    
    def kron(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return torch.kron(
            first if first.is_contiguous() else first.contiguous(),
            second if second.is_contiguous() else second.contiguous()
        )
    
    def transpose(self, array: torch.Tensor, axis: typing.Optional[typing.List[int] | typing.Tuple[int, ...]] = None) -> torch.Tensor:
        if axis is None or len(axis) == 0:
            axis = list(reversed(range(len(array.shape))))
        return array.permute(axis)
    
    def identity(self, dim: int, **kwargs) -> torch.Tensor:
        return torch.eye(dim, dim, **kwargs)
    
    def flip(self, array: torch.Tensor, axis: int | typing.Tuple[int, ...]):
        return torch.flip(array, axis)
    
    def iscomplexobj(self, array: torch.Tensor) -> bool:
        return array.is_complex()
    
    def maximum(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
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
