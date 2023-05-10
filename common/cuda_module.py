"""
Provide a smooth interface between .cu files and python code.
"""

import os
import inspect
import warnings
from dataclasses import dataclass, field
import cupy as cp

from os import PathLike
from typing import Callable, Iterable, Iterator, TypeVarTuple, Generic, Self

CUDA_DEVICE_COUNT: int = cp.cuda.runtime.getDeviceCount()
CUDA_THREAD_LIMIT      = tuple[int, ...](cp.cuda.runtime.getDeviceProperties(i)["maxThreadsPerBlock"] for i in range(CUDA_DEVICE_COUNT))
CUDA_BLOCK_SHAPE       = tuple[tuple[int, int, int], ...](cp.cuda.runtime.getDeviceProperties(i)["maxThreadsDim"] for i in range(CUDA_DEVICE_COUNT))
CUDA_GRID_SHAPE        = tuple[tuple[int, int, int], ...](cp.cuda.runtime.getDeviceProperties(i)["maxGridSize"] for i in range(CUDA_DEVICE_COUNT))


@dataclass(slots=True)
class Dim:

    x: int = field(default=1)
    y: int = field(default=1)
    z: int = field(default=1)

    @property
    def tuple(self) -> tuple[int, int, int]:
        return (self.x, self.y, self.z)
    
    def __iter__(self) -> Iterator[int]:
        def iterator():
            yield self.x
            yield self.y
            yield self.z
        return iterator()
    
    def __getitem__(self, idx: int) -> int:
        return getattr(self, ("x", "y", "z")[idx])
    
    def __setitem__(self, idx: int, value: int):
        setattr(self, ("x", "y", "z")[idx], value)

KernelArgs = TypeVarTuple("KernelArgs")
DimFun     = Callable[[*KernelArgs], tuple[Dim, Dim]]
RawKernel  = Callable[[Dim, Dim, *KernelArgs], None]


class CudaModule(object):

    @classmethod
    def __init_subclass__(cls, /,
                          path: PathLike,
                          name_expressions: Iterable[str] | None = None,
                          cpp_standard: str | None = None,
                          path_lexically_relative: bool = True,
                          **kwargs):
        super().__init_subclass__(**kwargs)

        if path_lexically_relative and not os.path.isabs(path):
            caller = inspect.stack()[0]
            dirname = os.path.dirname(caller.filename)
            path = os.path.join(dirname, path)

        with open(path) as file:
            code = file.read()
        
        options = (f"-std={cpp_standard}",) if cpp_standard else ()

        cls.module = cp.RawModule(code=code, name_expressions=name_expressions, options=options)

    @classmethod
    def get_function(cls: type[Self], name: str) -> RawKernel:
        return cls.module.get_function(name)


class CudaKernel(Generic[*KernelArgs]):

    def __init__(self: Self,
                 dimspec: DimFun[*KernelArgs],
                 f: RawKernel | None = None,
                 name: str | None = None):
        if callable(dimspec):
            self.dims = dimspec
        else:
            self.dims = lambda *_: dimspec
        self.__wrapped__ = f
        self.name = name

    def __set_name__(self, owner: CudaModule, name: str):
        self.name = name

    def __call__(self: Self, *args: *KernelArgs) -> None:
        if self.__wrapped__:
            gridspec, blockspec = self.dims(*args)
            self.__wrapped__(gridspec.tuple, blockspec.tuple, args)
        else:
            warnings.warn(f"")
        
    def __get__(self: Self, instance: CudaModule | None, owner: type[CudaModule] | None = None) -> Self:
        if not self.__wrapped__ and self.name:
            obj = instance if instance else owner
            self.__wrapped__ = obj.get_function(self.name)
        return self

    
class DimSpec:

    x, y, z = 0, 1, 2
    
    @classmethod
    def groupby(cls: type[Self], dim: int, arg: int = 0, shape: Callable[[tuple[int, ...]], Dim] = lambda x: Dim(*x)) -> DimFun[*KernelArgs]:
        def ret(*args: *KernelArgs) -> tuple[Dim, Dim]:
            s = shape(args[arg].shape)
            dev: int = cp.cuda.runtime.getDevice()
            thread_lim = min(CUDA_THREAD_LIMIT[dev], CUDA_BLOCK_SHAPE[dev][dim])
            gridspec, blockspec = list(s), [1] * 3
            blocks = (lambda d: d[0] + (1 if d[1] else 0))(divmod(s[dim], thread_lim))
            gridspec[dim] = blocks
            gridspec = Dim(*gridspec)
            if blocks > CUDA_GRID_SHAPE[dev][dim]:
                raise NotImplementedError(f"Grid {gridspec} too large for CUDA device {dev}")
            blockspec[dim] = thread_lim
            return gridspec, Dim(*blockspec)
        return ret
    
    @classmethod
    def groupby_first_x(cls: type[Self], shape: Callable[[tuple[int, ...]], Dim] = lambda x: Dim(*x)) -> DimFun[*KernelArgs]:
        return cls.groupby(cls.x, 0, shape)


def cuda_kernel(dimspec: DimFun[*KernelArgs]) -> Callable[[RawKernel[*KernelArgs]], CudaKernel[*KernelArgs]]:
    return lambda _: CudaKernel(dimspec, None)
