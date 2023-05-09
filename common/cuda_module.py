"""
Provide a smooth interface between .cu files and python code.
"""

import os
import inspect
import cupy as cp

from os import PathLike
from typing import Callable, Iterable, TypeVarTuple, Generic, Self

KernelArgs = TypeVarTuple("KernelArgs")
DimSpec    = int | tuple[int] | tuple[int, int] | tuple[int, int, int]
DimSpecs   = tuple[DimSpec, DimSpec]
DimFun     = Callable[[*KernelArgs], DimSpecs]
RawKernel  = Callable[[DimSpec, DimSpec, *KernelArgs], None]

CUDA_DEVICE_COUNT = cp.cuda.runtime.getDeviceCount()
CUDA_THREAD_LIMIT = tuple(cp.cuda.runtime.getDeviceProperties(i)["maxThreadsPerBlock"] for i in range(CUDA_DEVICE_COUNT))
CUDA_BLOCK_SHAPE  = tuple(cp.cuda.runtime.getDeviceProperties(i)["maxThreadsDim"] for i in range(CUDA_DEVICE_COUNT))
CUDA_GRID_SHAPE   = tuple(cp.cuda.runtime.getDeviceProperties(i)["maxGridSize"] for i in range(CUDA_DEVICE_COUNT))

class CudaKernel(Generic[*KernelArgs]):

    def __init__(self: Self,
                 dimspec: DimSpecs | DimFun[*KernelArgs],
                 f: RawKernel | None = None):
        if callable(dimspec):
            self.dims = dimspec
        else:
            self.dims = lambda *_: dimspec
        self.__wrapped__ = f

    def __call__(self: Self, *args: *KernelArgs):
        if self.__wrapped__:
            gridspec, blockspec = self.dims(*args)
            self.__wrapped__(gridspec, blockspec, args)
        else:
            raise AttributeError

def cuda_kernel(dimspec: DimSpecs | DimFun[*KernelArgs]) -> Callable[[RawKernel[*KernelArgs]], CudaKernel]:
    return lambda _: CudaKernel(dimspec, None)

class CudaModule(object):

    def __init__(self: Self,
                 path: PathLike,
                 name_expressions: Iterable[str] = (),
                 cpp_standard: str | None = None,
                 path_lexically_relative: bool = True):
        super().__init__()

        if path_lexically_relative and not os.path.isabs(path):
            caller = inspect.stack()[0]
            dirname = os.path.dirname(caller.filename)
            path = os.path.join(dirname, path)

        with open(path) as file:
            code = file.read()
        
        options = (f"-std={cpp_standard}",) if cpp_standard else ()

        self.module = cp.RawModule(code=code, name_expressions=name_expressions, options=options)

    def __getattr__(self: Self, name: str) -> Callable[..., None]:
        try:
            f = self.module.get_function(name)
            setattr(self, name, f)
            return f
        except:
            raise
