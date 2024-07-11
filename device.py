from abc    import ABC, abstractmethod
from typing import Any, List

from mpi4py import MPI
import numpy
from numpy.typing import NDArray

class Device(ABC):
   def __init__(self, xp, expm) -> None:
      self.xp = xp
      self.expm = expm

   @abstractmethod
   def __synchronize__(self, **kwargs):
      pass

   def synchronize(self, **kwargs):
      self.__synchronize__(**kwargs)

   @abstractmethod
   def __array__(self, a: NDArray, **kwargs) -> NDArray:
      pass

   def array(self, a: NDArray, *args, **kwargs) -> NDArray:
      return self.__array__(a, *args, **kwargs)

   @abstractmethod
   def __pinned__(self, *args, **kwargs) -> NDArray:
      pass

   def pinned(self, *args, **kwargs) -> NDArray:
      return self.__pinned__(*args, **kwargs)

   @abstractmethod
   def __to_host__(self, val, **kwargs):
      pass

   def to_host(self, val: Any, **kwargs) -> Any:
      return self.__to_host__(val, **kwargs)

class CpuDevice(Device):
   def __init__(self) -> None:
      from scipy.linalg import expm
      super().__init__(numpy, expm)
   
   def __synchronize__(self, **kwargs):
      pass

   def __array__(self, a: NDArray, *args, **kwargs) -> NDArray:
      return a

   def __pinned__(self, *args, **kwargs) -> NDArray:
      return self.xp.empty(*args, **kwargs)

   def __to_host__(self, val, **kwargs):
      return val

class CudaDevice(Device):

   def __init__(self, device_list: List[int]) -> None:
      # Delay imports, to avoid loading CUDA if not asked

      import cupy
      from wx_cupy import num_devices, expm
      super().__init__(cupy, expm)

      if num_devices <= 0:
         raise ValueError(f'Unable to create a CudaDevice object, no GPU devices were detected')

      import cupyx
      self.cupyx = cupyx
      self.cupy = cupy

      rank = MPI.COMM_WORLD.Get_rank()
      devnum = rank % len(device_list)
      cupy.cuda.Device(device_list[devnum]).use()

      # TODO don't use managed memory
      cupy.cuda.set_allocator(cupy.cuda.MemoryPool(cupy.cuda.malloc_managed).malloc)

      self.main_stream = cupy.cuda.get_current_stream()
      self.copy_stream = cupy.cuda.Stream(non_blocking=True)

   def __synchronize__(self, **kwargs):
      if 'copy_stream' in kwargs and kwargs['copy_stream']:
         self.copy_stream.synchronize()
      else:
         self.main_stream.synchronize()

   def __array__(self, a: NDArray, *args, **kwargs) -> NDArray:
      return self.xp.asarray(a)

   def __pinned__(self, *args, **kwargs) -> NDArray:
      return self.cupyx.empty_pinned(*args, **kwargs)

   def __to_host__(self, val, **kwargs):
      return val.get(**kwargs)
