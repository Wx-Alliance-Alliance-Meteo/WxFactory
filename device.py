from abc    import ABC, abstractmethod
from typing import List

from mpi4py import MPI
import numpy
from numpy.typing import NDArray

class Device(ABC):
   def __init__(self, xp) -> None:
      self.xp = xp

   @abstractmethod
   def __synchronize__(self):
      pass

   def synchronize(self):
      self.__synchronize__()

   @abstractmethod
   def __array__(self, a: NDArray):
      pass

   def array(self, a: NDArray):
      return self.__array__(a)

class CpuDevice(Device):
   def __init__(self) -> None:
      super().__init__(numpy)
   
   def __synchronize__(self):
      pass

   def __array__(self, a: NDArray):
      return a

class CudaDevice(Device):

   def __init__(self, device_list: List[int]) -> None:
      # Delay imports, to avoid loading CUDA if not asked

      import cupy
      super().__init__(cupy)

      from gef_cuda import num_devices

      if num_devices <= 0:
         raise ValueError(f'Unable to create a CudaDevice object, no GPU devices were detected')


      rank = MPI.COMM_WORLD.Get_rank()
      devnum = rank % len(device_list)
      cupy.cuda.Device(device_list[devnum]).use()

      # TODO don't use managed memory
      cupy.cuda.set_allocator(cupy.cuda.MemoryPool(cupy.cuda.malloc_managed).malloc)

   def __synchronize__(self):
      self.xp.cuda.get_current_stream().synchronize()

   def __array__(self, a: NDArray):
      return self.xp.asarray(a)
