from typing import List

from mpi4py import MPI

from gef_cuda import num_devices

class Device:
   def __init__(self) -> None:
      pass

class CpuDevice(Device):
   def __init__(self) -> None:
      super().__init__()

class CudaDevice(Device):
   def __init__(self, device_list: List[int]) -> None:
      super().__init__()

      if num_devices <= 0:
         raise ValueError(f'Unable to create a CudaDevice object, no GPU devices were detected')

      import cupy

      rank = MPI.COMM_WORLD.Get_rank()
      devnum = rank % len(device_list)
      cupy.cuda.Device(device_list[devnum]).use()

      # TODO don't use managed memory
      cupy.cuda.set_allocator(cupy.cuda.MemoryPool(cupy.cuda.malloc_managed).malloc)
