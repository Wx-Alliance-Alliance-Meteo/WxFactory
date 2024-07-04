
__all__ = ['cuda_avail', 'num_devices',
           'kiops_cuda', 'rhs_bubble_cuda', 'rhs_euler_cuda']

from mpi4py import MPI
import numpy


num_devices = 0
loading_error = None
try:
   import cupy

   num_devices = cupy.cuda.runtime.getDeviceCount()

   if num_devices <= 0:
      if MPI.COMM_WORLD.rank == 0: print(f'No cuda devices found')
      num_devices = 0

except ModuleNotFoundError as e:
   loading_error = e
   # if MPI.COMM_WORLD.rank == 0:
   #    print(f'cupy does not seem to be installed. '
   #          f'You need it (and GPUs) to be able run GEF with device=cuda.\n'
   #          f'We will run on CPU instead')

except ImportError as e:
   loading_error = e
   # if MPI.COMM_WORLD.rank == 0:
   #    print(f'Module cupy is installed, but we were unable to load it, so we will run on CPUs instead')

except Exception as e:
   loading_error = e

cuda_avail = num_devices > 0

if MPI.COMM_WORLD.rank == 0:
   avail = 'available' if cuda_avail else 'not available'
   print(f'CUDA is {avail}')

if cuda_avail:
   # import cuda-related modules
   from .kiops_cuda      import kiops_cuda
   from .rhs_bubble_cuda import rhs_bubble_cuda
   from .rhs_euler_cuda  import rhs_euler_cuda
else:
   # define stubs
   def dummy(*args, **kwargs):
      raise ValueError(f'No one should ever call me')

   kiops_cuda = dummy
   rhs_bubble_cuda = dummy
   rhs_euler_cuda = dummy
