from abc import ABC, abstractmethod
from typing import Any, List, Optional

from mpi4py import MPI
from numpy.typing import NDArray

from compiler.compile_utils import mpi_compile
from . import wx_cupy


class Device(ABC):
    """Description of a device on which code can be executed.

    The device is allowed to be the same as the host (so that code is executed on the host). In such
    a case, most operations do nothing

    :param comm: The MPI communicator associated with this device.
    :type comm: MPI.Comm
    :param xp: Basic math/array module for this device (numpy on the CPU, cupy or other on the GPU)
    :param xalg: Advanced math module for this device (scipy on the CPU, cupy equivalent on the GPU)
    :param libmodule: Module containing all the compiled code for this device
    """

    def __init__(self, comm: MPI.Comm, xp, xalg, libmodule) -> None:
        """Set a few modules and functions to have the same name, so that callers can use a single name."""
        self.comm = comm
        self.xp = xp
        self.xalg = xalg
        self.libmodule = libmodule

    @abstractmethod
    def __synchronize__(self, **kwargs):
        pass

    def synchronize(self, **kwargs):
        """Synchronize this device with the host. This is essentially a host-device barrier."""
        self.__synchronize__(**kwargs)

    @abstractmethod
    def __array__(self, a: NDArray, **kwargs) -> NDArray:
        pass

    def array(self, a: NDArray, *args, **kwargs) -> NDArray:
        """Copy the given array to this device, if it's not the same as the host."""
        return self.__array__(a, *args, **kwargs)

    @abstractmethod
    def __pinned__(self, *args, **kwargs) -> NDArray:
        pass

    def pinned(self, *args, **kwargs) -> NDArray:
        """Allocate a host array with pinned memory."""
        return self.__pinned__(*args, **kwargs)

    @abstractmethod
    def __to_host__(self, val, **kwargs):
        pass

    def to_host(self, val: Any, **kwargs) -> Any:
        """Copy the given array to the host (if it's not there already)."""
        return self.__to_host__(val, **kwargs)

    def has_128_bits_float(self) -> bool:
        """Not all devices can perform 128 bits floating point operation"""
        return hasattr(self.xp, "float128")


class CpuDevice(Device):
    def __init__(self, comm: MPI.Comm = MPI.COMM_WORLD) -> None:
        import numpy
        import scipy

        try:
            mpi_compile("cpp", force=False, comm=comm)
            from lib.pde import interface_c
        except (ModuleNotFoundError, SystemExit):
            if comm.rank == 0:
                print(f"Unable to find the interface_c module. You need to compile it.", flush=True)
            raise
        except:
            print(f"Unknown exception!", flush=True)
            raise

        super().__init__(comm, numpy, scipy, interface_c)

    def __synchronize__(self, **kwargs):
        """Don't do anything. This is to allow writing generic code when device is not the same as the host."""
        pass

    def __array__(self, a: NDArray, *args, **kwargs) -> NDArray:
        """Return the input array unchanged."""
        return a

    def __pinned__(self, *args, **kwargs) -> NDArray:
        """Return allocated space, without any special characteristic."""
        return self.xp.empty(*args, **kwargs)

    def __to_host__(self, val, **kwargs):
        """Return the input array unchanged."""
        return val


class CudaDevice(Device):

    def __init__(self, comm: MPI.Comm = MPI.COMM_WORLD, device_list: Optional[List[int]] = None) -> None:
        # Delay imports, to avoid loading CUDA if not asked
        import cupy
        import cupyx
        import cupyx.scipy.linalg

        try:
            mpi_compile("cuda", force=False, comm=comm)
            import lib.pde.interface_cuda as interface_cuda
        except (ModuleNotFoundError, SystemExit):
            if comm.rank == 0:
                print(
                    f"Unable to load the interface_cuda module, you need to compile it if you want to use the GPU",
                    flush=True,
                )
            raise
        except:
            print(f"{comm.rank} Unknown exception", flush=True)
            raise

        wx_cupy.load_cupy()

        super().__init__(comm, cupy, cupyx.scipy, interface_cuda)

        if not wx_cupy.cuda_avail:
            raise ValueError(f"Unable to create a CudaDevice object, no GPU devices were detected")

        if device_list is None:
            device_list = []
        device_list = [x for x in device_list if x < wx_cupy.num_devices]

        if len(device_list) == 0:
            device_list = range(wx_cupy.num_devices)

        self.cupyx = cupyx
        self.cupy = cupy

        # Select the CUDA device on which this PE will execute its kernels
        rank = self.comm.Get_rank()
        devnum = rank % len(device_list)
        cupy.cuda.Device(device_list[devnum]).use()

        # TODO don't use managed memory
        cupy.cuda.set_allocator(cupy.cuda.MemoryPool(cupy.cuda.malloc_managed).malloc)

        self.main_stream = cupy.cuda.get_current_stream()
        self.copy_stream = cupy.cuda.Stream(non_blocking=True)

    def __synchronize__(self, **kwargs):
        """Synchronize a stream, based on input arguments. By default, the main stream is synchronized.
        If copy_stream is True, synchronize that one instead."""
        if "copy_stream" in kwargs and kwargs["copy_stream"]:
            self.copy_stream.synchronize()
        else:
            self.main_stream.synchronize()

    def __array__(self, a: NDArray, *args, **kwargs) -> NDArray:
        """Copy given array to the device."""
        return self.xp.asarray(a)

    def __pinned__(self, *args, **kwargs) -> NDArray:
        """Allocate an array with pinned memory."""
        return self.cupyx.empty_pinned(*args, **kwargs)

    def __to_host__(self, val, **kwargs):
        """Copy given array to the host."""
        return val.get(**kwargs)


_default_device = None


def get_default_device():
    global _default_device
    if _default_device is None:
        _default_device = CpuDevice()

    return _default_device
