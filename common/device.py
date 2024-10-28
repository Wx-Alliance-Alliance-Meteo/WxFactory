from abc import ABC, abstractmethod
from typing import Any, List

from mpi4py import MPI
from numpy.typing import NDArray


class Device(ABC):
    """Description of a device on which code can be executed.

    The device is allowed to be the same as the host (so that code is executed on the host). In such
    a case, most operations do nothing
    """

    def __init__(self, xp, xalg) -> None:
        """Set a few modules and functions to have the same name, so that callers can use a single name."""
        self.xp = xp
        self.xalg = xalg

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
    def __init__(self) -> None:
        import numpy
        import scipy

        super().__init__(numpy, scipy)

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

    def __init__(self, device_list: List[int]) -> None:
        # Delay imports, to avoid loading CUDA if not asked

        import cupy
        import cupyx
        import cupyx.scipy.linalg

        from wx_cupy import num_devices

        super().__init__(cupy, cupyx.scipy)

        if num_devices <= 0:
            raise ValueError(f"Unable to create a CudaDevice object, no GPU devices were detected")

        self.cupyx = cupyx
        self.cupy = cupy

        # Select the CUDA device on which this PE will execute its kernels
        rank = MPI.COMM_WORLD.Get_rank()
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


default_device = CpuDevice()
