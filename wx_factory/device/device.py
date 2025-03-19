from abc import ABC, abstractmethod
import importlib
from time import time
from typing import Any, List, Optional, Tuple, TypeVar, Union

from mpi4py import MPI
from numpy.typing import NDArray

from compiler import compile_kernels
from . import wx_cupy


_Timestamp = TypeVar("Timestamp", bound=Union[float, "Event"])


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

    def __init__(self, comm: MPI.Comm, xp, xalg, pde_module) -> None:
        """Set a few modules and functions to have the same name, so that callers can use a single name."""
        self.comm = comm
        self.xp = xp
        self.xalg = xalg
        self.pde = pde_module

    @abstractmethod
    def synchronize(self, **kwargs):
        """Synchronize this device with the host. This is essentially a host-device barrier."""

    @abstractmethod
    def array(self, a: NDArray, *args, **kwargs) -> NDArray:
        """Copy the given array to this device, if it's not the same as the host."""

    @abstractmethod
    def pinned(self, *args, **kwargs) -> NDArray:
        """Allocate a host array with pinned memory."""

    @abstractmethod
    def to_host(self, val: Any, **kwargs) -> Any:
        """Copy the given array to the host (if it's not there already)."""

    @abstractmethod
    def timestamp(self, **kwargs) -> _Timestamp:
        """Get the "current" time in the flow of execution. On the GPU, execution may not have started yet,
        so this function will actually insert a timing event in the flow."""

    @abstractmethod
    def elapsed(self, timestamps: List[_Timestamp]) -> List[float]:
        """Get the set of elapsed times between the given list of timestamps. Also return
        the total elapsed time (between first and last timestamp)."""

    def has_128_bits_float(self) -> bool:
        """Not all devices can perform 128 bits floating point operation"""
        return hasattr(self.xp, "float128")

    @staticmethod
    def get_default() -> "CpuDevice":
        return CpuDevice.get_default()


class CpuDevice(Device):
    _default = None

    def __init__(self, comm: MPI.Comm = MPI.COMM_WORLD) -> None:
        import numpy
        import scipy

        try:
            compile_kernels.compile("pde", "cpp", force=False, comm=comm)
            pde = compile_kernels.load_module("pde", "cpp")
        except (ModuleNotFoundError, SystemExit):
            if comm.rank == 0:
                print(f"Unable to find the interface_c module. You need to compile it.", flush=True)
            raise
        except:
            print(f"Unknown exception!", flush=True)
            raise

        super().__init__(comm, numpy, scipy, pde)

        # If there is no default device yet, set it to self
        if CpuDevice._default is None and comm == MPI.COMM_WORLD:
            CpuDevice._default = self

    def synchronize(self, **kwargs):
        """Don't do anything. This is to allow writing generic code when device is not the same as the host."""

    def array(self, a: NDArray, *args, **kwargs) -> NDArray:
        """Return the input array unchanged."""
        return a

    def pinned(self, *args, **kwargs) -> NDArray:
        """Return allocated space, without any special characteristic."""
        return self.xp.empty(*args, **kwargs)

    def to_host(self, val, **kwargs):
        """Return the input array unchanged."""
        return val

    def timestamp(self, **kwargs):
        return time()

    def elapsed(self, timestamps):
        intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        intervals.append(timestamps[-1] - timestamps[0])
        return intervals

    @staticmethod
    def get_default() -> "CpuDevice":
        if CpuDevice._default is None:
            CpuDevice._default = CpuDevice()
        return CpuDevice._default


class CudaDevice(Device):
    _default = None

    def __init__(self, comm: MPI.Comm = MPI.COMM_WORLD, device_list: Optional[List[int]] = None) -> None:
        # Delay imports, to avoid loading CUDA if not asked

        wx_cupy.load_cupy()
        if not wx_cupy.cuda_avail:
            raise ValueError(f"Unable to create a CudaDevice object, no GPU devices were detected")

        import cupy
        import cupyx
        import cupyx.scipy.linalg

        # Get compiled library
        try:
            compile_kernels.compile("pde", "cuda", force=False, comm=comm)
            pde = compile_kernels.load_module("pde", "cuda")
        except (ModuleNotFoundError, ImportError, SystemExit):
            if comm.rank == 0:
                print(
                    f"Unable to load the interface_cuda module, you need to compile it if you want to use the GPU",
                    flush=True,
                )
            raise
        except:
            print(f"{comm.rank} Unknown exception", flush=True)
            raise

        # Set members
        super().__init__(comm, cupy, cupyx.scipy, pde)
        self.cupyx = cupyx
        self.cupy = cupy

        # Choose device on which to run kernels
        if device_list is None:
            device_list = []
        device_list = [x for x in device_list if x < wx_cupy.num_devices]

        if len(device_list) == 0:
            device_list = range(wx_cupy.num_devices)

        devnum = self.comm.rank % len(device_list)
        cupy.cuda.Device(device_list[devnum]).use()

        # TODO don't use managed memory
        cupy.cuda.set_allocator(cupy.cuda.MemoryPool(cupy.cuda.malloc_managed).malloc)

        # Set up compute and copy streams
        self.main_stream = cupy.cuda.get_current_stream()
        self.copy_stream = cupy.cuda.Stream(non_blocking=True)

        # If there is no default device yet, set it to self
        if CudaDevice._default is None and comm == MPI.COMM_WORLD:
            CudaDevice._default = self

    def synchronize(self, **kwargs):
        """Synchronize a stream, based on input arguments. By default, the main stream is synchronized.
        If copy_stream is True, synchronize that one instead."""
        if "copy_stream" in kwargs and kwargs["copy_stream"]:
            self.copy_stream.synchronize()
        else:
            self.main_stream.synchronize()

    def array(self, a: NDArray, *args, **kwargs) -> NDArray:
        """Copy given array to the device."""
        return self.xp.asarray(a)

    def pinned(self, *args, **kwargs) -> NDArray:
        """Allocate an array with pinned memory."""
        return self.cupyx.empty_pinned(*args, **kwargs)

    def to_host(self, val, **kwargs):
        """Copy given array to the host."""
        return val.get(**kwargs)

    def timestamp(self, **kwargs):
        ts = self.cupy.cuda.Event()
        if "copy_stream" in kwargs and kwargs["copy_stream"]:
            ts.record(self.copy_stream)
        else:
            ts.record(self.main_stream)

        return ts

    def elapsed(self, timestamps):
        get_time = self.cupy.cuda.get_elapsed_time
        intervals = [get_time(timestamps[i], timestamps[i + 1]) / 1000.0 for i in range(len(timestamps) - 1)]
        intervals.append(get_time(timestamps[0], timestamps[-1]) / 1000.0)
        return intervals

    @staticmethod
    def get_default() -> "CudaDevice":
        if CudaDevice._default is None:
            CudaDevice._default = CudaDevice()
        return CudaDevice._default
