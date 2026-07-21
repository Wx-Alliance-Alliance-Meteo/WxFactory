"""Global operations performed on distributed vectors.

These reductions are device-agnostic: the local contribution is computed with the array's own
library (numpy, cupy or torch) and brought to the host as a Python scalar (or a small host array)
before the MPI reduction, so they work whether the vectors live on the CPU or a GPU. Bringing the
scalar to the host also forces a device synchronization, which the callers need anyway."""

from typing import Optional

from mpi4py import MPI
import numpy
from numpy.typing import NDArray

from ..device import Device

__all__ = ["global_norm", "global_dotprod", "global_inf_norm", "global_allreduce"]


def _to_scalar(value):
    """Bring a 0-d device/host reduction result to a Python scalar."""
    return value.item() if hasattr(value, "item") else value


def global_norm(vec: NDArray, device: Optional[Device] = None):
    """Compute vector 2-norm across all PEs (from the given device, default CpuDevice).

    Returns a 0-d array of the device's array library, so callers can still use ``.item()`` on it."""
    if len(vec.shape) != 1:
        raise ValueError("This function only accept a vector (1 dimension tensor)")
    if device is None:
        device = Device.get_default()

    local_sum = _to_scalar(vec @ vec)
    total = device.comm.allreduce(local_sum)
    return device.xp.sqrt(device.xp.asarray(total))


def global_dotprod(vec1: NDArray, vec2: NDArray, comm: MPI.Comm = MPI.COMM_WORLD):
    """Compute dot product across all PEs in the communicator (default COMM_WORLD)."""
    local_sum = _to_scalar(vec1 @ vec2)
    return comm.allreduce(local_sum)


def global_inf_norm(vec: NDArray, comm: MPI.Comm = MPI.COMM_WORLD):
    """Compute infinity norm across all PEs in the communicator (default COMM_WORLD).

    Uses the array's own ``abs``/``max`` so it works for numpy, cupy and torch alike."""
    local_max = _to_scalar(abs(vec).max())
    return comm.allreduce(local_max, op=MPI.MAX)


def global_allreduce(array: NDArray, device: Optional[Device] = None):
    """Sum a small (host or device) array across all PEs, returning it on the device.

    Used for the FGMRES orthogonalization reduction, whose operand is a small matrix rather than a
    scalar. The buffer is reduced on the host (contiguous numpy) to stay compatible with non
    CUDA-aware MPI, then copied back to the device."""
    if device is None:
        device = Device.get_default()
    host = numpy.ascontiguousarray(device.to_host(array))
    total = numpy.empty_like(host)
    device.comm.Allreduce(host, total)
    return device.xp.asarray(total)
