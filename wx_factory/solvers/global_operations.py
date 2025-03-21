"""Global operations performed on distributed vectors."""

from typing import Optional

from mpi4py import MPI
import numpy
from numpy.typing import NDArray

from device import Device

__all__ = ["global_norm", "global_dotprod", "global_inf_norm"]


def global_norm(vec: NDArray, device: Optional[Device] = None):
    """Compute vector norm across all PEs in the communicator (from given device, default CpuDevice)"""
    if len(vec.shape) != 1:
        raise ValueError("This function only accept a vector (1 dimension tensor)")
    if device is None:
        device = Device.get_default()

    comm = device.comm

    local_sum = vec @ vec
    return device.xp.sqrt(comm.allreduce(local_sum))


def global_dotprod(vec1: NDArray, vec2: NDArray, comm: MPI.Comm = MPI.COMM_WORLD):
    """Compute dot product across all PEs in the communicator (default COMM_WORLD)"""
    local_sum = vec1 @ vec2
    return comm.allreduce(local_sum)


def global_inf_norm(vec: NDArray, comm: MPI.Comm = MPI.COMM_WORLD):
    """Compute infinity norm across all PEs in the communicator (default COMM_WORLD)"""
    local_max = numpy.amax(numpy.abs(vec))
    return comm.allreduce(local_max, op=MPI.MAX)
