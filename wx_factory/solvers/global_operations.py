"""Global operations performed on distributed vectors.

The local contribution is computed with PyTorch and brought to the host as a Python scalar (or a
small host array) before the MPI reduction. This works with tensors on either the CPU or a GPU.
Bringing the scalar to the host also forces the device synchronization required by the caller."""

from typing import Optional

from mpi4py import MPI
import numpy
from numpy.typing import NDArray
import torch
from torch import Tensor

from ..context import Context

__all__ = ["global_norm", "global_dotprod", "global_inf_norm", "global_allreduce"]


def _to_scalar(value):
    """Bring a 0-d device/host reduction result to a Python scalar."""
    return value.item() if hasattr(value, "item") else value


def global_norm(vec: Tensor, context: Optional[Context] = None):
    """Compute vector 2-norm across all PEs.

    Returns a 0-d array, so callers can still use ``.item()`` on it."""
    if len(vec.shape) != 1:
        raise ValueError("This function only accept a vector (1 dimension tensor)")
    if context is None:
        context = Context.get_default()

    local_sum = _to_scalar(vec @ vec)
    total = context.comm.allreduce(local_sum)
    return torch.sqrt(torch.asarray(total))


def global_dotprod(vec1: Tensor, vec2: Tensor, comm: MPI.Comm = MPI.COMM_WORLD):
    """Compute dot product across all PEs in the communicator (default COMM_WORLD)."""
    local_sum = _to_scalar(vec1 @ vec2)
    return comm.allreduce(local_sum)


def global_inf_norm(vec: NDArray, comm: MPI.Comm = MPI.COMM_WORLD):
    """Compute infinity norm across all PEs in the communicator (default COMM_WORLD)."""
    local_max = _to_scalar(abs(vec).max())
    return comm.allreduce(local_max, op=MPI.MAX)


def global_allreduce(array: Tensor, context: Optional[Context] = None):
    """Sum a small (host or device) array across all PEs, returning it on the device.

    Used for the FGMRES orthogonalization reduction, whose operand is a small matrix rather than a
    scalar. The buffer is reduced on the host (contiguous numpy) to stay compatible with non
    CUDA-aware MPI, then copied back to the device."""
    if context is None:
        context = Context.get_default()
    host = array.cpu().numpy()
    total = numpy.empty_like(host)
    context.comm.Allreduce(host, total)
    return context.tensor(total)
