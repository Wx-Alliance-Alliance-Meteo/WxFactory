"""The context of a simulation: the device on which the model runs (CPU/GPU), the MPI communicator and other
information related to the execution environment.

Now that the whole model is written directly against PyTorch there is a single device type. This
object no longer abstracts an array module; it just holds the process's ``comm``, the torch device
its tensors live on, the working floating-point precision, and a few host/device transfer and
timing helpers.
"""

import os
from time import perf_counter
from typing import Any, Self

import torch
from mpi4py import MPI

from .wx_mpi import split_nodes

__all__ = ["Context"]

# WxFactory speaks NumPy-flavoured method names in a few places; make torch tensors answer to them
# too, so the same call works whether an array happens to be a tensor or a host NumPy array.
torch.Tensor.astype = torch.Tensor.to
torch.Tensor.copy = torch.Tensor.clone


def _differentiable_requested() -> bool:
    """Whether the user asked to keep autograd on (opt out of inference mode)."""
    return os.environ.get("WX_FACTORY_DIFFERENTIABLE", "").lower() in ("1", "true", "yes", "on")


class Context:
    """The PyTorch compute device and its MPI communicator."""

    _default: Self = None

    def __init__(self, comm: MPI.Comm, device_type: str = "cuda") -> None:
        self.comm = comm
        self.torch = torch

        # Putting the model on a GPU is entirely a matter of where the tensors live. Ranks of a node
        # are spread over the available devices.
        if device_type == "cuda" and torch.cuda.is_available():
            node_comm, _ = split_nodes(comm)
            num_devices = torch.cuda.device_count()
            num_per_device = (node_comm.size + num_devices - 1) // num_devices
            self.torch_device = torch.device("cuda", node_comm.rank // num_per_device)
            torch.cuda.set_device(self.torch_device)
        else:
            if device_type == "cuda" and comm.rank == 0:
                print("No GPU available for the Pytorch backend, falling back to the CPU", flush=True)
            self.torch_device = torch.device("cpu")

        # Every tensor the code creates goes through torch's default device.
        torch.set_default_device(self.torch_device)

        # Disable autograd bookkeeping unless WX_FACTORY_DIFFERENTIABLE requests it.
        if not _differentiable_requested() and not torch.is_inference_mode_enabled():
            self._inference_mode_guard = torch.inference_mode()
            self._inference_mode_guard.__enter__()

        # Floating-point precision of the whole computation. Defaults to double; the Simulation
        # overrides these from the `precision` configuration option. Single precision halves the
        # memory footprint (and the bandwidth), which is what lets the finer resolutions fit on a GPU.
        self.real_dtype = torch.float64

        if comm.rank == 0:
            print(f"Pytorch backend running on {self.torch_device} (on rank {comm.rank})", flush=True)

    def tensor(self, a) -> torch.Tensor:
        return torch.tensor(a, device=self.torch_device)

    def synchronize(self, **kwargs):
        """Wait for the queued GPU work. A no-op when the tensors are already on the host."""
        if self.torch_device.type == "cuda":
            torch.cuda.synchronize(self.torch_device)

    def array(self, a: Any) -> torch.Tensor:
        """Bring an array-like onto this device as a tensor."""
        return torch.asarray(a, device=self.torch_device)

    def to_host(self, val: torch.Tensor, **kwargs) -> Any:
        """Copy a tensor back to the host as a NumPy array."""
        return val.cpu().numpy().copy()

    def timestamp(self, **kwargs) -> float | torch.cuda.Event:
        if self.torch_device.type == "cpu":
            return perf_counter()
        else:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event

    def elapsed(self, timestamps: list) -> list[float]:
        """Return the elapsed time between each pair of timestamps, in milliseconds.
        The last element is the total time between the first and last timestamps."""

        if isinstance(timestamps[0], float):
            intervals = [(timestamps[i + 1] - timestamps[i]) * 1000.0 for i in range(len(timestamps) - 1)]
            intervals.append((timestamps[-1] - timestamps[0]) * 1000.0)
        elif isinstance(timestamps[0], torch.cuda.Event):
            intervals = []
            timestamps[-1].synchronize()
            intervals = [timestamps[i].elapsed_time(timestamps[i + 1]) for i in range(len(timestamps) - 1)]
            intervals.append(timestamps[0].elapsed_time(timestamps[-1]))
        else:
            raise ValueError(f"Unknown timestamp type {type(timestamps[0])}")

        return intervals

    @staticmethod
    def set_default(context: "Context") -> None:
        """Set the default context."""
        Context._default = context

    @staticmethod
    def get_default() -> "Context":
        if Context._default is None:
            Context._default = Context(MPI.COMM_WORLD)
        return Context._default
