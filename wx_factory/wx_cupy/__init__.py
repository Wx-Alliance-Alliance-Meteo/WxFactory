__all__ = ["init_wx_cupy", "cuda_avail", "num_devices"]

from mpi4py import MPI

__initialized: bool = False
num_devices: int = 0
loading_error: None | Exception = None
cuda_avail: bool = False


def init_wx_cupy():
    global __initialized

    if __initialized:
        return

    global num_devices
    global loading_error
    global cuda_avail

    try:
        import cupy
        import cupy_backends

        num_devices = cupy.cuda.runtime.getDeviceCount()

        if num_devices <= 0:
            if MPI.COMM_WORLD.rank == 0:
                print(f"No cuda devices found")
            num_devices = 0

    except ModuleNotFoundError as e:
        loading_error = e
        if MPI.COMM_WORLD.rank == 0:
            print(
                f"cupy does not seem to be installed. "
                f"You need it (and GPUs) to be able run GEF with device=cuda.\n"
                f"We will run on CPU instead"
            )

    except ImportError as e:
        loading_error = e
        if MPI.COMM_WORLD.rank == 0:
            print(f"Module cupy is installed, but we were unable to load it, so we will run on CPUs instead")

    except cupy_backends.cuda.api.runtime.CUDARuntimeError as e:
        loading_error = e
        if MPI.COMM_WORLD.rank == 0:
            print(f"{e}")
            print(f"Module cupy is installed, but we could not find any CUDA device. Will run on CPU instead")

    except Exception as e:
        loading_error = e

    cuda_avail = num_devices > 0

    if MPI.COMM_WORLD.rank == 0:
        avail = "available" if cuda_avail else "not available"
        print(f"CUDA is {avail}")

    __initialized = True
