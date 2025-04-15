__all__ = ["load_cupy", "cuda_avail", "num_devices"]

from mpi4py import MPI

__initialized = False
num_devices = 0
cuda_avail = False


def load_cupy():
    global __initialized
    global num_devices
    global cuda_avail

    if __initialized:
        return cuda_avail

    try:
        import cupy
        import cupy_backends

        num_devices = cupy.cuda.runtime.getDeviceCount()

        if num_devices <= 0:
            if MPI.COMM_WORLD.rank == 0:
                print(f"No cuda devices found")
            num_devices = 0

    except ModuleNotFoundError as e:
        if MPI.COMM_WORLD.rank == 0:
            print(f"{e}")
            print(
                f"cupy does not seem to be installed. "
                f"You need it (and GPUs) to be able run GEF with device=cuda.\n"
                f"We will run on CPU instead"
            )

    except ImportError as e:
        if MPI.COMM_WORLD.rank == 0:
            print(f"{e}")
            print(f"Module cupy is installed, but we were unable to load it, so we will run on CPUs instead")

    except cupy_backends.cuda.api.runtime.CUDARuntimeError as e:
        if MPI.COMM_WORLD.rank == 0:
            print(f"{e}")
            print(f"Module cupy is installed, but we could not find any CUDA device. Will run on CPU instead")

    except Exception:
        pass

    cuda_avail = num_devices > 0

    if MPI.COMM_WORLD.rank == 0:
        avail = "available" if cuda_avail else "not available"
        print(f"CUDA is {avail}")

    __initialized = True

    return cuda_avail
