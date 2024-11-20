#!/usr/bin/env python3

import sys

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    print(f"mpi4py does not seem available, so we can't do anything")
    sys.exit(-1)

try:
    import cupy as cp

    cupy_avail = True
except (ModuleNotFoundError, ImportError):
    cupy_avail = False
    if MPI.COMM_WORLD.rank == 0:
        print(f"Unable to import module cupy")


def main():
    num_pes = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank

    node_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED, rank)

    node_rank = node_comm.rank
    node_size = node_comm.size

    node_roots_comm = MPI.COMM_WORLD.Split(node_rank == 0, rank)

    num_nodes = 0
    if node_rank == 0:
        num_nodes = 1

    num_nodes = MPI.COMM_WORLD.allreduce(num_nodes, op=MPI.SUM)

    if rank == 0:
        print(f"Launched with {num_pes} PEs on {num_nodes} nodes.")

        # if cupy_avail: cp.show_config()

    MPI.COMM_WORLD.Barrier()

    if node_rank == 0:
        node_id = node_roots_comm.rank

        num_devices = 0
        if cupy_avail:
            num_devices = cp.cuda.runtime.getDeviceCount()

        print(f"Node {node_id:3d}: \n" f"  node size = {node_size}\n" f"  num CUDA devices = {num_devices}\n")

        for i in range(num_devices):
            try:
                dev_info(i)
            except:
                print(f"{i} is a wrong number")

        # if cupy_avail: cp.show_config()


def dev_info(id):
    with cp.cuda.Device(id) as dev:
        free_mem, total_mem = dev.mem_info
        kb = 1024
        mb = kb * kb
        gb = kb * kb * kb
        print(f"  Device {id}: {free_mem / gb :.1f}/{total_mem / gb :.1f} GB available")


if __name__ == "__main__":
    main()
