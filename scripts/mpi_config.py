#!/usr/bin/env python3


try:
    from mpi4py import MPI
except ModuleNotFoundError:
    print("mpi4py does not seem available, so we can't do anything")
    raise

try:
    import torch

    num_devices = torch.cuda.device_count()
except (ModuleNotFoundError, ImportError, RuntimeError) as e:
    num_devices = 0
    if MPI.COMM_WORLD.rank == 0:
        print("Unable to import module torch")
        print(e)


def main():
    num_pes = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank

    node_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED, rank)

    node_rank = node_comm.rank
    node_size = node_comm.size

    node_roots_comm = MPI.COMM_WORLD.Split(node_rank == 0, rank)

    num_nodes = MPI.COMM_WORLD.allreduce(int(node_rank == 0), op=MPI.SUM)

    if rank == 0:
        print(f"Launched with {num_pes} PEs on {num_nodes} nodes.", flush=True)

    MPI.COMM_WORLD.Barrier()

    global_ranks_on_node = node_comm.gather(MPI.COMM_WORLD.rank)

    if node_rank == 0:
        node_id = node_roots_comm.rank

        print(
            f"Node {node_id:3d}: \n"
            f"  node size = {node_size}\n"
            f"  num CUDA devices = {num_devices}\n"
            f"  Global ranks: {global_ranks_on_node}\n",
            flush=True,
        )

        if node_id > 0:
            node_roots_comm.recv(source=node_id - 1)

        for i in range(num_devices):
            try:
                dev_info(i, node_id=node_id)
            except (RuntimeError, ValueError):
                print(f"{i} is a wrong number", flush=True)

        if node_id < node_roots_comm.size - 1:
            node_roots_comm.send(1, dest=node_id + 1)


def dev_info(device_id, node_id=-1):
    free_mem, total_mem = torch.cuda.mem_get_info(device_id)
    gb = 1024**3
    print(f"(Node {node_id:3d}) Device {device_id}: {free_mem / gb:.1f}/{total_mem / gb:.1f} GB available", flush=True)


if __name__ == "__main__":
    main()
