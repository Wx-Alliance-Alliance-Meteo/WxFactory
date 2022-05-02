
from mpi4py import MPI

GLOBAL_COMM_val = MPI.COMM_WORLD

def GLOBAL_COMM():
    return GLOBAL_COMM_val

def split_comm():
    global GLOBAL_COMM_val
    old_comm = GLOBAL_COMM_val

    GLOBAL_COMM_val = old_comm.Split(old_comm.rank // 6, old_comm.rank)
