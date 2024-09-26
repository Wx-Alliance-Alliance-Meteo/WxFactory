from mpi4py import MPI
import numpy

def send_string_to(string: str, process_index: int, comm: MPI.Comm = MPI.COMM_WORLD):
    data: bytes = bytes(string, 'utf-8')
    length: int = len(data)
    length_buffer: numpy.ndarray = numpy.empty((1), dtype=int)
    length_buffer[0] = length

    comm.Send([length_buffer, MPI.LONG], process_index, 0)
    comm.Send([data, MPI.CHAR], process_index, 1)

def receive_string_from(process_index: int, comm: MPI.Comm = MPI.COMM_WORLD):
    length_buffer: numpy.ndarray = numpy.empty((1), dtype=int)
    comm.Recv([length_buffer, MPI.LONG], process_index, 0)
    
    data: bytearray = bytearray(length_buffer[0])
    comm.Recv([data, MPI.CHAR], process_index, 1)
    return str(data, 'utf-8')

