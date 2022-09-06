import numpy
from gef_mpi import GLOBAL_COMM, MPI

def norm(vec):
   """Compute vector norm across all PEs"""
   local_sum = vec @ vec
   return numpy.sqrt( GLOBAL_COMM().allreduce(local_sum) )

def inf_norm(vec):
    """Compute infinity norm across all PEs"""
    local_max = numpy.amax(numpy.abs(vec))
    return GLOBAL_COMM().allreduce(local_max, op=MPI.MAX)

def dotprod(vec1, vec2):
   """Compute dot product across all PEs"""
   local_sum = vec1 @ vec2
   return GLOBAL_COMM().allreduce(local_sum)
