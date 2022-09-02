import numpy
import mpi4py

def norm(vec):
   """Compute vector norm across all PEs"""
   local_sum = vec @ vec
   return numpy.sqrt( mpi4py.MPI.COMM_WORLD.allreduce(local_sum) )

def inf_norm(vec):
    """Compute infinity norm across all PEs"""
    local_max = numpy.amax(numpy.abs(vec))
    return mpi4py.MPI.COMM_WORLD.allreduce(local_max, op=mpi4py.MPI.MAX)

def dotprod(vec1, vec2):
   """Compute dot product across all PEs"""
   local_sum = vec1 @ vec2
   return mpi4py.MPI.COMM_WORLD.allreduce(local_sum)
