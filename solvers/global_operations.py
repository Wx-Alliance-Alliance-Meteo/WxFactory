"""Global operations performed on distributed vectors."""

from mpi4py import MPI
import numpy

__all__ = ['global_norm', 'global_dotprod', 'global_inf_norm']

def global_norm(vec, comm=MPI.COMM_WORLD):
   """Compute vector norm across all PEs"""
   local_sum = vec @ vec
   return numpy.sqrt( comm.allreduce(local_sum) )

def global_dotprod(vec1, vec2, comm=MPI.COMM_WORLD):
   """Compute dot product across all PEs"""
   local_sum = vec1 @ vec2
   return comm.allreduce(local_sum)

def global_inf_norm(vec, comm=MPI.COMM_WORLD):
   """Compute infinity norm across all PEs"""
   local_max = numpy.amax(numpy.abs(vec))
   return comm.allreduce(local_max, op=MPI.MAX)
