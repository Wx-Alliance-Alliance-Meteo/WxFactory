"""Global operations performed on distributed vectors."""

from mpi4py import MPI
import numpy

__all__ = ['global_norm', 'global_dotprod', 'global_inf_norm']

def global_norm(vec):
   """Compute vector norm across all PEs"""
   local_sum = vec @ vec
   return numpy.sqrt( MPI.COMM_WORLD.allreduce(local_sum) )

def global_dotprod(vec1, vec2):
   """Compute dot product across all PEs"""
   local_sum = vec1 @ vec2
   return MPI.COMM_WORLD.allreduce(local_sum)

def global_inf_norm(vec):
   """Compute infinity norm across all PEs"""
   local_max = numpy.amax(numpy.abs(vec))
   return MPI.COMM_WORLD.allreduce(local_max, op=MPI.MAX)
