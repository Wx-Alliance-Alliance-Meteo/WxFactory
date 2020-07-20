import numpy
import mpi4py.MPI

from definitions import *

def blockstats(Q, step, case_number):

   h  = Q[:,:,0]
   if case_number > 1:
      uu = Q[:,:,1] / h
      vv = Q[:,:,2] / h

   print("\n==================================================================================")

   if step == 0:
      print("Blockstats for initial conditions")
   else:
      print("Blockstats for timestep ", step)

   print("h\t\tmean = %e\tmin = %e\tmax = %e\n" % (numpy.mean(h),  numpy.amin(h),  numpy.amax(h)) )
   if case_number > 1:
      print("u\t\tmean = %e\tmin = %e\tmax = %e\n" % (numpy.mean(uu), numpy.amin(uu), numpy.amax(uu)) )
      print("v\t\tmean = %e\tmin = %e\tmax = %e\n" % (numpy.mean(vv), numpy.amin(vv), numpy.amax(vv)) )

   print("==================================================================================")

def global_integral(field, geom, mtrx, nbsolpts, nb_elements_horiz):

   local_sum = 0.
   for elem in range(nb_elements_horiz):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)
      local_sum += numpy.sum( field[epais,epais] * mtrx.quad_weights )

   return mpi4py.MPI.COMM_WORLD.allreduce(local_sum)
