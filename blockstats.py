import numpy

from constants import *

def blockstats(Q, step):

   h  = Q[:,:,0]
   u1 = Q[:,:,1] / h TODO : problème ...
   u2 = Q[:,:,2] / h
  
   print("\n==================================================================================")

   if step == 0:
      print("Blockstats for initial conditions")
   else:
      print("Blockstats for timestep ", step)
   
   print("h\t\tmean = %e\tmin = %e\tmax = %e\n", numpy.mean(h),  numpy.amin(h),  numpy.amax(h) )
   print("u\t\tmean = %e\tmin = %e\tmax = %e\n", numpy.mean(u1), numpy.amin(u1), numpy.amax(u1) )
   print("v\t\tmean = %e\tmin = %e\tmax = %e\n", numpy.mean(u2), numpy.amin(u2), numpy.amax(u2) )

   print("==================================================================================")

