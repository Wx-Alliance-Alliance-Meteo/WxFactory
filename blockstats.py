import numpy

from constants import *

def blockstats(Q, step):

   h  = Q[:,:,0]
   uu = Q[:,:,1] / h
   ww = Q[:,:,2] / h

   print("\n==================================================================================")

   if step == 0:
      print("Blockstats for initial conditions")
   else:
      print("Blockstats for timestep ", step)

   print("h\t\tmean = %e\tmin = %e\tmax = %e\n" % (numpy.mean(h),  numpy.amin(h),  numpy.amax(h)) )
   print("u\t\tmean = %e\tmin = %e\tmax = %e\n" % (numpy.mean(uu), numpy.amin(uu), numpy.amax(uu)) )
   print("w\t\tmean = %e\tmin = %e\tmax = %e\n" % (numpy.mean(ww), numpy.amin(ww), numpy.amax(ww)) )

   print("==================================================================================")



