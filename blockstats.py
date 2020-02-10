import numpy

from definitions import *

def blockstats(Q, step, case_number):

   # TODO: paralléliser 

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
      print("w\t\tmean = %e\tmin = %e\tmax = %e\n" % (numpy.mean(vv), numpy.amin(vv), numpy.amax(vv)) )

   print("==================================================================================")
