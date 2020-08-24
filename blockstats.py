import numpy
import math
import mpi4py.MPI

from shallow_water_test import *

def blockstats(Q, geom, metric, mtrx, param, step):

   h  = Q[0,:,:]

   if param.case_number == 0:
      h_anal, _ = height_vortex(geom, metric, param, step)
   elif param.case_number == 1:
      h_anal = height_case1(geom, metric, param, step)
   else:
      print('not yet implemented')
      exit(1)

   if param.case_number > 1:
      uu = Q[1,:,:] / h
      vv = Q[2,:,:] / h

   print("\n==================================================================================")

   if step == 0:
      print("Blockstats for initial conditions")
   else:
      print("Blockstats for timestep ", step)

   absol_err = global_integral(abs(h - h_anal), mtrx, metric, param.nbsolpts, param.nb_elements) 
   int_h_anal = global_integral(abs(h_anal), mtrx, metric, param.nbsolpts, param.nb_elements) 

   absol_err2 = global_integral((h - h_anal)**2, mtrx, metric, param.nbsolpts, param.nb_elements) 
   int_h_anal2 = global_integral(h_anal**2, mtrx, metric, param.nbsolpts, param.nb_elements) 

   max_absol_err = mpi4py.MPI.COMM_WORLD.allreduce(numpy.max(abs(h - h_anal)), op=mpi4py.MPI.MAX)
   max_h_anal = mpi4py.MPI.COMM_WORLD.allreduce(numpy.max(h_anal), op=mpi4py.MPI.MAX)

   l1 = absol_err / int_h_anal
   l2 = math.sqrt( absol_err2 / int_h_anal2 )
   linf = max_absol_err / max_h_anal

   print(f'l1 = {l1} \t l2 = {l2} \t linf = {linf}')

#   print("h\t\tmean = %e\tmin = %e\tmax = %e\n" % (numpy.mean(inth),  numpy.amin(inth),  numpy.amax(inth)) )
#   if param.case_number > 1:
#      print("u\t\tmean = %e\tmin = %e\tmax = %e\n" % (numpy.mean(uu), numpy.amin(uu), numpy.amax(uu)) )
#      print("v\t\tmean = %e\tmin = %e\tmax = %e\n" % (numpy.mean(vv), numpy.amin(vv), numpy.amax(vv)) )

   print("==================================================================================")

def global_integral(field, mtrx, metric, nbsolpts, nb_elements_horiz):

   local_sum = 0.
   for line in range(nb_elements_horiz):
      epais_lin = line * nbsolpts + numpy.arange(nbsolpts)
      for column in range(nb_elements_horiz):
         epais_col = column * nbsolpts + numpy.arange(nbsolpts)
         local_sum += numpy.sum( field[epais_lin,epais_col] * metric.sqrtG[epais_lin,epais_col] * mtrx.quad_weights )

   return mpi4py.MPI.COMM_WORLD.allreduce(local_sum)
