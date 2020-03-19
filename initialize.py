import numpy
from definitions import *
from shallow_water_test import *

class Topo:
   def __init__(self, hsurf, dzdx1, dzdx2):
      self.hsurf = hsurf
      self.dzdx1 = dzdx1
      self.dzdx2 = dzdx2

def initialize(geom, metric, mtrx, param): 

   ni, nj = geom.lon.shape

   if param.case_number <= 1:
      # advection only, save u1 and u2
      Q = numpy.zeros((nb_equations+2, ni, nj)) # TODO : reviser
   else:
      Q = numpy.zeros((nb_equations, ni, nj))

   if param.case_number == 0:
      u1, u2, h, h_analytic = circular_vortex(geom, metric, param)

   elif param.case_number == 1:
      u1, u2, h, h_analytic = williamson_case1(geom, metric, param)

   elif param.case_number == 2:
      u1, u2, h, h_analytic = williamson_case2(geom, metric, param)

   elif param.case_number == 5:
      u1, u2, h, h_analytic, hsurf, dzdx1, dzdx2 = williamson_case5(geom, metric, mtrx, param)

   elif param.case_number == 6:
      u1, u2, h, h_analytic = williamson_case6(geom, metric, param)

   elif param.case_number == 8:
      u1, u2, h, h_analytic = case_galewsky(geom, metric, param)

   elif param.case_number == 9: 
      u1, u2, h, h_analytic = case_matsuno(geom, metric, param)
         
   Q[idx_h,:,:]   = h

   if param.case_number <= 1:
      # advection only
      Q[idx_u1,:,:] = u1
      Q[idx_u2,:,:] = u2
   else:
      Q[idx_hu1,:,:] = h * u1
      Q[idx_hu2,:,:] = h * u2

   return Q, Topo(hsurf, dzdx1, dzdx2), h_analytic
