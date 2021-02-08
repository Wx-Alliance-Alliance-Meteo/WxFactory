import numpy
from definitions import *
from shallow_water_test import *

class Topo:
   def __init__(self, hsurf, dzdx1, dzdx2, hsurf_itf_i, hsurf_itf_j):
      self.hsurf = hsurf
      self.dzdx1 = dzdx1
      self.dzdx2 = dzdx2
      self.hsurf_itf_i = hsurf_itf_i
      self.hsurf_itf_j = hsurf_itf_j

def initialize_euler(geom, metric, mtrx, param):
   return None

def initialize_sw(geom, metric, mtrx, param):

   ni, nj = geom.lon.shape

   if param.case_number != 5:
      hsurf = numpy.zeros((ni, nj))
      dzdx1 = numpy.zeros((ni, nj))
      dzdx2 = numpy.zeros((ni, nj))
      hsurf_itf_i = numpy.zeros((param.nb_elements_horizontal+2, param.nbsolpts*param.nb_elements_horizontal, 2))
      hsurf_itf_j = numpy.zeros((param.nb_elements_horizontal+2, 2, param.nbsolpts*param.nb_elements_horizontal))

   # --- Shallow water
   if param.case_number == 0:
      u1_contra, u2_contra, fluid_height = circular_vortex(geom, metric, param)

   elif param.case_number == 1:
      u1_contra, u2_contra, fluid_height = williamson_case1(geom, metric, param)

   elif param.case_number == 2:
      u1_contra, u2_contra, fluid_height = williamson_case2(geom, metric, param)

   elif param.case_number == 5:
      u1_contra, u2_contra, fluid_height, hsurf, dzdx1, dzdx2, hsurf_itf_i, hsurf_itf_j = williamson_case5(geom, metric, mtrx, param)

   elif param.case_number == 6:
      u1_contra, u2_contra, fluid_height = williamson_case6(geom, metric, param)

   elif param.case_number == 8:
      u1_contra, u2_contra, fluid_height = case_galewsky(geom, metric, param)

   elif param.case_number == 9:
      u1_contra, u2_contra, fluid_height = case_matsuno(geom, metric, param)

   elif param.case_number == 10:
      u1_contra, u2_contra, fluid_height, hsurf, dzdx1, dzdx2, hsurf_itf_i, hsurf_itf_j = case_unsteady_zonal(geom, metric, mtrx, param)

   Q = numpy.zeros((nb_equations, ni, nj))
   Q[idx_h, :, :] = fluid_height

   if param.case_number <= 1:
      # advection only
      Q[idx_u1, :, :] = u1_contra
      Q[idx_u2, :, :] = u2_contra
   else:
      Q[idx_hu1, :, :] = fluid_height * u1_contra
      Q[idx_hu2, :, :] = fluid_height * u2_contra

   return Q, Topo(hsurf, dzdx1, dzdx2, hsurf_itf_i, hsurf_itf_j)
