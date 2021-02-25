import numpy
from definitions import *
from shallow_water_test import *
from dcmip import *

class Topo:
   def __init__(self, hsurf, dzdx1, dzdx2, hsurf_itf_i, hsurf_itf_j):
      self.hsurf = hsurf
      self.dzdx1 = dzdx1
      self.dzdx2 = dzdx2
      self.hsurf_itf_i = hsurf_itf_i
      self.hsurf_itf_j = hsurf_itf_j

def initialize_euler(geom, metric, mtrx, param):

   #-------------------------------------------------------------------------|
   # case DCMIP 2012    | Pure advection                                     |
   #                    | ---------------------------------------------------|
   #                    | 11: 3D deformational flow                          |
   #                    | 12: 3D Hadley-like meridional circulation          |
   #                    | 13: 2D solid-body rotation of thin cloud-like      |
   #                    |     tracer in the presence of orography            |
   #                    | ---------------------------------------------------|
   #                    | 20: Steady-state at rest in presence of orography. |
   #                    | ---------------------------------------------------|
   #                    | Gravity waves, Non-rotating small-planet           |
   #                    | ---------------------------------------------------|
   #                    | 21: Mountain waves over a Schaer-type mountain     |
   #                    | 22: As 21 but with wind shear                      |
   #                    | 31: Gravity wave along the equator                 |
   #                    | ---------------------------------------------------|
   #                    | Rotating planet: Hydro. to non-hydro. scales (X)   |
   #                    | ---------------------------------------------------|
   #                    | 41X: Dry Baroclinic Instability Small Planet       |
   #                    | ---------------------------------------------------|
   #                    | 43 : Moist Baroclinic Instability Simple physics   |
   #--------------------|----------------------------------------------------|
   # case DCMIP 2016    | 161: Baroclinic wave with Toy Terminal Chemistry   |
   #                    | 162: Tropical cyclone                              |
   #                    | 163: Supercell (Small Planet)                      |
   #--------------------|----------------------------------------------------|
   # DCMIP_2012: https://www.earthsystemcog.org/projects/dcmip-2012/         |
   # DCMIP_2016: https://www.earthsystemcog.org/projects/dcmip-2016/         |
   #-------------------------------------------------------------------------|
   
   ni, nj, nk = geom.height.shape

   if param.case_number == 31:
      density, u1_contra, u2_contra, u3_contra, potential_temperature = dcmip_gravity_wave(geom, metric, mtrx, param)
   else:
      print('Something has gone horribly wrong in initialization. Back away slowly')
      exit(1)

   Q = numpy.zeros((5, ni, nj, nk))

   Q[idx_rho_u1, :, :] = density * u1_contra
   Q[idx_rho_u2, :, :] = density * u2_contra
   Q[idx_rho_u3, :, :] = density * u3_contra
   Q[idx_rho, :, :] = density
   Q[idx_rho_theta, :, :] = density * potential_temperature

   return Q, None

def initialize_sw(geom, metric, mtrx, param):

   ni, nj = geom.lon.shape

   if param.case_number != 5:
      hsurf = numpy.zeros((ni, nj))
      dzdx1 = numpy.zeros((ni, nj))
      dzdx2 = numpy.zeros((ni, nj))
      hsurf_itf_i = numpy.zeros((param.nb_elements_horizontal+2, param.nbsolpts*param.nb_elements_horizontal, 2))
      hsurf_itf_j = numpy.zeros((param.nb_elements_horizontal+2, 2, param.nbsolpts*param.nb_elements_horizontal))

   # --- Shallow water
   #   0 : deformation flow (passive advection only)
   #   1 : cosine hill (passive advection only)
   #   2 : zonal flow (shallow water)
   #   5 : zonal flow over an isolated mountain (shallow water)
   #   6 : Rossby-Haurvitz waves (shallow water)
   #   8 : Unstable jet (shallow water)
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

   Q = numpy.zeros((3, ni, nj))
   Q[idx_h, :, :] = fluid_height

   if param.case_number <= 1:
      # advection only
      Q[idx_u1, :, :] = u1_contra
      Q[idx_u2, :, :] = u2_contra
   else:
      Q[idx_hu1, :, :] = fluid_height * u1_contra
      Q[idx_hu2, :, :] = fluid_height * u2_contra

   return Q, Topo(hsurf, dzdx1, dzdx2, hsurf_itf_i, hsurf_itf_j)
