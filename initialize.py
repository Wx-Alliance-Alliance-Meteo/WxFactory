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
   
   nk, nj, ni = geom.height.shape

   nb_equations = 5

   topo = None
   
   if param.case_number == 11:
      nb_equations = 9
      rho, u1_contra, u2_contra, w, potential_temperature, q1, q2, q3, q4 = dcmip_advection_deformation(geom, metric, mtrx, param)
   elif param.case_number == 12:
      nb_equations = 6
      rho, u1_contra, u2_contra, w, potential_temperature, q1 = dcmip_advection_hadley(geom, metric, mtrx, param)
   elif param.case_number == 20:
      rho, u1_contra, u2_contra, w, potential_temperature = dcmip_steady_state_mountain(geom, metric, mtrx, param)
   elif param.case_number == 31:
      rho, u1_contra, u2_contra, w, potential_temperature = dcmip_gravity_wave(geom, metric, mtrx, param)
   else:
      print('Something has gone horribly wrong in initialization. Back away slowly')
      exit(1)

   Q = numpy.zeros((nb_equations, nk, nj, ni))

   Q[idx_rho   , :, :, :]    = rho
   Q[idx_rho_u1, :, :, :]    = rho * u1_contra
   Q[idx_rho_u2, :, :, :]    = rho * u2_contra
   Q[idx_rho_w, :, :, :]     = rho * w
   Q[idx_rho_theta, :, :, :] = rho * potential_temperature

   if param.case_number == 11 or param.case_number == 12:
      Q[5, :, :, :] = rho * q1
   if param.case_number == 11:
      Q[6, :, :, :] = rho * q2
      Q[7, :, :, :] = rho * q3
      Q[8, :, :, :] = rho * q4
   
   return Q, None

def initialize_sw(geom, metric, mtrx, param):

   ni, nj = geom.lon.shape
   nb_equations = 3

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

   Q = numpy.zeros((nb_equations, ni, nj))
   Q[idx_h, :, :] = fluid_height

   if param.case_number <= 1:
      # advection only
      Q[idx_u1, :, :] = u1_contra
      Q[idx_u2, :, :] = u2_contra
   else:
      Q[idx_hu1, :, :] = fluid_height * u1_contra
      Q[idx_hu2, :, :] = fluid_height * u2_contra

   # Note : we move the last axis of the first topo array so that both have similiar ordering
   return Q, Topo(hsurf, dzdx1, dzdx2, numpy.moveaxis(hsurf_itf_i, -1, -2), hsurf_itf_j)
