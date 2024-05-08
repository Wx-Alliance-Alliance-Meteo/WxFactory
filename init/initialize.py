import numpy

from common.definitions      import idx_rho, idx_rho_u1, idx_rho_u2, idx_rho_w, idx_rho_theta,                 \
                                    idx_h, idx_u1, idx_u2, idx_hu1, idx_hu2,                                   \
                                    idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta,                  \
                                    gravity, cpd, cvd, Rd, p0
from common.program_options  import Configuration
from init.dcmip              import dcmip_advection_deformation, dcmip_advection_hadley, dcmip_gravity_wave,   \
                                    dcmip_schar_waves, dcmip_steady_state_mountain, dcmip_baroclinic_instability, acoustic_wave
from init.shallow_water_test import case_galewsky, case_matsuno, case_unsteady_zonal, circular_vortex,         \
                                    williamson_case1, williamson_case2, williamson_case5, williamson_case6

from geometry                import Cartesian2D, CubedSphere

class Topo:
   def __init__(self, hsurf, dzdx1, dzdx2, hsurf_itf_i, hsurf_itf_j):
      self.hsurf = hsurf
      self.dzdx1 = dzdx1
      self.dzdx2 = dzdx2
      self.hsurf_itf_i = hsurf_itf_i
      self.hsurf_itf_j = hsurf_itf_j

def initialize_euler(geom: CubedSphere, metric, mtrx, param):

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

   if param.case_number == 11:
      nb_equations = 9
      rho, u1_contra, u2_contra, w, potential_temperature, q1, q2, q3, q4 = \
         dcmip_advection_deformation(geom, metric, mtrx, param)
   elif param.case_number == 12:
      nb_equations = 6
      rho, u1_contra, u2_contra, w, potential_temperature, q1 = dcmip_advection_hadley(geom, metric, mtrx, param)
   elif param.case_number == 20:
      rho, u1_contra, u2_contra, w, potential_temperature = dcmip_steady_state_mountain(geom, metric, mtrx, param)
   elif param.case_number == 21:
      rho, u1_contra, u2_contra, w, potential_temperature = dcmip_schar_waves(geom, metric, mtrx, param, False)
   elif param.case_number == 22:
      rho, u1_contra, u2_contra, w, potential_temperature = dcmip_schar_waves(geom, metric, mtrx, param, True)
   elif param.case_number == 31:
      rho, u1_contra, u2_contra, w, potential_temperature = dcmip_gravity_wave(geom, metric, mtrx, param)
   elif param.case_number == 41:
      rho, u1_contra, u2_contra, w, potential_temperature = dcmip_baroclinic_instability(geom, metric, mtrx, param)
   elif param.case_number == 77:
      rho, u1_contra, u2_contra, w, potential_temperature = acoustic_wave(geom, metric, mtrx, param)
   else:
      raise ValueError('Something has gone horribly wrong in initialization. Back away slowly')

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
      u1_contra, u2_contra, fluid_height, hsurf, dzdx1, dzdx2, hsurf_itf_i, hsurf_itf_j = \
         williamson_case5(geom, metric, mtrx, param)

   elif param.case_number == 6:
      u1_contra, u2_contra, fluid_height = williamson_case6(geom, metric, param)

   elif param.case_number == 8:
      u1_contra, u2_contra, fluid_height = case_galewsky(geom, metric, param)

   elif param.case_number == 9:
      u1_contra, u2_contra, fluid_height = case_matsuno(geom, metric, param)

   elif param.case_number == 10:
      u1_contra, u2_contra, fluid_height, hsurf, dzdx1, dzdx2, hsurf_itf_i, hsurf_itf_j = \
         case_unsteady_zonal(geom, metric, mtrx, param)

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

def initialize_cartesian2d(geom: Cartesian2D, param: Configuration):
   '''Initialize a problem on a 2D cartesian grid based on a case number.'''

   nb_equations = 4

   # Initial state at rest, isentropic, hydrostatic
   nk, ni = geom.X1.shape
   Q = numpy.zeros((nb_equations, nk, ni))
   uu    = numpy.zeros_like(geom.X1)
   ww    = numpy.zeros_like(geom.X1)
   exner = numpy.zeros_like(geom.X1)
   θ = numpy.ones_like(geom.X1)

   if param.case_number != 0:
      θ *= param.bubble_theta

   if param.case_number == 0:
      # Mountain wave
      geom.make_mountain()

      # Use periodic BC in x-direction
      geom.xperiodic = True

   elif param.case_number == 1:
      # Pill

      xc=500.0
      zc=260.0

      pert = 0.5

      for k in range(nk):
         for i in range(ni):
            r = (geom.X1[k,i]-xc)**2 + (geom.X3[k,i]-zc)**2
            if r < param.bubble_rad**2:
               θ[k,i] += pert

   elif param.case_number == 2:
      # Gaussian bubble

      A = 0.5
      a = 50
      s = 100
      x0 = 500
      z0 = 260
      r = numpy.sqrt( (geom.X1-x0)**2 + (geom.X3-z0)**2 )

      θ = numpy.where(r <= a,
                      θ + A,
                      θ + A * numpy.exp(-((r-a)/s)**2))

      # Enforce mirror symmetry
      if ni % 2 == 0:
         middle_col = ni / 2
      else:
         middle_col = ni / 2 + 1

      for i in range(int(middle_col)):
         θ[:, ni-i-1] = θ[:, i]

   elif param.case_number == 3:
      # Colliding bubbles

      A = 0.5
      a = 150
      s = 50
      x0 = 500
      z0 = 300
      for k in range(nk):
         for i in range(ni):
            r = numpy.sqrt( (geom.X1[k,i]-x0)**2 + (geom.X3[k,i]-z0)**2 )
            if r <= a:
               θ[k,i] += A
            else:
               θ[k,i] += A * numpy.exp(-((r-a)/s)**2)

      A = -0.15
      a = 0
      s = 50
      x0 = 560
      z0 = 640
      for k in range(nk):
         for i in range(ni):
            r = numpy.sqrt( (geom.X1[k,i]-x0)**2 + (geom.X3[k,i]-z0)**2 )
            if r <= a:
               θ[k,i] += A
            else:
               θ[k,i] += A * numpy.exp(-((r-a)/s)**2)

   elif param.case_number == 4:
      # Cold density current
      x0 = 0.
      z0 = 3000.
      xr = 4000.
      zr = 2000.
      θc = -15.

      # Use periodic BC in x-direction
      # geom.xperiodic = True

      for k in range(nk):
         for i in range(ni):
            r = numpy.sqrt( ((geom.X1[k,i]-x0)/xr)**2 + ((geom.X3[k,i]-z0)/zr)**2 )
            if r <= 1.:
               θ[k,i] += 0.5 * θc * (1. + numpy.cos(numpy.pi * r))

      geom.make_mountain(mountain_type='step')

   if param.case_number == 0:
      N_star = 0.01
      t0 = 288

      a00 = N_star**2 / gravity
      capc1 = gravity**2 / (N_star**2 * cpd * t0)

      exner = 1.0 - capc1 * (1.0 - numpy.exp(-a00 * geom.X3))
      θ = t0 * numpy.exp(a00 * geom.X3)

      uu[:,:] = 10.

   else:
      exner = (1.0 - gravity / (cpd * θ) * geom.X3)

   ρ = p0 / (Rd * θ) * exner**(cvd / Rd)

   Q[idx_2d_rho,:,:]       = ρ
   Q[idx_2d_rho_u,:,:]     = ρ * uu
   Q[idx_2d_rho_w,:,:]     = ρ * ww
   Q[idx_2d_rho_theta,:,:] = ρ * θ

   return Q
