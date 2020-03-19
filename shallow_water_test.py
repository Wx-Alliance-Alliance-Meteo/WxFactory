import math
import numpy
from definitions import *
from wind2contra import *

import matsuno

def eval_u_prime(lat):
   u_max = 80.0
   phi0 = math.pi / 7.0
   phi1 = math.pi / 2.0 - phi0

   if lat < phi0:
      return 0.0
   elif lat > phi1:
      return 0.0

   e_n = math.exp( -4.0 / ((phi1 - phi0)**2) )

   u_p = math.exp( 1.0 / ((lat - phi0) * (lat - phi1)) )

   return u_max / e_n * u_p

def solid_body_rotation(geom, metric, param):
   if param.case_number == 5:
      u0 = 20.0
      sinα = 0
      cosα = 1
   else:
      u0 = 2.0 * math.pi * earth_radius / (12.0 * day_in_secs)
      sinα = math.sin(param.Williamson_angle)
      cosα = math.cos(param.Williamson_angle)

   if geom.cube_face == 0:
      u1 = u0 / earth_radius * (cosα + geom.Y / (1.0 + geom.X**2) * sinα)
      u2 = u0 * geom.X / (earth_radius * (1.0 + geom.Y**2)) * (geom.Y * cosα - sinα)
   elif geom.cube_face == 1:
      u1 = u0 / earth_radius * (cosα - geom.X * geom.Y / (1.0 + geom.X**2) * sinα)
      u2 = u0 / earth_radius * (geom.X * geom.Y / (1.0 + geom.Y**2) * cosα - sinα)
   elif geom.cube_face == 2:
      u1 = u0 / earth_radius * (cosα - geom.Y / (1.0 + geom.X**2) * sinα)
      u2 = u0 * geom.X / (earth_radius * (1.0 + geom.Y**2)) * (geom.Y * cosα + sinα)
   elif geom.cube_face == 3:
      u1 = u0 / earth_radius * (cosα + geom.X * geom.Y / (1.0 + geom.X**2) * sinα)
      u2 = u0 / earth_radius * (geom.X * geom.Y / (1.0 + geom.Y**2) * cosα + sinα)
   elif geom.cube_face == 4:
      u1 = u0 / earth_radius * (- geom.Y / (1.0 + geom.X**2) * cosα + sinα)
      u2 = u0 * geom.X / (earth_radius * (1.0 + geom.Y**2)) * (cosα + geom.Y * sinα)
   elif geom.cube_face == 5:
      u1 = u0 / earth_radius * (geom.Y / (1.0 + geom.X**2) * cosα - sinα)
      u2 =-u0 * geom.X / (earth_radius * (1.0 + geom.Y**2)) * (cosα + geom.Y * sinα)

   return u1, u2

def circular_vortex(geom, metric, param):
   print("--------------------------------------------------------------")
   print("CASE 0 (Tracer): Circular vortex, Nair and Machenhauer,2002   ")
   print("--------------------------------------------------------------")

   # Deformational Flow (Nair and Machenhauer, 2002)
   lon_center = math.pi - 0.8
   lat_center = math.pi / 4.8

   V0    = 2.0 * math.pi / (12.0 * day_in_secs) * earth_radius
   rho_0 = 3.0
   gamma = 5.0

   lonR = numpy.arctan2( geom.coslat * numpy.sin(geom.lon - lon_center), \
      geom.coslat * math.sin(lat_center) * numpy.cos(geom.lon - lon_center) - math.cos(lat_center) * geom.sinlat )

   lonR[lonR<0.0] = lonR[lonR<0.0] + (2.0 * math.pi)

   latR = numpy.arcsin( geom.sinlat * math.sin(lat_center) + geom.coslat * math.cos(lat_center) * numpy.cos(geom.lon - lon_center) )

   rho = rho_0 * numpy.cos(latR)

   Vt  = V0 * (3.0/2.0 * math.sqrt(3)) * (1.0 / numpy.cosh(rho))**2 * numpy.tanh(rho)

   Omega = numpy.zeros_like(geom.lat)

   ni, nj = geom.lat.shape

   for i in range(ni):
      for j in range(nj):
         if (abs(rho[i,j]) > 1e-9):
               Omega[i,j] = Vt[i,j] / (earth_radius * rho[i,j])

   h          = 1.0 - numpy.tanh( (rho / gamma) * numpy.sin(lonR) )
   h_analytic = 1.0 - numpy.tanh( (rho / gamma) * numpy.sin(lonR - Omega * param.t_end) )

   u = earth_radius * Omega * (math.sin(lat_center) * geom.coslat - math.cos(lat_center) * numpy.cos(geom.lon - lon_center) * geom.sinlat)
   v = earth_radius * Omega * numpy.cos(lat_center) * numpy.sin(geom.lon - lon_center)
   u1, u2 = wind2contra(u, v, geom)

   return u1, u2, h, h_analytic

def williamson_case1(geom, metric, param):
   print("---------------------------------------------------------------")
   print("WILLIAMSON CASE 1 (Tracer): Cosine Bell, Williamson et al.,1992")
   print("---------------------------------------------------------------")

   u1, u2 = solid_body_rotation(geom, metric, param)

   # Initialize gaussian bell
   lon_center = 3.0 * math.pi / 2.0
   lat_center = 0.0

   h0 = 1000.0

   radius = 1.0 / 3.0

   dist = numpy.arccos(math.sin(lat_center) * geom.sinlat + math.cos(lat_center) * geom.coslat * numpy.cos(geom.lon - lon_center))

   h = 0.5 * h0 * (1.0 + numpy.cos(math.pi * dist / radius)) * (dist <= radius)
   h_analytic = h

   return u1, u2, h, h_analytic

def williamson_case2(geom, metric, param):
   print("--------------------------------------------")
   print("WILLIAMSON CASE 2, Williamson et al. (1992) ")
   print("Steady state nonlinear geostrophic flow     ")
   print("--------------------------------------------")

   if abs(param.Williamson_angle) > 0.0:
      print("param.Williamson_angle != 0 not yet implemented for case 2")
      exit(0)

   u1, u2 = solid_body_rotation(geom, metric, param)

   # Global Steady State Nonlinear Zonal Geostrophic Flow
   gh0 = 29400.0
   u0 = 2.0 * math.pi * earth_radius / (12.0 * day_in_secs)

   sinα = math.sin(param.Williamson_angle)
   cosα = math.cos(param.Williamson_angle)

   h = (gh0 - (earth_radius * rotation_speed * u0 + (0.5 * u0**2)) \
     * (-geom.coslon * geom.coslat * sinα + geom.sinlat * cosα)**2) / gravity

   h_analytic = h

   return u1, u2, h, h_analytic

def williamson_case5(geom, metric, mtrx, param):
   print('--------------------------------------------')
   print('WILLIAMSON CASE 5, Williamson et al. (1992) ')
   print('Zonal Flow over an isolated mountain        ')
   print('--------------------------------------------')

   u0 = 20.0   # Max wind (m/s)
   h0 = 5960.0 # Mean height (m)

   u1, u2 = solid_body_rotation(geom, metric, param)

   h_star = (gravity*h0 - (earth_radius * rotation_speed * u0 + 0.5*u0**2)*(geom.sinlat)**2) / gravity

   # Isolated mountain
   hs0 = 2000.0
   rr = math.pi / 9.0

   # Mountain location
   lon_mountain = math.pi / 2.0
   lat_mountain = math.pi / 6.0

   r = numpy.sqrt(numpy.minimum(rr**2,(geom.lon-lon_mountain)**2 + (geom.lat-lat_mountain)**2))

   r_itf_i = numpy.sqrt(numpy.minimum(rr**2,(geom.lon_itf_i-lon_mountain)**2 + (geom.lat_itf_i-lat_mountain)**2))
   r_itf_j = numpy.sqrt(numpy.minimum(rr**2,(geom.lon_itf_j-lon_mountain)**2 + (geom.lat_itf_j-lat_mountain)**2))

   hsurf = hs0 * (1 - r / rr)

   nb_interfaces_horiz = param.nb_elements + 1
   hsurf_itf_i = numpy.zeros((param.nb_elements+2, param.nbsolpts*param.nb_elements, 2))
   hsurf_itf_j = numpy.zeros((param.nb_elements+2, 2, param.nbsolpts*param.nb_elements))

   for itf in range(nb_interfaces_horiz):
      elem_L = itf
      elem_R = itf + 1

      hsurf_itf_i[elem_L, :, 1] = hs0 * (1 - r_itf_i[:, itf] / rr)
      hsurf_itf_i[elem_R, :, 0] = hsurf_itf_i[elem_L, :, 1]

      hsurf_itf_j[elem_L, 1, :] = hs0 * (1 - r_itf_j[itf, :] / rr)
      hsurf_itf_j[elem_R, 0, :] = hsurf_itf_j[elem_L, 1, :]

   ni, nj = geom.lon.shape
   dzdx1 = numpy.zeros((ni, nj))
   dzdx2 = numpy.zeros((ni, nj))

   offset = 1 # Offset due to the halo
   for elem in range(param.nb_elements):
      epais = elem * param.nbsolpts + numpy.arange(param.nbsolpts)

      # --- Direction x1
      dzdx1[:, epais] = ( hsurf[:,epais] @ mtrx.diff_solpt_tr + hsurf_itf_i[elem+offset,:,:] @ mtrx.correction_tr ) * 2.0 / geom.Δx1

      # --- Direction x2
      dzdx2[epais,:] = ( mtrx.diff_solpt @ hsurf[epais,:] + mtrx.correction @ hsurf_itf_j[elem+offset,:,:] ) * 2.0 / geom.Δx2

   h = h_star - hsurf

   return u1, u2, h, None, hsurf, dzdx1, dzdx2



def williamson_case6(geom, metric, param):
   print("--------------------------------------------")
   print("WILLIAMSON CASE 6, Williamson et al. (1992) ")
   print("Rossby-Haurwitz wave                        ")
   print("--------------------------------------------")

   # Rossby-Haurwitz wave

   R = 4

   omega = 7.848e-6
   K     = omega
   h0    = 8000.0

   A = omega/2.0 * (2.0 * rotation_speed + omega) * geom.coslat**2 + (K**2) / 4.0 * geom.coslat**(2*R) \
      * ( (R+1) * geom.coslat**2 + (2.0 * R**2 - R - 2.0) - 2.0 * (R**2) * geom.coslat**(-2) )

   B = 2.0 * (rotation_speed+omega) * K / ((R + 1) * (R + 2)) * geom.coslat**R * ( (R**2 + 2 * R + 2) - (R + 1)**2 * geom.coslat**2 )

   C = (K**2) / 4.0 * geom.coslat**(2*R) * ( (R + 1) * (geom.coslat**2) - (R + 2.0) )

   h = h0 + ( earth_radius**2 * A + earth_radius**2*B*numpy.cos(R * geom.lon) + earth_radius**2 * C * numpy.cos(2.0 * R * geom.lon) ) / gravity

   u = earth_radius * omega * geom.coslat + earth_radius * K * geom.coslat**(R-1) * \
         ( R*geom.sinlat**2 - geom.coslat**2 ) * numpy.cos(R*geom.lon)
   v = -earth_radius * K * R * geom.coslat**(R-1) * geom.sinlat * numpy.sin(R*geom.lon)

   u1, u2 = wind2contra(u, v, geom)

   return u1, u2, h, None

def case_galewsky(geom, metric, param):
   print("--------------------------------------------")
   print("CASE 8, Galewsky et al. (2004)              ")
   print("Barotropic wave                             ")
   print("--------------------------------------------")

   h0 = 10158.18617045463179
   h_hat = 120.0
   phi2 = math.pi / 4.0
   alpha = 1.0 / 3.0
   beta  = 1.0 / 15.0

   ni, nj = geom.lon.shape

   u = numpy.zeros((ni,nj))
   v = numpy.zeros((ni,nj))
   h = numpy.zeros((ni,nj))

   for i in range(ni):
      for j in range(nj):

         # Calculate height field via numerical integration
         nIntervals = int((geom.lat[i,j] + 0.5 * math.pi) / (1.0e-2))

         if nIntervals < 1:
            nIntervals = 1

         latX = numpy.zeros(nIntervals+1)

         for k in range(nIntervals+1):
            latX[k] = - 0.5 * math.pi + ((geom.lat[i,j] + 0.5 * math.pi) / nIntervals) * k

         h_integrand = 0.0

         for k in range(nIntervals):
            for m in range(-1,2,2):
               dXeval = 0.5 * (latX[k+1] + latX[k]) + m * math.sqrt(1.0 / 3.0) * 0.5 * (latX[k+1] - latX[k])

               dU = eval_u_prime(dXeval)

               h_integrand += (2.0 * earth_radius * rotation_speed * math.sin(dXeval) + dU * math.tan(dXeval)) * dU

         h_integrand *= 0.5 * (latX[1] - latX[0])

         h[i,j] = h0 - h_integrand / gravity

         # Add perturbation
         h[i,j] += h_hat * math.cos(geom.lat[i,j]) * math.exp(-(geom.lon[i,j] / alpha)**2) * math.exp(-((phi2 - geom.lat[i,j]) / beta)**2)

         # Evaluate the velocity field
         u_p = eval_u_prime(geom.lat[i,j])

         if abs(math.cos(geom.lon[i,j])) < 1.0e-13:
            u[i,j] = u_p
         else:
            u[i,j] = (v[i,j] * math.sin(geom.lat[i,j]) * math.sin(geom.lon[i,j]) + u_p * math.cos(geom.lon[i,j])) / math.cos(geom.lon[i,j])

   u1, u2 = wind2contra(u, v, geom)

   return u1, u2, h, None

def case_matsuno(geom, metric, param):
   print("--------------------------------------------")
   print("CASE 9, Shamir et al.,2019,GMD,12,2181-2193 ")

   if param.matsuno_wave_type == 'Rossby': 
      print("The Matsuno baroclinic wave (Rosby)         ")
   elif param.matsuno_wave_type == 'EIG': 
      print("The Matsuno baroclinic wave (EIG)           ")
      print("--------------------------------------------")
   elif param.matsuno_wave_type == 'WIG': 
      print("The Matsuno baroclinic wave (WIG)           ")
   print("--------------------------------------------")

   ni, nj = geom.lon.shape

   u = numpy.zeros((ni,nj))
   v = numpy.zeros((ni,nj))
   h = numpy.zeros((ni,nj))

   h_analytic = numpy.zeros((ni,nj))

   for i in range(ni):
      for j in range(nj):
         h[i, j] = matsuno.eval_field(geom.lat[i,j], geom.lon[i,j], 0., field='phi', wave_type=param.matsuno_wave_type) / gravity
         u[i, j] = matsuno.eval_field(geom.lat[i,j], geom.lon[i,j], 0., field='u', wave_type=param.matsuno_wave_type)
         v[i, j] = matsuno.eval_field(geom.lat[i,j], geom.lon[i,j], 0., field='v', wave_type=param.matsuno_wave_type)

         h_analytic[i, j] = matsuno.eval_field(geom.lat[i,j], geom.lon[i,j], param.t_end, field='phi', wave_type=param.matsuno_wave_type) / gravity

   u1, u2 = wind2contra(u, v, geom)

   return u1, u2, h, h_analytic
