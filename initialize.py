import math
import numpy
from definitions import *
from wind2contra import *

import matsuno

class Topo:
   def __init__(self, hsurf, dzdx1, dzdx2):
      self.hsurf = hsurf
      self.dzdx1 = dzdx1
      self.dzdx2 = dzdx2

def initialize(geom, metric, mtrx, nbsolpts, nb_elements_horiz, case_number, Williamson_angle, t_anal=0):

   ni, nj = geom.lon.shape

   h_analytic = None
   hsurf = numpy.zeros((ni, nj))
   dzdx1 = numpy.zeros((ni, nj))
   dzdx2 = numpy.zeros((ni, nj))

   if case_number <= 1:
      # advection only, save u1 and u2
      Q = numpy.zeros((nb_equations+2, ni, nj))
   else:
      Q = numpy.zeros((nb_equations, ni, nj))

   if case_number == -1 or \
      case_number == 1  or \
      case_number == 2  or \
      case_number == 5:
      # Solid body rotation

      if case_number == 5:
         u0 = 20.0
         sinα = 0
         cosα = 1
      else:
         u0 = 2.0 * math.pi * earth_radius / (12.0 * day_in_secs)
         sinα = math.sin(Williamson_angle)
         cosα = math.cos(Williamson_angle)

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


   if case_number == 0:
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
      h_analytic = 1.0 - numpy.tanh( (rho / gamma) * numpy.sin(lonR - Omega * t_anal) )

      u = earth_radius * Omega * (math.sin(lat_center) * geom.coslat - math.cos(lat_center) * numpy.cos(geom.lon - lon_center) * geom.sinlat)
      v = earth_radius * Omega * numpy.cos(lat_center) * numpy.sin(geom.lon - lon_center)
      u1, u2 = wind2contra(u, v, geom)


   elif case_number == 1:
      print("---------------------------------------------------------------")
      print("WILLIAMSON CASE 1 (Tracer): Cosine Bell, Williamson et al.,1992")
      print("---------------------------------------------------------------")

      # Initialize gaussian bell
      lon_center = 3.0 * math.pi / 2.0
      lat_center = 0.0

      h0 = 1000.0

      radius = 1.0 / 3.0

      dist = numpy.arccos(math.sin(lat_center) * geom.sinlat + math.cos(lat_center) * geom.coslat * numpy.cos(geom.lon - lon_center))

      h = 0.5 * h0 * (1.0 + numpy.cos(math.pi * dist / radius)) * (dist <= radius)
      h_analytic = h

   elif case_number == 2:
      print("--------------------------------------------")
      print("WILLIAMSON CASE 2, Williamson et al. (1992) ")
      print("Steady state nonlinear geostrophic flow     ")
      print("--------------------------------------------")

      if abs(Williamson_angle) > 0.0:
         print("Williamson_angle != 0 not yet implemented for case 2")
         exit(0)

      # Global Steady State Nonlinear Zonal Geostrophic Flow
      gh0 = 29400.0
      u0 = 2.0 * math.pi * earth_radius / (12.0 * day_in_secs)

      sinα = math.sin(Williamson_angle)
      cosα = math.cos(Williamson_angle)

      h = (gh0 - (earth_radius * rotation_speed * u0 + (0.5 * u0**2)) \
        * (-geom.coslon * geom.coslat * sinα + geom.sinlat * cosα)**2) / gravity

      h_analytic = h

   elif case_number == 5:

      u0 = 20.0   # Max wind (m/s)
      h0 = 5960.0 # Mean height (m)

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

      nb_interfaces_horiz = nb_elements_horiz + 1
      hsurf_itf_i = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz))
      hsurf_itf_j = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz))

      for itf in range(nb_interfaces_horiz):
         elem_L = itf
         elem_R = itf + 1

         hsurf_itf_i[elem_L, 1, :] = hs0 * (1 - r_itf_i[:, itf] / rr)
         hsurf_itf_i[elem_R, 0, :] = hsurf_itf_i[elem_L, 1, :]

         hsurf_itf_j[elem_L, 1, :] = hs0 * (1 - r_itf_j[itf, :] / rr)
         hsurf_itf_j[elem_R, 0, :] = hsurf_itf_j[elem_L, 1, :]

      offset = 1 # Offset due to the halo
      for elem in range(nb_elements_horiz):
         epais = elem * nbsolpts + numpy.arange(nbsolpts)

         # --- Direction x1
         dzdx1[:, epais] = ( mtrx.diff_solpt @ hsurf[:,epais].T + mtrx.correction @ hsurf_itf_i[elem+offset,:,:] ).T # TODO : éviter la transpose ...

         # --- Direction x2
         dzdx2[epais,:] = mtrx.diff_solpt @ hsurf[epais,:] + mtrx.correction @ hsurf_itf_j[elem+offset,:,:]

      # Convert derivative from standard element to the cubed-sphere domain
      dzdx1 *= 2.0 / geom.Δx1
      dzdx2 *= 2.0 / geom.Δx2

      h = h_star - hsurf

   elif case_number == 6:
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


   elif case_number == 8:
      print("--------------------------------------------")
      print("CASE 8, Galewsky et al. (2004)              ")
      print("Barotropic wave                             ")
      print("--------------------------------------------")

      H0 = 10158.18617045463179
      HHat = 120.0
      HPhi2 = math.pi / 4.0
      HAlpha = 1.0 / 3.0
      HBeta = 1.0 / 15.0
      Alpha = 0.0

      u = numpy.zeros((ni,nj))
      v = numpy.zeros((ni,nj))
      h = numpy.zeros((ni,nj))

      for i in range(ni):
         for j in range(nj):

            # Calculate height field via numerical integration
            nIntervals = int((geom.lat[i,j] + 0.5 * math.pi) / (1.0e-2))

            if nIntervals < 1:
               nIntervals = 1

            dLatX = numpy.zeros(nIntervals+1)

            for k in range(nIntervals+1):
               dLatX[k] = - 0.5 * math.pi + ((geom.lat[i,j] + 0.5 * math.pi) / nIntervals) * k

            dH = 0.0

            for k in range(nIntervals):
               for m in range(-1,2,2):
                  dXeval = 0.5 * (dLatX[k+1] + dLatX[k]) + m * math.sqrt(1.0 / 3.0) * 0.5 * (dLatX[k+1] - dLatX[k])

                  dU = EvaluateUPrime(geom.lon[i,j], dXeval)

                  dH += (2.0 * earth_radius * rotation_speed * math.sin(dXeval) + dU * math.tan(dXeval)) * dU

            dH *= 0.5 * (dLatX[1] - dLatX[0])

            h[i,j] = H0 - dH / gravity

            # Add perturbation
            h[i,j] += HHat * math.cos(geom.lat[i,j]) * math.exp(-(geom.lon[i,j]**2 / (HAlpha * HAlpha))) * math.exp(-((HPhi2 - geom.lat[i,j]) * (HPhi2 - geom.lat[i,j]) / (HBeta * HBeta)))

            # Evaluate the velocity field
            dUP = EvaluateUPrime(geom.lon[i,j], geom.lat[i,j])

            v[i,j] = (- dUP * math.sin(Alpha) * math.sin(geom.lon[i,j])) / math.cos(geom.lat[i,j])

            if abs(math.cos(geom.lon[i,j])) < 1.0e-13:
               if abs(Alpha) > 1.0e-13:
                  if math.cos(geom.lon[i,j]) > 0.0:
                     u[i,j] = - v[i,j] * math.cos(geo.lat[i,j]) / math.tan(Alpha)
                  else:
                     u[i,j] = v[i,j] * math.cos(geom.lat[i,j]) / math.tan(Alpha)
               else:
                  u[i,j] = dUP
            else:
               u[i,j] = (v[i,j] * math.sin(geom.lat[i,j]) * math.sin(geom.lon[i,j]) + dUP * math.cos(geom.lon[i,j])) / math.cos(geom.lon[i,j])

      u1, u2 = wind2contra(u, v, geom)

   elif case_number == 9:
      print("--------------------------------------------")
      print("CASE 9, Shamir et al.,2019,GMD,12,2181-2193 ")
      print("The Matsuno baroclinic wave                 ")
      print("--------------------------------------------")

      u = numpy.zeros((ni,nj))
      v = numpy.zeros((ni,nj))
      h = numpy.zeros((ni,nj))

      for i in range(ni):
         for j in range(nj):
            h[i, j] = matsuno.eval_field(geom.lat[i,j], geom.lon[i,j], 0., field='phi') / gravity
            u[i, j] = matsuno.eval_field(geom.lat[i,j], geom.lon[i,j], 0., field='u')
            v[i, j] = matsuno.eval_field(geom.lat[i,j], geom.lon[i,j], 0., field='v')

      u1, u2 = wind2contra(u, v, geom)


   Q[idx_h,:,:]   = h

   if case_number <= 1:
      # advection only
      Q[idx_u1,:,:] = u1
      Q[idx_u2,:,:] = u2
   else:
      Q[idx_hu1,:,:] = h * u1
      Q[idx_hu2,:,:] = h * u2

   return Q, Topo(hsurf, dzdx1, dzdx2), h_analytic


def EvaluateUPrime(dLonP, dLatP):
   U0 = 80.0
   Theta0 = math.pi / 7.0
   Theta1 = math.pi / 2.0 - Theta0

   if dLatP < Theta0:
      return 0.0
   elif dLatP > Theta1:
      return 0.0

   dNormalizer = math.exp(- 4.0 / (Theta1 - Theta0) / (Theta1 - Theta0))

   dUp = math.exp(1.0 / (dLatP - Theta0) / (dLatP - Theta1))

   return U0 / dNormalizer * dUp
