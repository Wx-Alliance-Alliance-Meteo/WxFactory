import math
import numpy
from definitions import *
from wind2contra import *

def initialize(geom, metric, case_number, Williamson_angle):

   ni, nj = geom.lon.shape

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
      print("CASE0 (Tracer): Circular vortex, Nair and Machenhauer,2002    ")
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

      h     = 1.0 - numpy.tanh( (rho / gamma) * numpy.sin(lonR) )
      hsurf = numpy.zeros_like(h)

      u = earth_radius * Omega * (math.sin(lat_center) * geom.coslat - math.cos(lat_center) * numpy.cos(geom.lon - lon_center) * geom.sinlat)
      v = earth_radius * Omega * numpy.cos(lat_center) * numpy.sin(geom.lon - lon_center)
      u1, u2 = wind2contra(u, v, geom)


   elif case_number == 1:
      print("--------------------------------------------------------------")
      print("WILLIAMSON CASE1 (Tracer): Cosine Bell, Williamson et al.,1992")
      print("--------------------------------------------------------------")

      # Initialize gaussian bell
      lon_center = 3.0 * math.pi / 2.0
      lat_center = 0.0

      h0 = 1000.0

      radius = 1.0 / 3.0

      dist = numpy.arccos(math.sin(lat_center) * geom.sinlat + math.cos(lat_center) * geom.coslat * numpy.cos(geom.lon - lon_center))

      h = 0.5 * h0 * (1.0 + numpy.cos(math.pi * dist / radius)) * (dist <= radius)

      hsurf = numpy.zeros_like(h)


   elif case_number == 2:
      print("--------------------------------------------")
      print("WILLIAMSON CASE2, Williamson et al. (1992)  ")
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

      hsurf = numpy.zeros_like(h)

   elif case_number == 5:

      print("Not yet implemented")
      exit(0)

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

      hsurf = hs0 * (1 - r / rr)

      h = h_star - hsurf

   elif case_number == 6:
      print("--------------------------------------------")
      print("WILLIAMSON CASE6, Williamson et al. (1992)  ")
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

      hsurf = numpy.zeros_like(h)


   elif case_number == 8:
      print("--------------------------------------------")
      print("CASE8, Galewsky et al. (2004)               ")
      print("Barotropic wave                             ")
      print("--------------------------------------------")

      print("Not yet implemented")
      exit(0)



   Q[idx_h,:,:]   = h

   if case_number <= 1:
      # advection only
      Q[idx_u1,:,:] = u1
      Q[idx_u2,:,:] = u2
   else:
      Q[idx_hu1,:,:] = h * u1
      Q[idx_hu2,:,:] = h * u2

   return Q, hsurf
