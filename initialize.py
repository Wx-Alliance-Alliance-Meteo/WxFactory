import math
import numpy
from constants import *
from wind2contra import *

def initialize(geom, case_number, α):

   ni, nj, _= geom.lon.shape

   if case_number == -1 or \
      case_number == 1  or \
      case_number == 2  or \
      case_number == 5:
      # Solid body rotation

      u1 = numpy.zeros((ni, nj, nbfaces))
      u2 = numpy.zeros((ni, nj, nbfaces))

      if case_number == 5:
         u0 = 20.0
         sinα = 0
         cosα = 1
      else:
         u0 = 2.0 * math.pi * earth_radius / (12.0 * day_in_secs)
         sinα = math.sin(α)
         cosα = math.cos(α)

      u1[:,:,0] = u0 / earth_radius * (cosα + geom.Y / (1.0 + geom.X**2) * sinα)
      u2[:,:,0] = u0 * geom.X / (earth_radius * (1.0 + geom.Y**2)) * (geom.Y * cosα - sinα)

      u1[:,:,1] = u0 / earth_radius * (cosα - geom.X * geom.Y / (1.0 + geom.X**2) * sinα)
      u2[:,:,1] = u0 / earth_radius * (geom.X * geom.Y / (1.0 + geom.Y**2) * cosα - sinα)

      u1[:,:,2] = u0 / earth_radius * (cosα - geom.Y / (1.0 + geom.X**2) * sinα)
      u2[:,:,2] = u0 * geom.X / (earth_radius * (1.0 + geom.Y**2)) * (geom.Y * cosα + sinα)

      u1[:,:,3] = u0 / earth_radius * (cosα + geom.X * geom.Y / (1.0 + geom.X**2) * sinα)
      u2[:,:,3] = u0 / earth_radius * (geom.X * geom.Y / (1.0 + geom.Y**2) * cosα + sinα)

      u1[:,:,4] = u0 / earth_radius * (- geom.Y / (1.0 + geom.X**2) * cosα + sinα)
      u2[:,:,4] = u0 * geom.X / (earth_radius * (1.0 + geom.Y**2)) * (cosα + geom.Y * sinα)

      u1[:,:,5] = u0 / earth_radius * (geom.Y / (1.0 + geom.X**2) * cosα - sinα)
      u2[:,:,5] =-u0 * geom.X / (earth_radius * (1.0 + geom.Y**2)) * (cosα + geom.Y * sinα)


   if case_number == 1:
      # Initialize gaussian bell

      lon_center = 3.0 * math.pi / 2.0
      lat_center = 0.0

      h0 = 1000.0

      radius = 1.0 / 3.0

      dist = numpy.arccos(math.sin(lat_center) * numpy.sin(geom.lat) + math.cos(lat_center) * numpy.cos(geom.lat) * numpy.cos(geom.lon - lon_center))

      h = 0.5 * h0 * (1.0 + numpy.cos(math.pi * dist / radius)) * (dist <= radius)

      h_analytic = h
      hsurf = numpy.zeros((ni, nj, nbfaces))


   elif case_number == 2:

      # Global Steady State Nonlinear Zonal Geostrophic Flow
      gh0 = 29400.0
      u0 = 2.0 * math.pi * earth_radius / (12.0 * day_in_secs)

      sinα = math.sin(α)
      cosα = math.cos(α)

      sinlat = numpy.sin(geom.lat)
      coslat = numpy.cos(geom.lat)
      coslon = numpy.cos(geom.lon)

      h = (gh0 - (earth_radius * rotation_speed * u0 + (0.5 * u0**2)) \
        * (-coslon * coslat * sinα + sinlat * cosα)**2) / gravity

      hsurf = numpy.zeros_like(h)

   elif case_number == 5:

      u0 = 20.0   # Max wind (m/s)
      h0 = 5960.0 # Mean height (m)

      h_star = (gravity*h0 - (earth_radius * rotation_speed * u0 + 0.5*u0**2)*(numpy.sin(geom.lat))**2) / gravity

      # Isolated mountain
      hs0 = 2000.0
      rr = math.pi / 9.0

      # Mountain location
      lon_mountain = math.pi / 2.0
      lat_mountain = math.pi / 6.0

      r = numpy.sqrt(numpy.minimum(rr**2,(geom.lon-lon_mountain)**2 + (geom.lat-lat_mountain)**2))

      hsurf = hs0 * (1-r / rr)

      h = h_star - hsurf

   elif case_number == 6:

      # Rossby-Haurwitz wave

      R = 4.0

      omega = 7.848e-6
      K     = omega
      h0    = 8000.0

      A = omega/2 * (2*rotation_speed+omega) * numpy.cos(geom.lat)**2 + (K**2)/4 * numpy.cos(geom.lat)**(2*R) \
         * ( (R+1)*numpy.cos(geom.lat)**2 + (2*R**2-R-2) - 2*(R**2)*numpy.cos(geom.lat)**(-2) )

      B = 2 * (rotation_speed+omega) * K / ((R+1)*(R+2)) * numpy.cos(geom.lat)**R * ( (R**2+2*R+2) - (R+1)**2*numpy.cos(geom.lat)**2 )

      C = (K**2)/4 * numpy.cos(geom.lat)**(2*R) * ( (R+1)*(numpy.cos(geom.lat)**2) - (R+2) )

      h = h0 + ( earth_radius**2*A + earth_radius**2*B*numpy.cos(R*geom.lon) + earth_radius**2*C*numpy.cos(2*R*geom.lon) ) / gravity

      u = earth_radius * omega * numpy.cos(geom.lat) + earth_radius * K * numpy.cos(geom.lat)**(R-1) * \
            ( R*numpy.sin(geom.lat)**2-numpy.cos(geom.lat)**2 ) * numpy.cos(R*geom.lon)
      v = -earth_radius * K * R * numpy.cos(geom.lat)**(R-1) * numpy.sin(geom.lat) * numpy.sin(R*geom.lon)

      u1, u2 = wind2contra(u, v, geom)

      hsurf = numpy.zeros_like(h)

#      u = u1_contra * earth_radius * 2/grd.elementSize
#      v = u2_contra * earth_radius * 2/grd.elementSize
   Q = numpy.zeros((ni, nj, nbfaces, nb_equations))

   Q[:,:,:,0] = h
   Q[:,:,:,1] = h * u1
   Q[:,:,:,2] = h * u2

   return Q, hsurf
