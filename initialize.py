import math
import numpy
from constants import *

def initialize(geom, case_number, α):

   ni, nj, _= geom.lon.shape

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

   if case_number == 1:
      # Solid body rotation

      u1 = numpy.zeros((ni, nj, nbfaces))
      u2 = numpy.zeros((ni, nj, nbfaces))

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

#      u = u1_contra * earth_radius * 2/grd.elementSize
#      v = u2_contra * earth_radius * 2/grd.elementSize


   Q = numpy.zeros((ni, nj, nbfaces, nb_equations))

   Q[:,:,:,0] = h
   Q[:,:,:,1] = h * u1
   Q[:,:,:,2] = h * u2

   return Q
