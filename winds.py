import numpy
import math

def wind2contra(u, v, geom):
   # Convert winds coords to spherical basis
   lambda_dot = u / (geom.earth_radius * geom.coslat)
   phi_dot    = v / geom.earth_radius

   denom = numpy.sqrt( (math.cos(geom.lat_p) + geom.X * math.sin(geom.lat_p)*math.sin(geom.angle_p) - geom.Y * math.sin(geom.lat_p)*math.cos(geom.angle_p))**2 + (geom.X * math.cos(geom.angle_p) + geom.Y * math.sin(geom.angle_p))**2 )

   dx1dlon = math.cos(geom.lat_p) * math.cos(geom.angle_p) + ( geom.X * geom.Y * math.cos(geom.lat_p) * math.sin(geom.angle_p) - geom.Y * math.sin(geom.lat_p) ) / (1. + geom.X**2)
   dx2dlon = ( geom.X * geom.Y * math.cos(geom.lat_p) * math.cos(geom.angle_p) + geom.X * math.sin(geom.lat_p) ) / (1. + geom.Y**2) + math.cos(geom.lat_p) * math.sin(geom.angle_p)

   dx1dlat = -geom.delta2 * ( (math.cos(geom.lat_p)*math.sin(geom.angle_p) + geom.X * math.sin(geom.lat_p))/(1. + geom.X**2) ) / denom
   dx2dlat = geom.delta2 * ( (math.cos(geom.lat_p)*math.cos(geom.angle_p) - geom.Y * math.sin(geom.lat_p))/(1. + geom.Y**2) ) / denom
   
   # transform to the reference element

   u1_contra = ( dx1dlon * lambda_dot + dx1dlat * phi_dot ) * 2. / geom.Δx1
   u2_contra = ( dx2dlon * lambda_dot + dx2dlat * phi_dot ) * 2. / geom.Δx2

   return u1_contra, u2_contra


def contra2wind(u1, u2, geom):
   # Convert from reference element to "physical winds"

   u1_contra = u1*geom.Δx1/2.
   u2_contra = u2*geom.Δx2/2.

   denom = (math.cos(geom.lat_p) + geom.X * math.sin(geom.lat_p) * math.sin(geom.angle_p) - geom.Y * math.sin(geom.lat_p) * math.cos(geom.angle_p))**2 + (geom.X * math.cos(geom.angle_p) + geom.Y * math.sin(geom.angle_p))**2

   dlondx1 = (math.cos(geom.lat_p) * math.cos(geom.angle_p) - geom.Y * math.sin(geom.lat_p)) * (1. + geom.X**2) / denom

   dlondx2 = (math.cos(geom.lat_p) * math.sin(geom.angle_p) + geom.X * math.sin(geom.lat_p)) * (1. + geom.Y**2) / denom

   denom[:,:] = numpy.sqrt( (math.cos(geom.lat_p) + geom.X * math.sin(geom.lat_p)*math.sin(geom.angle_p) - geom.Y * math.sin(geom.lat_p)*math.cos(geom.angle_p))**2 + (geom.X * math.cos(geom.angle_p) + geom.Y * math.sin(geom.angle_p))**2 )

   dlatdx1 = - ( (geom.X * geom.Y * math.cos(geom.lat_p) * math.cos(geom.angle_p) + geom.X * math.sin(geom.lat_p) + (1. + geom.Y**2) * math.cos(geom.lat_p) * math.sin(geom.angle_p)) * (1. + geom.X**2) ) / ( geom.delta2 * denom)

   dlatdx2 = ( ((1. + geom.X**2) * math.cos(geom.lat_p) * math.cos(geom.angle_p) + geom.X * geom.Y * math.cos(geom.lat_p) * math.sin(geom.angle_p) - geom.Y * math.sin(geom.lat_p)) * (1. + geom.Y**2) ) / ( geom.delta2 * denom)

   u = ( dlondx1 * u1_contra + dlondx2 * u2_contra ) * geom.coslat * geom.earth_radius
   v = ( dlatdx1 * u1_contra + dlatdx2 * u2_contra ) * geom.earth_radius

   return u, v
