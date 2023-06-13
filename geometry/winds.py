import numpy
import math
from typing import Union

from .cubed_sphere  import CubedSphere
from .metric        import Metric3DTopo

def wind2contra_2d(u : Union[float, numpy.ndarray], v : Union[float, numpy.ndarray], geom : CubedSphere):
   '''Convert wind fields from the spherical basis (zonal, meridional) to panel-appropriate contrvariant winds, in two dimensions

   Parameters:
   ----------
   u : float | numpy.ndarray
      Input zonal winds, in meters per second
   v : float | numpy.ndarray
      Input meridional winds, in meters per second
   geom : CubedSphere
      Geometry object (CubedSphere), describing the grid configuration and globe paramters.  Required parameters:
      earth_radius, coslat, lat_p, angle_p, X, Y, delta2

   Returns:
   -------
   (u1_contra, u2_contra) : tuple
      Tuple of contravariant winds'''
   
   # Convert winds coords to spherical basis

   if (geom.nk > 1 and geom.deep):
      # In 3D code with the deep atmosphere, the conversion to λ and φ
      # uses the full radial height of the grid point:
      lambda_dot = u / ((geom.earth_radius + geom.coordVec_gnom[2,:,:,:]) * geom.coslat)
      phi_dot    = v / (geom.earth_radius + geom.coordVec_gnom[2,:,:,:])
   else:
      # Otherwise, the conversion uses just the planetary radius, with no
      # correction for height above the surface
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

def wind2contra_3d(u : Union[float, numpy.ndarray],
                   v : Union[float, numpy.ndarray],
                   w : Union[float, numpy.ndarray],
                   geom : CubedSphere,
                   metric : Metric3DTopo):
   '''Convert wind fields from spherical values (zonal, meridional, vertical) to contravariant winds
   on a terrain-following grid.

   Parameters:
   ----------
   u : float | numpy.ndarray
      Input zonal winds, in meters per second
   v : float | numpy.ndarray
      Input meridional winds, in meters per second
   w : float | numpy.ndarray
      Input vertical winds, in meters per second
   geom : CubedSphere
      Geometry object (CubedSphere), describing the grid configuration and globe paramters.  Required parameters:
      earth_radius, coslat, lat_p, angle_p, X, Y, delta2
   metric : Metric3DTopo
      Metric object containing H_contra and inv_dzdeta parameters

   Returns:
   -------
   (u1_contra, u2_contra, u3_contra) : tuple
      Tuple of contravariant winds
   '''

   # First, re-use wind2contra_2d to get preliminary values for u1_contra and u2_contra.  We will update this with the
   # contribution from vertical velocity in a second step.

   (u1_contra, u2_contra) = wind2contra_2d(u,v,geom)

   # Second, convert w to _covariant_ u3, which points in the vertical direction regardless of topography.  We do this
   # by multiplying by dz/deta, or dividing by metric.inv_dzdeta  (equivalently, taking the dot product with the e_3 basis
   # vector)
   u3_cov = w/metric.inv_dzdeta

   # Now, convert covariant u3 to contravariant components.  Because topography, u^3 is normal to the terrain-following x1 and
   # x2 coordinates, implying that u^3 has horizontal components.  To cancel this, we need to adjust u^1 and u^2 accordingly.

   u1_contra += metric.H_contra[0,2,:,:,:]*u3_cov
   u2_contra += metric.H_contra[1,2,:,:,:]*u3_cov
   u3_contra = metric.H_contra[2,2,:,:,:]*u3_cov
   
   return (u1_contra, u2_contra, u3_contra)

def contra2wind_2d(u1 : Union[float, numpy.ndarray],
                   u2 : Union[float, numpy.ndarray],
                   geom : CubedSphere):
   ''' Convert from reference element to "physical winds", in two dimensions

   Parameters:
   -----------
   u1 : float | numpy.ndarray
      Contravariant winds along first component (X)
   u2 : float | numpy.ndarray
      Contravariant winds along second component (Y)
   geom : CubedSphere
      Geometry object, containing:
         Δx1, Δx2, lat_p, angle_p, X, Y, coslat, earth_radius

   Returns:
   --------
   (u, v) : tuple
      Zonal/meridional winds, in m/s
   '''

   u1_contra = u1*geom.Δx1/2.
   u2_contra = u2*geom.Δx2/2.

   denom = (math.cos(geom.lat_p) + geom.X * math.sin(geom.lat_p) * math.sin(geom.angle_p) - geom.Y * math.sin(geom.lat_p) * math.cos(geom.angle_p))**2 + (geom.X * math.cos(geom.angle_p) + geom.Y * math.sin(geom.angle_p))**2

   dlondx1 = (math.cos(geom.lat_p) * math.cos(geom.angle_p) - geom.Y * math.sin(geom.lat_p)) * (1. + geom.X**2) / denom

   dlondx2 = (math.cos(geom.lat_p) * math.sin(geom.angle_p) + geom.X * math.sin(geom.lat_p)) * (1. + geom.Y**2) / denom

   denom[:,:] = numpy.sqrt( (math.cos(geom.lat_p) + geom.X * math.sin(geom.lat_p)*math.sin(geom.angle_p) - geom.Y * math.sin(geom.lat_p)*math.cos(geom.angle_p))**2 + (geom.X * math.cos(geom.angle_p) + geom.Y * math.sin(geom.angle_p))**2 )

   dlatdx1 = - ( (geom.X * geom.Y * math.cos(geom.lat_p) * math.cos(geom.angle_p) + geom.X * math.sin(geom.lat_p) + (1. + geom.Y**2) * math.cos(geom.lat_p) * math.sin(geom.angle_p)) * (1. + geom.X**2) ) / ( geom.delta2 * denom)

   dlatdx2 = ( ((1. + geom.X**2) * math.cos(geom.lat_p) * math.cos(geom.angle_p) + geom.X * geom.Y * math.cos(geom.lat_p) * math.sin(geom.angle_p) - geom.Y * math.sin(geom.lat_p)) * (1. + geom.Y**2) ) / ( geom.delta2 * denom)

   if (geom.nk > 1 and geom.deep):
      # If we are in a 3D geometry with the deep atmosphere, the conversion from
      # contravariant → spherical → zonal/meridional winds uses the full radial distance
      # at the last step
      u = ( dlondx1 * u1_contra + dlondx2 * u2_contra ) * geom.coslat * (geom.earth_radius + geom.coordVec_gnom[2,:,:,:])
      v = ( dlatdx1 * u1_contra + dlatdx2 * u2_contra ) * (geom.earth_radius + geom.coordVec_gnom[2,:,:,:])
   else:
      # Otherwise, the conversion is based on the spherical radius only, with no height correction
      u = ( dlondx1 * u1_contra + dlondx2 * u2_contra ) * geom.coslat * geom.earth_radius
      v = ( dlatdx1 * u1_contra + dlatdx2 * u2_contra ) * geom.earth_radius

   return u, v

def contra2wind_3d(u1_contra : numpy.ndarray,
                   u2_contra : numpy.ndarray,
                   u3_contra : numpy.ndarray,
                   geom : CubedSphere,
                   metric : Metric3DTopo):
   ''' contra2wind_3d: convert from contravariant wind fields to "physical winds" in three dimensions
   
   This function transforms the contravariant fields u1, u2, and u3 into their physical equivalents, assuming a cubed-sphere-like
   grid.  In particular, we assume that the vertical coordinate η is the only mapped coordinate, so u3 can be ignored in computing
   the zonal (u) and meridional (w) wind.

   This function inverts the cubed-sphere latlon/XY coordinate transform to give u and v, taking into account the panel rotation,
   and it forms a covariant u3 field to derive w, removing any horizontal component that would otherwise be included inside the
   contravariant u3.

   Parameters:
   -----------
   u1_contra: numpy.ndarray
      contravariant wind, u1 component
   u2_contra: numpy.ndarray
      contravariant wind, u2 component
   u3_contra: numpy.ndarray
      contravariant wind, u3 component
   geom: CubedSphere
      geometry object, implementing:
         Δx1, Δx2, lat_p, angle_p, X, Y, coslat, earth_radius
   metric: Metric3DTopo
      metric object, implementing H_cov (covariant spatial metric) and inv_dzdeta

   Retruns:
   (u, v, w) : tuple
      zonal, meridional, and vertical winds (m/s)
   '''

   # First, re-use contra2wind_2d to compute outuput u and v.  No need to reinvent the wheel
   u, v = contra2wind_2d(u1_contra,u2_contra,geom)

   # Now, form covariant u3 by taking the appropriate multiplication with the covariant metric to lower the index
   u3_cov = u1_contra[:,:,:]*metric.H_cov[2,0,:,:,:] + \
            u2_contra[:,:,:]*metric.H_cov[2,1,:,:,:] + \
            u3_contra[:,:,:]*metric.H_cov[2,2,:,:,:]

   # Covariant u3 now points straight "up", but it is expressed in η units, implicitly multiplied
   # by the covariant basis vector.
   # To convert to physical units, think of dz/dz=1 m/m.  In the covariant expression, however:
   # dz/dz = 1m/m = (z)_(,3) * e^3
   # but (z)_(,3) is the definition of dzdeta, which absorbs the (Δη/4) scaling term of the numerical
   # differentiation.  To cancel this, e^3 = inv_dzeta * (zhat), giving:

   w = u3_cov[:,:,:]*metric.inv_dzdeta[:,:,:] # Multiply by (dz/dη)^(-1), or e^3

   return (u,v,w)

