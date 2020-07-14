import numpy
import math

def sph2cart(az, elev, radius):
   """ Transform spherical to Cartesian coordinates.
      [X,Y,Z] = sph2cart(TH,PHI,radius) transforms corresponding elements of
      data stored in spherical coordinates (azimuth TH, elevation PHI,
      radius radius) to Cartesian coordinates X,Y,Z.  The arrays TH, PHI, and
      radius must be the same size (or any of them can be scalar).  TH and
      PHI must be in radians.

      TH is the counterclockwise angle in the xy plane measured from the
      positive x axis.  PHI is the elevation angle from the xy plane.
   """

   z        = radius * numpy.sin(elev)
   rcoselev = radius * numpy.cos(elev)
   x        = rcoselev * numpy.cos(az)
   y        = rcoselev * numpy.sin(az)

   return x, y, z

def cart2sph(x,y,z):
   """ Transform Cartesian to spherical coordinates.
      az,elev,r = cart2sph(X,Y,Z) transforms corresponding elements of
      data stored in Cartesian coordinates X,Y,Z to spherical
      coordinates (azimuth, elevation, and radius).  The arrays
      X,Y, and Z must be the same size (or any of them can be scalar).
      az and elev are returned in radians.

      az is the counterclockwise angle in the xy plane measured from the
      positive x axis.  elev is the elevation angle from the xy plane.
   """

   hypotxy = numpy.hypot(x, y)
   r       = numpy.hypot(hypotxy, z)
   elev    = numpy.arctan2(z, hypotxy)
   az      = numpy.arctan2(y, x)

   # Map to the interval [0, 2 pi]
   az[az<0.] += 2. * math.pi

   return az, elev, r
