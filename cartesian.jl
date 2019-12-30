function sph2cart(az, elev, radius)
   """ SPH2CART Transform spherical to Cartesian coordinates.
      [X,Y,Z] = SPH2CART(TH,PHI,radius) transforms corresponding elements of
      data stored in spherical coordinates (azimuth TH, elevation PHI,
      radius radius) to Cartesian coordinates X,Y,Z.  The arrays TH, PHI, and
      radius must be the same size (or any of them can be scalar).  TH and
      PHI must be in radians.

      TH is the counterclockwise angle in the xy plane measured from the
      positive x axis.  PHI is the elevation angle from the xy plane.
   """

   z        = radius .* sin.(elev)
   rcoselev = radius .* cos.(elev)
   x        = rcoselev .* cos.(az)
   y        = rcoselev .* sin.(az)

   return x, y, z
end

function cart2sph(x,y,z)
   """ CART2SPH Transform Cartesian to spherical coordinates.
      [TH,PHI,R] = CART2SPH(X,Y,Z) transforms corresponding elements of
      data stored in Cartesian coordinates X,Y,Z to spherical
      coordinates (azimuth TH, elevation PHI, and radius R).  The arrays
      X,Y, and Z must be the same size (or any of them can be scalar).
      TH and PHI are returned in radians.

      TH is the counterclockwise angle in the xy plane measured from the
      positive x axis.  PHI is the elevation angle from the xy plane.
   """

   hypotxy = hypot.(x, y)
   r       = hypot.(hypotxy, z)
   elev    = atan.(z, hypotxy)
   az      = atan.(y, x)

   return az,elev,r
end
