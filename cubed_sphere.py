import math
import numpy
import sphere
import quadrature
import definitions
import sympy

# For type hints
from parallel import Distributed_World
from program_options import Configuration

class cubed_sphere:
   def __init__(self, nb_elements_horizontal:int , nb_elements_vertical: int, nbsolpts: int, λ0: float, ϕ0: float, α0: float, ztop: float, ptopo: Distributed_World, param: Configuration):
      '''Initialized the cubed sphere geometry, for an earthlike sphere with no topography.
      
      This function initializes the basic cubed_sphere geometry object, which provides the parameters necessary to define
      the coodinates in both spherical coordinates and Cartesian space.  These parameters are then used to define the metric
      object.
      
      The cubed-sphere panelization is as follows:
      ```
            +---+
            | 4 |
        +---+---+---+---+
        | 3 | 0 | 1 | 2 |
        +---+---+---+---+
            | 5 |
            +---+
      ```
      ... where each panel has its own local (x1,x2) coordinate axis.  With typical parameters, panel 0
      contains the intersection of the prime meridian and equator, the equator runs through panels 3-0-1-2
      from west to east, panel 4 contains the north pole, and panel 5 contains the south pole.
      
      Parameters:
      -----------
      nb_elements_horizontal: int
         Number of elements in the (x1,x2) directions, per panel
      nb_elements_vertical: int
         Number of elements in the vertical, between 0 and ztop
      nbsolpts: int
         Number of nodal points in each of the (x1,x2,x3) dimensions inside the element
      λ0: float
         Grid rotation: physical longitude of the central point of the 0 panel
         Valid range ]-π/2,0]
      ϕ0: float
         Grid rotation: physical latitude of the central point of the 0 panel
         Valid range ]-π/4,π/4]
      α0: float
         Grid rotation: rotation of the central meridian of the 0 panel, relatve to true north
         Valid range ]-π/2,0]
      ztop: float
         Physical height of the model top, in meters.  x3 values will range from 0 to ztop, with
         the center of the planet located at x3=(-radius_earth).  Note that this parameter is
         defined prior to any topography mapping.
      ptopo: Distributed_World
         Wraps the parameters and helper functions necessary for MPI parallelism.  By assumption,
         each panel is a separate MPI process.
      param: Configuration
         Wraps parameters from the configuration pole that are not otherwise specified in this
         constructor.      
      '''

      panel_domain_x1 = (-math.pi/4, math.pi/4)
      panel_domain_x2 = (-math.pi/4, math.pi/4)

      Δx1_PE = (panel_domain_x1[1] - panel_domain_x1[0]) / ptopo.nb_lines_per_panel
      Δx2_PE = (panel_domain_x2[1] - panel_domain_x2[0]) / ptopo.nb_lines_per_panel

      PE_start_x1 = -math.pi/4 + ptopo.my_col * Δx1_PE
      PE_end_x1 = PE_start_x1 + Δx1_PE

      PE_start_x2 = -math.pi/4 + ptopo.my_row * Δx2_PE
      PE_end_x2 = PE_start_x2 + Δx2_PE

      PE_start_x3 = 0.
      PE_end_x3 = ztop

      domain_x1 = (PE_start_x1, PE_end_x1)
      domain_x2 = (PE_start_x2, PE_end_x2)
      domain_x3 = (PE_start_x3, PE_end_x3)

      nb_elements_x1 = nb_elements_horizontal
      nb_elements_x2 = nb_elements_horizontal
      nb_elements_x3 = nb_elements_vertical

      # Gauss-Legendre solution points
      solutionPoints_sym, solutionPoints, glweights = quadrature.gauss_legendre(nbsolpts)
      print(f'Solution points : {solutionPoints}')
      print(f'GL weights : {glweights}')

      # Extend the solution points to include -1 and 1
      extension = numpy.append(numpy.append([-1], solutionPoints), [1])
      extension_sym = solutionPoints_sym.copy()
      extension_sym.insert(0, sympy.sympify('-1'))
      extension_sym.append(sympy.sympify('1'))

      scaled_points = 0.5 * (1.0 + solutionPoints)

      # Equiangular coordinates
      Δx1 = (domain_x1[1] - domain_x1[0]) / nb_elements_x1
      Δx2 = (domain_x2[1] - domain_x2[0]) / nb_elements_x2
      Δx3 = (domain_x3[1] - domain_x3[0]) / nb_elements_x3

      interfaces_x1 = numpy.linspace(start = domain_x1[0], stop = domain_x1[1], num = nb_elements_x1 + 1)
      interfaces_x2 = numpy.linspace(start = domain_x2[0], stop = domain_x2[1], num = nb_elements_x2 + 1)
      interfaces_x3 = numpy.linspace(start = domain_x3[0], stop = domain_x3[1], num = nb_elements_x3 + 1)

      ni = nb_elements_x1 * len(solutionPoints)
      x1 = numpy.zeros(ni)
      for i in range(nb_elements_x1):
         idx = i * nbsolpts
         x1[idx : idx + nbsolpts] = interfaces_x1[i] + scaled_points * Δx1

      nj = nb_elements_x2 * len(solutionPoints)
      x2 = numpy.zeros(nj)
      for j in range(nb_elements_x2):
         idx = j * nbsolpts
         x2[idx : idx + nbsolpts] = interfaces_x2[j] + scaled_points * Δx2

      # x3 is η, the generalized vertical coordinate.  With no topography, this vertical coordinate
      # coincides with height above the surface.  With topography, this coordinate will be mapped
      # as z(x1,x2,η)
      if ztop > 0:
         nk = nb_elements_x3 * len(solutionPoints)
         x3 = numpy.zeros(nk)
         for k in range(nb_elements_x3):
            idx = k * nbsolpts
            x3[idx : idx + nbsolpts] = interfaces_x3[k] + scaled_points * Δx3
      else:
         nk = 1
         x3 = numpy.zeros(nk)

      X1, X2 = numpy.meshgrid(x1, x2)

      X2_itf_i, X1_itf_i,  = numpy.meshgrid(x2, interfaces_x1)
      X1_itf_j, X2_itf_j = numpy.meshgrid(x1, interfaces_x2)

      # Gnomonic coordinates
      X = numpy.tan(X1)
      Y = numpy.tan(X2)

      # TODO : terrain following
      height, _, _ = numpy.meshgrid(x3, x1, x2, indexing = 'ij') # TODO: terrain-following coordinate system

      X_itf_i = numpy.tan(X1_itf_i)
      Y_itf_i = numpy.tan(X2_itf_i)
      X_itf_j = numpy.tan(X1_itf_j)
      Y_itf_j = numpy.tan(X2_itf_j)

      delta2 = 1.0 + X**2 + Y**2
      delta  = numpy.sqrt(delta2)

      delta2_itf_i = 1.0 + X_itf_i**2 + Y_itf_i**2
      delta_itf_i  = numpy.sqrt(delta2_itf_i)

      delta2_itf_j = 1.0 + X_itf_j**2 + Y_itf_j**2
      delta_itf_j  = numpy.sqrt(delta2_itf_j)

      # Compute the parameters of the rotated grid

      if (λ0 > 0.) or (λ0 <= -math.pi / 2.):
         print('lambda0 not within the acceptable range of ]-pi/2 , 0]. Stopping.')
         exit(1)

      if (ϕ0 <= -math.pi/4.) or (ϕ0 > math.pi/4.):
         print('phi0 not within the acceptable range of ]-pi/4 , pi/4]. Stopping.')
         exit(1)

      if (α0 <= -math.pi/2.) or (α0 > 0.):
         print('alpha0 not within the acceptable range of ]-pi/2 , 0]. Stopping.')
         exit(1)

      c1=math.cos(λ0)
      c2=math.cos(ϕ0)
      c3=math.cos(α0)
      s1=math.sin(λ0)
      s2=math.sin(ϕ0)
      s3=math.sin(α0)

      if ptopo.my_panel == 0:
         lon_p = λ0
         lat_p = ϕ0
         angle_p = α0

      elif ptopo.my_panel == 1:
         lon_p = math.atan2(s1*s2*s3+c1*c3, c1*s2*s3-s1*c3)
         lat_p = -math.asin(c2*s3)
         angle_p = math.atan2(s2, c2*c3)

      elif ptopo.my_panel == 2:
         lon_p = math.atan2(-s1, -c1)
         lat_p = -ϕ0
         angle_p = -math.atan2(s3, c3)

      elif ptopo.my_panel == 3:
         lon_p = math.atan2(-s1*s2*s3-c1*c3, -c1*s2*s3+s1*c3)
         lat_p = math.asin(c2*s3)
         angle_p = -math.atan2(s2, c2*c3)

      elif ptopo.my_panel == 4:
         if (abs(ϕ0) < 1e-13) and (abs(α0) < 1e-13):
            lon_p = 0.0
            lat_p = math.pi / 2.0
            angle_p = -λ0
         else:
            lon_p = math.atan2(-s1*s2*c3+c1*s3, -c1*s2*c3-s1*s3)
            lat_p = math.asin(c2*c3)
            angle_p = math.atan2(c2*s3, -s2)

      elif ptopo.my_panel == 5:
         if (abs(ϕ0)<1e-13) and (abs(α0)<1e-13):
            lon_p = 0.0
            lat_p = -math.pi/2.0
            angle_p = λ0
         else:
            lon_p = math.atan2(s1*s2*c3-c1*s3, c1*s2*c3+s1*s3)
            lat_p = -math.asin(c2*c3)
            angle_p = math.atan2(c2*s3, s2)

      # Cartesian coordinates on unit sphere

      cartX = 1.0 / delta * ( math.cos(lon_p) * math.cos(lat_p) \
            + X * ( math.cos(lon_p) * math.sin(lat_p) * math.sin(angle_p) - math.sin(lon_p) * math.cos(angle_p) ) \
            - Y * ( math.cos(lon_p) * math.sin(lat_p) * math.cos(angle_p) + math.sin(lon_p) * math.sin(angle_p) ) )

      cartY = 1.0 / delta * ( math.sin(lon_p) * math.cos(lat_p) \
            + X * ( math.sin(lon_p) * math.sin(lat_p) * math.sin(angle_p) + math.cos(lon_p) * math.cos(angle_p) ) \
            - Y * ( math.sin(lon_p) * math.sin(lat_p) * math.cos(angle_p) - math.cos(lon_p) * math.sin(angle_p) ) )

      cartZ = 1.0 / delta * ( math.sin(lat_p) - X * math.cos(lat_p) * math.sin(angle_p) + Y * math.cos(lat_p) * math.cos(angle_p) )

      # Spherical coordinates
      lon, lat, _ = sphere.cart2sph(cartX, cartY, cartZ)

      # Cartesian and spherical coordinates for elements interfaces

      cartX_itf_i = 1.0 / delta_itf_i * ( math.cos(lon_p) * math.cos(lat_p) \
            + X_itf_i * ( math.cos(lon_p) * math.sin(lat_p) * math.sin(angle_p) - math.sin(lon_p) * math.cos(angle_p) ) \
            - Y_itf_i * ( math.cos(lon_p) * math.sin(lat_p) * math.cos(angle_p) + math.sin(lon_p) * math.sin(angle_p) ) )

      cartY_itf_i = 1.0 / delta_itf_i * ( math.sin(lon_p) * math.cos(lat_p) \
            + X_itf_i * ( math.sin(lon_p) * math.sin(lat_p) * math.sin(angle_p) + math.cos(lon_p) * math.cos(angle_p) ) \
            - Y_itf_i * ( math.sin(lon_p) * math.sin(lat_p) * math.cos(angle_p) - math.cos(lon_p) * math.sin(angle_p) ) )

      cartZ_itf_i = 1.0 / delta_itf_i * ( math.sin(lat_p) - X_itf_i * math.cos(lat_p) * math.sin(angle_p) + Y_itf_i * math.cos(lat_p) * math.cos(angle_p) )

      lon_itf_i, lat_itf_i, _ = sphere.cart2sph(cartX_itf_i, cartY_itf_i, cartZ_itf_i)

      cartX_itf_j = 1.0 / delta_itf_j * ( math.cos(lon_p) * math.cos(lat_p) \
            + X_itf_j * ( math.cos(lon_p) * math.sin(lat_p) * math.sin(angle_p) - math.sin(lon_p) * math.cos(angle_p) ) \
            - Y_itf_j * ( math.cos(lon_p) * math.sin(lat_p) * math.cos(angle_p) + math.sin(lon_p) * math.sin(angle_p) ) )

      cartY_itf_j = 1.0 / delta_itf_j * ( math.sin(lon_p) * math.cos(lat_p) \
            + X_itf_j * ( math.sin(lon_p) * math.sin(lat_p) * math.sin(angle_p) + math.cos(lon_p) * math.cos(angle_p) ) \
            - Y_itf_j * ( math.sin(lon_p) * math.sin(lat_p) * math.cos(angle_p) - math.cos(lon_p) * math.sin(angle_p) ) )

      cartZ_itf_j = 1.0 / delta_itf_j * ( math.sin(lat_p) - X_itf_j * math.cos(lat_p) * math.sin(angle_p) + Y_itf_j * math.cos(lat_p) * math.cos(angle_p) )

      lon_itf_j, lat_itf_j, _ = sphere.cart2sph(cartX_itf_j, cartY_itf_j, cartZ_itf_j)

      # Map to the interval [0, 2 pi]
      lon_itf_j[lon_itf_j<0.0] = lon_itf_j[lon_itf_j<0.0] + (2.0 * math.pi)

      # --- define planet radius and rotation speed
      self.earth_radius   = 6371220.0      # Mean radius of the earth (m)
      self.rotation_speed = 7.29212e-5     # Angular speed of rotation of the earth (radians/s)

      # Check for resized planet
      planet_scaling_factor = 1.
      planet_is_rotating = 1.
      if param.equations == "Euler":
         if param.case_number == 31:
            planet_scaling_factor = 125.
            planet_is_rotating = 0.
      self.earth_radius   /= planet_scaling_factor
      self.rotation_speed *= planet_is_rotating / planet_scaling_factor

      self.solutionPoints = solutionPoints
      self.solutionPoints_sym = solutionPoints_sym
      self.glweights = glweights
      self.extension = extension
      self.extension_sym = extension_sym
      self.lon_p = lon_p
      self.lat_p = lat_p
      self.angle_p = angle_p
      self.ni = ni
      self.nj = nj
      self.nk = nk
      self.x1 = x1
      self.x2 = x2
      self.x3 = x3
      self.Δx1 = Δx1
      self.Δx2 = Δx2
      self.Δx3 = Δx3
      self.X = X
      self.Y = Y
      self.height = height
      self.delta2 = delta2
      self.delta = delta
      self.delta2_itf_i = delta2_itf_i
      self.delta_itf_i  = delta_itf_i
      self.delta2_itf_j = delta2_itf_j
      self.delta_itf_j  = delta_itf_j
      self.cartX = cartX
      self.cartY = cartY
      self.cartZ = cartZ
      self.lon = lon
      self.lat = lat
      self.X_itf_i = X_itf_i
      self.Y_itf_i = Y_itf_i
      self.X_itf_j = X_itf_j
      self.Y_itf_j = Y_itf_j
      self.lon_itf_i = lon_itf_i
      self.lat_itf_i = lat_itf_i
      self.lon_itf_j = lon_itf_j
      self.lat_itf_j = lat_itf_j

      self.coslon = numpy.cos(lon)
      self.sinlon = numpy.sin(lon)
      self.coslat = numpy.cos(lat)
      self.sinlat = numpy.sin(lat)
