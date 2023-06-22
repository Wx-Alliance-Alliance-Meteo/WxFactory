import math
import numpy
import sys

from .geometry   import Geometry
from .sphere     import cart2sph

# For type hints
from common.parallel          import DistributedWorld
from common.program_options   import Configuration

class CubedSphere(Geometry):
   def __init__(self, nb_elements_horizontal:int , nb_elements_vertical: int, nbsolpts: int, 
                λ0: float, ϕ0: float, α0: float, ztop: float, ptopo: DistributedWorld, param: Configuration):
      '''Initialize the cubed sphere geometry, for an earthlike sphere with no topography.

      This function initializes the basic CubedSphere geometry object, which provides the parameters necessary to define
      the values in numeric (x1, x2, η) coordinates, gnomonic (projected; X, Y, Z) coordinates, spherical (lat, lon, Z),
      Cartesian (Xc, Yc, Zc) coordinates.

      These coordinates respect the DG formulation, but they are themselves indifferent to the mechanics of
      differentiation.

      On initialization, the coordinate is defined with respect to a smooth sphere, with geometric height varying between
      0 and ztop.  To impose a topographic mapping, the CubedSphere object must be updated via the update_topo method.

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
      … where each panel has its own local (x1,x2) coordinate axis, representing the angular deviation from
      the panel center.  With typical parameters, panel 0 contains the intersection of the prime meridian and
      equator, the equator runs through panels 3-0-1-2 from west to east, panel 4 contains the north pole,
      and panel 5 contains the south pole.

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
         defined prior to any topography mapping, and this initializer defines the grid on
         a smooth sphere.
      ptopo: Distributed_World
         Wraps the parameters and helper functions necessary for MPI parallelism.  By assumption,
         each panel is a separate MPI process.
      param: Configuration
         Wraps parameters from the configuration pole that are not otherwise specified in this
         constructor.
      '''
      super().__init__(nbsolpts, 'cubed_sphere', param)

      ## Panel / parallel decomposition properties
      self.ptopo = ptopo

      # Full extent of the cubed-sphere panel
      panel_domain_x1 = (-math.pi/4, math.pi/4)
      panel_domain_x2 = (-math.pi/4, math.pi/4)

      # Find the extent covered by this particular processor
      Δx1_PE = (panel_domain_x1[1] - panel_domain_x1[0]) / ptopo.nb_lines_per_panel
      Δx2_PE = (panel_domain_x2[1] - panel_domain_x2[0]) / ptopo.nb_lines_per_panel

      # Find the lower and upper bounds of x1, x2 for this processor
      PE_start_x1 = -math.pi/4 + ptopo.my_col * Δx1_PE
      PE_end_x1 = PE_start_x1 + Δx1_PE

      PE_start_x2 = -math.pi/4 + ptopo.my_row * Δx2_PE
      PE_end_x2 = PE_start_x2 + Δx2_PE

      PE_start_x3 = 0.
      PE_end_x3 = ztop

      self.ztop = ztop

      # Define the computational η coordinate
      PE_start_eta = 0.
      PE_end_eta = 1.

      domain_x1 = (PE_start_x1, PE_end_x1)
      domain_x2 = (PE_start_x2, PE_end_x2)
      domain_x3 = (PE_start_x3, PE_end_x3)
      domain_eta = (PE_start_eta, PE_end_eta)

      nb_elements_x1 = nb_elements_horizontal
      nb_elements_x2 = nb_elements_horizontal
      nb_elements_x3 = nb_elements_vertical

      # Assign the number of elements and solution points to the CubedSphere
      # object, in order to generate compatible grids and loops elsewhere
      self.nb_elements_x1 = nb_elements_x1
      self.nb_elements_x2 = nb_elements_x2
      self.nb_elements_x3 = nb_elements_x3

      ## Element sizes

      # Equiangular coordinates
      Δx1 = (domain_x1[1] - domain_x1[0]) / nb_elements_x1
      Δx2 = (domain_x2[1] - domain_x2[0]) / nb_elements_x2
      Δx3 = (domain_x3[1] - domain_x3[0]) / nb_elements_x3
      Δeta = (domain_eta[1] - domain_eta[0]) / nb_elements_x3

      # Reset Δx3 to a nonzero value if ztop=0, as for shallow water
      if (Δx3 == 0): Δx3 = 1

      self.Δx1 = Δx1
      self.Δx2 = Δx2
      self.Δx3 = Δx3
      self.Δeta = Δeta

      # Helper variables to normalize the intra-element computational coordinate
      minComp = min(self.extension)
      maxComp = max(self.extension)
      Δcomp = maxComp - minComp

      # Define the coordinate values at the interfaces between elements
      # interfaces_x1 = numpy.linspace(start = domain_x1[0], stop = domain_x1[1], num = nb_elements_x1 + 1)
      # interfaces_x2 = numpy.linspace(start = domain_x2[0], stop = domain_x2[1], num = nb_elements_x2 + 1)
      # interfaces_x3 = numpy.linspace(start = domain_x3[0], stop = domain_x3[1], num = nb_elements_x3 + 1)
      # interfaces_eta = numpy.linspace(start = domain_eta[0], stop = domain_eta[1], num = nb_elements_x3 + 1)

      ni = nb_elements_x1*nbsolpts
      nj = nb_elements_x2*nbsolpts
      if (ztop > 0):
         nk = nb_elements_x3*nbsolpts
      else:
         nk = 1

      ## Save array size properties
      self.ni = ni
      self.nj = nj
      self.nk = nk

      # Define the shape of the coordinate grid on this process
      grid_shape_3d = (nk, nj, ni)
      self.grid_shape_3d = grid_shape_3d
      self.grid_shape_2d = (nj,ni)
      # And the shape of arrays corresponding to each of the three interfaces
      self.itf_i_shape_3d = (nk,nj,nb_elements_x1+1)
      self.itf_i_shape_2d = (nj,nb_elements_x1+1)
      self.itf_j_shape_3d = (nk,nb_elements_x2+1,ni)  
      self.itf_j_shape_2d = (nb_elements_x2+1,ni)  
      self.itf_k_shape_3d = (nb_elements_x3+1,nj,ni)  
      self.itf_k_shape_2d = (nj,ni)  

      # Assign a token zbot, potentially to be overridden later with supplied topography
      self.zbot = param.array_module.zeros(self.grid_shape_2d)

      ## Coordinate vectors for the numeric / angular coordinate system

      # Define the base coordinate.  x1 and x2 are fundamentally 1D arrays,
      # while x3 and eta are 3D arrays in support of coordinate mapping

      x1 = param.array_module.empty(ni)
      x2 = param.array_module.empty(nj)
      x3 = param.array_module.empty(self.grid_shape_3d)
      eta = param.array_module.empty(self.grid_shape_3d)

      # # 1D element-counting arrays, for coordinate assignment
      elements_x1 = numpy.arange(nb_elements_x1, like=x1)
      elements_x2 = numpy.arange(nb_elements_x2, like=x2)
      elements_x3 = numpy.arange(nb_elements_x3, like=x3)

      # Assign the coordinates, using numpy's broadcasting
      # First, reshape the coordinate to segregate the element interior into its own dimension
      x1.shape = (nb_elements_x1,nbsolpts)
      # Then broadcast the coordinate into the variable, using the structure:
      # <minimum> + <delta_element>*<element number> + <delta_inner>*<solutionPoints>
      x1[:] = domain_x1[0] + Δx1*elements_x1[:,numpy.newaxis] + \
                  Δx1/Δcomp*(-minComp + self.solutionPoints[numpy.newaxis,:])
      # Finally, reshape back to the unified view   
      x1.shape = (ni,)

      # Repeat for x2, x3, and eta
      x2.shape = (nb_elements_x2,nbsolpts)
      x2[:] = domain_x2[0] + Δx2*elements_x2[:,numpy.newaxis] + \
                  Δx2/Δcomp*(-minComp + self.solutionPoints[numpy.newaxis,:])
      x2.shape = (nj,)

      if (ztop > 0): # Note that x3 and eta are 3D arrays
         x3.shape = (nb_elements_x3,nbsolpts,nj,ni)
         x3[:] = domain_x3[0] + Δx3*elements_x3[:,numpy.newaxis,numpy.newaxis,numpy.newaxis] + \
                     Δx3/Δcomp*(-minComp + self.solutionPoints[numpy.newaxis,:,numpy.newaxis,numpy.newaxis])
         x3.shape = self.grid_shape_3d

         eta.shape = (nb_elements_x3,nbsolpts,nj,ni)
         eta[:] = domain_eta[0] + Δeta*elements_x3[:,numpy.newaxis,numpy.newaxis,numpy.newaxis] + \
                     Δeta/Δcomp*(-minComp + self.solutionPoints[numpy.newaxis,:,numpy.newaxis,numpy.newaxis])
         eta.shape = self.grid_shape_3d
      else:
         x3[:] = 0
         eta[:] = 0


      # Repeat for the interface values
      x1_itf_i = numpy.empty(nb_elements_x1 + 1, like=x1)  # 1D array
      x2_itf_i = numpy.empty(nj, like=x2)  # 1D array
      x3_itf_i = numpy.empty(self.itf_i_shape_3d, like=x3)  # 3D array
      eta_itf_i = numpy.empty(self.itf_i_shape_3d, like=eta) # 3D array

      x1_itf_i[:-1] = domain_x1[0] + Δx1*elements_x1[:] # Left edges
      x1_itf_i[-1] = domain_x1[1] # Right edge
      x2_itf_i[:] = x2[:] # Copy over x2, without change because of tensor product structure
      x3_itf_i[:,:,:] = x3[:,:,0:1] # Same for x3
      eta_itf_i[:,:,:] = eta[:,:,0:1]

      x1_itf_j = numpy.empty(ni, like=x1) # n.b. 1D array
      x2_itf_j = numpy.empty(nb_elements_x2 + 1, like=x2) # Also 1D array
      x3_itf_j = numpy.empty(self.itf_j_shape_3d, like=x3) # 3D array
      eta_itf_j = numpy.empty(self.itf_j_shape_3d, like=eta) # 3D array

      x1_itf_j[:] = x1[:]
      x2_itf_j[:-1] = domain_x2[0] + Δx2*elements_x2[:] # South edges
      x2_itf_j[-1] = domain_x2[1] # North edge
      x3_itf_j[:,:,:] = x3[:,0:1,:]
      eta_itf_j[:,:,:] = eta[:,0:1,:]

      x1_itf_k = numpy.empty(ni, like=x1)
      x2_itf_k = numpy.empty(nj, like=x2)
      x3_itf_k = numpy.empty(self.itf_k_shape_3d, like=x3)
      eta_itf_k = numpy.empty(self.itf_k_shape_3d, like=eta)

      x1_itf_k[:] = x1[:]
      x2_itf_k[:] = x2[:]
      x3_itf_k[:-1,:,:] = domain_x3[0] + Δx3*elements_x3[:,numpy.newaxis,numpy.newaxis] # Bottom edges
      x3_itf_k[-1,:,:] = domain_x3[1]
      eta_itf_k[:-1,:,:] = domain_eta[0] + Δeta*elements_x3[:,numpy.newaxis,numpy.newaxis]
      eta_itf_k[-1,:,:] = domain_eta[1]

      self.x1 = x1
      self.x2 = x2
      self.eta = eta
      self.x3 = x3

      self.x1_itf_i = x1_itf_i
      self.x2_itf_i = x2_itf_i
      self.x3_itf_i = x3_itf_i
      self.eta_itf_i = eta_itf_i

      self.x1_itf_j = x1_itf_j
      self.x2_itf_j = x2_itf_j
      self.x3_itf_j = x3_itf_j
      self.eta_itf_j = eta_itf_j

      self.x1_itf_k = x1_itf_k
      self.x2_itf_k = x2_itf_k
      self.x3_itf_k = x3_itf_k
      self.eta_itf_k = eta_itf_k

      ## Construct the combined coordinate vector for the numeric/equiangular coordinate (x1, x2, η)
      coordVec_num = numpy.stack((numpy.broadcast_to(x1[None, None, :], eta.shape),
                                  numpy.broadcast_to(x2[None, :, None], eta.shape),
                                  eta))

      # coordVec_num_itf_i = numpy.empty((3,) + (nk,nb_elements_x1+1,nj))
      # coordVec_num_itf_j = numpy.empty((3,) + (nk,nb_elements_x2+1,ni))
      # coordVec_num_itf_k = numpy.empty((3,) + (nb_elements_x3+1,nj,ni))

      # coordVec_num_itf_i[0,:,:,:] = x1_itf_i[numpy.newaxis,:,numpy.newaxis]
      # coordVec_num_itf_i[1,:,:,:] = x2_itf_i[numpy.newaxis,numpy.newaxis,:]
      # coordVec_num_itf_i[2,:,:,:] = eta_itf_i.transpose((0,2,1))

      coordVec_num_itf_i = numpy.stack((numpy.broadcast_to(x1_itf_i[None, None, :], eta_itf_i.shape),
                                        numpy.broadcast_to(x2_itf_i[None, :, None], eta_itf_i.shape),
                                        eta_itf_i))
      coordVec_num_itf_j = numpy.stack((numpy.broadcast_to(x1_itf_j[None, None, :], eta_itf_j.shape),
                                        numpy.broadcast_to(x2_itf_j[None, :, None], eta_itf_j.shape),
                                        eta_itf_j))
      coordVec_num_itf_k = numpy.stack((numpy.broadcast_to(x1_itf_k[None, None, :], eta_itf_k.shape),
                                        numpy.broadcast_to(x2_itf_k[None, :, None], eta_itf_k.shape),
                                        eta_itf_k))

      self.coordVec_num = coordVec_num
      self.coordVec_num_itf_i = coordVec_num_itf_i
      self.coordVec_num_itf_j = coordVec_num_itf_j
      self.coordVec_num_itf_k = coordVec_num_itf_k

      # Compute the parameters of the rotated grid

      # if (λ0 > 0.) or (λ0 <= -math.pi / 2.):
      #    print('lambda0 not within the acceptable range of ]-pi/2 , 0]. Stopping.')
      #    exit(1)

      # if (ϕ0 <= -math.pi/4.) or (ϕ0 > math.pi/4.):
      #    print('phi0 not within the acceptable range of ]-pi/4 , pi/4]. Stopping.')
      #    exit(1)

      # if (α0 <= -math.pi/2.) or (α0 > 0.):
      #    print('alpha0 not within the acceptable range of ]-pi/2 , 0]. Stopping.')
      #    exit(1)

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
      else:
         print(f'Invalid panel number {ptopo.my_panel}')
         exit(1)

      self.lon_p = lon_p
      self.lat_p = lat_p
      self.angle_p = angle_p

      # --- define planet radius and rotation speed
      self.earth_radius   = 6371220.0      # Mean radius of the earth (m)
      self.rotation_speed = 7.29212e-5     # Angular speed of rotation of the earth (radians/s)

      # Check for resized planet
      planet_scaling_factor = 1.
      planet_is_rotating = 1.
      if param.equations.lower() == "euler":
         if param.case_number == 31:
            planet_scaling_factor = 125.
            planet_is_rotating = 0.
         elif param.case_number == 20:
            # Normal planet, but no rotation
            planet_is_rotating = 0.0
         elif param.case_number == 21 or param.case_number == 22:
            # Small planet, no rotation
            planet_scaling_factor = 500
            planet_is_rotating = 0.0

         assert param.depth_approx is not None
         if param.depth_approx.lower() == "deep":
            self.deep = True
         elif param.depth_approx.lower() == "shallow":
            self.deep = False
         else:
            raise AssertionError(f'Invalid Euler atmosphere depth approximation ({param.depth_approx})')
      self.earth_radius   /= planet_scaling_factor
      self.rotation_speed *= planet_is_rotating / planet_scaling_factor

      # Call _build_physical_coordinates() to continue the construction of coordinate vectors
      # for the "physical" coordinates.  These are segregated into another method because they
      # will be redefined if this case involves topography mapping – x1/x2/η will remain the same,
      # as will the DG structures.

      self._build_physical_coordinates()

   def apply_topography(self, zbot, zbot_itf_i, zbot_itf_j):
      # Apply a topography field, given by heights (above the 0 reference sphere) specified at
      # interior points, i-boundaries, and j-boundaries.  This function applies a linear mapping,
      # where η=0 corresponds to the given surface and η=1 corresponds to the top.

      # First, preserve the topography
      self.zbot = zbot.copy()
      self.zbot_itf_i = zbot_itf_i.copy()
      self.zbot_itf_j = zbot_itf_j.copy()

      ztop = self.ztop

      # To apply the topography, we need to redefine self.x3 and its interfaced versions.

      self.x3[:,:,:] = zbot[numpy.newaxis,:,:] + (ztop - zbot[numpy.newaxis,:,:])*self.eta # shape k, j, i
      self.x3_itf_i[:,:,:] = zbot_itf_i[numpy.newaxis,:,:] + (ztop - zbot_itf_i[numpy.newaxis,:,:])*self.eta_itf_i # shape k, j, itf_i
      self.x3_itf_j[:,:,:] = zbot_itf_j[numpy.newaxis,:,:] + (ztop - zbot_itf_j[numpy.newaxis,:,:])*self.eta_itf_j # shape k, itf_j, i
      self.x3_itf_k[:,:,:] = zbot[numpy.newaxis,:,:] + (ztop - zbot[numpy.newaxis,:,:])*self.eta_itf_k # shape itf_k, j, i

      # Now, rebuild the physical coordinates to re-generate X/Y/Z and the Cartesian coordinates
      self._build_physical_coordinates()

   def _build_physical_coordinates(self):
      # Build the physical coordinate arrays and vectors (gnomonic plane, lat/lon, Cartesian)
      # based on the pre-defined equiangular coordinates (x1, x2) and height (x3)

      # Retrieve the numeric values for use here
      x1 = self.x1
      x2 = self.x2
      x3 = self.x3
      eta = self.eta

      x1_itf_i = self.x1_itf_i
      x2_itf_i = self.x2_itf_i
      x3_itf_i = self.x3_itf_i

      x1_itf_j = self.x1_itf_j
      x2_itf_j = self.x2_itf_j
      x3_itf_j = self.x3_itf_j

      x3_itf_k = self.x3_itf_k

      coordVec_num = self.coordVec_num
      coordVec_num_itf_i = self.coordVec_num_itf_i
      coordVec_num_itf_j = self.coordVec_num_itf_j
      coordVec_num_itf_k = self.coordVec_num_itf_k

      ni = self.ni
      nj = self.nj
      nk = self.nk

      nb_elements_x1 = self.nb_elements_x1
      nb_elements_x2 = self.nb_elements_x2
      nb_elements_x3 = self.nb_elements_x3

      lon_p = self.lon_p
      lat_p = self.lat_p
      angle_p = self.angle_p

      earth_radius = self.earth_radius
      rotation_speed = self.rotation_speed

      ## Gnomonic (projected plane) coordinate values
      # X and Y (and their interface variants) are 2D arrays on the ij plane;
      # height is still necessarily a 3D array.

      Y, X = numpy.meshgrid(numpy.tan(x2),numpy.tan(x1),indexing='ij')

      height = x3

      # Because of conventions used in the parallel exchanges, both i and j interface variables
      # are expected to be of size (#interface, #pts).  Compared to the usual (j,i) ordering,
      # this means that the i-interface variable should be transposed

      X_itf_i = numpy.broadcast_to(numpy.tan(x1_itf_i)[numpy.newaxis,:],(nj,nb_elements_x1+1)).T
      Y_itf_i = numpy.broadcast_to(numpy.tan(x2_itf_i)[:,numpy.newaxis],(nj,nb_elements_x1+1)).T
      X_itf_j = numpy.broadcast_to(numpy.tan(x1_itf_j)[numpy.newaxis,:],(nb_elements_x2+1,ni))
      Y_itf_j = numpy.broadcast_to(numpy.tan(x2_itf_j)[:,numpy.newaxis],(nb_elements_x2+1,ni))

      delta2 = 1.0 + X**2 + Y**2
      delta  = numpy.sqrt(delta2)

      delta2_itf_i = 1.0 + X_itf_i**2 + Y_itf_i**2
      delta_itf_i  = numpy.sqrt(delta2_itf_i)

      delta2_itf_j = 1.0 + X_itf_j**2 + Y_itf_j**2
      delta_itf_j  = numpy.sqrt(delta2_itf_j)

      self.X = X
      self.Y = Y
      self.height = height
      self.delta2 = delta2
      self.delta = delta
      self.delta2_itf_i = delta2_itf_i
      self.delta_itf_i  = delta_itf_i
      self.delta2_itf_j = delta2_itf_j
      self.delta_itf_j  = delta_itf_j

      ## Other coordinate vectors:
      # * gnonomic coordinates (X, Y, Z)

      coordVec_gnom = numpy.empty_like(coordVec_num) # (X,Y,Z)
      coordVec_gnom_itf_i = numpy.empty_like(coordVec_num_itf_i)
      coordVec_gnom_itf_j = numpy.empty_like(coordVec_num_itf_j)
      coordVec_gnom_itf_k = numpy.empty_like(coordVec_num_itf_k)

      # for (coord, x, y, z) in zip([coordVec_gnom, coordVec_gnom_itf_i, coordVec_gnom_itf_j, coordVec_gnom_itf_k],
      #                             [X, X_itf_i, X_itf_j, X],
      #                             [Y, Y_itf_i, Y_itf_j, Y],
      #                             [x3, x3_itf_i.transpose((0,2,1)), x3_itf_j, x3_itf_k]):
      #    coord[0,:,:,:] = x[numpy.newaxis,:,:]
      #    coord[1,:,:,:] = y[numpy.newaxis,:,:]
      #    coord[2,:,:,:] = z  

      for (coordgnom, coordnum, z) in zip([coordVec_gnom, coordVec_gnom_itf_i, coordVec_gnom_itf_j, coordVec_gnom_itf_k],
                                          [coordVec_num, coordVec_num_itf_i, coordVec_num_itf_j, coordVec_num_itf_k],
                                          [x3, x3_itf_i, x3_itf_j, x3_itf_k]):
         coordgnom[0,:,:,:] = numpy.tan(coordnum[0,:,:,:])
         coordgnom[1,:,:,:] = numpy.tan(coordnum[1,:,:,:])
         coordgnom[2,:,:,:] = z

      # * Cartesian coordinates on the deep sphere (Xc, Yc, Zc)

      coordVec_cart = numpy.empty_like(coordVec_num) # (Xc,Yc,Zc)
      coordVec_cart_itf_i = numpy.empty_like(coordVec_num_itf_i)
      coordVec_cart_itf_j = numpy.empty_like(coordVec_num_itf_j)
      coordVec_cart_itf_k = numpy.empty_like(coordVec_num_itf_k)

      # Built the Cartesian coordinates, by inverting the gnomonic projection.  At the north pole without grid
      # rotation, the formulas are:
      # Xc = (r+Z)*X/sqrt(1+X^2+Y^2)
      # Yc = (r+Z)*Y/sqrt(1+X^2+Y^2)
      # Zc = (r+Z)/sqrt(1+X^2+Y^2)
      for (coord_cart, coord_gnom) in zip([coordVec_cart, coordVec_cart_itf_i, coordVec_cart_itf_j, coordVec_cart_itf_k],
                                             [coordVec_gnom, coordVec_gnom_itf_i, coordVec_gnom_itf_j, coordVec_gnom_itf_k]):
         delt = numpy.sqrt(1.0 + coord_gnom[0,:,:,:]**2 + coord_gnom[1,:,:,:]**2)
         coord_cart[0,:] = (self.earth_radius + coord_gnom[2,:]) / delt * ( math.cos(lon_p) * math.cos(lat_p) \
               + coord_gnom[0,:] * ( math.cos(lon_p) * math.sin(lat_p) * math.sin(angle_p) - math.sin(lon_p) * math.cos(angle_p) ) \
               - coord_gnom[1,:] * ( math.cos(lon_p) * math.sin(lat_p) * math.cos(angle_p) + math.sin(lon_p) * math.sin(angle_p) ) )

         coord_cart[1,:] = (self.earth_radius + coord_gnom[2,:]) / delt * ( math.sin(lon_p) * math.cos(lat_p) \
               + coord_gnom[0,:] * ( math.sin(lon_p) * math.sin(lat_p) * math.sin(angle_p) + math.cos(lon_p) * math.cos(angle_p) ) \
               - coord_gnom[1,:] * ( math.sin(lon_p) * math.sin(lat_p) * math.cos(angle_p) - math.cos(lon_p) * math.sin(angle_p) ) )

         coord_cart[2,:] = (self.earth_radius + coord_gnom[2,:]) / delt * ( math.sin(lat_p) \
               - coord_gnom[0,:] * math.cos(lat_p) * math.sin(angle_p) \
               + coord_gnom[1,:] * math.cos(lat_p) * math.cos(angle_p) )

      coordVec_latlon = numpy.empty_like(coordVec_num) # (lat, lon, Z)
      coordVec_latlon_itf_i = numpy.empty_like(coordVec_num_itf_i)
      coordVec_latlon_itf_j = numpy.empty_like(coordVec_num_itf_j)
      coordVec_latlon_itf_k = numpy.empty_like(coordVec_num_itf_k)

      # * Polar coordinates (lat, lon, Z)

      for (latlon, cart, gnom) in zip([coordVec_latlon, coordVec_latlon_itf_i, coordVec_latlon_itf_j, coordVec_latlon_itf_k],
                                      [coordVec_cart, coordVec_cart_itf_i, coordVec_cart_itf_j, coordVec_cart_itf_k],
                                      [coordVec_gnom, coordVec_gnom_itf_i, coordVec_gnom_itf_j, coordVec_gnom_itf_k]):
         [latlon[0,:], latlon[1,:], _] = cart2sph(cart[0,:],cart[1,:],cart[2,:])
         latlon[2,:] = gnom[2,:]

      self.coordVec_gnom = coordVec_gnom
      self.coordVec_gnom_itf_i = coordVec_gnom_itf_i
      self.coordVec_gnom_itf_j = coordVec_gnom_itf_j
      self.coordVec_gnom_itf_k = coordVec_gnom_itf_k

      self.coordVec_cart = coordVec_cart
      self.coordVec_cart_itf_i = coordVec_cart_itf_i
      self.coordVec_cart_itf_j = coordVec_cart_itf_j
      self.coordVec_cart_itf_k = coordVec_cart_itf_k

      self.coordVec_latlon = coordVec_latlon
      self.coordVec_latlon_itf_i = coordVec_latlon_itf_i
      self.coordVec_latlon_itf_j = coordVec_latlon_itf_j
      self.coordVec_latlon_itf_k = coordVec_latlon_itf_k

      cartX = coordVec_cart[0,:]
      cartY = coordVec_cart[1,:]
      cartZ = coordVec_cart[2,:]

      cartX_itf_i = coordVec_cart_itf_i[0,:]
      cartX_itf_j = coordVec_cart_itf_j[0,:]
      cartX_itf_k = coordVec_cart_itf_k[0,:]
      
      cartY_itf_i = coordVec_cart_itf_i[1,:]
      cartY_itf_j = coordVec_cart_itf_j[1,:]
      cartY_itf_k = coordVec_cart_itf_k[1,:]

      cartZ_itf_i = coordVec_cart_itf_i[2,:]
      cartZ_itf_j = coordVec_cart_itf_j[2,:]
      cartZ_itf_k = coordVec_cart_itf_k[2,:]

      lon = coordVec_latlon[0,0,:,:]
      lat = coordVec_latlon[1,0,:,:]

      # The _itf_i variables are transposed here compared to the coordVec
      # order to retain compatibility with the shallow water test cases.
      # Those cases assume that the 2D interface variables are written
      # with [interface#,interior#] indexing, and for the _itf_i variables
      # this is effective the opposite of the geometric [k,j,i] order.

      lon_itf_i = coordVec_latlon_itf_i[0,0,:,:].T
      lon_itf_j = coordVec_latlon_itf_j[0,0,:,:]
      lon_itf_k = coordVec_latlon_itf_k[0,0,:,:]

      lat_itf_i = coordVec_latlon_itf_i[1,0,:,:].T
      lat_itf_j = coordVec_latlon_itf_j[1,0,:,:]
      lat_itf_k = coordVec_latlon_itf_k[1,0,:,:]

      # Map to the interval [0, 2 pi]
      #lon_itf_j[lon_itf_j<0.0] = lon_itf_j[lon_itf_j<0.0] + (2.0 * math.pi)

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
