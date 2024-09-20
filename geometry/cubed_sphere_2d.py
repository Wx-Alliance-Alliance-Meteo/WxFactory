import math
import numpy
from   numpy.typing import NDArray

from mpi4py import MPI

from .geometry   import Geometry
from .sphere     import cart2sph

# For type hints
from common.process_topology import ProcessTopology
from common.configuration    import Configuration

class CubedSphere2D(Geometry):
   def __init__(self, nb_elements_horizontal:int , nbsolpts: int,
                λ0: float, ϕ0: float, α0: float, ptopo: ProcessTopology, param: Configuration):
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
      rank = MPI.COMM_WORLD.rank

      super().__init__(nbsolpts, 'cubed_sphere', param.array_module)
      xp = self.array_module

      ## Panel / parallel decomposition properties
      self.ptopo = ptopo

      # Full extent of the cubed-sphere panel, in radians
      panel_domain_x1 = (-math.pi/4, math.pi/4)
      panel_domain_x2 = (-math.pi/4, math.pi/4)

      # Find the extent covered by this particular processor (a tile), in radians
      delta_x1_panel = (panel_domain_x1[1] - panel_domain_x1[0]) / ptopo.nb_lines_per_panel
      delta_x2_panel = (panel_domain_x2[1] - panel_domain_x2[0]) / ptopo.nb_lines_per_panel

      # Find the lower and upper bounds of x1, x2 for this processor
      start_x1_tile = -math.pi/4 + ptopo.my_col * delta_x1_panel
      end_x1_tile = start_x1_tile + delta_x1_panel

      start_x2_tile = -math.pi/4 + ptopo.my_row * delta_x2_panel
      end_x2_tile = start_x2_tile + delta_x2_panel

      domain_x1 = (start_x1_tile, end_x1_tile) # in radians
      domain_x2 = (start_x2_tile, end_x2_tile) # in radians

      # Assign the number of elements and solution points to the CubedSphere
      # object, in order to generate compatible grids and loops elsewhere
      nb_elements_x1 = nb_elements_horizontal
      nb_elements_x2 = nb_elements_horizontal
      self.nb_elements_x1 = nb_elements_x1
      self.nb_elements_x2 = nb_elements_x2

      ## Element sizes

      # Equiangular coordinates
      self.delta_x1 = (domain_x1[1] - domain_x1[0]) / nb_elements_x1 # west-east grid spacing in radians
      self.delta_x2 = (domain_x2[1] - domain_x2[0]) / nb_elements_x2 # south-north grid spacing in radians
      delta_x1 = self.delta_x1
      delta_x2 = self.delta_x2

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
      nk = 1

      # Save array size properties
      # NOTE: ni is the number of "columns" and nj is the number of "rows" in a 2D matrix storing solution points
      # Since this is python, matrices are stored in row-major order (columns are the fast-varying index) 
      self.ni = ni
      self.nj = nj
      self.nk = nk

      # Define the shape of the coordinate grid on this process
      self.grid_shape_2d = (nj, ni)
      self.block_shape = (nj, ni)
      # And the shape of arrays corresponding to each of the three interfaces
      self.itf_i_shape_2d = (nj, nb_elements_x1+1)
      self.itf_j_shape_2d = (nb_elements_x2+1, ni)

      # The shapes. For a single field (e.g. coordinates of solution points, in the current case), we
      # have an array of elements, where each element has a total nbsolpts**2 (in 2D) or nbsolpts**3 (in 3D) solution
      # points.
      # For the interfaces we have 3 cases:
      # - In 2D, we have 2 arrays (for west-east and south-north interfaces), with the same shape: an array of all
      #   elements, each with 2*nbsolpts interface points (nbsolpts for each side of the element along that direction)
      # - In 3D, horizontally, we also have two arrays (west-east, south-north), but the shape is to be determined
      # - In 3D vertically, we only have one array, with shape similar to horizontally, TBD
      self.grid_shape = (nb_elements_x1 * nb_elements_x2, nbsolpts * nbsolpts)
      # self.grid_shape_3d_new = (nb_elements_x1 * nb_elements_x2 * nb_elements_x3, nbsolpts * nbsolpts * nbsolpts)
      self.itf_shape = (nb_elements_x1 * (nb_elements_x2 + 2), nbsolpts * 2)   # Same for i and j interfaces, because same size

      self.west_edge = numpy.s_[..., ::nb_elements_x1 + 2, :nbsolpts]    # West boundary of the western halo elements
      self.east_edge = numpy.s_[..., (nb_elements_x1 + 1)::nb_elements_x1 + 2, nbsolpts:] # East boundary of the eastern halo elements
      self.south_edge = numpy.s_[..., :nb_elements_x1, :nbsolpts] # South boundary of the southern halo elements
      self.north_edge = numpy.s_[..., -nb_elements_x1:, nbsolpts:] # North boundary of the northern halo elements

      ## Coordinate vectors for the numeric / angular coordinate system

      # --- 1D element-counting arrays, for coordinate assignment

      # Element interior
      offsets_x1 = domain_x1[0] + delta_x1 * xp.arange(nb_elements_x1)
      ref_solpts_x1 = delta_x1 / Δcomp * (-minComp + self.solutionPoints)
      self.x1 = numpy.repeat(offsets_x1, nbsolpts) + numpy.tile(ref_solpts_x1, nb_elements_x1)

      offsets_x2 = domain_x2[0] + delta_x2 * xp.arange(nb_elements_x2)
      ref_solpts_x2 = delta_x2 / Δcomp * (-minComp + self.solutionPoints)
      self.x2 = numpy.repeat(offsets_x2, nbsolpts) + numpy.tile(ref_solpts_x2, nb_elements_x2)

      # Element interfaces
      self.x1_itf_i = numpy.linspace(domain_x1[0], domain_x1[1], nb_elements_x1 + 1) # At every boundary between elements
      self.x2_itf_i = self.x2.copy() # Copy over x2, without change because of tensor product structure

      self.x1_itf_j = self.x1.copy()
      self.x2_itf_j = numpy.linspace(domain_x2[0], domain_x2[1], nb_elements_x2 + 1)

      ## Construct the combined coordinate vector for the numeric/equiangular coordinate (x1, x2)
      self.block_radians_x1, self.block_radians_x2 = numpy.meshgrid(self.x1, self.x2)
      i_x1, i_x2 = numpy.meshgrid(self.x1_itf_i, self.x2_itf_i, indexing='ij')
      j_x1, j_x2 = numpy.meshgrid(self.x1_itf_j, self.x2_itf_j)

      self.radians = self._to_new(numpy.array([self.block_radians_x1, self.block_radians_x2]))
      self.coordVec_itf_i = self._to_new_itf_i(numpy.array([i_x1, i_x2]))
      self.coordVec_itf_j = self._to_new_itf_j(numpy.array([j_x1, j_x2]))


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
         raise ValueError(f'Invalid panel number {ptopo.my_panel}')

      self.lon_p = lon_p
      self.lat_p = lat_p
      self.angle_p = angle_p

      # --- define planet radius and rotation speed
      self.earth_radius   = 6371220.0      # Mean radius of the earth (m)
      self.rotation_speed = 7.29212e-5     # Angular speed of rotation of the earth (radians/s)

      # Check for resized planet
      planet_scaling_factor = 1.
      planet_is_rotating = 1.
      self.deep = False
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

   def _build_physical_coordinates(self):
      """
      Build the physical coordinate arrays and vectors (gnomonic plane, lat/lon, Cartesian)
      based on the pre-defined equiangular coordinates (x1, x2) and height (x3)
      """

      rank = MPI.COMM_WORLD.rank

      # Retrieve the numeric values for use here
      x1 = self.x1
      x2 = self.x2

      x1_itf_i = self.x1_itf_i
      x2_itf_i = self.x2_itf_i

      x1_itf_j = self.x1_itf_j
      x2_itf_j = self.x2_itf_j


      ni = self.ni
      nj = self.nj

      nb_elements_x1 = self.nb_elements_x1
      nb_elements_x2 = self.nb_elements_x2

      lon_p = self.lon_p
      lat_p = self.lat_p
      angle_p = self.angle_p

      ## Gnomonic (projected plane) coordinate values
      # X and Y (and their interface variants) are 2D arrays on the ij plane;
      # x comes before y in the indices -> Y is the "fast-varying" index

      Y, X = numpy.meshgrid(numpy.tan(x2),numpy.tan(x1),indexing='ij')

      Y_new = self._to_new(Y)
      X_new = self._to_new(X)

      self.boundary_sn = X[0, :] # Coordinates of the south and north boundaries along the X (west-east) axis
      self.boundary_we = Y[:, 0] # Coordinates of the west and east boundaries along the Y (south-north) axis

      # Because of conventions used in the parallel exchanges, both i and j interface variables
      # are expected to be of size (#interface, #pts).  Compared to the usual (j,i) ordering,
      # this means that the i-interface variable should be transposed

      X_itf_i = numpy.broadcast_to(numpy.tan(x1_itf_i)[numpy.newaxis,:],(nj,nb_elements_x1+1)).T
      Y_itf_i = numpy.broadcast_to(numpy.tan(x2_itf_i)[:,numpy.newaxis],(nj,nb_elements_x1+1)).T
      X_itf_j = numpy.broadcast_to(numpy.tan(x1_itf_j)[numpy.newaxis,:],(nb_elements_x2+1,ni))
      Y_itf_j = numpy.broadcast_to(numpy.tan(x2_itf_j)[:,numpy.newaxis],(nb_elements_x2+1,ni))

      X_itf_i_new = self._to_new_itf_i(X_itf_i)
      Y_itf_i_new = self._to_new_itf_i(Y_itf_i)
      X_itf_j_new = self._to_new_itf_j(X_itf_j)
      Y_itf_j_new = self._to_new_itf_j(Y_itf_j)

      delta2_new = 1.0 + X_new**2 + Y_new**2
      delta_new  = numpy.sqrt(delta2_new)

      delta2_itf_i_new = 1.0 + X_itf_i_new**2 + Y_itf_i_new**2
      delta_itf_i_new  = numpy.sqrt(delta2_itf_i_new)

      delta2_itf_j_new = 1.0 + X_itf_j_new**2 + Y_itf_j_new**2
      delta_itf_j_new  = numpy.sqrt(delta2_itf_j_new)

      self.X = X
      self.Y = Y
      self.X_new = X_new
      self.Y_new = Y_new
      self.delta2_new = delta2_new
      self.delta_new = delta_new
      self.delta2_itf_i_new = delta2_itf_i_new
      self.delta_itf_i_new  = delta_itf_i_new
      self.delta2_itf_j_new = delta2_itf_j_new
      self.delta_itf_j_new  = delta_itf_j_new

      ## Other coordinate vectors:
      # * gnonomic coordinates (X, Y, Z)

      def to_gnomonic(coord_num, z=None):
         gnom = numpy.empty_like(coord_num)
         gnom[0, ...] = numpy.tan(coord_num[0])
         gnom[1, ...] = numpy.tan(coord_num[1])
         if z is not None: gnom[2, ...] = z

         return gnom

      self.coordVec_gnom_new = to_gnomonic(self.radians)
      self.gnom_itf_i_new    = to_gnomonic(self.coordVec_itf_i)
      self.gnom_itf_j_new    = to_gnomonic(self.coordVec_itf_j)

      # * Cartesian coordinates on the deep sphere (Xc, Yc, Zc)

      def to_cartesian(gnom):
         ''' Built the Cartesian coordinates, by inverting the gnomonic projection.  At the north pole without grid
         rotation, the formulas are:
         Xc = (r+Z) * X / sqrt(1+X^2+Y^2)
         Yc = (r+Z) * Y / sqrt(1+X^2+Y^2)
         Zc = (r+Z) / sqrt(1+X^2+Y^2)
         '''
         base_shape = gnom.shape[1:]
         cart = numpy.empty((3,) + base_shape, dtype=gnom.dtype)
         delt = numpy.sqrt(1.0 + gnom[0, ...]**2 + gnom[1, ...]**2)
         r = self.earth_radius + (0. if gnom.shape[0] < 3 else gnom[2, ...])
         cart[0,:] = r / delt * ( math.cos(lon_p) * math.cos(lat_p) \
               + gnom[0, ...] * (math.cos(lon_p) * math.sin(lat_p) * math.sin(angle_p) - \
                                 math.sin(lon_p) * math.cos(angle_p)) \
               - gnom[1, ...] * (math.cos(lon_p) * math.sin(lat_p) * math.cos(angle_p) + \
                                 math.sin(lon_p) * math.sin(angle_p)) )

         cart[1,:] = r / delt * ( math.sin(lon_p) * math.cos(lat_p) \
               + gnom[0, ...] * (math.sin(lon_p) * math.sin(lat_p) * math.sin(angle_p) + \
                                 math.cos(lon_p) * math.cos(angle_p)) \
               - gnom[1, ...] * (math.sin(lon_p) * math.sin(lat_p) * math.cos(angle_p) - \
                                 math.cos(lon_p) * math.sin(angle_p)) )

         cart[2,:] = r / delt * ( math.sin(lat_p) \
               - gnom[0, ...] * math.cos(lat_p) * math.sin(angle_p) \
               + gnom[1, ...] * math.cos(lat_p) * math.cos(angle_p) )


         return cart


      self.cart_new = to_cartesian(self.coordVec_gnom_new)
      self.cart_itf_i = to_cartesian(self.gnom_itf_i_new)
      self.cart_itf_j = to_cartesian(self.gnom_itf_j_new)


      # * Polar coordinates (lat, lon, Z)
      def to_polar(cart, z=None):
         if z is not None:
            polar = numpy.empty_like(cart)
            polar[2, ...] = z
         else:
            base_shape = cart.shape[1:]
            polar = numpy.empty((2,) + base_shape, dtype=cart.dtype)

         polar[0, ...], polar[1, ...], _ = cart2sph(cart[0, ...], cart[1, ...], cart[2, ...])

         return polar


      self.polar_new = to_polar(self.cart_new)
      self.polar_itf_i = to_polar(self.cart_itf_i)
      self.polar_itf_j = to_polar(self.cart_itf_j)

      self.polar_itf_i[self.west_edge]  = 0.0
      self.polar_itf_i[self.east_edge]  = 0.0
      self.polar_itf_j[self.south_edge] = 0.0
      self.polar_itf_j[self.north_edge] = 0.0

      self.lon_new = self.polar_new[0, ...]
      self.lat_new = self.polar_new[1, ...]
      self.coslon_new = numpy.cos(self.lon_new)
      self.coslat_new = numpy.cos(self.lat_new)
      self.sinlon_new = numpy.sin(self.lon_new)
      self.sinlat_new = numpy.sin(self.lat_new)

      self.block_lon = self.to_single_block(self.lon_new)
      self.block_lat = self.to_single_block(self.lat_new)

      self.X_itf_i_new = X_itf_i_new
      self.Y_itf_i_new = Y_itf_i_new
      self.X_itf_j_new = X_itf_j_new
      self.Y_itf_j_new = Y_itf_j_new

      self.lon_itf_i_new = self.polar_itf_i[0, ...]
      self.lat_itf_i_new = self.polar_itf_i[1, ...]
      self.lon_itf_j_new = self.polar_itf_j[0, ...]
      self.lat_itf_j_new = self.polar_itf_j[1, ...]


   def _to_new(self, a: NDArray) -> NDArray:
      """Convert input array to new memory layout"""

      if isinstance(a, float): return a

      expected_shape =  (self.nb_elements_x2 * self.nbsolpts, self.nb_elements_x1 * self.nbsolpts)
      if a.ndim == 2 and a.shape == expected_shape:
         tmp_shape = (self.nb_elements_x2, self.nbsolpts, self.nb_elements_x1, self.nbsolpts)
         new_shape = self.grid_shape
         return a.reshape(tmp_shape).transpose(0, 2, 1, 3).reshape(new_shape)

      elif (a.ndim == 3 and a.shape[1:] == expected_shape) or \
           (a.ndim == 4 and a.shape[2:] == expected_shape and a.shape[1] == 1):
         tmp_shape = (a.shape[0], self.nb_elements_x2, self.nbsolpts, self.nb_elements_x1, self.nbsolpts)
         new_shape = (a.shape[0],) + self.grid_shape
         return a.reshape(tmp_shape).transpose(0, 1, 3, 2, 4).reshape(new_shape)

      raise ValueError(f'Unhandled number of dimensions... Shape is {a.shape}')

   def to_single_block(self, a: NDArray) -> NDArray:
      """Convert input array from a list of elements to a single block of points layout."""
      expected_shape = (self.nb_elements_x1 * self.nb_elements_x2, self.nbsolpts * self.nbsolpts)
      if a.shape == expected_shape:
         tmp_shape = (self.nb_elements_x2, self.nb_elements_x1, self.nbsolpts, self.nbsolpts)
         new_shape = self.grid_shape_2d
         return a.reshape(tmp_shape).transpose(0, 2, 1, 3).reshape(new_shape)
      elif a.ndim == 3 and a.shape[1:] == expected_shape:
         tmp_shape = (a.shape[0], self.nb_elements_x2, self.nb_elements_x1, self.nbsolpts, self.nbsolpts)
         new_shape = (a.shape[0],) + self.grid_shape_2d
         return a.reshape(tmp_shape).transpose(0, 1, 3, 2, 4).reshape(new_shape)
      else:
         raise ValueError(f'Unexpected shape {a.shape} (expected {expected_shape})')

   def middle_itf_i(self, a):
      '''Extract the non-halo part of the given west-east interface array'''
      tmp_shape = a.shape[:-2] + (self.nb_elements_x2, self.nb_elements_x1 + 2, self.nbsolpts * 2)
      new_shape = a.shape[:-2] + (self.nb_elements_x2 * self.nb_elements_x1, self.nbsolpts * 2)
      return a.reshape(tmp_shape)[..., 1:-1, :].reshape(new_shape)

   def middle_itf_j(self, a):
      '''Extract the non-halo part of the given south-north interface array'''
      tmp_shape = a.shape[:-2] + (self.nb_elements_x2 + 2, self.nb_elements_x1, self.nbsolpts * 2)
      new_shape = a.shape[:-2] + (self.nb_elements_x2 * self.nb_elements_x2, self.nbsolpts * 2)
      return a.reshape(tmp_shape)[..., 1:-1, :, :].reshape(new_shape)

   def _to_new_itf_i(self, a):
      """Convert input array (west and east interface) to new memory layout"""
      # expected_shape = (self.nb_elements_x2 * self.nbsolpts, self.nb_elements_x1 + 1)
      expected_shape = (self.nb_elements_x1 + 1, self.nb_elements_x2 * self.nbsolpts)
      if a.shape[-2:] == expected_shape:
         plane_shape = (self.nb_elements_x2, self.nb_elements_x1 + 2, self.nbsolpts * 2)
         new = numpy.empty(a.shape[:-2] + plane_shape, dtype=a.dtype)
         west_itf  = numpy.arange(self.nbsolpts)
         east_itf = numpy.arange(self.nbsolpts, 2*self.nbsolpts)

         tmp_shape = a.shape[:-2] + (self.nb_elements_x1 + 1, self.nb_elements_x2, self.nbsolpts)
         offset = len(a.shape) - 2
         transp = tuple([i for i in range(offset)]) + (1 + offset, 0 + offset, 2 + offset)

         tmp_array = a.reshape(tmp_shape).transpose(transp)
         new[..., :-1, east_itf] = tmp_array
         new[..., 1:,  west_itf] = tmp_array

         new_shape = a.shape[:-2] + (self.nb_elements_x2 * (self.nb_elements_x1 + 2), self.nbsolpts * 2)
         new = new.reshape(new_shape)
         new[self.east_edge] = 0.0
         new[self.west_edge] = 0.0

         return numpy.squeeze(new)
      else:
         raise ValueError(f'Unexpected array shape {a.shape} (expected {expected_shape})')

   def _to_new_itf_j(self, a):
      """Convert input array (south and north interface) to new memory layout"""
      expected_shape = (self.nb_elements_x2 + 1, self.nb_elements_x1 * self.nbsolpts)
      if a.shape[-2:] == expected_shape:
         plane_shape = (self.nb_elements_x2 + 2, self.nb_elements_x1, self.nbsolpts * 2)
         new_tmp = numpy.empty(a.shape[:-2] + plane_shape, dtype=a.dtype)
         # new_tmp[...] = 1000.0
         south = numpy.s_[..., 1:,  :, :self.nbsolpts]  # South boundary of elements, including northern halo
         north = numpy.s_[..., :-1, :, self.nbsolpts:]  # North boundary of elements, including southern halo

         tmp_shape = a.shape[:-2] + (self.nb_elements_x2 + 1, self.nb_elements_x1, self.nbsolpts)
         tmp_array = a.reshape(tmp_shape)
         new_tmp[north] = tmp_array
         new_tmp[south] = tmp_array

         new_shape = a.shape[:-2] + ((self.nb_elements_x2 + 2) * self.nb_elements_x1, self.nbsolpts * 2)
         new = new_tmp.reshape(new_shape)
         new[self.south_edge] = 0.0
         new[self.north_edge] = 0.0

         # raise ValueError(f'a = \n{a[0]}\nnew_tmp = \n{new_tmp[0]}\nnew = \n{new[0]}')

         return numpy.squeeze(new)
      else:
         raise ValueError(f'Unexpected array shape {a.shape} (expected {expected_shape})')


   def wind2contra(self, u: float | numpy.ndarray, v: float | numpy.ndarray):
      '''Convert wind fields from the spherical basis (zonal, meridional) to panel-appropriate contravariant winds, in two dimensions

      Parameters:
      ----------
      u : float | numpy.ndarray
         Input zonal winds, in meters per second
      v : float | numpy.ndarray
         Input meridional winds, in meters per second

      Returns:
      -------
      (u1_contra, u2_contra) : tuple
         Tuple of contravariant winds'''

      # Convert winds coords to spherical basis

      if (self.nk > 1 and self.deep):
         # In 3D code with the deep atmosphere, the conversion to λ and φ
         # uses the full radial height of the grid point:
         lambda_dot = u / ((self.earth_radius + self.coordVec_gnom_new[2, ...]) * self.coslat)
         phi_dot    = v / (self.earth_radius + self.coordVec_gnom_new[2, ...])
      else:
         # Otherwise, the conversion uses just the planetary radius, with no
         # correction for height above the surface
         lambda_dot = u / (self.earth_radius * self.coslat_new)
         phi_dot    = v / self.earth_radius

      denom = numpy.sqrt( (math.cos(self.lat_p) + \
                           self.X_new * math.sin(self.lat_p) * math.sin(self.angle_p) - \
                           self.Y_new * math.sin(self.lat_p) * math.cos(self.angle_p))**2 + \
                          (self.X_new * math.cos(self.angle_p) + self.Y_new * math.sin(self.angle_p))**2 )

      dx1dlon = math.cos(self.lat_p) * math.cos(self.angle_p) + \
                ( self.X_new * self.Y_new * math.cos(self.lat_p) * math.sin(self.angle_p) - \
                  self.Y_new * math.sin(self.lat_p) ) / (1. + self.X_new**2)
      dx2dlon = ( self.X_new * self.Y_new * math.cos(self.lat_p) * math.cos(self.angle_p) + self.X_new * math.sin(self.lat_p) ) / (1. + self.Y_new**2) + \
                math.cos(self.lat_p) * math.sin(self.angle_p)

      dx1dlat = -self.delta2_new * ( (math.cos(self.lat_p)*math.sin(self.angle_p) + self.X_new * math.sin(self.lat_p))/(1. + self.X_new**2) ) / denom
      dx2dlat = self.delta2_new * ( (math.cos(self.lat_p)*math.cos(self.angle_p) - self.Y_new * math.sin(self.lat_p))/(1. + self.Y_new**2) ) / denom
      
      # transform to the reference element

      u1_contra = ( dx1dlon * lambda_dot + dx1dlat * phi_dot ) * 2. / self.delta_x1
      u2_contra = ( dx2dlon * lambda_dot + dx2dlat * phi_dot ) * 2. / self.delta_x2

      return u1_contra, u2_contra

   def contra2wind(self, u1 : float | NDArray, u2 : float | NDArray) -> tuple[NDArray, NDArray]:
      ''' Convert from reference element to "physical winds", in two dimensions

      Parameters:
      -----------
      u1 : float | numpy.ndarray
         Contravariant winds along first component (X)
      u2 : float | numpy.ndarray
         Contravariant winds along second component (Y)

      Returns:
      --------
      (u, v) : tuple
         Zonal/meridional winds, in m/s
      '''

      u1_contra = u1 * self.delta_x1/2.
      u2_contra = u2 * self.delta_x2/2.

      denom = (math.cos(self.lat_p) + \
               self.X_new * math.sin(self.lat_p) * math.sin(self.angle_p) - \
               self.Y_new * math.sin(self.lat_p) * math.cos(self.angle_p))**2 + \
              (self.X_new * math.cos(self.angle_p) + \
               self.Y_new * math.sin(self.angle_p))**2

      dlondx1 = (math.cos(self.lat_p) * math.cos(self.angle_p) - self.Y_new * math.sin(self.lat_p)) * \
                (1. + self.X_new**2) / denom

      dlondx2 = (math.cos(self.lat_p) * math.sin(self.angle_p) + self.X_new * math.sin(self.lat_p)) * \
                (1. + self.Y_new**2) / denom

      denom[...] = numpy.sqrt( (math.cos(self.lat_p) + \
                                self.X_new * math.sin(self.lat_p)*math.sin(self.angle_p) - \
                                self.Y_new * math.sin(self.lat_p)*math.cos(self.angle_p))**2 + \
                               (self.X_new * math.cos(self.angle_p) + \
                                self.Y_new * math.sin(self.angle_p))**2 )

      dlatdx1 = - ( (self.X_new * self.Y_new * math.cos(self.lat_p) * math.cos(self.angle_p) + \
                     self.X_new * math.sin(self.lat_p) + \
                     (1. + self.Y_new**2) * math.cos(self.lat_p) * math.sin(self.angle_p)) * 
                    (1. + self.X_new**2) ) / ( self.delta2_new * denom)

      dlatdx2 = ( ((1. + self.X_new**2) * math.cos(self.lat_p) * math.cos(self.angle_p) + \
                   self.X_new * self.Y_new * math.cos(self.lat_p) * math.sin(self.angle_p) - \
                   self.Y_new * math.sin(self.lat_p)) * \
                  (1. + self.Y_new**2) ) / ( self.delta2_new * denom)

      u = ( dlondx1 * u1_contra + dlondx2 * u2_contra ) * self.coslat_new * self.earth_radius
      v = ( dlatdx1 * u1_contra + dlatdx2 * u2_contra ) * self.earth_radius

      return u, v
