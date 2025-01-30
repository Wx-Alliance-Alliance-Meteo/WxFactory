import math
import numpy
from numpy.typing import NDArray

from mpi4py import MPI

from .cubed_sphere import CubedSphere
from .sphere import cart2sph

# For type hints
from common import Configuration
from device import Device
from wx_mpi import ProcessTopology


class CubedSphere2D(CubedSphere):
    def __init__(
        self,
        num_elements_horizontal: int,
        num_solpts: int,
        lambda0: float,
        phi0: float,
        alpha0: float,
        ptopo: ProcessTopology,
        param: Configuration,
        device: Device,
    ):
        """Initialize the cubed sphere geometry, for an earthlike sphere with no topography.

        This function initializes the basic CubedSphere geometry object, which provides the parameters necessary
        to define the values in numeric (x1, x2, η) coordinates, gnomonic (projected; X, Y, Z) coordinates,
        spherical (lat, lon, Z), Cartesian (Xc, Yc, Zc) coordinates.

        These coordinates respect the DG formulation, but they are themselves indifferent to the mechanics of
        differentiation.

        On initialization, the coordinate is defined with respect to a smooth sphere, with geometric height
        varying between 0 and ztop.
        To impose a topographic mapping, the CubedSphere object must be updated via the update_topo method.

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
        num_elements_horizontal: int
           Number of elements in the (x1,x2) directions, per panel
        num_solpts: int
           Number of nodal points in each of the (x1,x2,x3) dimensions inside the element
        lambda0: float
           Grid rotation: physical longitude of the central point of the 0 panel
           Valid range ]-π/2,0]
        phi0: float
           Grid rotation: physical latitude of the central point of the 0 panel
           Valid range ]-π/4,π/4]
        alpha0: float
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
        """
        rank = MPI.COMM_WORLD.rank

        super().__init__(num_solpts, lambda0, phi0, alpha0, device)
        xp = self.device.xp

        ## Panel / parallel decomposition properties
        self.ptopo = ptopo

        # Full extent of the cubed-sphere panel, in radians
        panel_domain_x1 = (-math.pi / 4, math.pi / 4)
        panel_domain_x2 = (-math.pi / 4, math.pi / 4)

        # Find the extent covered by this particular processor (a tile), in radians
        delta_x1_panel = (panel_domain_x1[1] - panel_domain_x1[0]) / ptopo.num_lines_per_panel
        delta_x2_panel = (panel_domain_x2[1] - panel_domain_x2[0]) / ptopo.num_lines_per_panel

        # Find the lower and upper bounds of x1, x2 for this processor
        start_x1_tile = -math.pi / 4 + ptopo.my_col * delta_x1_panel
        end_x1_tile = start_x1_tile + delta_x1_panel

        start_x2_tile = -math.pi / 4 + ptopo.my_row * delta_x2_panel
        end_x2_tile = start_x2_tile + delta_x2_panel

        domain_x1 = (start_x1_tile, end_x1_tile)  # in radians
        domain_x2 = (start_x2_tile, end_x2_tile)  # in radians

        # Assign the number of elements and solution points to the CubedSphere
        # object, in order to generate compatible grids and loops elsewhere
        num_elements_x1 = num_elements_horizontal
        num_elements_x2 = num_elements_horizontal
        self.num_elements_x1 = num_elements_x1
        self.num_elements_x2 = num_elements_x2

        ## Element sizes

        # Equiangular coordinates
        self.delta_x1 = (domain_x1[1] - domain_x1[0]) / num_elements_x1  # west-east grid spacing in radians
        self.delta_x2 = (domain_x2[1] - domain_x2[0]) / num_elements_x2  # south-north grid spacing in radians
        delta_x1 = self.delta_x1
        delta_x2 = self.delta_x2

        # Helper variables to normalize the intra-element computational coordinate
        minComp = min(self.extension)
        maxComp = max(self.extension)
        Δcomp = maxComp - minComp

        # Define the coordinate values at the interfaces between elements
        # interfaces_x1 = numpy.linspace(start = domain_x1[0], stop = domain_x1[1], num = num_elements_x1 + 1)
        # interfaces_x2 = numpy.linspace(start = domain_x2[0], stop = domain_x2[1], num = num_elements_x2 + 1)
        # interfaces_x3 = numpy.linspace(start = domain_x3[0], stop = domain_x3[1], num = num_elements_x3 + 1)
        # interfaces_eta = numpy.linspace(start = domain_eta[0], stop = domain_eta[1], num = num_elements_x3 + 1)

        ni = num_elements_x1 * num_solpts
        nj = num_elements_x2 * num_solpts
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
        self.itf_i_shape_2d = (nj, num_elements_x1 + 1)
        self.itf_j_shape_2d = (num_elements_x2 + 1, ni)

        # The shapes. For a single field (e.g. coordinates of solution points, in the current case), we
        # have an array of elements, where each element has a total num_solpts**2 (in 2D) or num_solpts**3 (in 3D)
        # solution points.
        # For the interfaces we have 3 cases:
        # - In 2D, we have 2 arrays (for west-east and south-north interfaces), with the same shape: an array of all
        #   elements, each with 2*num_solpts interface points (num_solpts for each side of the element along that
        #   direction)
        # - In 3D, horizontally, we also have two arrays (west-east, south-north), but the shape is to be determined
        # - In 3D vertically, we only have one array, with shape similar to horizontally, TBD
        self.grid_shape = (num_elements_x1, num_elements_x2, num_solpts * num_solpts)
        # self.grid_shape_3d_new = (num_elements_x1 * num_elements_x2 * num_elements_x3, num_solpts * num_solpts * num_solpts)
        # i and j interfaces have same shape (because same number of elements in each direction)
        self.itf_i_shape = (num_elements_x2, num_elements_x1 + 2, num_solpts * 2)
        self.itf_j_shape = (num_elements_x2 + 2, num_elements_x1, num_solpts * 2)

        self.west_edge = numpy.s_[..., 0, :num_solpts]  # West boundary of the western halo elements
        self.east_edge = numpy.s_[..., -1, num_solpts:]  # East boundary of the eastern halo elements
        self.south_edge = numpy.s_[..., 0, :, :num_solpts]  # South boundary of the southern halo elements
        self.north_edge = numpy.s_[..., -1, :, num_solpts:]  # North boundary of the northern halo elements

        ## Coordinate vectors for the numeric / angular coordinate system

        # --- 1D element-counting arrays, for coordinate assignment

        # Element interior
        offsets_x1 = domain_x1[0] + delta_x1 * xp.arange(num_elements_x1)
        ref_solpts_x1 = delta_x1 / Δcomp * (-minComp + self.solutionPoints)
        self.x1 = xp.repeat(offsets_x1, num_solpts) + xp.tile(ref_solpts_x1, num_elements_x1)

        offsets_x2 = domain_x2[0] + delta_x2 * xp.arange(num_elements_x2)
        ref_solpts_x2 = delta_x2 / Δcomp * (-minComp + self.solutionPoints)
        self.x2 = xp.repeat(offsets_x2, num_solpts) + xp.tile(ref_solpts_x2, num_elements_x2)

        # Element interfaces
        self.x1_itf_i = xp.linspace(
            domain_x1[0], domain_x1[1], num_elements_x1 + 1
        )  # At every boundary between elements
        self.x2_itf_i = self.x2.copy()  # Copy over x2, without change because of tensor product structure

        self.x1_itf_j = self.x1.copy()
        self.x2_itf_j = xp.linspace(domain_x2[0], domain_x2[1], num_elements_x2 + 1)

        ## Construct the combined coordinate vector for the numeric/equiangular coordinate (x1, x2)
        self.block_radians_x1, self.block_radians_x2 = xp.meshgrid(self.x1, self.x2)
        i_x1, i_x2 = xp.meshgrid(self.x1_itf_i, self.x2_itf_i, indexing="ij")
        j_x1, j_x2 = xp.meshgrid(self.x1_itf_j, self.x2_itf_j)

        self.radians = self._to_new(xp.array([self.block_radians_x1, self.block_radians_x2]))
        self.coordVec_itf_i = self._to_new_itf_i(xp.array([i_x1, i_x2]))
        self.coordVec_itf_j = self._to_new_itf_j(xp.array([j_x1, j_x2]))

        # Compute the parameters of the rotated grid

        # if (lambda0 > 0.) or (lambda0 <= -math.pi / 2.):
        #    print('lambda0 not within the acceptable range of ]-pi/2 , 0]. Stopping.')
        #    exit(1)

        # if (phi0 <= -math.pi/4.) or (phi0 > math.pi/4.):
        #    print('phi0 not within the acceptable range of ]-pi/4 , pi/4]. Stopping.')
        #    exit(1)

        # if (alpha0 <= -math.pi/2.) or (alpha0 > 0.):
        #    print('alpha0 not within the acceptable range of ]-pi/2 , 0]. Stopping.')
        #    exit(1)

        c1 = math.cos(lambda0)
        c2 = math.cos(phi0)
        c3 = math.cos(alpha0)
        s1 = math.sin(lambda0)
        s2 = math.sin(phi0)
        s3 = math.sin(alpha0)

        if ptopo.my_panel == 0:
            lon_p = lambda0
            lat_p = phi0
            angle_p = alpha0

        elif ptopo.my_panel == 1:
            lon_p = math.atan2(s1 * s2 * s3 + c1 * c3, c1 * s2 * s3 - s1 * c3)
            lat_p = -math.asin(c2 * s3)
            angle_p = math.atan2(s2, c2 * c3)

        elif ptopo.my_panel == 2:
            lon_p = math.atan2(-s1, -c1)
            lat_p = -phi0
            angle_p = -math.atan2(s3, c3)

        elif ptopo.my_panel == 3:
            lon_p = math.atan2(-s1 * s2 * s3 - c1 * c3, -c1 * s2 * s3 + s1 * c3)
            lat_p = math.asin(c2 * s3)
            angle_p = -math.atan2(s2, c2 * c3)

        elif ptopo.my_panel == 4:
            if (abs(phi0) < 1e-13) and (abs(alpha0) < 1e-13):
                lon_p = 0.0
                lat_p = math.pi / 2.0
                angle_p = -lambda0
            else:
                lon_p = math.atan2(-s1 * s2 * c3 + c1 * s3, -c1 * s2 * c3 - s1 * s3)
                lat_p = math.asin(c2 * c3)
                angle_p = math.atan2(c2 * s3, -s2)

        elif ptopo.my_panel == 5:
            if (abs(phi0) < 1e-13) and (abs(alpha0) < 1e-13):
                lon_p = 0.0
                lat_p = -math.pi / 2.0
                angle_p = lambda0
            else:
                lon_p = math.atan2(s1 * s2 * c3 - c1 * s3, c1 * s2 * c3 + s1 * s3)
                lat_p = -math.asin(c2 * c3)
                angle_p = math.atan2(c2 * s3, s2)
        else:
            raise ValueError(f"Invalid panel number {ptopo.my_panel}")

        self.lon_p = lon_p
        self.lat_p = lat_p
        self.angle_p = angle_p

        # --- define planet radius and rotation speed
        self.earth_radius = 6371220.0  # Mean radius of the earth (m)
        self.rotation_speed = 7.29212e-5  # Angular speed of rotation of the earth (radians/s)

        # Check for resized planet
        planet_scaling_factor = 1.0
        planet_is_rotating = 1.0
        self.deep = False
        self.earth_radius /= planet_scaling_factor
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
        xp = self.device.xp

        # Retrieve the numeric values for use here
        x1 = self.x1
        x2 = self.x2

        x1_itf_i = self.x1_itf_i
        x2_itf_i = self.x2_itf_i

        x1_itf_j = self.x1_itf_j
        x2_itf_j = self.x2_itf_j

        ni = self.ni
        nj = self.nj

        num_elements_x1 = self.num_elements_x1
        num_elements_x2 = self.num_elements_x2

        lon_p = self.lon_p
        lat_p = self.lat_p
        angle_p = self.angle_p

        ## Gnomonic (projected plane) coordinate values
        # X and Y (and their interface variants) are 2D arrays on the ij plane;
        # x comes before y in the indices -> Y is the "fast-varying" index

        Y_block, X_block = xp.meshgrid(xp.tan(x2), xp.tan(x1), indexing="ij")

        Y = self._to_new(Y_block)
        X = self._to_new(X_block)

        self.boundary_sn = X_block[0, :]  # Coordinates of the south and north boundaries along the X (west-east) axis
        self.boundary_we = Y_block[:, 0]  # Coordinates of the west and east boundaries along the Y (south-north) axis

        # Because of conventions used in the parallel exchanges, both i and j interface variables
        # are expected to be of size (#interface, #pts).  Compared to the usual (j,i) ordering,
        # this means that the i-interface variable should be transposed

        X_block_itf_i = xp.broadcast_to(xp.tan(x1_itf_i)[numpy.newaxis, :], (nj, num_elements_x1 + 1)).T
        Y_block_itf_i = xp.broadcast_to(xp.tan(x2_itf_i)[:, numpy.newaxis], (nj, num_elements_x1 + 1)).T
        X_block_itf_j = xp.broadcast_to(xp.tan(x1_itf_j)[numpy.newaxis, :], (num_elements_x2 + 1, ni))
        Y_block_itf_j = xp.broadcast_to(xp.tan(x2_itf_j)[:, numpy.newaxis], (num_elements_x2 + 1, ni))

        X_itf_i = self._to_new_itf_i(X_block_itf_i)
        Y_itf_i = self._to_new_itf_i(Y_block_itf_i)
        X_itf_j = self._to_new_itf_j(X_block_itf_j)
        Y_itf_j = self._to_new_itf_j(Y_block_itf_j)

        delta2 = 1.0 + X**2 + Y**2
        delta = xp.sqrt(delta2)

        delta2_itf_i = 1.0 + X_itf_i**2 + Y_itf_i**2
        delta_itf_i = xp.sqrt(delta2_itf_i)

        delta2_itf_j = 1.0 + X_itf_j**2 + Y_itf_j**2
        delta_itf_j = xp.sqrt(delta2_itf_j)

        self.X_block = X_block
        self.Y_block = Y_block
        self.X = X
        self.Y = Y
        self.delta2 = delta2
        self.delta = delta
        self.delta2_itf_i = delta2_itf_i
        self.delta_itf_i = delta_itf_i
        self.delta2_itf_j = delta2_itf_j
        self.delta_itf_j = delta_itf_j

        ## Other coordinate vectors:
        # * gnonomic coordinates (X, Y, Z)

        def to_gnomonic(coord_num, z=None):
            gnom = xp.empty_like(coord_num)
            gnom[0, ...] = xp.tan(coord_num[0])
            gnom[1, ...] = xp.tan(coord_num[1])
            if z is not None:
                gnom[2, ...] = z

            return gnom

        self.coordVec_gnom = to_gnomonic(self.radians)
        self.gnom_itf_i = to_gnomonic(self.coordVec_itf_i)
        self.gnom_itf_j = to_gnomonic(self.coordVec_itf_j)

        # * Cartesian coordinates on the deep sphere (Xc, Yc, Zc)

        def to_cartesian(gnom):
            """Built the Cartesian coordinates, by inverting the gnomonic projection.  At the north pole without grid
            rotation, the formulas are:
            Xc = (r+Z) * X / sqrt(1+X^2+Y^2)
            Yc = (r+Z) * Y / sqrt(1+X^2+Y^2)
            Zc = (r+Z) / sqrt(1+X^2+Y^2)
            """
            base_shape = gnom.shape[1:]
            cart = xp.empty((3,) + base_shape, dtype=gnom.dtype)
            delt = xp.sqrt(1.0 + gnom[0, ...] ** 2 + gnom[1, ...] ** 2)
            r = self.earth_radius + (0.0 if gnom.shape[0] < 3 else gnom[2, ...])
            cart[0, :] = (
                r
                / delt
                * (
                    math.cos(lon_p) * math.cos(lat_p)
                    + gnom[0, ...]
                    * (math.cos(lon_p) * math.sin(lat_p) * math.sin(angle_p) - math.sin(lon_p) * math.cos(angle_p))
                    - gnom[1, ...]
                    * (math.cos(lon_p) * math.sin(lat_p) * math.cos(angle_p) + math.sin(lon_p) * math.sin(angle_p))
                )
            )

            cart[1, :] = (
                r
                / delt
                * (
                    math.sin(lon_p) * math.cos(lat_p)
                    + gnom[0, ...]
                    * (math.sin(lon_p) * math.sin(lat_p) * math.sin(angle_p) + math.cos(lon_p) * math.cos(angle_p))
                    - gnom[1, ...]
                    * (math.sin(lon_p) * math.sin(lat_p) * math.cos(angle_p) - math.cos(lon_p) * math.sin(angle_p))
                )
            )

            cart[2, :] = (
                r
                / delt
                * (
                    math.sin(lat_p)
                    - gnom[0, ...] * math.cos(lat_p) * math.sin(angle_p)
                    + gnom[1, ...] * math.cos(lat_p) * math.cos(angle_p)
                )
            )

            return cart

        self.cart = to_cartesian(self.coordVec_gnom)
        self.cart_itf_i = to_cartesian(self.gnom_itf_i)
        self.cart_itf_j = to_cartesian(self.gnom_itf_j)

        # * Polar coordinates (lat, lon, Z)
        def to_polar(cart, z=None):
            if z is not None:
                polar = xp.empty_like(cart)
                polar[2, ...] = z
            else:
                base_shape = cart.shape[1:]
                polar = xp.empty((2,) + base_shape, dtype=cart.dtype)

            polar[0, ...], polar[1, ...], _ = cart2sph(cart[0, ...], cart[1, ...], cart[2, ...])

            return polar

        self.polar = to_polar(self.cart)
        self.polar_itf_i = to_polar(self.cart_itf_i)
        self.polar_itf_j = to_polar(self.cart_itf_j)

        self.polar_itf_i[self.west_edge] = 0.0
        self.polar_itf_i[self.east_edge] = 0.0
        self.polar_itf_j[self.south_edge] = 0.0
        self.polar_itf_j[self.north_edge] = 0.0

        self.lon = self.polar[0, ...]
        self.lat = self.polar[1, ...]
        self.coslon = xp.cos(self.lon)
        self.coslat = xp.cos(self.lat)
        self.sinlon = xp.sin(self.lon)
        self.sinlat = xp.sin(self.lat)

        self.block_lon = self.to_single_block(self.lon)
        self.block_lat = self.to_single_block(self.lat)

        self.X_itf_i = X_itf_i
        self.Y_itf_i = Y_itf_i
        self.X_itf_j = X_itf_j
        self.Y_itf_j = Y_itf_j

        self.lon_itf_i = self.polar_itf_i[0, ...]
        self.lat_itf_i = self.polar_itf_i[1, ...]
        self.lon_itf_j = self.polar_itf_j[0, ...]
        self.lat_itf_j = self.polar_itf_j[1, ...]

    def _to_new(self, a: NDArray) -> NDArray:
        """Convert input array to new memory layout"""

        if isinstance(a, float):
            return a

        expected_shape = (self.num_elements_x2 * self.num_solpts, self.num_elements_x1 * self.num_solpts)
        if a.ndim == 2 and a.shape == expected_shape:
            tmp_shape = (self.num_elements_x2, self.num_solpts, self.num_elements_x1, self.num_solpts)
            new_shape = self.grid_shape
            return a.reshape(tmp_shape).transpose(0, 2, 1, 3).reshape(new_shape)

        elif (a.ndim == 3 and a.shape[1:] == expected_shape) or (
            a.ndim == 4 and a.shape[2:] == expected_shape and a.shape[1] == 1
        ):
            tmp_shape = (a.shape[0], self.num_elements_x2, self.num_solpts, self.num_elements_x1, self.num_solpts)
            new_shape = (a.shape[0],) + self.grid_shape
            return a.reshape(tmp_shape).transpose(0, 1, 3, 2, 4).reshape(new_shape)

        raise ValueError(f"Unhandled number of dimensions... Shape is {a.shape}")

    def to_single_block(self, a: NDArray) -> NDArray:
        """Convert input array from a list of elements to a single block of points layout."""
        expected_shape = (self.num_elements_x2, self.num_elements_x1, self.num_solpts * self.num_solpts)
        xp = self.device.xp

        if a.shape[-3:] != expected_shape:
            raise ValueError(f"Unexpected shape {a.shape} (expected (..., ) + {expected_shape})")

        tmp_shape = a.shape[:-3] + (self.num_elements_x2, self.num_elements_x1, self.num_solpts, self.num_solpts)
        new_shape = a.shape[:-3] + self.grid_shape_2d
        return xp.moveaxis(a.reshape(tmp_shape), -3, -2).reshape(new_shape)

    def middle_itf_i(self, a):
        """Extract the non-halo part of the given west-east interface array"""
        return a[..., 1:-1, :]

    def middle_itf_j(self, a):
        """Extract the non-halo part of the given south-north interface array"""
        return a[..., 1:-1, :, :]

    def _to_new_itf_i(self, a):
        """Convert input array (west and east interface) to new memory layout"""
        # expected_shape = (self.num_elements_x2 * self.num_solpts, self.num_elements_x1 + 1)
        xp = self.device.xp
        expected_shape = (self.num_elements_x1 + 1, self.num_elements_x2 * self.num_solpts)
        if a.shape[-2:] == expected_shape:
            plane_shape = (self.num_elements_x2, self.num_elements_x1 + 2, self.num_solpts * 2)
            new = xp.empty(a.shape[:-2] + plane_shape, dtype=a.dtype)
            west_itf = xp.arange(self.num_solpts)
            east_itf = xp.arange(self.num_solpts, 2 * self.num_solpts)

            tmp_shape = a.shape[:-2] + (self.num_elements_x1 + 1, self.num_elements_x2, self.num_solpts)
            offset = len(a.shape) - 2
            transp = tuple([i for i in range(offset)]) + (1 + offset, 0 + offset, 2 + offset)

            tmp_array = a.reshape(tmp_shape).transpose(transp)
            new[..., :-1, east_itf] = tmp_array
            new[..., 1:, west_itf] = tmp_array

            new[self.east_edge] = 0.0
            new[self.west_edge] = 0.0

            return xp.squeeze(new)
        else:
            raise ValueError(f"Unexpected array shape {a.shape} (expected {expected_shape})")

    def _to_new_itf_j(self, a):
        """Convert input array (south and north interface) to new memory layout"""
        xp = self.device.xp
        expected_shape = (self.num_elements_x2 + 1, self.num_elements_x1 * self.num_solpts)
        if a.shape[-2:] == expected_shape:
            plane_shape = (self.num_elements_x2 + 2, self.num_elements_x1, self.num_solpts * 2)
            new = xp.empty(a.shape[:-2] + plane_shape, dtype=a.dtype)
            # new[...] = 1000.0
            south = numpy.s_[..., 1:, :, : self.num_solpts]  # South boundary of elements, including northern halo
            north = numpy.s_[..., :-1, :, self.num_solpts :]  # North boundary of elements, including southern halo

            tmp_shape = a.shape[:-2] + (self.num_elements_x2 + 1, self.num_elements_x1, self.num_solpts)
            tmp_array = a.reshape(tmp_shape)
            new[north] = tmp_array
            new[south] = tmp_array

            new[self.south_edge] = 0.0
            new[self.north_edge] = 0.0

            # raise ValueError(f'a = \n{a[0]}\nnew_tmp = \n{new_tmp[0]}\nnew = \n{new[0]}')

            return xp.squeeze(new)
        else:
            raise ValueError(f"Unexpected array shape {a.shape} (expected {expected_shape})")

    def wind2contra(self, u: float | NDArray, v: float | NDArray):
        """Convert wind fields from the spherical basis (zonal, meridional) to panel-appropriate contravariant winds

        Parameters:
        ----------
        u : float | NDArray
           Input zonal winds, in meters per second
        v : float | NDArray
           Input meridional winds, in meters per second

        Returns:
        -------
        (u1_contra, u2_contra) : tuple
           Tuple of contravariant winds"""

        xp = self.device.xp

        # Convert winds coords to spherical basis

        if self.nk > 1 and self.deep:
            # In 3D code with the deep atmosphere, the conversion to λ and φ
            # uses the full radial height of the grid point:
            lambda_dot = u / ((self.earth_radius + self.coordVec_gnom[2, ...]) * self.coslat)
            phi_dot = v / (self.earth_radius + self.coordVec_gnom[2, ...])
        else:
            # Otherwise, the conversion uses just the planetary radius, with no
            # correction for height above the surface
            lambda_dot = u / (self.earth_radius * self.coslat)
            phi_dot = v / self.earth_radius

        denom = xp.sqrt(
            (
                math.cos(self.lat_p)
                + self.X * math.sin(self.lat_p) * math.sin(self.angle_p)
                - self.Y * math.sin(self.lat_p) * math.cos(self.angle_p)
            )
            ** 2
            + (self.X * math.cos(self.angle_p) + self.Y * math.sin(self.angle_p)) ** 2
        )

        dx1dlon = math.cos(self.lat_p) * math.cos(self.angle_p) + (
            self.X * self.Y * math.cos(self.lat_p) * math.sin(self.angle_p) - self.Y * math.sin(self.lat_p)
        ) / (1.0 + self.X**2)
        dx2dlon = (self.X * self.Y * math.cos(self.lat_p) * math.cos(self.angle_p) + self.X * math.sin(self.lat_p)) / (
            1.0 + self.Y**2
        ) + math.cos(self.lat_p) * math.sin(self.angle_p)

        dx1dlat = (
            -self.delta2
            * ((math.cos(self.lat_p) * math.sin(self.angle_p) + self.X * math.sin(self.lat_p)) / (1.0 + self.X**2))
            / denom
        )
        dx2dlat = (
            self.delta2
            * ((math.cos(self.lat_p) * math.cos(self.angle_p) - self.Y * math.sin(self.lat_p)) / (1.0 + self.Y**2))
            / denom
        )

        # transform to the reference element

        u1_contra = (dx1dlon * lambda_dot + dx1dlat * phi_dot) * 2.0 / self.delta_x1
        u2_contra = (dx2dlon * lambda_dot + dx2dlat * phi_dot) * 2.0 / self.delta_x2

        return u1_contra, u2_contra

    def contra2wind(self, u1: float | NDArray, u2: float | NDArray) -> tuple[NDArray, NDArray]:
        """Convert from reference element to "physical winds", in two dimensions

        Parameters:
        -----------
        u1 : float | NDArray
           Contravariant winds along first component (X)
        u2 : float | NDArray
           Contravariant winds along second component (Y)

        Returns:
        --------
        (u, v) : tuple
           Zonal/meridional winds, in m/s
        """

        xp = self.device.xp

        u1_contra = u1 * self.delta_x1 / 2.0
        u2_contra = u2 * self.delta_x2 / 2.0

        denom = (
            math.cos(self.lat_p)
            + self.X * math.sin(self.lat_p) * math.sin(self.angle_p)
            - self.Y * math.sin(self.lat_p) * math.cos(self.angle_p)
        ) ** 2 + (self.X * math.cos(self.angle_p) + self.Y * math.sin(self.angle_p)) ** 2

        dlondx1 = (
            (math.cos(self.lat_p) * math.cos(self.angle_p) - self.Y * math.sin(self.lat_p)) * (1.0 + self.X**2) / denom
        )

        dlondx2 = (
            (math.cos(self.lat_p) * math.sin(self.angle_p) + self.X * math.sin(self.lat_p)) * (1.0 + self.Y**2) / denom
        )

        denom[...] = xp.sqrt(
            (
                math.cos(self.lat_p)
                + self.X * math.sin(self.lat_p) * math.sin(self.angle_p)
                - self.Y * math.sin(self.lat_p) * math.cos(self.angle_p)
            )
            ** 2
            + (self.X * math.cos(self.angle_p) + self.Y * math.sin(self.angle_p)) ** 2
        )

        dlatdx1 = -(
            (
                self.X * self.Y * math.cos(self.lat_p) * math.cos(self.angle_p)
                + self.X * math.sin(self.lat_p)
                + (1.0 + self.Y**2) * math.cos(self.lat_p) * math.sin(self.angle_p)
            )
            * (1.0 + self.X**2)
        ) / (self.delta2 * denom)

        dlatdx2 = (
            (
                (1.0 + self.X**2) * math.cos(self.lat_p) * math.cos(self.angle_p)
                + self.X * self.Y * math.cos(self.lat_p) * math.sin(self.angle_p)
                - self.Y * math.sin(self.lat_p)
            )
            * (1.0 + self.Y**2)
        ) / (self.delta2 * denom)

        u = (dlondx1 * u1_contra + dlondx2 * u2_contra) * self.coslat * self.earth_radius
        v = (dlatdx1 * u1_contra + dlatdx2 * u2_contra) * self.earth_radius

        return u, v
