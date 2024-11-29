import math

from mpi4py import MPI
import numpy
from numpy.typing import NDArray

from .cubed_sphere import CubedSphere
from .sphere import cart2sph

# For type hints
from common.process_topology import ProcessTopology
from common.configuration import Configuration
from common.device import Device, default_device


class CubedSphere3D(CubedSphere):

    def __init__(
        self,
        nb_elements_horizontal: int,
        nb_elements_vertical: int,
        nbsolpts: int,
        λ0: float,
        ϕ0: float,
        α0: float,
        ztop: float,
        ptopo: ProcessTopology,
        param: Configuration,
        device: Device = default_device,
    ):
        """Initialize the cubed sphere geometry, for an earthlike sphere with no topography.

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
        """
        super().__init__(nbsolpts, device)
        xp = device.xp
        self.dtype = numpy.dtype(numpy.float64)

        rank = MPI.COMM_WORLD.rank

        ## Panel / parallel decomposition properties
        self.ptopo = ptopo

        # Full extent of the cubed-sphere panel
        panel_domain_x1 = (-math.pi / 4, math.pi / 4)
        panel_domain_x2 = (-math.pi / 4, math.pi / 4)

        # Find the extent covered by this particular processor
        delta_x1_PE = (panel_domain_x1[1] - panel_domain_x1[0]) / ptopo.nb_lines_per_panel
        delta_x2_PE = (panel_domain_x2[1] - panel_domain_x2[0]) / ptopo.nb_lines_per_panel

        # Find the lower and upper bounds of x1, x2 for this processor
        PE_start_x1 = -math.pi / 4 + ptopo.my_col * delta_x1_PE
        PE_end_x1 = PE_start_x1 + delta_x1_PE

        PE_start_x2 = -math.pi / 4 + ptopo.my_row * delta_x2_PE
        PE_end_x2 = PE_start_x2 + delta_x2_PE

        PE_start_x3 = 0.0
        PE_end_x3 = ztop

        self.ztop = ztop

        # Define the computational η coordinate
        PE_start_eta = 0.0
        PE_end_eta = 1.0

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
        self.nb_elements = nb_elements_x1 * nb_elements_x2 * nb_elements_x3

        ## Element sizes

        # Equiangular coordinates
        delta_x1 = (domain_x1[1] - domain_x1[0]) / nb_elements_x1
        delta_x2 = (domain_x2[1] - domain_x2[0]) / nb_elements_x2
        delta_x3 = (domain_x3[1] - domain_x3[0]) / nb_elements_x3
        delta_eta = (domain_eta[1] - domain_eta[0]) / nb_elements_x3

        self.delta_x1 = delta_x1
        self.delta_x2 = delta_x2
        self.delta_x3 = delta_x3
        self.delta_eta = delta_eta

        # Helper variables to normalize the intra-element computational coordinate
        minComp = min(self.extension)
        maxComp = max(self.extension)
        delta_comp = maxComp - minComp

        # Define the coordinate values at the interfaces between elements
        # interfaces_x1 = numpy.linspace(start = domain_x1[0], stop = domain_x1[1], num = nb_elements_x1 + 1)
        # interfaces_x2 = numpy.linspace(start = domain_x2[0], stop = domain_x2[1], num = nb_elements_x2 + 1)
        # interfaces_x3 = numpy.linspace(start = domain_x3[0], stop = domain_x3[1], num = nb_elements_x3 + 1)
        # interfaces_eta = numpy.linspace(start = domain_eta[0], stop = domain_eta[1], num = nb_elements_x3 + 1)

        ni = nb_elements_x1 * nbsolpts
        nj = nb_elements_x2 * nbsolpts
        nk = nb_elements_x3 * nbsolpts

        # Save array size properties
        # NOTE: ni is the number of "columns" and nj is the number of "rows" in a 2D matrix storing solution points
        # Since this is python, matrices are stored in row-major order (columns are the fast-varying index)
        self.ni = ni
        self.nj = nj
        self.nk = nk

        # Define the shape of the coordinate grid on this process
        grid_shape_3d = (nk, nj, ni)
        self.grid_shape_3d = grid_shape_3d
        self.grid_shape_2d = (nj, ni)
        # And the shape of arrays corresponding to each of the three interfaces
        self.itf_i_shape_3d = (nk, nj, nb_elements_x1 + 1)
        self.itf_j_shape_3d = (nk, nb_elements_x2 + 1, ni)
        self.itf_k_shape_3d = (nb_elements_x3 + 1, nj, ni)

        # The shapes. For a single field (e.g. coordinates of solution points, in the current case), we
        # have an array of elements, where each element has a total nbsolpts**2 (in 2D) or nbsolpts**3 (in 3D) solution
        # points.
        # For the interfaces we have 3 cases:
        # - In 2D, we have 2 arrays (for west-east and south-north interfaces), with the same shape: an array of all
        #   elements, each with 2*nbsolpts interface points (nbsolpts for each side of the element along that direction)
        # - In 3D, horizontally, we also have two arrays (west-east, south-north), but the shape is to be determined
        # - In 3D vertically, we only have one array, with shape similar to horizontally, TBD
        self.block_shape = (nk, nj, ni)
        self.grid_shape_3d_new = (self.nb_elements_x3, self.nb_elements_x2, self.nb_elements_x1, nbsolpts**3)
        self.floor_shape = (self.nb_elements_x2, self.nb_elements_x1, nbsolpts**2)

        # Interface shapes include a halo of one element along the direction of the interface
        self.itf_i_shape = (self.nb_elements_x3, self.nb_elements_x2, self.nb_elements_x1 + 2, (nbsolpts**2) * 2)
        self.itf_j_shape = (self.nb_elements_x3, self.nb_elements_x2 + 2, self.nb_elements_x1, (nbsolpts**2) * 2)
        self.itf_k_shape = (self.nb_elements_x3 + 2, self.nb_elements_x2, self.nb_elements_x1, (nbsolpts**2) * 2)
        self.itf_i_floor_shape = (self.nb_elements_x2, self.nb_elements_x1 + 2, nbsolpts * 2)
        self.itf_j_floor_shape = (self.nb_elements_x2 + 2, self.nb_elements_x1, nbsolpts * 2)

        # Interface array edges
        self.west_edge = numpy.s_[..., 0, : nbsolpts**2]  # West boundary of the western halo elements
        self.east_edge = numpy.s_[..., -1, nbsolpts**2 :]  # East boundary of the eastern halo elements
        self.south_edge = numpy.s_[..., 0, :, : nbsolpts**2]  # South boundary of the southern halo elements
        self.north_edge = numpy.s_[..., -1, :, nbsolpts**2 :]  # North boundary of the northern halo elements
        self.bottom_edge = numpy.s_[..., 0, :, :, : nbsolpts**2]  # Bottom boundary of bottom halo elements
        self.top_edge = numpy.s_[..., -1, :, :, nbsolpts**2 :]  # Top boundary of top halo elements

        self.floor_west_edge = numpy.s_[..., 0, :nbsolpts]
        self.floor_east_edge = numpy.s_[..., -1, nbsolpts:]
        self.floor_south_edge = numpy.s_[..., 0, :, :nbsolpts]
        self.floor_north_edge = numpy.s_[..., -1, :, nbsolpts:]

        # Assign a token zbot, potentially to be overridden later with supplied topography
        self.zbot = xp.zeros(self.grid_shape_2d)

        ## Coordinate vectors for the numeric / angular coordinate system

        # Define the base coordinate.  x1 and x2 are fundamentally 1D arrays,
        # while x3 and eta are 3D arrays in support of coordinate mapping
        offsets_x1 = domain_x1[0] + delta_x1 * xp.arange(nb_elements_x1)
        ref_solpts_x1 = delta_x1 / delta_comp * (-minComp + self.solutionPoints)
        x1 = xp.repeat(offsets_x1, nbsolpts) + xp.tile(ref_solpts_x1, nb_elements_x1)

        offsets_x2 = domain_x2[0] + delta_x2 * xp.arange(nb_elements_x2)
        ref_solpts_x2 = delta_x2 / delta_comp * (-minComp + self.solutionPoints)
        x2 = xp.repeat(offsets_x2, nbsolpts) + xp.tile(ref_solpts_x2, nb_elements_x2)

        offsets_x3 = domain_x3[0] + delta_x3 * xp.arange(nb_elements_x3)
        ref_solpts_x3 = delta_x3 / delta_comp * (-minComp + self.solutionPoints)
        x3_linear = xp.repeat(offsets_x3, nbsolpts) + xp.tile(ref_solpts_x3, nb_elements_x3)
        x3 = xp.repeat(x3_linear, ni * nj).reshape(self.grid_shape_3d)

        offsets_eta = domain_eta[0] + delta_eta * xp.arange(nb_elements_x3)
        ref_solpts_eta = delta_eta / delta_comp * (-minComp + self.solutionPoints)
        eta_linear = xp.repeat(offsets_eta, nbsolpts) + xp.tile(ref_solpts_eta, nb_elements_x3)
        eta = xp.repeat(eta_linear, ni * nj).reshape(self.grid_shape_3d)

        # Repeat for the interface values
        x1_itf_i = xp.linspace(domain_x1[0], domain_x1[1], nb_elements_x1 + 1)  # At every boundary between elements
        x2_itf_i = x2.copy()
        x3_itf_i = xp.repeat(x3[:, :, 0], nb_elements_x1 + 1).reshape(self.itf_i_shape_3d)  # Repeat zy plane
        eta_itf_i = xp.repeat(eta[:, :, 0], nb_elements_x1 + 1).reshape(self.itf_i_shape_3d)

        x1_itf_j = x1.copy()
        x2_itf_j = xp.linspace(domain_x2[0], domain_x2[1], nb_elements_x2 + 1)
        x3_itf_j = xp.repeat(x3[:, 0, :], nb_elements_x2 + 1).reshape(self.itf_j_shape_3d)
        eta_itf_j = xp.repeat(eta[:, 0, :], nb_elements_x2 + 1).reshape(self.itf_j_shape_3d)

        x1_itf_k = x1.copy()
        x2_itf_k = x2.copy()
        x3_itf_k_linear = xp.linspace(domain_x3[0], domain_x3[1], nb_elements_x3 + 1)
        x3_itf_k = xp.repeat(x3_itf_k_linear, ni * nj).reshape(self.itf_k_shape_3d)
        eta_itf_k_linear = xp.linspace(domain_eta[0], domain_eta[1], nb_elements_x3 + 1)
        eta_itf_k = xp.repeat(eta_itf_k_linear, ni * nj).reshape(self.itf_k_shape_3d)

        # x1, x2, x3 and eta coordinates of interior grid points
        self.x1 = x1
        self.x2 = x2
        self.eta = eta
        self.x3 = x3

        # x1, x2, x3 and eta coordinates of west-east interface points
        self.x1_itf_i = x1_itf_i
        self.x2_itf_i = x2_itf_i
        self.x3_itf_i = x3_itf_i
        self.eta_itf_i = eta_itf_i

        # x1, x2, x3 and eta coordinates of south-north interface points
        self.x1_itf_j = x1_itf_j
        self.x2_itf_j = x2_itf_j
        self.x3_itf_j = x3_itf_j
        self.eta_itf_j = eta_itf_j

        # x1, x2, x3 and eta coordinates of bottom-top interface points
        self.x1_itf_k = x1_itf_k
        self.x2_itf_k = x2_itf_k
        self.x3_itf_k = x3_itf_k
        self.eta_itf_k = eta_itf_k

        ## Construct the combined coordinate vector for the numeric/equiangular coordinate (x1, x2, η)
        ## This is the numeric coordinate at every grid (and interface) point
        coordVec_num = xp.stack(
            (xp.broadcast_to(x1[None, None, :], eta.shape), xp.broadcast_to(x2[None, :, None], eta.shape), eta)
        )

        coordVec_num_itf_i = xp.stack(
            (
                xp.broadcast_to(x1_itf_i[None, None, :], eta_itf_i.shape),
                xp.broadcast_to(x2_itf_i[None, :, None], eta_itf_i.shape),
                eta_itf_i,
            )
        )
        coordVec_num_itf_j = xp.stack(
            (
                xp.broadcast_to(x1_itf_j[None, None, :], eta_itf_j.shape),
                xp.broadcast_to(x2_itf_j[None, :, None], eta_itf_j.shape),
                eta_itf_j,
            )
        )
        coordVec_num_itf_k = xp.stack(
            (
                xp.broadcast_to(x1_itf_k[None, None, :], eta_itf_k.shape),
                xp.broadcast_to(x2_itf_k[None, :, None], eta_itf_k.shape),
                eta_itf_k,
            )
        )

        self.coordVec_num = coordVec_num
        self.coordVec_num_itf_i = coordVec_num_itf_i
        self.coordVec_num_itf_j = coordVec_num_itf_j
        self.coordVec_num_itf_k = coordVec_num_itf_k

        self.radians = self._to_new(coordVec_num)
        self.radians_itf_i = self._to_new_itf_i(coordVec_num_itf_i)
        self.radians_itf_j = self._to_new_itf_j(coordVec_num_itf_j)
        self.radians_itf_k = self._to_new_itf_k(coordVec_num_itf_k)

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

        c1 = math.cos(λ0)
        c2 = math.cos(ϕ0)
        c3 = math.cos(α0)
        s1 = math.sin(λ0)
        s2 = math.sin(ϕ0)
        s3 = math.sin(α0)

        if ptopo.my_panel == 0:
            lon_p = λ0
            lat_p = ϕ0
            angle_p = α0

        elif ptopo.my_panel == 1:
            lon_p = math.atan2(s1 * s2 * s3 + c1 * c3, c1 * s2 * s3 - s1 * c3)
            lat_p = -math.asin(c2 * s3)
            angle_p = math.atan2(s2, c2 * c3)

        elif ptopo.my_panel == 2:
            lon_p = math.atan2(-s1, -c1)
            lat_p = -ϕ0
            angle_p = -math.atan2(s3, c3)

        elif ptopo.my_panel == 3:
            lon_p = math.atan2(-s1 * s2 * s3 - c1 * c3, -c1 * s2 * s3 + s1 * c3)
            lat_p = math.asin(c2 * s3)
            angle_p = -math.atan2(s2, c2 * c3)

        elif ptopo.my_panel == 4:
            if (abs(ϕ0) < 1e-13) and (abs(α0) < 1e-13):
                lon_p = 0.0
                lat_p = math.pi / 2.0
                angle_p = -λ0
            else:
                lon_p = math.atan2(-s1 * s2 * c3 + c1 * s3, -c1 * s2 * c3 - s1 * s3)
                lat_p = math.asin(c2 * c3)
                angle_p = math.atan2(c2 * s3, -s2)

        elif ptopo.my_panel == 5:
            if (abs(ϕ0) < 1e-13) and (abs(α0) < 1e-13):
                lon_p = 0.0
                lat_p = -math.pi / 2.0
                angle_p = λ0
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

        if param.case_number == 31:
            planet_scaling_factor = 125.0
            planet_is_rotating = 0.0
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
            raise AssertionError(f"Invalid Euler atmosphere depth approximation ({param.depth_approx})")
        self.earth_radius /= planet_scaling_factor
        self.rotation_speed *= planet_is_rotating / planet_scaling_factor

        # Call _build_physical_coordinates() to continue the construction of coordinate vectors
        # for the "physical" coordinates.  These are segregated into another method because they
        # will be redefined if this case involves topography mapping – x1/x2/η will remain the same,
        # as will the DG structures.
        self._build_physical_coordinates()

    def apply_topography(self, zbot, zbot_itf_i, zbot_itf_j, zbot_new, zbot_itf_i_new, zbot_itf_j_new):
        """
        Apply a topography field, given by heights (above the 0 reference sphere) specified at
        interior points, i-boundaries, and j-boundaries.  This function applies a linear mapping,
        where η=0 corresponds to the given surface and η=1 corresponds to the top.
        """

        xp = self.device.xp

        # First, preserve the topography
        self.zbot = zbot.copy()
        self.zbot_itf_i = zbot_itf_i.copy()
        self.zbot_itf_j = zbot_itf_j.copy()

        ztop = self.ztop

        # To apply the topography, we need to redefine self.x3 and its interfaced versions.

        self.x3[:, :, :] = zbot[xp.newaxis, :, :] + (ztop - zbot[xp.newaxis, :, :]) * self.eta
        self.x3_itf_i[:, :, :] = zbot_itf_i[xp.newaxis, :, :] + (ztop - zbot_itf_i[xp.newaxis, :, :]) * self.eta_itf_i
        self.x3_itf_j[:, :, :] = zbot_itf_j[xp.newaxis, :, :] + (ztop - zbot_itf_j[xp.newaxis, :, :]) * self.eta_itf_j
        self.x3_itf_k[:, :, :] = zbot[xp.newaxis, :, :] + (ztop - zbot[xp.newaxis, :, :]) * self.eta_itf_k

        # Now, rebuild the physical coordinates to re-generate X/Y/Z and the Cartesian coordinates
        self._build_physical_coordinates()

    def _build_physical_coordinates(self):
        """
        Build the physical coordinate arrays and vectors (gnomonic plane, lat/lon, Cartesian)
        based on the pre-defined equiangular coordinates (x1, x2) and height (x3)
        """
        xp = self.device.xp
        rank = MPI.COMM_WORLD.rank

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
        # x comes before y in the indices -> Y is the "fast-varying" index

        Y_block, X_block = xp.meshgrid(xp.tan(x2), xp.tan(x1), indexing="ij")

        # X_new = self.to_new_floor(X_block)
        # Y_new = self.to_new_floor(Y_block)
        X_new = self._to_new(xp.tile(X_block, (nb_elements_x3 * self.nbsolpts, 1, 1)))
        Y_new = self._to_new(xp.tile(Y_block, (nb_elements_x3 * self.nbsolpts, 1, 1)))

        self.boundary_sn = X_block[0, :]  # Coordinates of the south and north boundaries along the X (west-east) axis
        self.boundary_we = Y_block[:, 0]  # Coordinates of the west and east boundaries along the Y (south-north) axis

        self.boundary_sn_new = xp.tile(
            self.boundary_sn.reshape(self.nb_elements_x1, self.nbsolpts), self.nbsolpts
        ).reshape((self.nb_elements_x1, self.nbsolpts, self.nbsolpts))
        self.boundary_we_new = xp.tile(
            self.boundary_we.reshape(self.nb_elements_x2, self.nbsolpts), self.nbsolpts
        ).reshape((self.nb_elements_x2, self.nbsolpts, self.nbsolpts))

        # if MPI.COMM_WORLD.rank == 0:
        #     print(f"boundary sn (old): \n{self.boundary_sn}")
        #     print(f"boundary sn (new): \n{self.boundary_sn_new}")

        # if MPI.COMM_WORLD.rank == 0:
        #    print(f'old X = \n{X_block}')
        #    print(f'new X = \n{X_new}')

        height = x3

        # Because of conventions used in the parallel exchanges, both i and j interface variables
        # are expected to be of size (#interface, #pts).  Compared to the usual (j,i) ordering,
        # this means that the i-interface variable should be transposed

        X_itf_i = xp.broadcast_to(xp.tan(x1_itf_i)[xp.newaxis, :], (nj, nb_elements_x1 + 1)).T
        Y_itf_i = xp.broadcast_to(xp.tan(x2_itf_i)[:, xp.newaxis], (nj, nb_elements_x1 + 1)).T
        X_itf_j = xp.broadcast_to(xp.tan(x1_itf_j)[xp.newaxis, :], (nb_elements_x2 + 1, ni))
        Y_itf_j = xp.broadcast_to(xp.tan(x2_itf_j)[:, xp.newaxis], (nb_elements_x2 + 1, ni))

        # if MPI.COMM_WORLD.rank == 0:
        #     print(f'x itf i (shape {X_itf_i.shape})= \n{X_itf_i}')

        self.delta2_block = 1.0 + X_block**2 + Y_block**2
        self.delta_block = xp.sqrt(self.delta2_block)

        self.delta2_new = 1.0 + X_new**2 + Y_new**2
        self.delta_new = xp.sqrt(self.delta2_new)

        self.X_block = X_block
        self.Y_block = Y_block
        self.X_new = X_new
        self.Y_new = Y_new
        self.height = height

        ## Other coordinate vectors:
        # * gnonomic coordinates (X, Y, Z)

        def to_gnomonic(coord_num, z):
            gnom = xp.empty_like(coord_num)
            gnom[0] = xp.tan(coord_num[0])
            gnom[1] = xp.tan(coord_num[1])
            gnom[2] = z
            return gnom

        coordVec_gnom = to_gnomonic(coordVec_num, x3)
        coordVec_gnom_itf_i = to_gnomonic(coordVec_num_itf_i, x3_itf_i)
        coordVec_gnom_itf_j = to_gnomonic(coordVec_num_itf_j, x3_itf_j)
        coordVec_gnom_itf_k = to_gnomonic(coordVec_num_itf_k, x3_itf_k)

        self.gnomonic = to_gnomonic(self.radians, self._to_new(x3))
        self.gnomonic_itf_i = to_gnomonic(self.radians_itf_i, self._to_new_itf_i(x3_itf_i))
        self.gnomonic_itf_j = to_gnomonic(self.radians_itf_j, self._to_new_itf_j(x3_itf_j))
        self.gnomonic_itf_k = to_gnomonic(self.radians_itf_k, self._to_new_itf_k(x3_itf_k))

        # * Cartesian coordinates on the deep sphere (Xc, Yc, Zc)

        # Built the Cartesian coordinates, by inverting the gnomonic projection.  At the north pole without grid
        # rotation, the formulas are:
        # Xc = (r+Z)*X/sqrt(1+X^2+Y^2)
        # Yc = (r+Z)*Y/sqrt(1+X^2+Y^2)
        # Zc = (r+Z)/sqrt(1+X^2+Y^2)
        def gnomonic_to_cartesian(gnom):
            cart = xp.empty_like(gnom)
            delt = xp.sqrt(1.0 + gnom[0, ...] ** 2 + gnom[1, ...] ** 2)
            cart[0, ...] = (
                (self.earth_radius + gnom[2, ...])
                / delt
                * (
                    math.cos(lon_p) * math.cos(lat_p)
                    + gnom[0, ...]
                    * (math.cos(lon_p) * math.sin(lat_p) * math.sin(angle_p) - math.sin(lon_p) * math.cos(angle_p))
                    - gnom[1, ...]
                    * (math.cos(lon_p) * math.sin(lat_p) * math.cos(angle_p) + math.sin(lon_p) * math.sin(angle_p))
                )
            )

            cart[1, ...] = (
                (self.earth_radius + gnom[2, ...])
                / delt
                * (
                    math.sin(lon_p) * math.cos(lat_p)
                    + gnom[0, ...]
                    * (math.sin(lon_p) * math.sin(lat_p) * math.sin(angle_p) + math.cos(lon_p) * math.cos(angle_p))
                    - gnom[1, ...]
                    * (math.sin(lon_p) * math.sin(lat_p) * math.cos(angle_p) - math.cos(lon_p) * math.sin(angle_p))
                )
            )

            cart[2, ...] = (
                (self.earth_radius + gnom[2, ...])
                / delt
                * (
                    math.sin(lat_p)
                    - gnom[0, ...] * math.cos(lat_p) * math.sin(angle_p)
                    + gnom[1, ...] * math.cos(lat_p) * math.cos(angle_p)
                )
            )

            return cart

        coordVec_cart = gnomonic_to_cartesian(coordVec_gnom)
        coordVec_cart_itf_i = gnomonic_to_cartesian(coordVec_gnom_itf_i)
        coordVec_cart_itf_j = gnomonic_to_cartesian(coordVec_gnom_itf_j)
        coordVec_cart_itf_k = gnomonic_to_cartesian(coordVec_gnom_itf_k)

        self.cart = gnomonic_to_cartesian(self.gnomonic)
        self.cart_itf_i = gnomonic_to_cartesian(self.gnomonic_itf_i)
        self.cart_itf_j = gnomonic_to_cartesian(self.gnomonic_itf_j)
        self.cart_itf_k = gnomonic_to_cartesian(self.gnomonic_itf_k)

        # * Polar coordinates (lat, lon, Z)

        def cartesian_to_polar(cart, gnom):
            polar = xp.empty_like(cart)
            [polar[0, :], polar[1, :], _] = cart2sph(cart[0, :], cart[1, :], cart[2, :])
            polar[2, :] = gnom[2, :]

            return polar

        coordVec_latlon = cartesian_to_polar(coordVec_cart, coordVec_gnom)
        coordVec_latlon_itf_i = cartesian_to_polar(coordVec_cart_itf_i, coordVec_gnom_itf_i)
        coordVec_latlon_itf_j = cartesian_to_polar(coordVec_cart_itf_j, coordVec_gnom_itf_j)
        coordVec_latlon_itf_k = cartesian_to_polar(coordVec_cart_itf_k, coordVec_gnom_itf_k)

        self.polar = cartesian_to_polar(self.cart, self.gnomonic)
        self.polar_itf_i = cartesian_to_polar(self.cart_itf_i, self.gnomonic_itf_i)
        self.polar_itf_j = cartesian_to_polar(self.cart_itf_j, self.gnomonic_itf_j)
        self.polar_itf_k = cartesian_to_polar(self.cart_itf_k, self.gnomonic_itf_k)

        self.polar_itf_i[self.west_edge] = 0.0
        self.polar_itf_i[self.east_edge] = 0.0
        self.polar_itf_j[self.south_edge] = 0.0
        self.polar_itf_j[self.north_edge] = 0.0
        self.polar_itf_k[self.bottom_edge] = 0.0
        self.polar_itf_k[self.top_edge] = 0.0

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

        lon = coordVec_latlon[0, 0, :, :]
        lat = coordVec_latlon[1, 0, :, :]

        # The _itf_i variables are transposed here compared to the coordVec
        # order to retain compatibility with the shallow water test cases.
        # Those cases assume that the 2D interface variables are written
        # with [interface#,interior#] indexing, and for the _itf_i variables
        # this is effective the opposite of the geometric [k,j,i] order.

        lon_itf_i = coordVec_latlon_itf_i[0, 0, :, :].T
        lon_itf_j = coordVec_latlon_itf_j[0, 0, :, :]

        lat_itf_i = coordVec_latlon_itf_i[1, 0, :, :].T
        lat_itf_j = coordVec_latlon_itf_j[1, 0, :, :]

        # Map to the interval [0, 2 pi]
        # lon_itf_j[lon_itf_j<0.0] = lon_itf_j[lon_itf_j<0.0] + (2.0 * math.pi)

        self.lon = lon
        self.lat = lat
        self.block_lon = lon
        self.block_lat = lat
        self.X_itf_i = X_itf_i
        self.Y_itf_i = Y_itf_i
        self.X_itf_j = X_itf_j
        self.Y_itf_j = Y_itf_j
        self.lon_itf_i = lon_itf_i
        self.lat_itf_i = lat_itf_i
        self.lon_itf_j = lon_itf_j
        self.lat_itf_j = lat_itf_j

        self.coslon = xp.cos(lon)
        self.sinlon = xp.sin(lon)
        self.coslat = xp.cos(lat)
        self.sinlat = xp.sin(lat)

        self.coslon_new = xp.cos(self.polar[0, ...])
        self.coslat_new = xp.cos(self.polar[1, ...])

    def _to_new(self, a: NDArray) -> NDArray:
        """Convert input array to new memory layout"""
        if a.shape[-3:] != self.block_shape:
            raise ValueError(f"Unhandled shape {a.shape}, expected (...,) + {self.block_shape}")

        tmp_shape = a.shape[:-3] + (
            self.nb_elements_x3,
            self.nbsolpts,
            self.nb_elements_x2,
            self.nbsolpts,
            self.nb_elements_x1,
            self.nbsolpts,
        )
        new_shape = a.shape[:-3] + self.grid_shape_3d_new
        xp = self.device.xp

        return xp.moveaxis(a.reshape(tmp_shape), (-5, -3), (-3, -2)).reshape(new_shape)

    def to_single_block(self, a):
        """Convert input array to old memory layout"""
        if a.shape[-4:] != self.grid_shape_3d_new:
            raise ValueError(f"Unhandled shape {a.shape}, expected (...,) + {self.grid_shape_3d_new}")

        tmp_shape = a.shape[:-4] + (
            self.nb_elements_x3,
            self.nb_elements_x2,
            self.nb_elements_x1,
            self.nbsolpts,
            self.nbsolpts,
            self.nbsolpts,
        )
        new_shape = a.shape[:-4] + self.block_shape
        xp = self.device.xp

        return xp.moveaxis(a.reshape(tmp_shape), (-3, -2), (-5, -3)).reshape(new_shape)

    def _to_new_itf_i(self, a):
        """Convert input array (west and east interface) to new memory layout"""

        if a.shape[-3:] != self.itf_i_shape_3d:
            raise ValueError(f"Unexpected array shape {a.shape}, expected (...,) {self.itf_i_shape_3d})")

        xp = self.device.xp

        new = xp.empty(a.shape[:-3] + self.itf_i_shape, dtype=a.dtype)

        tmp_shape1 = a.shape[:-3] + (
            self.nb_elements_x3,
            self.nbsolpts,
            self.nb_elements_x2,
            self.nbsolpts,
            self.nb_elements_x1 + 1,
        )
        tmp_shape2 = a.shape[:-3] + (
            self.nb_elements_x3,
            self.nb_elements_x2,
            self.nb_elements_x1 + 1,
            self.nbsolpts**2,
        )
        tmp_array = xp.moveaxis(a.reshape(tmp_shape1), (-4, -2), (-2, -1)).reshape(tmp_shape2)

        west = numpy.s_[..., 1:, : self.nbsolpts**2]
        east = numpy.s_[..., :-1, self.nbsolpts**2 :]
        new[west] = tmp_array
        new[east] = tmp_array

        new[self.west_edge] = 0.0
        new[self.east_edge] = 0.0

        return new

    def _to_new_itf_j(self, a):
        """Convert input array (south and north interface) to new memory layout"""
        xp = self.device.xp

        if a.shape[-3:] != self.itf_j_shape_3d:
            raise ValueError(f"Unexpected array shape {a.shape}, expected (...,) {self.itf_j_shape_3d})")

        new = xp.zeros(a.shape[:-3] + self.itf_j_shape, dtype=a.dtype)

        tmp_shape1 = a.shape[:-3] + (
            self.nb_elements_x3,
            self.nbsolpts,
            self.nb_elements_x2 + 1,
            self.nb_elements_x1,
            self.nbsolpts,
        )
        tmp_shape2 = a.shape[:-3] + (
            self.nb_elements_x3,
            self.nb_elements_x2 + 1,
            self.nb_elements_x1,
            self.nbsolpts**2,
        )
        tmp_array = xp.moveaxis(a.reshape(tmp_shape1), -4, -2).reshape(tmp_shape2)

        south = xp.s_[..., 1:, :, : self.nbsolpts**2]
        north = xp.s_[..., :-1, :, self.nbsolpts**2 :]
        new[south] = tmp_array
        new[north] = tmp_array

        new[self.south_edge] = 0.0
        new[self.north_edge] = 0.0

        return new

    def _to_new_itf_k(self, a):
        """Convert input array (bottom and top interface) to new memory layout"""
        xp = self.device.xp

        if a.shape[-3:] != self.itf_k_shape_3d:
            raise ValueError(f"Unexpected array shape {a.shape}, expected (...,) {self.itf_k_shape_3d})")

        # plane_shape = (self.nb_elements_x3 + 2, self.nb_elements_x2, self.nb_elements_x1, (self.nbsolpts**2) * 2)
        new = xp.zeros(a.shape[:-3] + self.itf_k_shape, dtype=a.dtype)

        tmp_shape1 = a.shape[:-3] + (
            self.nb_elements_x3 + 1,
            self.nb_elements_x2,
            self.nbsolpts,
            self.nb_elements_x1,
            self.nbsolpts,
        )
        tmp_shape2 = a.shape[:-3] + (
            self.nb_elements_x3 + 1,
            self.nb_elements_x2,
            self.nb_elements_x1,
            self.nbsolpts**2,
        )
        tmp_array = xp.swapaxes(a.reshape(tmp_shape1), -3, -2).reshape(tmp_shape2)

        bottom = xp.s_[..., 1:, :, :, : self.nbsolpts**2]
        top = xp.s_[..., :-1, :, :, self.nbsolpts**2 :]
        new[bottom] = tmp_array
        new[top] = tmp_array

        new[self.bottom_edge] = 0.0
        new[self.top_edge] = 0.0

        return new

    def get_floor(self, a):
        """Retrieve slice of 'a' that's on the bottom (floor)"""
        if a.shape[-4:] != self.grid_shape_3d_new:
            raise ValueError(f"Unhandled shape {a.shape}, expected ... + {self.grid_shape_3d_new}")

        tmp_shape1 = a.shape[:-1] + (self.nbsolpts, self.nbsolpts, self.nbsolpts)
        tmp_shape2 = a.shape[:-4] + self.floor_shape
        floor = numpy.s_[..., 0, :, :, 0, :, :]
        return a.reshape(tmp_shape1)[floor].reshape(tmp_shape2)

    def get_itf_i_floor(self, a):
        """Retrieve slice of interface-i array 'a' that's on the floor"""
        if a.shape[-4:] != self.itf_i_shape:
            raise ValueError(f"Unhandled shape {a.shape}, expected ... + {self.itf_i_shape}")

        tmp_shape1 = a.shape[:-1] + (2, self.nbsolpts, self.nbsolpts)
        tmp_shape2 = a.shape[:-4] + self.itf_i_floor_shape
        floor = numpy.s_[..., 0, :, :, :, 0, :]
        return a.reshape(tmp_shape1)[floor].reshape(tmp_shape2)

    def get_itf_j_floor(self, a):
        """Retrieve slice of interface-j array 'a' that's on the floor"""
        if a.shape[-4:] != self.itf_j_shape:
            raise ValueError(f"Unhandled shape {a.shape}, expected ... + {self.itf_j_shape}")

        tmp_shape1 = a.shape[:-1] + (2, self.nbsolpts, self.nbsolpts)
        tmp_shape2 = a.shape[:-4] + self.itf_j_floor_shape
        floor = numpy.s_[..., 0, :, :, :, 0, :]
        return a.reshape(tmp_shape1)[floor].reshape(tmp_shape2)

    def to_new_floor(self, a: NDArray) -> NDArray:
        """Convert floor array from old to new layout"""
        if a.shape[-2:] != self.grid_shape_2d:
            raise ValueError(f"Unhandled shape {a.shape}, expected ... + {self.grid_shape_2d}")

        tmp_shape1 = a.shape[:-2] + (self.nb_elements_x2, self.nbsolpts, self.nb_elements_x1, self.nbsolpts)
        end_shape = a.shape[:-2] + self.floor_shape

        return numpy.swapaxes(a.reshape(tmp_shape1), -3, -2).reshape(end_shape)

    def to_new_itf_i_floor(self, a: NDArray) -> NDArray:
        """Convert itf-i array from old to new layout"""
        expected_shape = self.itf_i_shape_3d[1:]
        if a.shape[-2:] != expected_shape:
            raise ValueError(f"Unhandled shape {a.shape}, expected ... + {expected_shape}")

        xp = self.device.xp

        new = xp.zeros(a.shape[:-2] + self.itf_i_floor_shape, dtype=a.dtype)
        tmp_shape1 = a.shape[:-2] + (self.nb_elements_x2, self.nbsolpts, self.nb_elements_x1 + 1)

        tmp_array = numpy.moveaxis(a.reshape(tmp_shape1), -2, -1)

        west = numpy.s_[..., 1:, : self.nbsolpts]
        east = numpy.s_[..., :-1, self.nbsolpts :]
        new[west] = tmp_array
        new[east] = tmp_array

        return new

    def to_new_itf_j_floor(self, a: NDArray) -> NDArray:
        """Convert itf-j array from old to new layout"""
        expected_shape = self.itf_j_shape_3d[1:]
        if a.shape[-2:] != expected_shape:
            raise ValueError(f"Unhandled shape {a.shape}, expected ... + {expected_shape}")

        xp = self.device.xp

        new = xp.zeros(a.shape[:-2] + self.itf_j_floor_shape, dtype=a.dtype)
        tmp_shape1 = a.shape[:-2] + (self.nb_elements_x2 + 1, self.nb_elements_x1, self.nbsolpts)
        tmp_array = a.reshape(tmp_shape1)

        south = numpy.s_[..., 1:, :, : self.nbsolpts]
        north = numpy.s_[..., :-1, :, self.nbsolpts :]
        new[south] = tmp_array
        new[north] = tmp_array

        return new

    # def _to_new_itf(self, a):
    #    """Convert input array (interface) to new memory layout"""

    #    expected_shape_1 = (self.nb_elements_x2 * self.nbsolpts, self.nb_elements_x1 + 1)
    #    expected_shape_2 = (self.nb_elements_x2 + 1, self.nb_elements_x1 * self.nbsolpts)
    #    expected_shapes = [expected_shape_1, expected_shape_2]

    #    if a.ndim == 2 and a.shape in expected_shapes:
    #       new_shape = self.itf_shape_2d
    #       if a.shape == expected_shape_1:
    #          tmp_shape = (self.nb_elements_x2, self.nbsolpts, self.nb_elements_x1 + 1)
    #          return a.reshape(tmp_shape).transpose

    #       elif a.shape == expected_shape_2:
    #          tmp_shape = (self.nb_elements_x2 + 1, self.nb_elements_x1, self.nbsolpts)
    #       else: raise ValueError

    def wind2contra(
        self,
        u: float | NDArray,
        v: float | NDArray,
        w: float | NDArray,
        metric: "Metric3DTopo",
    ):
        """Convert wind fields from spherical values (zonal, meridional, vertical) to contravariant winds
        on a terrain-following grid.

        Parameters:
        ----------
        u : float | numpy.ndarray
        Input zonal winds, in meters per second
        v : float | numpy.ndarray
        Input meridional winds, in meters per second
        w : float | numpy.ndarray
        Input vertical winds, in meters per second
        metric : Metric3DTopo
            Metric object containing H_contra and inv_dzdeta parameters

        Returns:
        -------
        (u1_contra, u2_contra, u3_contra) : tuple
        Tuple of contravariant winds
        """

        xp = self.device.xp

        # First, re-use wind2contra_2d to get preliminary values for u1_contra and u2_contra.  We will update this
        # with the contribution from vertical velocity in a second step.

        if self.deep:
            # In 3D code with the deep atmosphere, the conversion to λ and φ
            # uses the full radial height of the grid point:
            lambda_dot = u / ((self.earth_radius + self.gnomonic[2, ...]) * self.coslat_new)
            phi_dot = v / (self.earth_radius + self.gnomonic[2, ...])
        else:
            # Otherwise, the conversion uses just the planetary radius, with no
            # correction for height above the surface
            lambda_dot = u / (self.earth_radius * self.coslat_new)
            phi_dot = v / self.earth_radius

        # if MPI.COMM_WORLD.rank == 0:
        #     print(f"u shape = {u.shape}, lambda_dot shape = {lambda_dot.shape}")
        #     print(f"X (new) shape = {self.X_new.shape}")

        denom = xp.sqrt(
            (
                math.cos(self.lat_p)
                + self.X_new * math.sin(self.lat_p) * math.sin(self.angle_p)
                - self.Y_new * math.sin(self.lat_p) * math.cos(self.angle_p)
            )
            ** 2
            + (self.X_new * math.cos(self.angle_p) + self.Y_new * math.sin(self.angle_p)) ** 2
        )

        dx1dlon = math.cos(self.lat_p) * math.cos(self.angle_p) + (
            self.X_new * self.Y_new * math.cos(self.lat_p) * math.sin(self.angle_p) - self.Y_new * math.sin(self.lat_p)
        ) / (1.0 + self.X_new**2)
        dx2dlon = (
            self.X_new * self.Y_new * math.cos(self.lat_p) * math.cos(self.angle_p) + self.X_new * math.sin(self.lat_p)
        ) / (1.0 + self.Y_new**2) + math.cos(self.lat_p) * math.sin(self.angle_p)

        dx1dlat = (
            -self.delta2_new
            * (
                (math.cos(self.lat_p) * math.sin(self.angle_p) + self.X_new * math.sin(self.lat_p))
                / (1.0 + self.X_new**2)
            )
            / denom
        )
        dx2dlat = (
            self.delta2_new
            * (
                (math.cos(self.lat_p) * math.cos(self.angle_p) - self.Y_new * math.sin(self.lat_p))
                / (1.0 + self.Y_new**2)
            )
            / denom
        )

        # transform to the reference element

        u1_contra = (dx1dlon * lambda_dot + dx1dlat * phi_dot) * 2.0 / self.delta_x1
        u2_contra = (dx2dlon * lambda_dot + dx2dlat * phi_dot) * 2.0 / self.delta_x2

        # Second, convert w to _covariant_ u3, which points in the vertical direction regardless of topography.
        # We do this by multiplying by dz/deta, or dividing by metric.inv_dzdeta  (equivalently, taking the dot product
        # with the e_3 basis vector)
        u3_cov = w / metric.inv_dzdeta_new

        # Now, convert covariant u3 to contravariant components.  Because topography, u^3 is normal to the
        # terrain-following x1 and x2 coordinates, implying that u^3 has horizontal components.
        # To cancel this, we need to adjust u^1 and u^2 accordingly.

        u1_contra += metric.h_contra_new[0, 2, ...] * u3_cov
        u2_contra += metric.h_contra_new[1, 2, ...] * u3_cov
        u3_contra = metric.h_contra_new[2, 2, ...] * u3_cov

        return (u1_contra, u2_contra, u3_contra)

    def contra2wind_2d(self, u1: float | NDArray, u2: float | NDArray):
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

        u1_contra = u1 * self.delta_x1 / 2.0
        u2_contra = u2 * self.delta_x2 / 2.0

        denom = (
            math.cos(self.lat_p)
            + self.X_new * math.sin(self.lat_p) * math.sin(self.angle_p)
            - self.Y_new * math.sin(self.lat_p) * math.cos(self.angle_p)
        ) ** 2 + (self.X_new * math.cos(self.angle_p) + self.Y_new * math.sin(self.angle_p)) ** 2

        dlondx1 = (
            (math.cos(self.lat_p) * math.cos(self.angle_p) - self.Y_new * math.sin(self.lat_p))
            * (1.0 + self.X_new**2)
            / denom
        )

        dlondx2 = (
            (math.cos(self.lat_p) * math.sin(self.angle_p) + self.X_new * math.sin(self.lat_p))
            * (1.0 + self.Y_new**2)
            / denom
        )

        denom[:, :] = numpy.sqrt(
            (
                math.cos(self.lat_p)
                + self.X_new * math.sin(self.lat_p) * math.sin(self.angle_p)
                - self.Y_new * math.sin(self.lat_p) * math.cos(self.angle_p)
            )
            ** 2
            + (self.X_new * math.cos(self.angle_p) + self.Y_new * math.sin(self.angle_p)) ** 2
        )

        dlatdx1 = -(
            (
                self.X_new * self.Y_new * math.cos(self.lat_p) * math.cos(self.angle_p)
                + self.X_new * math.sin(self.lat_p)
                + (1.0 + self.Y_new**2) * math.cos(self.lat_p) * math.sin(self.angle_p)
            )
            * (1.0 + self.X_new**2)
        ) / (self.delta2_new * denom)

        dlatdx2 = (
            (
                (1.0 + self.X_new**2) * math.cos(self.lat_p) * math.cos(self.angle_p)
                + self.X_new * self.Y_new * math.cos(self.lat_p) * math.sin(self.angle_p)
                - self.Y_new * math.sin(self.lat_p)
            )
            * (1.0 + self.Y_new**2)
        ) / (self.delta2_new * denom)

        if self.deep:
            # If we are in a 3D geometry with the deep atmosphere, the conversion from
            # contravariant → spherical → zonal/meridional winds uses the full radial distance
            # at the last step
            u = (
                (dlondx1 * u1_contra + dlondx2 * u2_contra)
                * self.coslat_new
                * (self.earth_radius + self.gnomonic[2, ...])
            )
            v = (dlatdx1 * u1_contra + dlatdx2 * u2_contra) * (self.earth_radius + self.gnomonic[2, ...])
        else:
            # Otherwise, the conversion is based on the spherical radius only, with no height correction
            u = (dlondx1 * u1_contra + dlondx2 * u2_contra) * self.coslat_new * self.earth_radius
            v = (dlatdx1 * u1_contra + dlatdx2 * u2_contra) * self.earth_radius

        return u, v

    def contra2wind_3d(
        self,
        u1_contra: NDArray,
        u2_contra: NDArray,
        u3_contra: NDArray,
        metric: "Metric3DTopo",
    ):
        """contra2wind_3d: convert from contravariant wind fields to "physical winds" in three dimensions

        This function transforms the contravariant fields u1, u2, and u3 into their physical equivalents, assuming
        a cubed-sphere-like grid.  In particular, we assume that the vertical coordinate η is the only mapped
        coordinate, so u3 can be ignored in computing the zonal (u) and meridional (w) wind.

        This function inverts the cubed-sphere latlon/XY coordinate transform to give u and v, taking into account
        the panel rotation, and it forms a covariant u3 field to derive w, removing any horizontal component that
        would otherwise be included inside the contravariant u3.

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
            delta_x1, delta_x2, lat_p, angle_p, X, Y, coslat, earth_radius
        metric: Metric3DTopo
        metric object, implementing H_cov (covariant spatial metric) and inv_dzdeta

        Retruns:
        (u, v, w) : tuple
        zonal, meridional, and vertical winds (m/s)
        """

        # First, re-use contra2wind_2d to compute outuput u and v.  No need to reinvent the wheel
        u, v = self.contra2wind_2d(u1_contra, u2_contra)

        # Now, form covariant u3 by taking the appropriate multiplication with the covariant metric to lower the index
        u3_cov = (
            u1_contra[...] * metric.h_cov_new[2, 0, ...]
            + u2_contra[...] * metric.h_cov_new[2, 1, ...]
            + u3_contra[...] * metric.h_cov_new[2, 2, ...]
        )

        # Covariant u3 now points straight "up", but it is expressed in η units, implicitly multiplied
        # by the covariant basis vector.
        # To convert to physical units, think of dz/dz=1 m/m.  In the covariant expression, however:
        # dz/dz = 1m/m = (z)_(,3) * e^3
        # but (z)_(,3) is the definition of dzdeta, which absorbs the (Δη/4) scaling term of the numerical
        # differentiation.  To cancel this, e^3 = inv_dzeta * (zhat), giving:

        w = u3_cov[...] * metric.inv_dzdeta_new[...]  # Multiply by (dz/dη)^(-1), or e^3

        return (u, v, w)
