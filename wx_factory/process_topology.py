import math
from typing import Callable, Optional, Tuple

from mpi4py import MPI
from numpy.typing import NDArray

from device import Device
from wx_mpi import SingleProcess, Conditional

ExchangedVector = Tuple[NDArray, ...] | NDArray

SOUTH = 0
NORTH = 1
WEST = 2
EAST = 3


class ProcessTopology:
    """Describes a cube-sphere process topology, where each process (tile) is linked to its 4 neighbors.

    Numbering of PEs starts at the bottom left. Panel ranks increase towards the east (right) in the x1 direction
    and increases towards the north (up) in the x2 direction:

    .. code-block::

               0 1 2 3 4
            +-----------+
          4 |           |
          3 |  x_2      |
          2 |  ^        |
          1 |  |        |
          0 |  + -->x_1 |
            +-----------+

    For instance, with n=96, panel 0 will be have a topology of 4x4 tiles like this

    .. code-block::

         +---+---+---+---+
         | 12| 13| 14| 15|
         |---+---+---+---|
         | 8 | 9 | 10| 11|
         |---+---+---+---|
         | 4 | 5 | 6 | 7 |
         |---+---+---+---|
         | 0 | 1 | 2 | 3 |
         +---+---+---+---+
    """

    def __init__(self, device: Device, rank: Optional[int] = None, comm: MPI.Comm = MPI.COMM_WORLD):
        """Create a cube-sphere process topology.

        :param device: Device on which MPI exchanges are to be made, like a CPU or a GPU.
        :type device: Device
        :param rank: Rank of the tile the topology will manage. This is useful for creating a tile with a
                    rank different from the process rank, for debugging purposes.
        :type rank: int, optional
        :param comm: MPI communicator through which the exchanges will occur. Defaults to COMM_WORLD
        :type comm: MPI.Comm

        """

        self.device = device

        self.comm = comm
        self.size = comm.Get_size()
        self.rank = comm.Get_rank() if rank is None else rank

        self.num_pe_per_panel = int(self.size / 6)
        self.num_lines_per_panel = int(math.sqrt(self.num_pe_per_panel))

        if (
            self.size < 6
            or self.num_pe_per_panel != self.num_lines_per_panel**2
            or self.num_pe_per_panel * 6 != self.size
        ):
            allowed_low = self.num_lines_per_panel**2 * 6
            allowed_high = (self.num_lines_per_panel + 1) ** 2 * 6
            raise ValueError(
                f"Wrong number of PEs ({self.size}). "
                f"Closest allowed processor counts are {allowed_low} and {allowed_high}"
            )

        def rank_from_location(panel, row, col):
            if row < 0:
                row += self.num_lines_per_panel
            if col < 0:
                col += self.num_lines_per_panel
            return panel * self.num_pe_per_panel + row * self.num_lines_per_panel + col

        self.my_panel = math.floor(self.rank / self.num_pe_per_panel)
        self.my_rank_in_panel = self.rank - (self.my_panel * self.num_pe_per_panel)
        self.my_row = math.floor(self.my_rank_in_panel / self.num_lines_per_panel)
        self.my_col = int(self.my_rank_in_panel % self.num_lines_per_panel)

        # --- List of panel neighbours for my panel
        #
        #      +---+
        #      | 4 |
        #  +---+---+---+---+
        #  | 3 | 0 | 1 | 2 |
        #  +---+---+---+---+
        #      | 5 |
        #      +---+
        all_neighbors = [
            # S N  W  E
            [5, 4, 3, 1],  # Panel 0
            [5, 4, 0, 2],  # Panel 1
            [5, 4, 1, 3],  # Panel 2
            [5, 4, 2, 0],  # Panel 3
            [0, 2, 3, 1],  # Panel 4
            [2, 0, 3, 1],  # Panel 5
        ]

        # fmt: off
        # (row, col) pair of the neighbor of each PE located at an edge (panel given by previous table)
        edge_coords = [
            # South/North/West/East edge
            [(-1, self.my_col),      (0, self.my_col),       (self.my_row, -1),      (self.my_row, 0)],      # panel 0
            [(-self.my_col - 1, -1), (self.my_col, -1),      (self.my_row, -1),      (self.my_row, 0)],      # panel 1
            [(0, -self.my_col - 1),  (-1, -self.my_col - 1), (self.my_row, -1),      (self.my_row, 0)],      # panel 2
            [(self.my_col, 0),       (-self.my_col - 1, 0),  (self.my_row, -1),      (self.my_row, 0)],      # panel 3
            [(-1, self.my_col),      (-1, -self.my_col - 1), (-1, -self.my_row - 1), (-1, self.my_row)],     # panel 4
            [(0, -self.my_col - 1),  (0, self.my_col),       (0, self.my_row),       (0, -self.my_row - 1)], # panel 5
        ]

        # Whether to flip data, if we are at the edge of a panel
        flips = [
            [False, False, False, False],
            [True,  False, False, False],
            [True,  True,  False, False],
            [False, True,  False, False],
            [False, True,  True,  False],
            [True,  False, False, True ],
        ]
        # fmt: on

        convert_contras = [
            [  # Panel 0
                lambda a1, a2, coord: (a1 + 2.0 * coord / (1.0 + coord**2) * a2, a2),  # South neighbor
                lambda a1, a2, coord: (a1 - 2.0 * coord / (1.0 + coord**2) * a2, a2),  # North neighbor
                lambda a1, a2, coord: (a1, 2.0 * coord / (1.0 + coord**2) * a1 + a2),  # West neighbor
                lambda a1, a2, coord: (a1, -2.0 * coord / (1.0 + coord**2) * a1 + a2),  # East neighbor
            ],
            [  # Panel 1
                lambda a1, a2, coord: (a2, -a1 - 2.0 * coord / (1.0 + coord**2) * a2),  # South neighbor
                lambda a1, a2, coord: (-a2, a1 - 2.0 * coord / (1.0 + coord**2) * a2),  # North neighbor
                lambda a1, a2, coord: (a1, 2.0 * coord / (1.0 + coord**2) * a1 + a2),  # West neighbor
                lambda a1, a2, coord: (a1, -2.0 * coord / (1.0 + coord**2) * a1 + a2),  # East neighbor
            ],
            [  # Panel 2
                lambda a1, a2, coord: (-a1 - 2.0 * coord / (1.0 + coord**2) * a2, -a2),  # South neighbor
                lambda a1, a2, coord: (-a1 + 2.0 * coord / (1.0 + coord**2) * a2, -a2),  # North neighbor
                lambda a1, a2, coord: (a1, 2.0 * coord / (1.0 + coord**2) * a1 + a2),  # West neighbor
                lambda a1, a2, coord: (a1, -2.0 * coord / (1.0 + coord**2) * a1 + a2),  # East neighbor
            ],
            [  # Panel 3
                lambda a1, a2, coord: (-a2, a1 + 2.0 * coord / (1.0 + coord**2) * a2),  # South neighbor
                lambda a1, a2, coord: (a2, -a1 + 2.0 * coord / (1.0 + coord**2) * a2),  # North neighbor
                lambda a1, a2, coord: (a1, 2.0 * coord / (1.0 + coord**2) * a1 + a2),  # West neighbor
                lambda a1, a2, coord: (a1, -2.0 * coord / (1.0 + coord**2) * a1 + a2),  # East neighbor
            ],
            [  # Panel 4
                lambda a1, a2, coord: (a1 + 2.0 * coord / (1.0 + coord**2) * a2, a2),  # South neighbor
                lambda a1, a2, coord: (-a1 + 2.0 * coord / (1.0 + coord**2) * a2, -a2),  # North neighbor
                lambda a1, a2, coord: (-2.0 * coord / (1.0 + coord**2) * a1 - a2, a1),  # West neighbor
                lambda a1, a2, coord: (-2.0 * coord / (1.0 + coord**2) * a1 + a2, -a1),  # East neigbor
            ],
            [  # Panel 5
                lambda a1, a2, coord: (-a1 - 2.0 * coord / (1.0 + coord**2) * a2, -a2),  # South neighbor
                lambda a1, a2, coord: (a1 - 2.0 * coord / (1.0 + coord**2) * a2, a2),  # North neighbor
                lambda a1, a2, coord: (2.0 * coord / (1.0 + coord**2) * a1 + a2, -a1),  # West neighbor
                lambda a1, a2, coord: (2.0 * coord / (1.0 + coord**2) * a1 - a2, a1),  # East neighbor
            ],
        ]

        convert_covs = [
            [  # Panel 0
                lambda a1, a2, x: (a1, a2 - 2.0 * x / (1.0 + x**2) * a1),  # South neighbor
                lambda a1, a2, x: (a1, a2 + 2.0 * x / (1.0 + x**2) * a1),  # North neighbor
                lambda a1, a2, x: (a1 - 2.0 * x / (1.0 + x**2) * a2, a2),  # West neighbor
                lambda a1, a2, x: (a1 + 2.0 * x / (1.0 + x**2) * a2, a2),  # East neighbor
            ],
            [  # Panel 1
                lambda a1, a2, x: (a2 - 2.0 * x / (1.0 + x**2) * a1, -a1),  # South neighbor
                lambda a1, a2, x: (-a2 - 2.0 * x / (1.0 + x**2) * a1, a1),  # North neighbor
                lambda a1, a2, x: (a1 - 2.0 * x / (1.0 + x**2) * a2, a2),  # West neighbor
                lambda a1, a2, x: (a1 + 2.0 * x / (1.0 + x**2) * a2, a2),  # East neighbor
            ],
            [  # Panel 2
                lambda a1, a2, x: (-a2, -a2 + 2.0 * x / (1.0 + x**2) * a1),  # South neighbor
                lambda a1, a2, x: (-a1, -a2 - 2.0 * x / (1.0 + x**2) * a1),  # North neighbor
                lambda a1, a2, x: (a1 - 2.0 * x / (1.0 + x**2) * a2, a2),  # West neighbor
                lambda a1, a2, x: (a1 + 2.0 * x / (1.0 + x**2) * a2, a2),  # East neighbor
            ],
            [  # Panel 3
                lambda a1, a2, x: (-a2 + 2.0 * x / (1.0 + x**2) * a1, a1),  # South neighbor
                lambda a1, a2, x: (a2 + 2.0 * x / (1.0 + x**2) * a1, -a1),  # North neighbor
                lambda a1, a2, x: (a1 - 2.0 * x / (1.0 + x**2) * a2, a2),  # West neighbor
                lambda a1, a2, x: (a1 + 2.0 * x / (1.0 + x**2) * a2, a2),  # East neighbor
            ],
            [  # Panel 4
                lambda a1, a2, x: (a1, a2 - 2.0 * x / (1.0 + x**2) * a1),  # South neighbor
                lambda a1, a2, x: (-a1, -a2 - 2.0 * x / (1.0 + x**2) * a1),  # North neighbor
                lambda a1, a2, x: (-a2, a1 - 2.0 * x / (1.0 + x**2) * a2),  # West neighbor
                lambda a1, a2, x: (a2, -a1 - 2.0 * x / (1.0 + x**2) * a2),  # East neighbor
            ],
            [  # Panel 5
                lambda a1, a2, x: (-a1, -a2 + 2.0 * x / (1.0 + x**2) * a1),  # South neighbor
                lambda a1, a2, x: (a1, a2 + 2.0 * x / (1.0 + x**2) * a1),  # North neighbor
                lambda a1, a2, x: (a2, -a1 + 2.0 * x / (1.0 + x**2) * a2),  # West neighbor
                lambda a1, a2, x: (-a2, a1 + 2.0 * x / (1.0 + x**2) * a2),  # East neighbor
            ],
        ]

        neighbor_panels = all_neighbors[self.my_panel]

        # --- Middle panel tile
        def convert_default(a1, a2, _):
            return (a1, a2)

        self.convert_contra = [convert_default, convert_default, convert_default, convert_default]
        self.convert_cov = [convert_default, convert_default, convert_default, convert_default]
        my_south = rank_from_location(self.my_panel, (self.my_row - 1), self.my_col)
        my_north = rank_from_location(self.my_panel, (self.my_row + 1), self.my_col)
        my_west = rank_from_location(self.my_panel, self.my_row, (self.my_col - 1))
        my_east = rank_from_location(self.my_panel, self.my_row, self.my_col + 1)
        self.flip = [False, False, False, False]

        # --- North panel edge
        if self.my_row == self.num_lines_per_panel - 1:
            my_north = rank_from_location(neighbor_panels[NORTH], *edge_coords[self.my_panel][NORTH])
            self.convert_contra[NORTH] = convert_contras[self.my_panel][NORTH]
            self.convert_cov[NORTH] = convert_covs[self.my_panel][NORTH]
            self.flip[NORTH] = flips[self.my_panel][NORTH]

        # --- South panel edge
        if self.my_row == 0:
            my_south = rank_from_location(neighbor_panels[SOUTH], *edge_coords[self.my_panel][SOUTH])
            self.convert_contra[SOUTH] = convert_contras[self.my_panel][SOUTH]
            self.convert_cov[SOUTH] = convert_covs[self.my_panel][SOUTH]
            self.flip[SOUTH] = flips[self.my_panel][SOUTH]

        # --- West panel edge
        if self.my_col == 0:
            my_west = rank_from_location(neighbor_panels[WEST], *edge_coords[self.my_panel][WEST])
            self.convert_contra[WEST] = convert_contras[self.my_panel][WEST]
            self.convert_cov[WEST] = convert_covs[self.my_panel][WEST]
            self.flip[WEST] = flips[self.my_panel][WEST]

        # --- East panel edge
        if self.my_col == self.num_lines_per_panel - 1:
            my_east = rank_from_location(neighbor_panels[EAST], *edge_coords[self.my_panel][EAST])
            self.convert_contra[EAST] = convert_contras[self.my_panel][EAST]
            self.convert_cov[EAST] = convert_covs[self.my_panel][EAST]
            self.flip[EAST] = flips[self.my_panel][EAST]

        # Distributed Graph
        self.sources = [my_south, my_north, my_west, my_east]  # Must correspond to values of SOUTH, NORTH, WEST, EAST
        self.destinations = self.sources
        self.comm_dist_graph = comm.Create_dist_graph_adjacent(self.sources, self.destinations)

        # Panel communicators
        self.panel_comm = self.comm.Split(self.my_panel, self.rank)
        self.panel_roots_comm = self.comm.Split(self.panel_comm.rank == 0, self.rank)
        if self.panel_comm.rank != 0:
            self.panel_roots_comm = MPI.COMM_NULL

    def start_exchange_scalars(
        self,
        south: NDArray,
        north: NDArray,
        west: NDArray,
        east: NDArray,
        boundary_shape: Tuple[int, ...],
        flip_dim: int | Tuple[int, ...] = -1,
    ):
        """Create a request for exchanging scalar data with neighboring tiles. The 4 input arrays must have the same
        shape and size.
        Input array values are assumed to be ordered so that the last dimension(s) exactly span the
        boundary (horizontally); that boundary can have any shape, as long as its total size matches `boundary_length`.
        Values along the boundary will be horizontally flipped if the neighbor has a flipped coordinate system.
        Received data will always be in local coordinates

        :param south: Array of values for the south boundary
        :param north: Array of values for the north boundary
        :param west:  Array of values for the west boundary
        :param east:  Array of values for the east boundary
        :param boundary_shape: Array shape at the boundary. It may be different (same total size or smaller) from that
            of the incoming arrays. The MPI exchange occurs with sets of lines along the boundary, but we need to
            know the shape to be able to properly flip the data between certain pairs of panels.
        :type boundary_shape: Tuple[int]
        :param flip_dim: Along which dimension to flip the data, when needed. We may need a custom flip_dim when the
            boundary shape is not linear. Defaults to -1 (the last dimension of the input array, after the boundary has
            been reshaped to a line)
        :type flip_dim: int | Tuple[int]


        :return: A request (MPI-like) for the transfer of the scalar values. Waiting on the request will return the
            resulting arrays in the same way as the input.
        :rtype: ExchangeRequest

        """
        xp = self.device.xp

        base_shape = get_base_shape(south.shape, boundary_shape)
        send_buffer = xp.empty((4,) + base_shape, dtype=south[0].dtype)
        recv_buffer = xp.empty_like(send_buffer)

        # Fill send buffer
        for i, data in enumerate([south, north, west, east]):
            tmp = data.reshape(base_shape)
            send_buffer[i] = xp.flip(tmp, axis=flip_dim) if self.flip[i] else tmp

        self.device.synchronize()  # When using GPU

        # Initiate MPI transfer
        mpi_request = self.comm_dist_graph.Ineighbor_alltoall(send_buffer, recv_buffer)

        return ExchangeRequest(recv_buffer, mpi_request, shape=south.shape, is_vector=False)

    def start_exchange_vectors(
        self,
        south: ExchangedVector,
        north: ExchangedVector,
        west: ExchangedVector,
        east: ExchangedVector,
        boundary_sn: NDArray,
        boundary_we: NDArray,
        flip_dim: int | Tuple[int, ...] = -1,
        covariant: bool = False,
    ):
        """Create a request for exchanging vectors with neighboring tiles. The 4 input vectors must have the same shape
        and size.
        Input array values are assumed to be ordered so that the last dimension(s) exactly span the
        boundary (horizontally); that boundary can have any shape, as long as its total size matches that of the given
        boundary points (`boundary_sn` and `boundary_we`)
        Vector Values are converted to the neighboring coordinate system before being transmitted through MPI.
        They will be flipped horizontally along the boundary if the neighbor has a flipped coordinate system.
        When receiving data, it will always be in local coordinates

        :param south: Tuple/List of vector components for the south boundary (one array-like for each component,
                        so there should be 2 or 3 arrays in the tuple)
        :param north: Tuple/List of vector components for the north boundary (one array-like for each component,
                        so there should be 2 or 3 arrays in the tuple)
        :param west:  Tuple/List of vector components for the west boundary (one array-like for each component,
                        so there should be 2 or 3 arrays in the tuple)
        :param east:  Tuple/List of vector components for the east boundary (one array-like for each component,
                        so there should be 2 or 3 arrays in the tuple)
        :param boundary_sn: Coordinates along the west-east axis at the south and north boundaries.
                            The entries in that vector *must* match the entries in the input arrays
        :param boundary_we: Coordinates along the south-north axis at the west and east boundaries. The entries in
                            that vector *must* match the entries in the input arrays
        :param flip_dim: Along which dimension to flip the data, when needed. We may need a custom flip_dim when the
            boundary shape is not linear. Defaults to -1 (the last dimension of the input array, after the boundary has
            been reshaped to a line)
        :type flip_dim: int | Tuple[int]

        :return: A request (MPI-like) for the transfer of the arrays. Waiting on the request will return the
                    resulting arrays in the same way as the input.
        """
        xp = self.device.xp

        convert = self.convert_cov if covariant else self.convert_contra

        base_shape = get_base_shape(south[0].shape, boundary_sn.shape)
        send_buffer = xp.empty((4, len(south)) + base_shape, dtype=south[0].dtype)
        recv_buffer = xp.empty_like(send_buffer)

        inputs = [south, north, west, east]
        boundaries = [boundary_sn, boundary_sn, boundary_we, boundary_we]
        for i, (data, bd) in enumerate(zip(inputs, boundaries)):
            send_buffer[i, 0], send_buffer[i, 1] = convert[i](
                data[0].reshape(base_shape), data[1].reshape(base_shape), bd
            )
            if len(data) == 3:
                send_buffer[i, 2] = data[2].reshape(base_shape)  # 3rd dimension if present
            if self.flip[i]:
                send_buffer[i] = xp.flip(send_buffer[i], axis=flip_dim)  # Flip arrays, if needed

        self.device.synchronize()  # When using GPU

        # Initiate MPI transfer
        mpi_request = self.comm_dist_graph.Ineighbor_alltoall(send_buffer, recv_buffer)

        return ExchangeRequest(recv_buffer, mpi_request, shape=south[0].shape, is_vector=True)

    def gather_tiles_to_panel(self, field: NDArray, num_dim: int) -> Optional[NDArray]:
        """Send given tile data (`field`) to one PE (the root) on current panel.
        The root collects all tiles and assembles them into one array. Other PEs on the panel
        simply return None.
        Each dimension along the plane of the panel is combined across tiles, like this:

        .. code-block::

            4 tiles:

             ----- -----                     ---------
            | x x | x x |   gather tiles    | x x x x |
            | x x | x x |                   | x x x x |
             ----- -----        ==>         | x x x x |
            | x x | x x |                   | x x x x |
            | x x | x x |                    ---------
             ----- -----

            Where `x` can be of any shape

        The only exception is when `field` is 1-dimensional; in that case, the arrays will be joined
        end-to-end rather that into a 2D grid.

        This function is a collective call that must be done by every PE within a panel.

        :param field: Tile data we want to send to the panel root (from this current tile)
        :param axis: Number of data dimensions on the tile (this is different from the number of array dimensions).
            This corresponds to 2 for a single shallow-water variable,
            3 for single 3d-euler variable, 4 for a set of 3d-euler variables, etc.
            This parameter is ignored when the tile is made of 1D data.
        :type axis: int
        :return: The assembled panel, as a single NDArray, on root PE; None on every non-root PE.
        """
        xp = self.device.xp

        if self.panel_comm.size == 1:
            return field

        panel_fields = self.panel_comm.gather(field, root=0)

        if panel_fields is None:  # non-root PEs
            return None

        side = self.num_lines_per_panel
        if field.ndim == 1:
            # When gathering a 1D array, put them all end-to-end
            panel_field = xp.concatenate(panel_fields[:side])
        else:
            # When gathering 2D+ array, join the tiles in a square
            panel_field = xp.concatenate(
                [xp.concatenate(panel_fields[i * side : (i + 1) * side], axis=num_dim - 1) for i in range(side)],
                axis=num_dim - 2,
            )

        return panel_field

    def gather_cube(self, field: NDArray, num_dim: int) -> Optional[NDArray]:
        """Gather given tile data into a single array on the root of this process topology. The first dimension
        will necessarily be 6; the rest will depend on the number of dimensions in the data and the number of tiles.
        This function is a collective call that must be made by every process member of this topology.

        :param field: Tile data
        :param num_dim: Number of dimensions in the data. For example:
            for a single shallow-water variable, it should be 2;
            for a set of shallow-water variables, it should be 3;
            for a set of 3D-euler variables, it should be 4
        :return: A single array with the entire cube-sphere data on process with rank 0 (in this topology);
            None on other processes
        """
        panel = self.gather_tiles_to_panel(field, num_dim)

        # Only panel roots can go further
        if panel is None:
            return None

        panels = self.panel_roots_comm.gather(panel, root=0)

        # Only the root of the entire cubesphere topology with continue
        if panels is None:
            return None

        return self.device.xp.stack(panels)

    def distribute_cube(self, field: Optional[NDArray], num_dim: int):
        """Split the given single array into its component tiles (according to this topology) and send
        each tile to its corresponding process.

        :param field: The data we want to split and distribute, if on root; ignored otherwise. *It has to
            be contiguous in memory, and not have transposed dimensions*.
        :param num_dim: Number of dimensions in the data. For example:
            for a single shallow-water variable, it should be 2;
            for a set of shallow-water variables, it should be 3;
            for a set of 3D-euler variables, it should be 4

        :return: The tile corresponding to this process within the topology
        :rtype: NDArray
        """
        side = self.num_lines_per_panel

        panel_list = None
        with SingleProcess(self.comm) as s, Conditional(s):

            # Verifications
            if field.ndim < num_dim + 1 or field.shape[0] != 6 or field.shape[num_dim - 1] != field.shape[num_dim]:
                print(f"This is not a cube with square panels {field.shape}, num_dim {num_dim}", flush=True)
                raise ValueError

            panel_side = field.shape[num_dim - 1]
            tile_side = panel_side // side
            if tile_side * side != panel_side:
                acceptable = [6 * i**2 for i in range(1, panel_side + 1) if panel_side % i == 0]
                print(
                    f"The given field shape {field.shape} cannot be distributed to this process topology\n"
                    f"Acceptable number of processes are {acceptable}",
                    flush=True,
                )
                raise ValueError

            # Group panels into list
            panel_list = [field[i] for i in range(6)]

        tile_list = None
        if self.panel_comm.rank == 0:
            # Send panel list
            panel = self.panel_roots_comm.scatter(panel_list, root=0)

            # Tile == panel if we only have 1 proc per panel
            if self.size == 6:
                return panel

            # A bit of reshaping is needed to avoid having different lines for different numbers of data dimensions
            # We flatten individual elements of each tile so that they have exactly 1 dimension (even scalars),
            # then extract the tile from the panel, then give it it's proper shape
            panel_side = panel.shape[num_dim - 2]
            tile_side = panel_side // side
            panel_base_shape = panel.shape[:num_dim]
            elem_shape = panel.shape[num_dim:]

            lin_panel = panel.reshape(panel_base_shape + (-1,))
            tile_shape = panel_base_shape[:-2] + (tile_side, tile_side) + elem_shape

            tile_list = [
                lin_panel[..., i * tile_side : (i + 1) * tile_side, j * tile_side : (j + 1) * tile_side, :].reshape(
                    tile_shape
                )
                for i in range(side)
                for j in range(side)
            ]

        tile = self.panel_comm.scatter(tile_list, root=0)

        return tile


def get_base_shape(initial_shape: tuple, length_shape: tuple[int, ...]):
    """Determine the base shape of an exchangeable array, from the given `initial_shape`. The last dimension
    of the resulting shape will horizontally span the boundary, which has size `length`

    :param initial_shape: Original shape of the array we want to transmit
    :param length: Horizontal size of boundary along which the array will be transmitted, in number of grid points

    :return: Shape of the array that will be passed to the MPI call
    :raise ValueError: If the given initial shape cannot fit the given boundary length
    """
    length = math.prod(length_shape)
    num_fuse = 1
    product = initial_shape[-1]
    while product < length:
        num_fuse += 1
        product *= initial_shape[-num_fuse]

    if product != length:
        raise ValueError(f"Unable to match data array shape ({initial_shape}) with boundary length {length}")
    return initial_shape[:-num_fuse] + length_shape


class ExchangeRequest:
    """Wrapper around an MPI request. Provides a wait function that will wait for MPI transfers to be complete and
    return the received arrays in the specified shape, split among the 4 directions/neighbors."""

    def __init__(self, recv_buffer: NDArray, request: MPI.Request, shape: tuple, is_vector: bool = False):
        """Create a request

        :param recv_buffer: Array where all the data will be received
        :param request: MPI request
        :param shape: Desired final shape of the data (can be different from the shape of the received data, as long as
         the total sizes match)
        :param is_vector: Whether the data are vectors (vectors have an additional dimension compared to scalars, but we
         can't tell just from the shape of the received data)
        """
        self.recv_buffer = recv_buffer
        self.request = request
        self.shape = shape
        self.is_vector = is_vector

        # Scalar data
        self.to_tuple: Callable[[NDArray], ExchangedVector] = lambda a: a.reshape(self.shape)

        # Vector data
        if self.is_vector:
            if self.recv_buffer.shape[1] == 2:  # 2D
                self.to_tuple = lambda a: (a[0].reshape(self.shape), a[1].reshape(self.shape))
            elif self.recv_buffer.shape[1] == 3:  # 3D
                self.to_tuple = lambda a: (a[0].reshape(self.shape), a[1].reshape(self.shape), a[2].reshape(self.shape))
            else:
                raise ValueError(f"Can only handle vectors with 2 or 3 components, not {self.recv_buffer.shape[1]}")

    def wait(self) -> Tuple[ExchangedVector, ExchangedVector, ExchangedVector, ExchangedVector]:
        """Wait for the exchange started when creating this object to be done.

        :return: The received data as a tuple of 4, in the same shape as the data that were sent
        """
        self.request.Wait()
        return (
            self.to_tuple(self.recv_buffer[SOUTH]),
            self.to_tuple(self.recv_buffer[NORTH]),
            self.to_tuple(self.recv_buffer[WEST]),
            self.to_tuple(self.recv_buffer[EAST]),
        )
