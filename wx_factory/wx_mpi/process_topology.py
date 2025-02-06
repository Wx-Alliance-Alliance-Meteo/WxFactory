import math
from typing import Callable, Optional, Tuple

from mpi4py import MPI
from numpy.typing import NDArray

from common.definitions import *
from device import Device

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
