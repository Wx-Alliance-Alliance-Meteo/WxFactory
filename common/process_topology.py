import math
from   typing import Callable, List, Optional, Tuple, TypeVar

from   mpi4py import MPI
import numpy
from   numpy.typing import NDArray

from .definitions import *
from .device import Device,default_device

ExchangedVector = Tuple[NDArray, ...] | List[NDArray] | NDArray

SOUTH = 0
NORTH = 1
WEST  = 2
EAST  = 3

class ProcessTopology:

   def __init__(self, device: Device, rank: Optional[int]=None):
      """Create a cube-sphere process topology.

      Args:
         :param device (Device): Device on which MPI exchanges are to be made, like a CPU or a GPU.
         :param rank: [Optional] Rank of the tile the topology will manage. This is useful for creating a tile with a rank
                      different from the process rank, for debugging purposes.

      The numbering of the PEs starts at the bottom right. Panel ranks increase towards the east in the x1 direction and increases towards the north in the x2 direction:
      
                 0 1 2 3 4
              +-----------+
            4 |           |
            3 |  x_2      |
            2 |  ^        |
            1 |  |        |
            0 |  + -->x_1 |
              +-----------+
      
      For instance, with n=96 the panel 0 will be endowed with a 4x4 topology like this
      
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

      self.device = device

      self.size = MPI.COMM_WORLD.Get_size()
      self.rank = MPI.COMM_WORLD.Get_rank() if rank is None else rank

      self.nb_pe_per_panel = int(self.size / 6)
      self.nb_lines_per_panel = int(math.sqrt(self.nb_pe_per_panel))

      if self.size < 6 or self.nb_pe_per_panel != self.nb_lines_per_panel**2 or self.nb_pe_per_panel * 6 != self.size:
         allowed_low = self.nb_lines_per_panel**2 * 6
         allowed_high = (self.nb_lines_per_panel + 1)**2 * 6
         raise Exception(f'Wrong number of PEs ({self.size}). '
                         f'Closest allowed processor counts are {allowed_low} and {allowed_high}')

      self.nb_elems_per_line = self.nb_lines_per_panel

      rank_from_location = lambda panel, row, col: panel * self.nb_pe_per_panel + row * self.nb_lines_per_panel + col

      self.my_panel = math.floor(self.rank / self.nb_pe_per_panel)
      self.my_rank_in_panel = self.rank - (self.my_panel * self.nb_pe_per_panel)
      self.my_row = math.floor(self.my_rank_in_panel / self.nb_lines_per_panel)
      self.my_col = int(self.my_rank_in_panel % self.nb_lines_per_panel)

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
      #   S  N  W  E
         [5, 4, 3, 1],  # Panel 0
         [5, 4, 0, 2],  # Panel 1
         [5, 4, 1, 3],  # Panel 2
         [5, 4, 2, 0],  # Panel 3
         [0, 2, 3, 1],  # Panel 4
         [2, 0, 3, 1],  # Panel 5
      ]

      neighbor_panels = all_neighbors[self.my_panel]

      # --- List of PE neighbours for my PE

      self.flip = [False, False, False, False]

      def convert_contra(a1, a2, coord):
         return (a1, a2)

      self.convert_contra = [convert_contra, convert_contra, convert_contra, convert_contra]

      # North
      my_north = rank_from_location(self.my_panel, (self.my_row + 1), self.my_col)
      if self.my_row == self.nb_lines_per_panel - 1:
         if self.my_panel == 0:
            my_north = rank_from_location(neighbor_panels[NORTH], 0, self.my_col)
            self.convert_contra[NORTH] = lambda a1, a2, coord: (a1 - 2.0 * coord / (1.0 + coord**2) * a2, a2)
         elif self.my_panel == 1:
            my_north = rank_from_location(neighbor_panels[NORTH], self.my_col, self.nb_lines_per_panel - 1)
            self.convert_contra[NORTH] = lambda a1, a2, coord: (-a2, a1 - 2.0 * coord / (1.0 + coord**2) * a2)
         elif self.my_panel == 2:
            my_north = rank_from_location(neighbor_panels[NORTH], self.nb_lines_per_panel - 1, self.nb_lines_per_panel - 1 - self.my_col)
            self.convert_contra[NORTH] = lambda a1, a2, coord: (-a1 + 2.0 * coord / (1.0 + coord**2) * a2, -a2)
            self.flip[NORTH] = True
         elif self.my_panel == 3:
            my_north = rank_from_location(neighbor_panels[NORTH], self.nb_lines_per_panel - 1 - self.my_col, 0)
            self.convert_contra[NORTH] = lambda a1, a2, coord: (a2 , -a1 + 2.0 * coord / (1.0 + coord**2) * a2)
            self.flip[NORTH] = True
         elif self.my_panel == 4:
            my_north = rank_from_location(neighbor_panels[NORTH], self.nb_lines_per_panel - 1, self.nb_lines_per_panel - 1 - self.my_col)
            self.convert_contra[NORTH] = lambda a1, a2, coord: (-a1 + 2.0 * coord / (1.0 + coord**2) * a2, -a2)
            self.flip[NORTH] = True
         elif self.my_panel == 5:
            my_north = rank_from_location(neighbor_panels[NORTH], 0, self.my_col)
            self.convert_contra[NORTH] = lambda a1, a2, coord: (a1 - 2.0 * coord / (1.0 + coord**2) * a2, a2)

      # South
      my_south = rank_from_location(self.my_panel, (self.my_row - 1), self.my_col)
      if self.my_row == 0:
         if self.my_panel == 0:
            my_south = rank_from_location(neighbor_panels[SOUTH], self.nb_elems_per_line-1, self.my_col)
            self.convert_contra[SOUTH] = lambda a1, a2, coord: (a1 + 2.0 * coord / (1.0 + coord**2) * a2, a2)
         elif self.my_panel == 1:
            my_south = rank_from_location(neighbor_panels[SOUTH], self.nb_elems_per_line - 1 - self.my_col, self.nb_lines_per_panel-1)
            self.convert_contra[SOUTH] = lambda a1, a2, coord: (a2, -a1 - 2.0 * coord / (1.0 + coord**2) * a2)
            self.flip[SOUTH] = True
         elif self.my_panel == 2:
            my_south = rank_from_location(neighbor_panels[SOUTH], 0, self.nb_lines_per_panel - 1 - self.my_col)
            self.convert_contra[SOUTH] = lambda a1, a2, coord: (-a1 - 2.0 * coord / (1.0 + coord**2) * a2, -a2)
            self.flip[SOUTH] = True
         elif self.my_panel == 3:
            my_south = rank_from_location(neighbor_panels[SOUTH], self.my_col, 0)
            self.convert_contra[SOUTH] = lambda a1, a2, coord: (-a2, a1 + 2.0 * coord / (1.0 + coord**2) * a2)
         elif self.my_panel == 4:
            my_south = rank_from_location(neighbor_panels[SOUTH], self.nb_elems_per_line-1, self.my_col)
            self.convert_contra[SOUTH] = lambda a1, a2, coord: (a1 + 2.0 * coord / (1.0 + coord**2) * a2, a2)
         elif self.my_panel == 5:
            my_south = rank_from_location(neighbor_panels[SOUTH], 0, self.nb_elems_per_line - 1 - self.my_col)
            self.convert_contra[SOUTH] = lambda a1, a2, coord: (-a1 - 2.0 * coord / (1.0 + coord**2) * a2, -a2)
            self.flip[SOUTH] = True

      # West
      if self.my_col == 0:
         if self.my_panel == 4:
            my_west = rank_from_location(neighbor_panels[WEST], self.nb_lines_per_panel-1, self.nb_lines_per_panel - 1 - self.my_row)
            self.flip[WEST] = True
            self.convert_contra[WEST] = lambda a1, a2, coord: (-2. * coord / ( 1. + coord**2 ) * a1 - a2, a1)
         elif self.my_panel == 5:
            my_west = rank_from_location(neighbor_panels[WEST], 0, self.my_row)
            self.convert_contra[WEST] = lambda a1, a2, coord: (2. * coord / ( 1. + coord**2 ) * a1 + a2, -a1)
         else:
            my_west = rank_from_location(neighbor_panels[WEST], self.my_row, self.nb_lines_per_panel-1)
            self.convert_contra[WEST] = lambda a1, a2, coord: (a1, 2. * coord / ( 1. + coord**2 ) * a1 + a2)
      else:
         my_west = rank_from_location(self.my_panel, self.my_row, (self.my_col-1))

      # East
      if self.my_col == self.nb_elems_per_line-1:
         if self.my_panel == 4:
            my_east = rank_from_location(neighbor_panels[EAST], self.nb_lines_per_panel-1, self.my_row)
            self.convert_contra[EAST] = lambda a1, a2, coord: (-2. * coord / ( 1. + coord**2) * a1 + a2, -a1)
         elif self.my_panel == 5:
            my_east = rank_from_location(neighbor_panels[EAST], 0, self.nb_lines_per_panel - 1 - self.my_row)
            self.flip[EAST] = True
            self.convert_contra[EAST] = lambda a1, a2, coord: (2. * coord / ( 1. + coord**2 ) * a1 - a2, a1)
         else:
            my_east = rank_from_location(neighbor_panels[EAST], self.my_row, 0)
            self.convert_contra[EAST] = lambda a1, a2, coord: (a1, -2. * coord / (1. + coord**2 ) * a1 + a2)
      else:
         my_east = rank_from_location(self.my_panel, self.my_row, self.my_col+1)

      # Distributed Graph
      self.sources = [my_south, my_north, my_west, my_east] # Must correspond to values of SOUTH, NORTH, WEST, EAST
      self.destinations = self.sources

      self.comm_dist_graph = MPI.COMM_WORLD.Create_dist_graph_adjacent(self.sources, self.destinations)

      self.get_rows_3d = lambda array, index1, index2: array[:, index1, index2, :] if array is not None else None
      self.get_rows_2d = lambda array, index1, index2: array[index1, index2, :] if array is not None else None


   def send_recv_neighbors(self,
                           south_send: NDArray,
                           north_send: NDArray,
                           west_send: NDArray,
                           east_send: NDArray,
                           flip_dim: int,
                           sync: Optional[bool]=True):

      xp = self.device.xp
      send_buffer = xp.empty(((4,) + north_send.shape), dtype=north_send.dtype)
      for do_flip, data, buffer in zip(self.flip,
                                       [south_send, north_send, west_send, east_send],
                                       [send_buffer[0], send_buffer[1], send_buffer[2], send_buffer[3]]):
         buffer[:] = xp.flip(data, flip_dim) if do_flip else data

      receive_buffer = xp.empty_like(send_buffer)
      self.device.synchronize()
      request = self.comm_dist_graph.Ineighbor_alltoall(send_buffer, receive_buffer)

      if sync:
         request.Wait()
      return request, receive_buffer[0], receive_buffer[1], receive_buffer[2], receive_buffer[3]


   def xchange_scalars(self, geom, field_itf_i, field_itf_j, blocking=True):

      is_3d = field_itf_i.ndim >= 4
      if is_3d:
         get_rows = self.get_rows_3d
         flip_dim = 1
      else:
         get_rows = self.get_rows_2d
         flip_dim = 0

      n_send = get_rows(field_itf_j, -2, 1)
      s_send = get_rows(field_itf_j, 1, 0)
      w_send = get_rows(field_itf_i, 1, 0)
      e_send = get_rows(field_itf_i, -2, 1)

      n_recv = get_rows(field_itf_j, -1, 0)
      s_recv = get_rows(field_itf_j, 0, 1)
      w_recv = get_rows(field_itf_i, 0, 1)
      e_recv = get_rows(field_itf_i, -1, 0)

      request, n_recv_buf, s_recv_buf, w_recv_buf, e_recv_buf = self.send_recv_neighbors(s_send, n_send, w_send, e_send, flip_dim, sync=False)
      request = ScalarNonBlockingExchangeRequest((n_recv_buf, s_recv_buf, w_recv_buf, e_recv_buf), (n_recv, s_recv, w_recv, e_recv), request)

      if blocking:
         request.wait()

      return request


   def xchange_simple_vectors(self, X, Y, u1_n, u2_n, u1_s, u2_s, u1_w, u2_w, u1_e, u2_e, u3_n=None, u3_s=None,
                              u3_w=None, u3_e=None, sync=True):
      ndim = 2
      if u3_n is not None: ndim = 3

      flip_dim = ndim - 1
      sendbuf = numpy.empty((4, ndim) + u1_n.shape, dtype=u1_n.dtype, like=u1_n)

      sendbuf[0, 0, :], sendbuf[0, 1, :] = self.convert_contra[NORTH](u1_n, u2_n, X)
      sendbuf[1, 0, :], sendbuf[1, 1, :] = self.convert_contra[SOUTH](u1_s, u2_s, X)
      sendbuf[2, 0, :], sendbuf[2, 1, :] = self.convert_contra[WEST](u1_w, u2_w, Y)
      sendbuf[3, 0, :], sendbuf[3, 1, :] = self.convert_contra[EAST](u1_e, u2_e, Y)

      if u3_n is not None:
         sendbuf[0, 2, :] = u3_n
         sendbuf[1, 2, :] = u3_s
         sendbuf[2, 2, :] = u3_w
         sendbuf[3, 2, :] = u3_e

      return self.send_recv_neighbors(sendbuf[1], sendbuf[0], sendbuf[2], sendbuf[3], flip_dim, sync=sync)


   def xchange_halo(self, f, halo_size=1):
      h = halo_size
      f_ext = numpy.zeros((f.shape[0] + halo_size*2, f.shape[1] + halo_size*2))

      f_ext[h:-h, h:-h] = f[:, :]

      request, f_ext[-1, h:-h], f_ext[0, h:-h], f_ext[h:-h, 0], f_ext[h:-h, -1] = self.send_recv_neighbors(
         f[0, :], f[-1, :], f[:, 0], f[:, -1], 0)

      return request, f_ext


   def xchange_halo_vector(self, geom, f_x1, f_x2, halo_size=1):

      h = halo_size

      f_x1_ext = numpy.zeros((f_x1.shape[0] + halo_size*2, f_x1.shape[1] + halo_size*2))
      f_x2_ext = numpy.zeros_like(f_x1_ext)

      f_x1_ext[h:-h, h:-h] = f_x1[:, :]
      f_x2_ext[h:-h, h:-h] = f_x2[:, :]

      X = geom.X_block[0, :]
      Y = geom.Y_block[:, 0]

      request, f_n, f_s, f_w, f_e = self.xchange_simple_vectors(
         X, Y, f_x1[-1, :], f_x2[-1, :], f_x1[0, :], f_x2[0, :], f_x1[:, 0], f_x2[:, 0], f_x1[:, -1], f_x2[:, -1])

      f_x1_ext[-1, h:-h] = f_n[0]
      f_x2_ext[-1, h:-h] = f_n[1]
      f_x1_ext[0, h:-h]  = f_s[0]
      f_x2_ext[0, h:-h]  = f_s[1]

      f_x1_ext[h:-h, 0]  = f_w[0]
      f_x2_ext[h:-h, 0]  = f_w[1]
      f_x1_ext[h:-h, -1] = f_e[0]
      f_x2_ext[h:-h, -1] = f_e[1]

      return request, f_x1_ext, f_x2_ext


   def xchange_Euler_interfaces(self,
                                geom: 'CubedSphere',
                                variables_itf_i: NDArray,
                                variables_itf_j: NDArray,
                                blocking: Optional[bool]=True):

      xp = self.device.xp

      X = self.device.array(geom.X_block[0, :]) # TODO these should not be necessary (should already have device arrays in geometry)
      Y = self.device.array(geom.Y_block[:, 0])
      flip_dim = 1
      id_first_tracer = 5

      init_shape = variables_itf_i.shape
      dtype = variables_itf_i.dtype
      send_buffer = xp.empty((4, init_shape[0], init_shape[1], init_shape[4]), dtype=dtype)
      recv_buffer = xp.empty_like(send_buffer)

      var_n = variables_itf_j[:, :, -2, 1, :]
      var_s = variables_itf_j[:, :, 1, 0, :]
      var_w = variables_itf_i[:, :, 1, 0, :]
      var_e = variables_itf_i[:, :, -2, 1, :]

      # Fill the send buffer, flipping when necessary and converting vector values
      for do_flip, convert, positions, var, buffer in zip(
         self.flip,
         self.convert_contra,
         [X, X, Y, Y],
         [var_s, var_n, var_w, var_e],
         [send_buffer[0], send_buffer[1], send_buffer[2], send_buffer[3]]):

         for id in [idx_rho, idx_rho_w, idx_rho_theta]:
            buffer[id, :] = xp.flip(var[id], flip_dim) if do_flip else var[id]

         tmp1, tmp2 = convert(var[idx_rho_u1], var[idx_rho_u2], positions)
         buffer[idx_rho_u1] = xp.flip(tmp1, flip_dim) if do_flip else tmp1
         buffer[idx_rho_u2] = xp.flip(tmp2, flip_dim) if do_flip else tmp2

         buffer[id_first_tracer:] = xp.flip(var[id_first_tracer:], flip_dim + 1) if do_flip else var[id_first_tracer:]

      # Initiate MPI transfer
      self.device.synchronize()
      mpi_request = self.comm_dist_graph.Ineighbor_alltoall(send_buffer, recv_buffer)

      # Setup request to that data ends up in the right arrays when the wait() function is called
      var_n_dest = variables_itf_j[:, :, -1, 0, :]
      var_s_dest = variables_itf_j[:, :, 0, 1, :]
      var_w_dest = variables_itf_i[:, :, 0, 1, :]
      var_e_dest = variables_itf_i[:, :, -1, 0, :]

      request = EulerExchangeRequest(recv_buffer, (var_s_dest, var_n_dest, var_w_dest, var_e_dest), mpi_request)

      if blocking:
         request.wait()

      return request

   
   def start_exchange_scalars(self,
                              south: ExchangedVector,
                              north: ExchangedVector,
                              west: ExchangedVector,
                              east: ExchangedVector):
      """Create a request for exchanging scalar data with neighboring tiles. The 4 input arrays or lists must have the same shape and size.
      Values will be flipped if the neighbor has a flipped coordinate system. This means that when receiving data,
      it will always be in local coordinates
      
      Args:
         :param south: Array (or tuple/list of arrays) of values for the south boundary (one array-like for each component, so there should be 2 or 3 arrays in the tuple)
         :param north: Array (or tuple/list of arrays) of values for the north boundary (one array-like for each component, so there should be 2 or 3 arrays in the tuple)
         :param west:  Array (or tuple/list of arrays) of values for the west boundary (one array-like for each component, so there should be 2 or 3 arrays in the tuple)
         :param east:  Array (or tuple/list of arrays) of values for the east boundary (one array-like for each component, so there should be 2 or 3 arrays in the tuple)

      Returns:
         A request (MPI-like) for the transfer of the scalar values. Waiting on the request will return the resulting arrays in the same way as the input.
      """
      xp = self.device.xp

      if isinstance(south, list) or isinstance(south, tuple):
         base_shape = (4, len(south)) + south[0].shape
         flip_dim = south[0].ndim - 2
      else:
         base_shape = (4,) + south.shape
         flip_dim = south.ndim - 2

      send_buffer = xp.empty(base_shape, dtype=south[0].dtype)
      # print(f'send buffer shape = {send_buffer.shape}')
      input = [south, north, west, east]
      for i in range(4):
         send_buffer[i] = xp.flip(input[i], flip_dim) if self.flip[i] else input[i]

      recv_buffer = xp.empty_like(send_buffer)
      self.device.synchronize() # When using GPU

      # Initiate MPI transfer
      mpi_request = self.comm_dist_graph.Ineighbor_alltoall(send_buffer, recv_buffer)

      return ExchangeRequest(recv_buffer, mpi_request, is_vector=False)

   def start_exchange_vectors(self, south, north, west, east, boundary_sn, boundary_we):
      """Create a request for exchanging vectors with neighboring tiles. The 4 input vectors must have the same shape and size.
      Vector Values are converted to the neighboring coordinate system before being transmitted through MPI. They will be flipped
      if the neighbor has a flipped coordinate system. This means that when receiving data, it will always be in local coordinates
      
      Args:
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
         :param boundary_we: Coordinates along the south-north axis at the west and east boundaries. The entries in that vector *must* match the entries in the input arrays

      Returns:
         A request (MPI-like) for the transfer of the arrays. Waiting on the request will return the resulting arrays in the same way as the input.
      """
      xp = self.device.xp

      flip_dim = south[0].ndim - 2

      base_shape = (4, len(south)) + south[0].shape
      send_buffer = xp.empty(base_shape, dtype=south[0].dtype)

      input = [south, north, west, east]
      boundaries = [boundary_sn, boundary_sn, boundary_we, boundary_we]
      for i in range(4):
         send_buffer[i, 0], send_buffer[i, 1] = self.convert_contra[i](input[i][0], input[i][1], boundaries[i])
         if len(input[i]) == 3: send_buffer[i, 2] = input[i][2]               # 3rd dimension if present
         if self.flip[i]: send_buffer[i] = xp.flip(send_buffer[i], flip_dim)  # Flip arrays, if needed

      recv_buffer = xp.empty_like(send_buffer)
      self.device.synchronize() # When using GPU

      # Initiate MPI transfer
      mpi_request = self.comm_dist_graph.Ineighbor_alltoall(send_buffer, recv_buffer)

      return ExchangeRequest(recv_buffer, mpi_request, is_vector=True)

   def xchange_vectors(self, geom, u1_itf_i, u2_itf_i, u1_itf_j, u2_itf_j, u3_itf_i=None, u3_itf_j=None, blocking=True):

      # --- 2D/3D setup

      is_3d = u1_itf_i.ndim >= 4
      if is_3d:
         if u3_itf_i is None:
            print(f'Calling xchange_vectors with arrays that look like they are from a 3D problem, '
                  f'but you don\'t provide the 3rd component!')
            raise ValueError

         get_rows = self.get_rows_3d
         X = geom.X_block[0, :] # TODO : debug avec niveau 0
         Y = geom.Y_block[:, 0]

      else:
         if u3_itf_i is not None:
            print(f'Calling xchange_vectors with arrays that look like they are from a 2D problem, '
                  f'but you also provide a 3rd component! We will just ignore it.')

         get_rows = self.get_rows_2d
         X = geom.X_block[0, :]
         Y = geom.Y_block[:, 0]

      # --- Get the right vectors

      u1_n, u2_n, u3_n = (get_rows(u1_itf_j, -2, 1), get_rows(u2_itf_j, -2, 1), get_rows(u3_itf_j, -2, 1))
      u1_s, u2_s, u3_s = (get_rows(u1_itf_j, 1, 0),  get_rows(u2_itf_j, 1, 0),  get_rows(u3_itf_j, 1, 0))
      u1_w, u2_w, u3_w = (get_rows(u1_itf_i, 1, 0),  get_rows(u2_itf_i, 1, 0),  get_rows(u3_itf_i, 1, 0))
      u1_e, u2_e, u3_e = (get_rows(u1_itf_i, -2, 1), get_rows(u2_itf_i, -2, 1), get_rows(u3_itf_i, -2, 1))

      # --- Convert vector coords and transmit them

      request, n_recv_buf, s_recv_buf, w_recv_buf, e_recv_buf = self.xchange_simple_vectors(
         X, Y, u1_n, u2_n, u1_s, u2_s, u1_w, u2_w, u1_e, u2_e, u3_n, u3_s, u3_w, u3_e, sync=False)

      u1_n_recv, u2_n_recv, u3_n_recv = (get_rows(u1_itf_j, -1, 0), get_rows(u2_itf_j, -1, 0), get_rows(u3_itf_j, -1, 0))
      u1_s_recv, u2_s_recv, u3_s_recv = (get_rows(u1_itf_j, 0, 1),  get_rows(u2_itf_j, 0, 1),  get_rows(u3_itf_j, 0, 1))
      u1_w_recv, u2_w_recv, u3_w_recv = (get_rows(u1_itf_i, 0, 1),  get_rows(u2_itf_i, 0, 1),  get_rows(u3_itf_i, 0, 1))
      u1_e_recv, u2_e_recv, u3_e_recv = (get_rows(u1_itf_i, -1, 0), get_rows(u2_itf_i, -1, 0), get_rows(u3_itf_i, -1, 0))

      request = VectorNonBlockingExchangeRequest( \
         (n_recv_buf, s_recv_buf, w_recv_buf, e_recv_buf), \
         ((u1_n_recv, u2_n_recv, u3_n_recv), (u1_s_recv, u2_s_recv, u3_s_recv), (u1_w_recv, u2_w_recv, u3_w_recv), (u1_e_recv, u2_e_recv, u3_e_recv)), \
         request, is_3d)

      if blocking:
         request.wait()

      return request


class ScalarNonBlockingExchangeRequest():
   def __init__(self, recv_buffers, outputs, request) -> None:
      self.recv_buffers = recv_buffers
      self.outputs      = outputs
      self.request      = request

      self.is_complete = False

   def wait(self):
      if not self.is_complete:
         self.request.Wait()
         self.outputs[0][:] = self.recv_buffers[0]
         self.outputs[1][:] = self.recv_buffers[1]
         self.outputs[2][:] = self.recv_buffers[2]
         self.outputs[3][:] = self.recv_buffers[3]
         self.is_complete = True


class ExchangeRequest:
   def __init__(self, recv_buffer, request, is_vector=False):
      self.recv_buffer = recv_buffer
      self.request = request
      self.is_vector = is_vector

      self.to_tuple: Callable[[ExchangedVector], ExchangedVector] = lambda a: a    # Do nothing by default
      if self.is_vector:
         if self.recv_buffer.shape[1] == 2:
            self.to_tuple = lambda a: (a[0], a[1])
         elif self.recv_buffer.shape[1] == 3:
            self.to_tuple = lambda a: (a[0], a[1], a[2])
         else:
            raise ValueError(f'Can only handle vectors with 2 or 3 components, not {self.recv_buffer.shape[1]}')

   def wait(self) -> Tuple[ExchangedVector, ExchangedVector, ExchangedVector, ExchangedVector]:
      """Wait for the exchange started when creating this object to be done.
      
      Returns:
         The received vectors, in the same shape as the vectors that were sent
      """
      self.request.Wait()
      # if MPI.COMM_WORLD.rank == 1:
      #    print(f'recv buffer = \n{self.recv_buffer}')
      return (self.to_tuple(self.recv_buffer[SOUTH]),
              self.to_tuple(self.recv_buffer[NORTH]),
              self.to_tuple(self.recv_buffer[WEST]),
              self.to_tuple(self.recv_buffer[EAST]))


class VectorNonBlockingExchangeRequest():
   def __init__(self, recv_buffers, outputs, request, is_3d) -> None:
      self.recv_buffers = recv_buffers
      self.outputs      = outputs
      self.request      = request
      self.is_3d        = is_3d

      self.is_complete = False

   def wait(self):
      if not self.is_complete:
         self.request.Wait()
         self.outputs[0][0][:] = self.recv_buffers[0][0]
         self.outputs[1][0][:] = self.recv_buffers[1][0]
         self.outputs[2][0][:] = self.recv_buffers[2][0]
         self.outputs[3][0][:] = self.recv_buffers[3][0]

         self.outputs[0][1][:] = self.recv_buffers[0][1]
         self.outputs[1][1][:] = self.recv_buffers[1][1]
         self.outputs[2][1][:] = self.recv_buffers[2][1]
         self.outputs[3][1][:] = self.recv_buffers[3][1]

         if self.is_3d:
            self.outputs[0][2][:] = self.recv_buffers[0][2]
            self.outputs[1][2][:] = self.recv_buffers[1][2]
            self.outputs[2][2][:] = self.recv_buffers[2][2]
            self.outputs[3][2][:] = self.recv_buffers[3][2]

         self.is_complete = True


class EulerExchangeRequest():
   def __init__(self, recv_buffers, outputs, mpi_request) -> None:
      self.recv_buffers = recv_buffers
      self.outputs = outputs
      self.mpi_request = mpi_request
      self.is_complete = False

   def wait(self):
      if not self.is_complete:
         self.mpi_request.Wait()
         for i in range(self.recv_buffers.shape[0]):
            for j in range(self.recv_buffers.shape[1]):
               self.outputs[i][j][:] = self.recv_buffers[i, j]
         self.is_complete = True
