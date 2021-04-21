import mpi4py.MPI
import numpy
import math

class Distributed_World:
   def __init__(self):

      # The numbering of the PEs starts at the bottom right. Pannel ranks increase towards the east in the x1 direction and increases towards the north in the x2 direction:
      #
      #            0 1 2 3 4
      #         +-----------+
      #       4 |           |
      #       3 |  x_2      |
      #       2 |  ^        |
      #       1 |  |        |
      #       0 |  + -->x_1 |
      #         +-----------+
      #
      # For instance, with n=96 the panel 0 will be endowed with a 4x4 topology like this
      #
      #      +---+---+---+---+
      #      | 12| 13| 14| 15|
      #      |---+---+---+---|
      #      | 8 | 9 | 10| 11|
      #      |---+---+---+---|
      #      | 4 | 5 | 6 | 7 |
      #      |---+---+---+---|
      #      | 0 | 1 | 2 | 3 |
      #      +---+---+---+---+

      self.size = mpi4py.MPI.COMM_WORLD.Get_size()
      self.rank = mpi4py.MPI.COMM_WORLD.Get_rank()

      self.nb_pe_per_panel = int(self.size / 6)
      self.nb_lines_per_panel = math.isqrt(self.nb_pe_per_panel)

      if self.size < 6 or self.nb_pe_per_panel != self.nb_lines_per_panel**2 or self.nb_pe_per_panel * 6 != self.size:
         raise Exception('Wrong number of PEs. This topology is not allowed')

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

      my_north_panel = -1
      my_south_panel = -1
      my_west_panel = -1
      my_east_panel = -1

      if self.my_panel == 0:
         my_north_panel = 4
         my_south_panel = 5
         my_west_panel = 3
         my_east_panel = 1
      elif self.my_panel == 1:
         my_north_panel = 4
         my_south_panel = 5
         my_west_panel = 0
         my_east_panel = 2
      elif self.my_panel == 2:
         my_north_panel = 4
         my_south_panel = 5
         my_west_panel = 1
         my_east_panel = 3
      elif self.my_panel == 3:
         my_north_panel = 4
         my_south_panel = 5
         my_west_panel = 2
         my_east_panel = 0
      elif self.my_panel == 4:
         my_north_panel = 2
         my_south_panel = 0
         my_west_panel = 3
         my_east_panel = 1
      elif self.my_panel == 5:
         my_north_panel = 0
         my_south_panel = 2
         my_west_panel = 3
         my_east_panel = 1

      # --- List of PE neighbours for my PE

      self.flip_north = False
      self.flip_south = False
      self.flip_west  = False
      self.flip_east  = False

      self.convert_contra_north = lambda a1, a2, X: (a1, a2)
      self.convert_contra_south = lambda a1, a2, X: (a1, a2)
      self.convert_contra_west  = lambda a1, a2, Y: (a1, a2)
      self.convert_contra_east  = lambda a1, a2, Y: (a1, a2)

      # North
      my_north = rank_from_location(self.my_panel, (self.my_row + 1), self.my_col)
      if self.my_row == self.nb_lines_per_panel - 1:
         if self.my_panel == 0:
            my_north = rank_from_location(my_north_panel, 0, self.my_col)
            self.convert_contra_north = lambda a1, a2, X: (a1 - 2.0 * X / (1.0 + X**2) * a2, a2)
         elif self.my_panel == 1:
            my_north = rank_from_location(my_north_panel, self.my_col, self.nb_lines_per_panel - 1)
            self.convert_contra_north = lambda a1, a2, X: (-a2, a1 - 2.0 * X / (1.0 + X**2) * a2)
         elif self.my_panel == 2:
            my_north = rank_from_location(my_north_panel, self.nb_lines_per_panel - 1, self.nb_lines_per_panel - 1 - self.my_col)
            self.convert_contra_north = lambda a1, a2, X: (-a1 + 2.0 * X / (1.0 + X**2) * a2, -a2)
            self.flip_north = True
         elif self.my_panel == 3:
            my_north = rank_from_location(my_north_panel, self.nb_lines_per_panel - 1 - self.my_col, 0)
            self.convert_contra_north = lambda a1, a2, X: (a2 , -a1 + 2.0 * X / (1.0 + X**2) * a2)
            self.flip_north = True
         elif self.my_panel == 4:
            my_north = rank_from_location(my_north_panel, self.nb_lines_per_panel - 1, self.nb_lines_per_panel - 1 - self.my_col)
            self.convert_contra_north = lambda a1, a2, X: (-a1 + 2.0 * X / (1.0 + X**2) * a2, -a2)
            self.flip_north = True
         elif self.my_panel == 5:
            my_north = rank_from_location(my_north_panel, 0, self.my_col)
            self.convert_contra_north = lambda a1, a2, X: (a1 - 2.0 * X / (1.0 + X**2) * a2, a2)

      # South
      my_south = rank_from_location(self.my_panel, (self.my_row - 1), self.my_col)
      if self.my_row == 0:
         if self.my_panel == 0:
            my_south = rank_from_location(my_south_panel, self.nb_elems_per_line-1, self.my_col)
            self.convert_contra_south = lambda a1, a2, X: (a1 + 2.0 * X / (1.0 + X**2) * a2, a2)
         elif self.my_panel == 1:
            my_south = rank_from_location(my_south_panel, self.nb_elems_per_line - 1 - self.my_col, self.nb_lines_per_panel-1)
            self.convert_contra_south = lambda a1, a2, X: (a2, -a1 - 2.0 * X / (1.0 + X**2) * a2)
            self.flip_south = True
         elif self.my_panel == 2:
            my_south = rank_from_location(my_south_panel, 0, self.nb_lines_per_panel - 1 - self.my_col)
            self.convert_contra_south = lambda a1, a2, X: (-a1 - 2.0 * X / (1.0 + X**2) * a2, -a2)
            self.flip_south = True
         elif self.my_panel == 3:
            my_south = rank_from_location(my_south_panel, self.my_col, 0)
            self.convert_contra_south = lambda a1, a2, X: (-a2, a1 + 2.0 * X / (1.0 + X**2) * a2)
         elif self.my_panel == 4:
            my_south = rank_from_location(my_south_panel, self.nb_elems_per_line-1, self.my_col)
            self.convert_contra_south = lambda a1, a2, X: (a1 + 2.0 * X / (1.0 + X**2) * a2, a2)
         elif self.my_panel == 5:
            my_south = rank_from_location(my_south_panel, 0, self.nb_elems_per_line - 1 - self.my_col)
            self.convert_contra_south = lambda a1, a2, X: (-a1 - 2.0 * X / (1.0 + X**2) * a2, -a2)
            self.flip_south = True

      # West
      if self.my_col == 0:
         if self.my_panel == 4:
            my_west = rank_from_location(my_west_panel, self.nb_lines_per_panel-1, self.nb_lines_per_panel - 1 - self.my_row)
            self.flip_west = True
            self.convert_contra_west = lambda a1, a2, Y: (-2. * Y / ( 1. + Y**2 ) * a1 - a2, a1)
         elif self.my_panel == 5:
            my_west = rank_from_location(my_west_panel, 0, self.my_row)
            self.convert_contra_west = lambda a1, a2, Y: (2. * Y / ( 1. + Y**2 ) * a1 + a2, -a1)
         else:
            my_west = rank_from_location(my_west_panel, self.my_row, self.nb_lines_per_panel-1)
            self.convert_contra_west = lambda a1, a2, Y: (a1, 2. * Y / ( 1. + Y**2 ) * a1 + a2)
      else:
         my_west = rank_from_location(self.my_panel, self.my_row, (self.my_col-1))

      # East
      if self.my_col == self.nb_elems_per_line-1:
         if self.my_panel == 4:
            my_east = rank_from_location(my_east_panel, self.nb_lines_per_panel-1, self.my_row)
            self.convert_contra_east = lambda a1, a2, Y: (-2. * Y / ( 1. + Y**2) * a1 + a2, -a1)
         elif self.my_panel == 5:
            my_east = rank_from_location(my_east_panel, 0, self.nb_lines_per_panel - 1 - self.my_row)
            self.flip_east = True
            self.convert_contra_east = lambda a1, a2, Y: (2. * Y / ( 1. + Y**2 ) * a1 - a2, a1)
         else:
            my_east = rank_from_location(my_east_panel, self.my_row, 0)
            self.convert_contra_east = lambda a1, a2, Y: (a1, -2. * Y / (1. + Y**2 ) * a1 + a2)
      else:
         my_east = rank_from_location(self.my_panel, self.my_row, self.my_col+1)

      # Distributed Graph
      self.sources = [my_north, my_south, my_west, my_east]
      self.destinations = self.sources

      self.comm_dist_graph = mpi4py.MPI.COMM_WORLD.Create_dist_graph_adjacent(self.sources, self.destinations)

      self.get_rows_3d = lambda array, index1, index2: array[:, index1, index2, :] if array is not None else None
      self.get_rows_2d = lambda array, index1, index2: array[index1, index2, :] if array is not None else None


   def send_recv_neighbors(self, north_send, south_send, west_send, east_send, flip_dim):

      send_buffer = numpy.empty(((4,) + north_send.shape), dtype=north_send.dtype)
      for do_flip, data, buffer in zip([self.flip_north, self.flip_south, self.flip_west, self.flip_east],
                                       [north_send, south_send, west_send, east_send],
                                       [send_buffer[0], send_buffer[1], send_buffer[2], send_buffer[3]]):
         buffer[:] = numpy.flip(data, flip_dim) if do_flip else data

      receive_buffer = self.comm_dist_graph.neighbor_alltoall(send_buffer)
      return receive_buffer[0], receive_buffer[1], receive_buffer[2], receive_buffer[3]


   def xchange_scalars(self, geom, field_itf_i, field_itf_j):

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

      n_recv[:], s_recv[:], w_recv[:], e_recv[:] = self.send_recv_neighbors(n_send, s_send, w_send, e_send, flip_dim)


   def xchange_simple_vectors(self, X, Y, u1_n, u2_n, u1_s, u2_s, u1_w, u2_w, u1_e, u2_e, u3_n=None, u3_s=None,
                              u3_w=None, u3_e=None):
      ndim = 2
      if u3_n is not None: ndim = 3

      flip_dim = ndim - 1
      sendbuf = numpy.empty((4, ndim) + u1_n.shape, dtype=u1_n.dtype)

      sendbuf[0, 0, :], sendbuf[0, 1, :] = self.convert_contra_north(u1_n, u2_n, X)
      sendbuf[1, 0, :], sendbuf[1, 1, :] = self.convert_contra_south(u1_s, u2_s, X)
      sendbuf[2, 0, :], sendbuf[2, 1, :] = self.convert_contra_west(u1_w, u2_w, Y)
      sendbuf[3, 0, :], sendbuf[3, 1, :] = self.convert_contra_east(u1_e, u2_e, Y)

      if u3_n is not None:
         sendbuf[0, 2, :] = u3_n
         sendbuf[1, 2, :] = u3_s
         sendbuf[2, 2, :] = u3_w
         sendbuf[3, 2, :] = u3_e

      return self.send_recv_neighbors(sendbuf[0], sendbuf[1], sendbuf[2], sendbuf[3], flip_dim)


   def xchange_vectors(self, geom, u1_itf_i, u2_itf_i, u1_itf_j, u2_itf_j, u3_itf_i=None, u3_itf_j=None):

      # --- 2D/3D setup

      is_3d = u1_itf_i.ndim >= 4
      if is_3d:
         if u3_itf_i is None:
            print(f'Calling xchange_vectors with arrays that look like they are from a 3D problem, '
                  f'but you don\'t provide the 3rd component!')
            raise ValueError

         get_rows = self.get_rows_3d
         X = geom.X[0, 0, :] # TODO : debug avec niveau 0
         Y = geom.Y[0, :, 0]

      else:
         if u3_itf_i is not None:
            print(f'Calling xchange_vectors with arrays that look like they are from a 3D problem, '
                  f'but you also provide a 3rd component! We will just ignore it.')

         get_rows = self.get_rows_2d
         X = geom.X[0, :]
         Y = geom.Y[:, 0]

      # --- Get the right vectors

      u1_n, u2_n, u3_n = (get_rows(u1_itf_j, -2, 1), get_rows(u2_itf_j, -2, 1), get_rows(u3_itf_j, -2, 1))
      u1_s, u2_s, u3_s = (get_rows(u1_itf_j, 1, 0),  get_rows(u2_itf_j, 1, 0),  get_rows(u3_itf_j, 1, 0))
      u1_w, u2_w, u3_w = (get_rows(u1_itf_i, 1, 0),  get_rows(u2_itf_i, 1, 0),  get_rows(u3_itf_i, 1, 0))
      u1_e, u2_e, u3_e = (get_rows(u1_itf_i, -2, 1), get_rows(u2_itf_i, -2, 1), get_rows(u3_itf_i, -2, 1))

      # --- Convert vector coords and transmit them

      n_recv, s_recv, w_recv, e_recv = self.xchange_simple_vectors(
         X, Y, u1_n, u2_n, u1_s, u2_s, u1_w, u2_w, u1_e, u2_e, u3_n, u3_s, u3_w, u3_e)

      # --- Unpack received data

      get_rows(u1_itf_j, -1, 0)[:] = n_recv[0]
      get_rows(u1_itf_j, 0, 1)[:]  = s_recv[0]
      get_rows(u1_itf_i, 0, 1)[:]  = w_recv[0]
      get_rows(u1_itf_i, -1, 0)[:] = e_recv[0]

      get_rows(u2_itf_j, -1, 0)[:] = n_recv[1]
      get_rows(u2_itf_j, 0, 1)[:]  = s_recv[1]
      get_rows(u2_itf_i, 0, 1)[:]  = w_recv[1]
      get_rows(u2_itf_i, -1, 0)[:] = e_recv[1]

      if is_3d:
         get_rows(u3_itf_j, -1, 0)[:] = n_recv[2]
         get_rows(u3_itf_j, 0, 1)[:]  = s_recv[2]
         get_rows(u3_itf_i, 0, 1)[:]  = w_recv[2]
         get_rows(u3_itf_i, -1, 0)[:] = e_recv[2]


   def xchange_covectors(self, geom, u1_itf_i, u2_itf_i, u1_itf_j, u2_itf_j):

      #      +---+
      #      | 4 |
      #  +---+---+---+---+
      #  | 3 | 0 | 1 | 2 |
      #  +---+---+---+---+
      #      | 5 |
      #      +---+

      data_type = u1_itf_i.dtype
      sendbuf_u1 = numpy.zeros((4, len(u1_itf_i[0, 0, :])), dtype=data_type)
      sendbuf_u2 = numpy.zeros_like(sendbuf_u1)

      X = geom.X[0,:]
      Y = geom.Y[:,0]

      # --- Send to northern neighbours

      if self.my_row == self.nb_lines_per_panel - 1:

         if self.my_panel == 0:

            sendbuf_u1[0,:] = u1_itf_j[-2, 1, :]
            sendbuf_u2[0,:] = u2_itf_j[-2, 1, :] + 2. * X / ( 1. + X**2) * u1_itf_j[-2, 1, :]

         elif self.my_panel == 1:

            sendbuf_u1[0,:] = -u2_itf_j[-2, 1, :] - 2. * X / ( 1. + X**2 ) * u1_itf_j[-2, 1, :]
            sendbuf_u2[0,:] = u1_itf_j[-2, 1, :]

         elif self.my_panel == 2:

            sendbuf_u1[0,:] = numpy.flipud( -u1_itf_j[-2, 1, :] )
            sendbuf_u2[0,:] = numpy.flipud( -u2_itf_j[-2, 1, :] - 2. * X / ( 1. + X**2 ) * u1_itf_j[-2, 1, :] )

         elif self.my_panel == 3:

            sendbuf_u1[0,:] = numpy.flipud( u2_itf_j[-2, 1, :] + 2. * X / ( 1. + X**2 ) * u1_itf_j[-2, 1, :] )
            sendbuf_u2[0,:] = numpy.flipud(-u1_itf_j[-2, 1, :] )

         elif self.my_panel == 4:

            sendbuf_u1[0,:] = numpy.flipud( -u1_itf_j[-2, 1, :] )
            sendbuf_u2[0,:] = numpy.flipud( -u2_itf_j[-2, 1, :] - 2. * X / ( 1. + X**2 ) * u1_itf_j[-2, 1, :] )

         elif self.my_panel == 5:

            sendbuf_u1[0,:] = u1_itf_j[-2, 1, :]
            sendbuf_u2[0,:] = u2_itf_j[-2, 1, :] + 2. * X / ( 1. + X**2) * u1_itf_j[-2, 1, :]

      else:

         sendbuf_u1[0,:] = u1_itf_j[-2, 1, :]
         sendbuf_u2[0,:] = u2_itf_j[-2, 1, :]

      # --- Send to southern neighbours

      if self.my_row == 0:

         if self.my_panel == 0:

            sendbuf_u1[1,:] = u1_itf_j[1, 0, :]
            sendbuf_u2[1,:] = u2_itf_j[1, 0, :] - 2. * X / ( 1. + X**2 ) * u1_itf_j[1, 0, :]

         elif self.my_panel == 1:

            sendbuf_u1[1,:] = numpy.flipud( u2_itf_j[1, 0, :] - 2. * X / ( 1. + X**2 ) * u1_itf_j[1, 0, :] )
            sendbuf_u2[1,:] = numpy.flipud( -u1_itf_j[1, 0, :] )

         elif self.my_panel == 2:

            sendbuf_u1[1,:] = numpy.flipud( -u2_itf_j[1, 0, :] )
            sendbuf_u2[1,:] = numpy.flipud( -u2_itf_j[1, 0, :] + 2. * X / ( 1. + X**2 ) * u1_itf_j[1, 0, :] )

         elif self.my_panel == 3:

            sendbuf_u1[1,:] =-u2_itf_j[1, 0, :] + 2. * X / ( 1. + X**2 ) * u1_itf_j[1, 0, :]
            sendbuf_u2[1,:] = u1_itf_j[1, 0, :]

         elif self.my_panel == 4:

            sendbuf_u1[1,:] = u1_itf_j[1, 0, :]
            sendbuf_u2[1,:] = u2_itf_j[1, 0, :] - 2. * X / ( 1. + X**2) * u1_itf_j[1, 0, :]

         elif self.my_panel == 5:

            sendbuf_u1[1,:] = numpy.flipud( -u1_itf_j[1, 0, :] )
            sendbuf_u2[1,:] = numpy.flipud( -u2_itf_j[1, 0, :] + 2. * X / ( 1. + X**2 ) * u1_itf_j[1, 0, :] )

      else:

         sendbuf_u1[1,:] = u1_itf_j[1, 0, :]
         sendbuf_u2[1,:] = u2_itf_j[1, 0, :]

      # --- Send to western neighbours

      if self.my_col == 0:

         if self.my_panel <= 3:

            sendbuf_u1[2,:] = -2. * Y / ( 1. + Y**2 ) * u2_itf_i[1, 0, :] + u1_itf_i[1, 0, :]
            sendbuf_u2[2,:] = u2_itf_i[1, 0, :]


         elif self.my_panel == 4:

            sendbuf_u1[2,:] = numpy.flipud(-u2_itf_i[1, 0, :] )
            sendbuf_u2[2,:] = numpy.flipud( -2. * Y / ( 1. + Y**2 ) * u2_itf_i[1, 0, :] + u1_itf_i[1, 0, :] )

         elif self.my_panel == 5:

            sendbuf_u1[2,:] = u2_itf_i[1, 0, :]
            sendbuf_u2[2,:] = 2. * Y / ( 1. + Y**2 ) * u2_itf_i[1, 0, :] - u1_itf_i[1, 0, :]

      else:

         sendbuf_u1[2,:] = u1_itf_i[1, 0, :]
         sendbuf_u2[2,:] = u2_itf_i[1, 0, :]

      # --- Send to eastern neighbours

      if self.my_col == self.nb_elems_per_line-1:

         if self.my_panel <= 3:

            sendbuf_u1[3,:] = 2. * Y / (1. + Y**2 ) * u2_itf_i[-2, 1, :] + u1_itf_i[-2, 1, :]
            sendbuf_u2[3,:] = u2_itf_i[-2, 1, :]

         elif self.my_panel == 4:

            sendbuf_u1[3,:] = u2_itf_i[-2, 1, :]
            sendbuf_u2[3,:] = -2. * Y / ( 1. + Y**2) * u2_itf_i[-2, 1, :] - u1_itf_i[-2, 1, :]


         elif self.my_panel == 5:

            sendbuf_u1[3,:] = numpy.flipud(-u2_itf_i[-2, 1, :] )
            sendbuf_u2[3,:] = numpy.flipud( 2. * Y / ( 1. + Y**2 ) * u2_itf_i[-2, 1, :] + u1_itf_i[-2, 1, :] )

      else:

         sendbuf_u1[3,:] = u1_itf_i[-2, 1, :]
         sendbuf_u2[3,:] = u2_itf_i[-2, 1, :]

      # --- All to all communication

      recvbuf_u1 = self.comm_dist_graph.neighbor_alltoall(sendbuf_u1)
      recvbuf_u2 = self.comm_dist_graph.neighbor_alltoall(sendbuf_u2)

      # --- Unpack received messages

      u1_itf_j[-1, 0, :] = recvbuf_u1[0]
      u1_itf_j[0, 1, :]  = recvbuf_u1[1]
      u1_itf_i[0, 1, :]  = recvbuf_u1[2]
      u1_itf_i[-1, 0, :] = recvbuf_u1[3]

      u2_itf_j[-1, 0, :] = recvbuf_u2[0]
      u2_itf_j[0, 1, :]  = recvbuf_u2[1]
      u2_itf_i[0, 1, :]  = recvbuf_u2[2]
      u2_itf_i[-1, 0, :] = recvbuf_u2[3]


   def xchange_fluxes(self, geom, T01_itf_i, T02_itf_i, T11_itf_i, T12_itf_i, T22_itf_i, T01_itf_j, T02_itf_j, T11_itf_j, T12_itf_j, T22_itf_j):

      #      +---+
      #      | 4 |
      #  +---+---+---+---+
      #  | 3 | 0 | 1 | 2 |
      #  +---+---+---+---+
      #      | 5 |
      #      +---+

      data_type = T11_itf_i.dtype
      sendbuf_T01 = numpy.zeros((4, len(T01_itf_i[0, 0, :])), dtype=data_type)
      sendbuf_T02 = numpy.zeros_like(sendbuf_T01)
      sendbuf_T11 = numpy.zeros_like(sendbuf_T01)
      sendbuf_T12 = numpy.zeros_like(sendbuf_T01)
      sendbuf_T22 = numpy.zeros_like(sendbuf_T01)

      X = geom.X[0,:]
      Y = geom.Y[:,0]

      # --- Send to northern neighbours

      if self.my_row == self.nb_lines_per_panel - 1:

         if self.my_panel == 0:

            sendbuf_T01[0,:] = T01_itf_j[-2, 1, :] - 2. * X / (1. + X**2) * T02_itf_j[-2, 1, :]
            sendbuf_T02[0,:] = T02_itf_j[-2, 1, :]
            sendbuf_T11[0,:] = T11_itf_j[-2, 1, :] - 4. * X / ( 1. + X**2) * T12_itf_j[-2, 1, :] + (2. * X/(1. + X**2))**2 * T22_itf_j[-2, 1, :]
            sendbuf_T12[0,:] = T12_itf_j[-2, 1, :] - 2. * X / (1. + X**2) * T22_itf_j[-2, 1, :]
            sendbuf_T22[0,:] = T22_itf_j[-2, 1, :]

         elif self.my_panel == 1:

            sendbuf_T01[0,:] = -T02_itf_j[-2, 1, :]
            sendbuf_T02[0,:] = T01_itf_j[-2, 1, :] - 2. * X / ( 1. + X**2 ) * T02_itf_j[-2, 1, :]
            sendbuf_T11[0,:] = T22_itf_j[-2, 1, :]
            sendbuf_T12[0,:] = -T12_itf_j[-2, 1, :] + 2. * X / ( 1. + X**2 ) * T22_itf_j[-2, 1, :]
            sendbuf_T22[0,:] =  T11_itf_j[-2, 1, :] - 4. * X / ( 1. + X**2 ) * T12_itf_j[-2, 1, :] + (2. * X / ( 1. + X**2 ))**2 * T22_itf_j[-2, 1, :]

         elif self.my_panel == 2:

            sendbuf_T01[0,:] = numpy.flipud( -T01_itf_j[-2, 1, :] + 2. * X / ( 1. + X**2 ) * T02_itf_j[-2, 1, :] )
            sendbuf_T02[0,:] = numpy.flipud( -T02_itf_j[-2, 1, :] )
            sendbuf_T11[0,:] = numpy.flipud(  T11_itf_j[-2, 1, :] - 4. * X / ( 1. + X**2 ) * T12_itf_j[-2, 1, :] + (2. * X / ( 1. + X**2 ))**2 * T22_itf_j[-2, 1, :] )
            sendbuf_T12[0,:] = numpy.flipud( T12_itf_j[-2, 1, :] - 2. * X / ( 1. + X**2 ) * T22_itf_j[-2, 1, :] )
            sendbuf_T22[0,:] = numpy.flipud( T22_itf_j[-2, 1, :] )

         elif self.my_panel == 3:

            sendbuf_T01[0,:] = numpy.flipud( T02_itf_j[-2, 1, :] )
            sendbuf_T02[0,:] = numpy.flipud( -T01_itf_j[-2, 1, :] + 2. * X / ( 1. + X**2 ) * T02_itf_j[-2, 1, :] )
            sendbuf_T11[0,:] = numpy.flipud( T22_itf_j[-2, 1, :] )
            sendbuf_T12[0,:] = numpy.flipud( -T12_itf_j[-2, 1, :] + 2. * X / ( 1. + X**2 ) * T22_itf_j[-2, 1, :] )
            sendbuf_T22[0,:] = numpy.flipud( T11_itf_j[-2, 1, :] - 4. * X / ( 1. + X**2 ) * T12_itf_j[-2, 1, :] + (2. * X / ( 1. + X**2 ))**2 * T22_itf_j[-2, 1, :] )

         elif self.my_panel == 4:

            sendbuf_T01[0,:] = numpy.flipud( -T01_itf_j[-2, 1, :] + 2. * X / ( 1. + X**2 ) * T02_itf_j[-2, 1, :] )
            sendbuf_T02[0,:] = numpy.flipud( -T02_itf_j[-2, 1, :] )
            sendbuf_T11[0,:] = numpy.flipud( T11_itf_j[-2, 1, :] - 4. * X / ( 1. + X**2 ) * T12_itf_j[-2, 1, :] + (2. * X / ( 1. + X**2 ))**2 * T22_itf_j[-2, 1, :])
            sendbuf_T12[0,:] = numpy.flipud( T12_itf_j[-2, 1, :] - 2. * X / ( 1. + X**2 ) * T22_itf_j[-2, 1, :] )
            sendbuf_T22[0,:] = numpy.flipud( T22_itf_j[-2, 1, :] )

         elif self.my_panel == 5:

            sendbuf_T01[0,:] = T01_itf_j[-2, 1, :] - 2. * X / ( 1. + X**2) * T02_itf_j[-2, 1, :]
            sendbuf_T02[0,:] = T02_itf_j[-2, 1, :]
            sendbuf_T11[0,:] = T11_itf_j[-2, 1, :] - 4. * X / ( 1. + X**2) * T12_itf_j[-2, 1, :] + (2. * X / ( 1. + X**2))**2 * T22_itf_j[-2, 1, :]
            sendbuf_T12[0,:] = T12_itf_j[-2, 1, :] - 2. * X / ( 1. + X**2) * T22_itf_j[-2, 1, :]
            sendbuf_T22[0,:] = T22_itf_j[-2, 1, :]

      else:

         sendbuf_T01[0,:] = T01_itf_j[-2, 1, :]
         sendbuf_T02[0,:] = T02_itf_j[-2, 1, :]
         sendbuf_T11[0,:] = T11_itf_j[-2, 1, :]
         sendbuf_T12[0,:] = T12_itf_j[-2, 1, :]
         sendbuf_T22[0,:] = T22_itf_j[-2, 1, :]

      # --- Send to southern neighbours

      if self.my_row == 0:

         if self.my_panel == 0:

            sendbuf_T01[1,:] = T01_itf_j[1, 0, :] + 2. * X / ( 1. + X**2 ) * T02_itf_j[1, 0, :]
            sendbuf_T02[1,:] = T02_itf_j[1, 0, :]
            sendbuf_T11[1,:] = T11_itf_j[1, 0, :] + 4. * X / ( 1. + X**2 ) * T12_itf_j[1, 0, :] + (2. * X / ( 1. + X**2 ))**2 * T22_itf_j[1, 0, :]
            sendbuf_T12[1,:] = T12_itf_j[1, 0, :] + 2. * X / ( 1. + X**2 ) * T22_itf_j[1, 0, :]
            sendbuf_T22[1,:] = T22_itf_j[1, 0, :]

         elif self.my_panel == 1:

            sendbuf_T01[1,:] = numpy.flipud( T02_itf_j[1, 0, :] )
            sendbuf_T02[1,:] = numpy.flipud( -T01_itf_j[1, 0, :] -2 * X / ( 1 + X**2 ) * T02_itf_j[1, 0, :] )
            sendbuf_T11[1,:] = numpy.flipud( T22_itf_j[1, 0, :] )
            sendbuf_T12[1,:] = numpy.flipud( -T12_itf_j[1, 0, :] -2 * X / ( 1 + X**2 ) * T22_itf_j[1, 0, :] )
            sendbuf_T22[1,:] = numpy.flipud( T11_itf_j[1, 0, :] + 4 * X / ( 1 + X**2 ) * T12_itf_j[1, 0, :] + (2 * X / ( 1 + X**2 ))**2 * T22_itf_j[1, 0, :])

         elif self.my_panel == 2:

            sendbuf_T01[1,:] = numpy.flipud( -T01_itf_j[1, 0, :] - 2. * X / ( 1. + X**2 ) * T02_itf_j[1, 0, :] )
            sendbuf_T02[1,:] = numpy.flipud( -T02_itf_j[1, 0, :] )
            sendbuf_T11[1,:] = numpy.flipud( T11_itf_j[1, 0, :] + 4. * X / ( 1. + X**2 ) * T12_itf_j[1, 0, :] + (2. * X / ( 1. + X**2 ))**2 * T22_itf_j[1, 0, :])
            sendbuf_T12[1,:] = numpy.flipud( T12_itf_j[1, 0, :] + 2. * X / ( 1. + X**2 ) * T22_itf_j[1, 0, :] )
            sendbuf_T22[1,:] = numpy.flipud( T22_itf_j[1, 0, :] )

         elif self.my_panel == 3:

            sendbuf_T01[1,:] = -T02_itf_j[1, 0, :]
            sendbuf_T02[1,:] = T01_itf_j[1, 0, :] + 2. * X / ( 1. + X**2 ) * T02_itf_j[1, 0, :]
            sendbuf_T11[1,:] = T22_itf_j[1, 0, :]
            sendbuf_T12[1,:] = -T12_itf_j[1, 0, :] - 2. * X / ( 1. + X**2 ) * T22_itf_j[1, 0, :]
            sendbuf_T22[1,:] = T11_itf_j[1, 0, :] + 4. * X / ( 1. + X**2 ) * T12_itf_j[1, 0, :] + (2. * X / ( 1. + X**2 ))**2 * T22_itf_j[1, 0, :]

         elif self.my_panel == 4:

            sendbuf_T01[1,:] = T01_itf_j[1, 0, :] + 2. * X / ( 1. + X**2) * T02_itf_j[1, 0, :]
            sendbuf_T02[1,:] = T02_itf_j[1, 0, :]
            sendbuf_T11[1,:] = T11_itf_j[1, 0, :] + 4. * X / ( 1. + X**2) * T12_itf_j[1, 0, :] + (2. * X / (1. + X**2))**2 * T22_itf_j[1, 0, :]
            sendbuf_T12[1,:] = T12_itf_j[1, 0, :] + 2. * X / ( 1. + X**2) * T22_itf_j[1, 0, :]
            sendbuf_T22[1,:] = T22_itf_j[1, 0, :]

         elif self.my_panel == 5:

            sendbuf_T01[1,:] = numpy.flipud( -T01_itf_j[1, 0, :] - 2. * X / ( 1. + X**2 ) * T02_itf_j[1, 0, :] )
            sendbuf_T02[1,:] = numpy.flipud( -T02_itf_j[1, 0, :] )
            sendbuf_T11[1,:] = numpy.flipud( T11_itf_j[1, 0, :] + 4. * X / ( 1. + X**2 ) * T12_itf_j[1, 0, :] + (2. * X / ( 1. + X**2 ))**2 * T22_itf_j[1, 0, :])
            sendbuf_T12[1,:] = numpy.flipud( T12_itf_j[1, 0, :] + 2. * X / ( 1. + X**2 ) * T22_itf_j[1, 0, :] )
            sendbuf_T22[1,:] = numpy.flipud( T22_itf_j[1, 0, :] )
      else:

         sendbuf_T01[1,:] = T01_itf_j[1, 0, :]
         sendbuf_T02[1,:] = T02_itf_j[1, 0, :]
         sendbuf_T11[1,:] = T11_itf_j[1, 0, :]
         sendbuf_T12[1,:] = T12_itf_j[1, 0, :]
         sendbuf_T22[1,:] = T22_itf_j[1, 0, :]

      # --- Send to western neighbours

      if self.my_col == 0:

         if self.my_panel <= 3:

            sendbuf_T01[2,:] = T01_itf_i[1, 0, :]
            sendbuf_T02[2,:] = 2. * Y / ( 1 + Y**2 ) * T01_itf_i[1, 0, :] + T02_itf_i[1, 0, :]
            sendbuf_T11[2,:] = T11_itf_i[1, 0, :]
            sendbuf_T12[2,:] = 2. * Y / ( 1 + Y**2 ) * T11_itf_i[1, 0, :] + T12_itf_i[1, 0, :]
            sendbuf_T22[2,:] = (2. * Y / ( 1 + Y**2 ))**2 * T11_itf_i[1, 0, :] + 4. * Y / (1. + Y**2) * T12_itf_i[1, 0, :] + T22_itf_i[1, 0, :]

         elif self.my_panel == 4:

            sendbuf_T01[2,:] = numpy.flipud( -2. * Y / ( 1. + Y**2 ) * T01_itf_i[1, 0, :] - T02_itf_i[1, 0, :] )
            sendbuf_T02[2,:] = numpy.flipud( T01_itf_i[1, 0, :] )
            sendbuf_T11[2,:] = numpy.flipud( (2. * Y / ( 1. + Y**2 ))**2 * T11_itf_i[1, 0, :] + 4. * Y / ( 1. + Y**2 ) * T12_itf_i[1, 0, :] + T22_itf_i[1, 0, :] )
            sendbuf_T12[2,:] = numpy.flipud( -2. * Y / ( 1. + Y**2 ) * T11_itf_i[1, 0, :] - T12_itf_i[1, 0, :] )
            sendbuf_T22[2,:] = numpy.flipud( T11_itf_i[1, 0, :] )

         elif self.my_panel == 5:

            sendbuf_T01[2,:] = 2. * Y / ( 1. + Y**2 ) * T01_itf_i[1, 0, :] + T02_itf_i[1, 0, :]
            sendbuf_T02[2,:] = -T01_itf_i[1, 0, :]
            sendbuf_T11[2,:] = (2. * Y / ( 1. + Y**2 ))**2 * T11_itf_i[1, 0, :] + 4. * Y / ( 1. + Y**2 ) * T12_itf_i[1, 0, :] + T22_itf_i[1, 0, :]
            sendbuf_T12[2,:] = -2. * Y / ( 1. + Y**2 ) * T11_itf_i[1, 0, :] - T12_itf_i[1, 0, :]
            sendbuf_T22[2,:] = T11_itf_i[1, 0, :]
      else:

         sendbuf_T01[2,:] = T01_itf_i[1, 0, :]
         sendbuf_T02[2,:] = T02_itf_i[1, 0, :]
         sendbuf_T11[2,:] = T11_itf_i[1, 0, :]
         sendbuf_T12[2,:] = T12_itf_i[1, 0, :]
         sendbuf_T22[2,:] = T22_itf_i[1, 0, :]

      # --- Send to eastern neighbours

      if self.my_col == self.nb_elems_per_line-1:

         if self.my_panel <= 3:

            sendbuf_T01[3,:] = T01_itf_i[-2, 1, :]
            sendbuf_T02[3,:] = -2 * Y / ( 1 + Y**2 ) * T01_itf_i[-2, 1, :] + T02_itf_i[-2, 1, :]
            sendbuf_T11[3,:] = T11_itf_i[-2, 1, :]
            sendbuf_T12[3,:] = -2 * Y / ( 1 + Y**2 ) * T11_itf_i[-2, 1, :] + T12_itf_i[-2, 1, :]
            sendbuf_T22[3,:] = (2 * Y / ( 1 + Y**2 ))**2 * T11_itf_i[-2, 1, :] - 4. * Y/(1. + Y**2) * T12_itf_i[-2, 1, :] + T22_itf_i[-2, 1, :]

         elif self.my_panel == 4:

            sendbuf_T01[3,:] = -2 * Y / ( 1 + Y**2) * T01_itf_i[-2, 1, :] + T02_itf_i[-2, 1, :]
            sendbuf_T02[3,:] = -T01_itf_i[-2, 1, :]
            sendbuf_T11[3,:] = (2. * Y / ( 1. + Y**2))**2 * T11_itf_i[-2, 1, :] - 4. * Y / ( 1. + Y**2) * T12_itf_i[-2, 1, :] + T22_itf_i[-2, 1, :]
            sendbuf_T12[3,:] = 2 * Y / ( 1 + Y**2) * T11_itf_i[-2, 1, :] - T12_itf_i[-2, 1, :]
            sendbuf_T22[3,:] = T11_itf_i[-2, 1, :]

         elif self.my_panel == 5:

            sendbuf_T01[3,:] = numpy.flipud( 2. * Y / ( 1. + Y**2 ) * T01_itf_i[-2, 1, :] - T02_itf_i[-2, 1, :] )
            sendbuf_T02[3,:] = numpy.flipud( T01_itf_i[-2, 1, :] )
            sendbuf_T11[3,:] = numpy.flipud( (2. * Y / ( 1. + Y**2 ))**2 * T11_itf_i[-2, 1, :] - 4. * Y / ( 1. + Y**2 ) * T12_itf_i[-2, 1, :] + T22_itf_i[-2, 1, :] )
            sendbuf_T12[3,:] = numpy.flipud( 2. * Y / ( 1. + Y**2 ) * T11_itf_i[-2, 1, :] - T12_itf_i[-2, 1, :] )
            sendbuf_T22[3,:] = numpy.flipud( T11_itf_i[-2, 1, :] )
      else:

         sendbuf_T01[3,:] = T01_itf_i[-2, 1, :]
         sendbuf_T02[3,:] = T02_itf_i[-2, 1, :]
         sendbuf_T11[3,:] = T11_itf_i[-2, 1, :]
         sendbuf_T12[3,:] = T12_itf_i[-2, 1, :]
         sendbuf_T22[3,:] = T22_itf_i[-2, 1, :]

      # --- All to all communication

      recvbuf_T01 = self.comm_dist_graph.neighbor_alltoall(sendbuf_T01)
      recvbuf_T02 = self.comm_dist_graph.neighbor_alltoall(sendbuf_T02)
      recvbuf_T11 = self.comm_dist_graph.neighbor_alltoall(sendbuf_T11)
      recvbuf_T12 = self.comm_dist_graph.neighbor_alltoall(sendbuf_T12)
      recvbuf_T22 = self.comm_dist_graph.neighbor_alltoall(sendbuf_T22)

      # --- Unpack received messages
      T01_itf_j[-1, 0, :] = recvbuf_T01[0]
      T01_itf_j[0, 1, :]  = recvbuf_T01[1]
      T01_itf_i[0, 1, :]  = recvbuf_T01[2]
      T01_itf_i[-1, 0, :] = recvbuf_T01[3]

      T02_itf_j[-1, 0, :] = recvbuf_T02[0]
      T02_itf_j[0, 1, :]  = recvbuf_T02[1]
      T02_itf_i[0, 1, :]  = recvbuf_T02[2]
      T02_itf_i[-1, 0, :] = recvbuf_T02[3]

      T11_itf_j[-1, 0, :] = recvbuf_T11[0]
      T11_itf_j[0, 1, :]  = recvbuf_T11[1]
      T11_itf_i[0, 1, :]  = recvbuf_T11[2]
      T11_itf_i[-1, 0, :] = recvbuf_T11[3]

      T12_itf_j[-1, 0, :] = recvbuf_T12[0]
      T12_itf_j[0, 1, :]  = recvbuf_T12[1]
      T12_itf_i[0, 1, :]  = recvbuf_T12[2]
      T12_itf_i[-1, 0, :] = recvbuf_T12[3]

      T22_itf_j[-1, 0, :] = recvbuf_T22[0]
      T22_itf_j[0, 1, :]  = recvbuf_T22[1]
      T22_itf_i[0, 1, :]  = recvbuf_T22[2]
      T22_itf_i[-1, 0, :] = recvbuf_T22[3]
