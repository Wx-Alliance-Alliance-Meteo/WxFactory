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
      # For instance, the panel 0 with n=24 will be endowed with a 4x4 topology like this
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

      if self.size < 6 or self.nb_pe_per_panel != math.isqrt(self.nb_pe_per_panel) ** 2 or self.nb_pe_per_panel * 6 != self.size:
         raise Exception('Wrong number of PEs. This topology is not allowed')

      self.nb_lines_per_panel = int(math.sqrt(self.nb_pe_per_panel))
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

      # North
      if self.my_row == self.nb_lines_per_panel -1:
         if self.my_panel == 0:
            my_north = rank_from_location(my_north_panel, 0, self.my_col)
         elif self.my_panel == 1:
            my_north = rank_from_location(my_north_panel, self.my_col, self.nb_lines_per_panel - 1)
         elif self.my_panel == 2:
            my_north = rank_from_location(my_north_panel, self.nb_lines_per_panel - 1, self.nb_lines_per_panel - 1 - self.my_col)
         elif self.my_panel == 3:
            my_north = rank_from_location(my_north_panel, self.nb_lines_per_panel - 1 - self.my_col, 0)
         elif self.my_panel == 4:
            my_north = rank_from_location(my_north_panel, self.nb_lines_per_panel - 1, self.nb_lines_per_panel - 1 - self.my_col)
         elif self.my_panel == 5:
            my_north = rank_from_location(my_north_panel, 0, self.my_col)
      else:
         my_north = rank_from_location(self.my_panel, (self.my_row+1), self.my_col)

      # South
      if self.my_row == 0:
         if self.my_panel == 0:
            my_south = rank_from_location(my_south_panel, self.nb_elems_per_line-1, self.my_col)
         elif self.my_panel == 1:
            my_south = rank_from_location(my_south_panel, self.nb_elems_per_line - 1 - self.my_col, self.nb_lines_per_panel-1)
         elif self.my_panel == 2:
            my_south = rank_from_location(my_south_panel, 0, self.nb_lines_per_panel - 1 - self.my_col)
         elif self.my_panel == 3:
            my_south = rank_from_location(my_south_panel, self.my_col, 0)
         elif self.my_panel == 4:
            my_south = rank_from_location(my_south_panel, self.nb_elems_per_line-1, self.my_col)
         elif self.my_panel == 5:
            my_south = rank_from_location(my_south_panel, 0, self.nb_elems_per_line - 1 - self.my_col)
      else:
         my_south = rank_from_location(self.my_panel, (self.my_row-1), self.my_col)

      # West
      if self.my_col == 0:
         if self.my_panel == 4:
            my_west = rank_from_location(my_west_panel, self.nb_lines_per_panel-1, self.nb_lines_per_panel - 1 - self.my_row)
         elif self.my_panel == 5:
            my_west = rank_from_location(my_west_panel, 0, self.my_row)
         else:
            my_west = rank_from_location(my_west_panel, self.my_row, self.nb_lines_per_panel-1)
      else:
         my_west = rank_from_location(self.my_panel, self.my_row, (self.my_col-1))

      # East
      if self.my_col == self.nb_elems_per_line-1:
         if self.my_panel == 4:
            my_east = rank_from_location(my_east_panel, self.nb_lines_per_panel-1, self.my_row)
         elif self.my_panel == 5:
            my_east = rank_from_location(my_east_panel, 0, self.nb_lines_per_panel - 1 - self.my_row)
         else:
            my_east = rank_from_location(my_east_panel, self.my_row, 0)
      else:
         my_east = rank_from_location(self.my_panel, self.my_row, self.my_col+1)

      # Distributed Graph
      sources = [my_north, my_south, my_west, my_east]
      destinations = sources

      self.comm_dist_graph = mpi4py.MPI.COMM_WORLD.Create_dist_graph_adjacent(sources, destinations)


   def xchange_scalars(self, geom, field_itf_i, field_itf_j):

      type_itf = type(field_itf_i[0, 0, :])
      sendbuf = numpy.empty((4, len(field_itf_i[0, 0, :])), dtype=type_itf)

      # --- Send to northern neighbours

      if self.my_row == self.nb_lines_per_panel -1 and ( self.my_panel == 2 or self.my_panel == 3 or self.my_panel == 4 ):
         sendbuf[0,:] = numpy.flipud( field_itf_j[-2, 1, :] )
      else:
         sendbuf[0,:] = field_itf_j[-2, 1, :]

      # --- Send to southern neighbours

      if self.my_row == 0 and ( self.my_panel == 1 or self.my_panel == 2 or self.my_panel == 5 ):
         sendbuf[1,:] = numpy.flipud( field_itf_j[1, 0, :] )
      else:
         sendbuf[1,:] = field_itf_j[1, 0, :]

      # --- Send to western neighbours

      if self.my_col == 0 and self.my_panel == 4:
         sendbuf[2,:] = numpy.flipud( field_itf_i[1, 0, :] )
      else:
         sendbuf[2,:] = field_itf_i[1, 0, :]

      # --- Send to eastern neighbours

      if self.my_col == self.nb_elems_per_line-1 and self.my_panel == 5:
         sendbuf[3,:] = numpy.flipud( field_itf_i[-2, 1, :] )
      else:
         sendbuf[3,:] = field_itf_i[-2, 1, :]

      # --- All to all communication
      
      recvbuf = self.comm_dist_graph.neighbor_alltoall(sendbuf)

      # --- Unpack received messages
         
      field_itf_j[-1, 0, :] = recvbuf[0]
      field_itf_j[0, 1, :]  = recvbuf[1]
      field_itf_i[0, 1, :]  = recvbuf[2]
      field_itf_i[-1, 0, :] = recvbuf[3]


   def xchange_vectors(self, geom, u1_itf_i, u2_itf_i, u1_itf_j, u2_itf_j):

      #      +---+
      #      | 4 |
      #  +---+---+---+---+
      #  | 3 | 0 | 1 | 2 |
      #  +---+---+---+---+
      #      | 5 |
      #      +---+

      type_itf = type(u1_itf_i[0, 0, :])
      sendbuf_u1 = numpy.zeros((4, len(u1_itf_i[0, 0, :])), dtype=type_itf)
      sendbuf_u2 = numpy.zeros_like(sendbuf_u1)

      X = geom.X[0,:]
      Y = geom.Y[:,0]

      # --- Send to northern neighbours

      if self.my_row == self.nb_lines_per_panel -1:

         if self.my_panel == 0:
   
            sendbuf_u1[0,:] = u1_itf_j[-2, 1, :] - 2 * X / ( 1 + X**2) * u2_itf_j[-2, 1, :]
            sendbuf_u2[0,:] = u2_itf_j[-2, 1, :]
   
         elif self.my_panel == 1:
   
            sendbuf_u1[0,:] = -u2_itf_j[-2, 1, :]
            sendbuf_u2[0,:] = u1_itf_j[-2, 1, :] - 2 * X / ( 1 + X**2 ) * u2_itf_j[-2, 1, :]
   
         elif self.my_panel == 2:
   
            sendbuf_u1[0,:] = numpy.flipud( -u1_itf_j[-2, 1, :] + 2 * X / ( 1 + X**2 ) * u2_itf_j[-2, 1, :] )
            sendbuf_u2[0,:] = numpy.flipud( -u2_itf_j[-2, 1, :] )
   
         elif self.my_panel == 3:
   
            sendbuf_u1[0,:] = numpy.flipud( u2_itf_j[-2, 1, :] )
            sendbuf_u2[0,:] = numpy.flipud( -u1_itf_j[-2, 1, :] + 2 * X / ( 1 + X**2 ) * u2_itf_j[-2, 1, :] )
   
         elif self.my_panel == 4:
   
            sendbuf_u1[0,:] = numpy.flipud( -u1_itf_j[-2, 1, :] + 2 * X / ( 1 + X**2 ) * u2_itf_j[-2, 1, :] )
            sendbuf_u2[0,:] = numpy.flipud( -u2_itf_j[-2, 1, :] )
   
         elif self.my_panel == 5:
   
            sendbuf_u1[0,:] = u1_itf_j[-2, 1, :] - 2 * X / ( 1 + X**2) * u2_itf_j[-2, 1, :]
            sendbuf_u2[0,:] = u2_itf_j[-2, 1, :]

      else:

         sendbuf_u1[0,:] = u1_itf_j[-2, 1, :]
         sendbuf_u2[0,:] = u2_itf_j[-2, 1, :]

   
      # --- Send to southern neighbours

      if self.my_row == 0:

         if self.my_panel == 0:
   
            sendbuf_u1[1,:] = u1_itf_j[1, 0, :] + 2 * X / ( 1 + X**2 ) * u2_itf_j[1, 0, :]
            sendbuf_u2[1,:] = u2_itf_j[1, 0, :]
   
         elif self.my_panel == 1:
   
            sendbuf_u1[1,:] = numpy.flipud( u2_itf_j[1, 0, :] )
            sendbuf_u2[1,:] = numpy.flipud( -u1_itf_j[1, 0, :] -2 * X / ( 1 + X**2 ) * u2_itf_j[1, 0, :] )
   
         elif self.my_panel == 2:
   
            sendbuf_u1[1,:] = numpy.flipud( -u1_itf_j[1, 0, :] -2 * X / ( 1 + X**2 ) * u2_itf_j[1, 0, :] )
            sendbuf_u2[1,:] = numpy.flipud( -u2_itf_j[1, 0, :] )
   
         elif self.my_panel == 3:
   
            sendbuf_u1[1,:] = -u2_itf_j[1, 0, :]
            sendbuf_u2[1,:] = u1_itf_j[1, 0, :] + 2 * X / ( 1 + X**2 ) * u2_itf_j[1, 0, :]
   
         elif self.my_panel == 4:
   
            sendbuf_u1[1,:] = u1_itf_j[1, 0, :] + 2 * X / ( 1 + X**2) * u2_itf_j[1, 0, :]
            sendbuf_u2[1,:] = u2_itf_j[1, 0, :]
   
         elif self.my_panel == 5:
   
            sendbuf_u1[1,:] = numpy.flipud( -u1_itf_j[1, 0, :] - 2 * X / ( 1 + X**2 ) * u2_itf_j[1, 0, :] )
            sendbuf_u2[1,:] = numpy.flipud( -u2_itf_j[1, 0, :] )
      else:

         sendbuf_u1[1,:] = u1_itf_j[1, 0, :]
         sendbuf_u2[1,:] = u2_itf_j[1, 0, :]

      # --- Send to western neighbours

      if self.my_col == 0:

         if self.my_panel == 0:

            sendbuf_u1[2,:] = u1_itf_i[1, 0, :]
            sendbuf_u2[2,:] = 2 * Y / ( 1 + Y**2 ) * u1_itf_i[1, 0, :] + u2_itf_i[1, 0, :]
                        
   
         elif self.my_panel == 1:
   
            sendbuf_u1[2,:] = u1_itf_i[1, 0, :]
            sendbuf_u2[2,:] = 2 * Y / ( 1 + Y**2 ) * u1_itf_i[1, 0, :] + u2_itf_i[1, 0, :]
                        
   
         elif self.my_panel == 2:
   
            sendbuf_u1[2,:] = u1_itf_i[1, 0, :]
            sendbuf_u2[2,:] = 2 * Y / ( 1 + Y**2 ) * u1_itf_i[1, 0, :] + u2_itf_i[1, 0, :]
                         
   
         elif self.my_panel == 3:
   
            sendbuf_u1[2,:] = u1_itf_i[1, 0, :]
            sendbuf_u2[2,:] = 2 * Y / ( 1 + Y**2 ) * u1_itf_i[1, 0, :] + u2_itf_i[1, 0, :]
                         
   
         elif self.my_panel == 4:
   
            sendbuf_u1[2,:] = numpy.flipud( -2 * Y / ( 1 + Y**2 ) * u1_itf_i[1, 0, :] - u2_itf_i[1, 0, :] )
            sendbuf_u2[2,:] = numpy.flipud( u1_itf_i[1, 0, :] )
   
         elif self.my_panel == 5:

            sendbuf_u1[2,:] = 2 * Y / ( 1 + Y**2 ) * u1_itf_i[1, 0, :] + u2_itf_i[1, 0, :]
            sendbuf_u2[2,:] = -u1_itf_i[1, 0, :]
      else:

         sendbuf_u1[2,:] = u1_itf_i[1, 0, :]
         sendbuf_u2[2,:] = u2_itf_i[1, 0, :]                        

      # --- Send to eastern neighbours

      if self.my_col == self.nb_elems_per_line-1:

         if self.my_panel == 0:

            sendbuf_u1[3,:] = u1_itf_i[-2, 1, :] 
            sendbuf_u2[3,:] = -2 * Y / (1 + Y**2 ) * u1_itf_i[-2, 1, :] + u2_itf_i[-2, 1, :] 
   
         elif self.my_panel == 1:
   
            sendbuf_u1[3,:] = u1_itf_i[-2, 1, :] 
            sendbuf_u2[3,:] = -2 * Y / ( 1 + Y**2 ) * u1_itf_i[-2, 1, :] + u2_itf_i[-2, 1, :] 
   
         elif self.my_panel == 2:
   
            sendbuf_u1[3,:] = u1_itf_i[-2, 1, :] 
            sendbuf_u2[3,:] = -2 * Y / ( 1 + Y**2 ) * u1_itf_i[-2, 1, :] + u2_itf_i[-2, 1, :] 
   
         elif self.my_panel == 3:
   
            sendbuf_u1[3,:] = u1_itf_i[-2, 1, :] 
            sendbuf_u2[3,:] = -2 * Y / (1 + Y**2 ) * u1_itf_i[-2, 1, :] + u2_itf_i[-2, 1, :] 
   
         elif self.my_panel == 4:
   
            sendbuf_u1[3,:] = -2 * Y / ( 1 + Y**2) * u1_itf_i[-2, 1, :] + u2_itf_i[-2, 1, :] 
            sendbuf_u2[3,:] = -u1_itf_i[-2, 1, :]
   
   
         elif self.my_panel == 5:
   
            sendbuf_u1[3,:] = numpy.flipud( 2 * Y / ( 1 + Y**2 ) * u1_itf_i[-2, 1, :] - u2_itf_i[-2, 1, :] )
            sendbuf_u2[3,:] = numpy.flipud( u1_itf_i[-2, 1, :] ) 
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
