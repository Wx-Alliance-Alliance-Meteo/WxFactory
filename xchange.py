import mpi4py.MPI
import numpy

#      +---+
#      | 4 |
#  +---+---+---+---+
#  | 3 | 0 | 1 | 2 |
#  +---+---+---+---+
#      | 5 |
#      +---+
#
# N, S, W, E ordering

def xchange_scalars(comm_dist_graph, geom, field_itf_i, field_itf_j):

   if geom.cube_face == 0:
      # neighbors = [4, 5, 3, 1]
      sendbuf = [ field_itf_j[-2, 1, :], field_itf_j[1, 0, :], field_itf_i[1, 0, :], field_itf_i[-2, 1, :] ]

   elif geom.cube_face == 1:
      # neighbors = [4, 5, 0, 2]
      sendbuf = [ field_itf_j[-2, 1, :], numpy.flipud( field_itf_j[1, 0, :] ), field_itf_i[1, 0, :], field_itf_i[-2, 1, :] ]

   elif geom.cube_face == 2:
      # neighbors = [4, 5, 1, 3]
      sendbuf = [ numpy.flipud( field_itf_j[-2, 1, :] ), numpy.flipud( field_itf_j[1, 0, :] ), field_itf_i[1, 0, :], field_itf_i[-2, 1, :] ]

   elif geom.cube_face == 3:
      # neighbors = [4, 5, 0, 2]
      sendbuf = [ numpy.flipud( field_itf_j[-2, 1, :] ), field_itf_j[1, 0, :], field_itf_i[-2, 1, :], field_itf_i[1, 0, :] ]

   elif geom.cube_face == 4:
      # neighbors = [2, 0, 3, 1]
      sendbuf = [ numpy.flipud( field_itf_j[-2, 1, :] ), field_itf_j[1, 0, :], numpy.flipud( field_itf_i[1, 0, :] ), field_itf_j[1, 0, :] ]

   elif geom.cube_face == 5:
      # neighbors = [0, 2, 3, 1]
      sendbuf = [ field_itf_j[-2, 1, :], numpy.flipud( field_itf_j[1, 0, :] ), field_itf_i[1, 0, :], numpy.flipud( field_itf_i[-2, 1, :] ) ]

   recvbuf = comm_dist_graph.neighbor_alltoall(sendbuf)

   if geom.cube_face == 0:

      field_itf_j[-1, 0, :] = recvbuf[0]
      field_itf_j[0, 1, :]  = recvbuf[1]
      field_itf_i[0, 1, :]  = recvbuf[2]
      field_itf_i[-1, 0, :] = recvbuf[3]

   elif geom.cube_face == 1:

      field_itf_j[-1, 0, :] = recvbuf[0]
      field_itf_j[0, 1, :]  = recvbuf[1]
      field_itf_i[0, 1, :]  = recvbuf[2]
      field_itf_i[-1, 0, :] = recvbuf[3]

   elif geom.cube_face == 2:

      field_itf_j[-1, 0, :] = recvbuf[0]
      field_itf_j[0, 1, :]  = recvbuf[1]
      field_itf_i[0, 1, :]  = recvbuf[2]
      field_itf_i[-1, 0, :] = recvbuf[3]

   elif geom.cube_face == 3:

      field_itf_j[-1, 0, :] = recvbuf[0]
      field_itf_j[0, 1, :]  = recvbuf[1]
      field_itf_i[-1, 0, :] = recvbuf[2]
      field_itf_i[0, 1, :]  = recvbuf[3]

   elif geom.cube_face == 4:

      field_itf_j[-1, 0, :] = recvbuf[0]
      field_itf_j[0, 1, :]  = recvbuf[1]
      field_itf_i[0, 1, :]  = recvbuf[2]
      field_itf_i[-1, 0, :] = recvbuf[3]

   elif geom.cube_face == 5:

      field_itf_j[-1, 0, :] = recvbuf[0]
      field_itf_j[0, 1, :]  = recvbuf[1]
      field_itf_i[0, 1, :]  = recvbuf[2]
      field_itf_i[-1, 0, :] = recvbuf[3]

   return


def xchange_vectors(comm_dist_graph, geom, u1_itf_i, u2_itf_i, u1_itf_j, u2_itf_j):

   X = geom.X[0,:]
   Y = geom.Y[:,0]
   X_flip = numpy.flipud(X)
   Y_flip = numpy.flipud(Y)

   if geom.cube_face == 0:
      # neighbors = [4, 5, 3, 1]

      sendbuf_u1 = [ u1_itf_j[-2, 1, :] - 2 * X / ( 1 + X**2) * u2_itf_j[-2, 1, :], \
                     u1_itf_j[1, 0, :] + 2 * X / ( 1 + X**2 ) * u2_itf_j[1, 0, :], \
                     u1_itf_i[1, 0, :], \
                     u1_itf_i[-2, 1, :] ]

      sendbuf_u2 = [ u2_itf_j[-2, 1, :], \
                     u2_itf_j[1, 0, :], \
                     2 * Y / ( 1 + Y**2 ) * u1_itf_i[1, 0, :] + u2_itf_i[1, 0, :], \
                    -2 * Y / (1 + Y**2 ) * u1_itf_i[-2, 1, :] + u2_itf_i[-2, 1, :] ]

   elif geom.cube_face == 1:
      # neighbors = [4, 5, 0, 2]

      sendbuf_u1 = [ - u2_itf_j[-2, 1, :], \
                     numpy.flipud( u2_itf_j[1, 0, :] ), \
                     u1_itf_i[1, 0, :], \
                     u1_itf_i[-2, 1, :] ]

      sendbuf_u2 = [ u1_itf_j[-2, 1, :] - 2 * X / ( 1 + X**2 ) * u2_itf_j[-2, 1, :], \
                     numpy.flipud( -u1_itf_j[1, 0, :] -2 * X / ( 1 + X**2 ) * u2_itf_j[1, 0, :] ), \
                     2 * Y / ( 1 + Y**2 ) * u1_itf_i[1, 0, :] + u2_itf_i[1, 0, :], \
                    -2 * Y / ( 1 + Y**2 ) * u1_itf_i[-2, 1, :] + u2_itf_i[-2, 1, :] ]

   elif geom.cube_face == 2:
      # neighbors = [4, 5, 1, 3]

      sendbuf_u1 = [ numpy.flipud( -u1_itf_j[-2, 1, :] + 2 * X / ( 1 + X**2 ) * u2_itf_j[-2, 1, :] ), \
                     numpy.flipud( -u1_itf_j[1, 0, :] -2 * X / ( 1 + X**2 ) * u2_itf_j[1, 0, :] ), \
                     u1_itf_i[1, 0, :], \
                     u1_itf_i[-2, 1, :] ]

      sendbuf_u2 = [ numpy.flipud( -u2_itf_j[-2, 1, :] ), \
                     numpy.flipud( -u2_itf_j[1, 0, :] ), \
                     2 * Y / ( 1 + Y**2 ) * u1_itf_i[1, 0, :] + u2_itf_i[1, 0, :], \
                    -2 * Y / ( 1 + Y**2 ) * u1_itf_i[-2, 1, :] + u2_itf_i[-2, 1, :] ]

   elif geom.cube_face == 3:
      # neighbors = [4, 5, 0, 2]

      sendbuf_u1 = [ numpy.flipud( u2_itf_j[-2, 1, :] ), \
                    -u2_itf_j[1, 0, :], \
                     u1_itf_i[-2, 1, :], \
                     u1_itf_i[1, 0, :] ]

      sendbuf_u2 = [ numpy.flipud( -u1_itf_j[-2, 1, :] + 2 * X / ( 1 + X**2 ) * u2_itf_j[-2, 1, :] ), \
                     u1_itf_j[1, 0, :] + 2 * X / ( 1 + X**2 ) * u2_itf_j[1, 0, :], \
                    -2 * Y / (1 + Y**2 ) * u1_itf_i[-2, 1, :] + u2_itf_i[-2, 1, :], \
                     2 * Y / ( 1 + Y**2 ) * u1_itf_i[1, 0, :] + u2_itf_i[1, 0, :] ]

   elif geom.cube_face == 4:
      # neighbors = [2, 0, 3, 1]

      sendbuf_u1 = [ numpy.flipud( -u1_itf_j[-2, 1, :] - 2 * X_flip / ( 1 + X_flip**2 ) * u2_itf_j[-2, 1, :] ), \
                     u1_itf_j[1, 0, :] + 2 * X / ( 1 + X**2) * u2_itf_j[1, 0, :], \
                     numpy.flipud( 2 * X_flip / ( 1 + X_flip**2 ) * u1_itf_i[1, 0, :] - u2_itf_i[1, 0, :] ), \
                     u1_itf_j[1, 0, :] + 2 * X / ( 1 + X**2) * u2_itf_j[1, 0, :] ]

      sendbuf_u2 = [ numpy.flipud( -u2_itf_j[-2, 1, :] ), \
                     u2_itf_j[1, 0, :], \
                     numpy.flipud( u1_itf_i[1, 0, :] ), \
                     -u1_itf_i[-2, 1, :] ]


   elif geom.cube_face == 5:
      # neighbors = [0, 2, 3, 1]

      sendbuf_u1 = [ u1_itf_j[-2, 1, :] - 2 * X / ( 1 + X**2) * u2_itf_j[-2, 1, :], \
                     numpy.flipud( -u1_itf_j[1, 0, :] + 2 * X_flip / ( 1 + X_flip**2 ) * u2_itf_j[1, 0, :] ), \
                     2 * X / ( 1 + X**2 ) * u1_itf_i[1, 0, :] + u2_itf_i[1, 0, :], \
                     numpy.flipud( -2 * X_flip / ( 1 + X_flip**2 ) * u1_itf_i[-2, 1, :] - u2_itf_i[-2, 1, :] ) ]

      sendbuf_u2 = [ u2_itf_j[-2, 1, :], \
                     numpy.flipud( -u2_itf_j[1, 0, :] ), \
                    -u1_itf_i[1, 0, :], \
                     numpy.flipud( u1_itf_i[-2, 1, :] ) ]

   recvbuf_u1 = comm_dist_graph.neighbor_alltoall(sendbuf_u1)
   recvbuf_u2 = comm_dist_graph.neighbor_alltoall(sendbuf_u2)

   if geom.cube_face == 0:

      u1_itf_j[-1, 0, :] = recvbuf_u1[0]
      u1_itf_j[0, 1, :]  = recvbuf_u1[1]
      u1_itf_i[0, 1, :]  = recvbuf_u1[2]
      u1_itf_i[-1, 0, :] = recvbuf_u1[3]

      u2_itf_j[-1, 0, :] = recvbuf_u2[0]
      u2_itf_j[0, 1, :]  = recvbuf_u2[1]
      u2_itf_i[0, 1, :]  = recvbuf_u2[2]
      u2_itf_i[-1, 0, :] = recvbuf_u2[3]

   elif geom.cube_face == 1:

      u1_itf_j[-1, 0, :] = recvbuf_u1[0]
      u1_itf_j[0, 1, :]  = recvbuf_u1[1]
      u1_itf_i[0, 1, :]  = recvbuf_u1[2]
      u1_itf_i[-1, 0, :] = recvbuf_u1[3]

      u2_itf_j[-1, 0, :] = recvbuf_u2[0]
      u2_itf_j[0, 1, :]  = recvbuf_u2[1]
      u2_itf_i[0, 1, :]  = recvbuf_u2[2]
      u2_itf_i[-1, 0, :] = recvbuf_u2[3]

   elif geom.cube_face == 2:

      u1_itf_j[-1, 0, :] = recvbuf_u1[0]
      u1_itf_j[0, 1, :]  = recvbuf_u1[1]
      u1_itf_i[0, 1, :]  = recvbuf_u1[2]
      u1_itf_i[-1, 0, :] = recvbuf_u1[3]

      u2_itf_j[-1, 0, :] = recvbuf_u2[0]
      u2_itf_j[0, 1, :]  = recvbuf_u2[1]
      u2_itf_i[0, 1, :]  = recvbuf_u2[2]
      u2_itf_i[-1, 0, :] = recvbuf_u2[3]

   elif geom.cube_face == 3:

      u1_itf_j[-1, 0, :] = recvbuf_u1[0]
      u1_itf_j[0, 1, :]  = recvbuf_u1[1]
      u1_itf_i[-1, 0, :] = recvbuf_u1[2]
      u1_itf_i[0, 1, :]  = recvbuf_u1[3]

      u2_itf_j[-1, 0, :] = recvbuf_u2[0]
      u2_itf_j[0, 1, :]  = recvbuf_u2[1]
      u2_itf_i[-1, 0, :] = recvbuf_u2[2]
      u2_itf_i[0, 1, :]  = recvbuf_u2[3]

   elif geom.cube_face == 4:

      u1_itf_j[-1, 0, :] = recvbuf_u1[0]
      u1_itf_j[0, 1, :]  = recvbuf_u1[1]
      u1_itf_i[0, 1, :]  = recvbuf_u1[2]
      u1_itf_i[-1, 0, :] = recvbuf_u1[3]

      u2_itf_j[-1, 0, :] = recvbuf_u2[0]
      u2_itf_j[0, 1, :]  = recvbuf_u2[1]
      u2_itf_i[0, 1, :]  = recvbuf_u2[2]
      u2_itf_i[-1, 0, :] = recvbuf_u2[3]

   elif geom.cube_face == 5:

      u1_itf_j[-1, 0, :] = recvbuf_u1[0]
      u1_itf_j[0, 1, :]  = recvbuf_u1[1]
      u1_itf_i[0, 1, :]  = recvbuf_u1[2]
      u1_itf_i[-1, 0, :] = recvbuf_u1[3]

      u2_itf_j[-1, 0, :] = recvbuf_u2[0]
      u2_itf_j[0, 1, :]  = recvbuf_u2[1]
      u2_itf_i[0, 1, :]  = recvbuf_u2[2]
      u2_itf_i[-1, 0, :] = recvbuf_u2[3]

   return
