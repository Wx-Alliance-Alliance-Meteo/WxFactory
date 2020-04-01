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


def xchange_vectors(geom, u1_itf_i, u2_itf_i, u1_itf_j, u2_itf_j):

   glb_u1_itf_i = mpi4py.MPI.COMM_WORLD.allgather(u1_itf_i)
   glb_u2_itf_i = mpi4py.MPI.COMM_WORLD.allgather(u2_itf_i)
   glb_u1_itf_j = mpi4py.MPI.COMM_WORLD.allgather(u1_itf_j)
   glb_u2_itf_j = mpi4py.MPI.COMM_WORLD.allgather(u2_itf_j)

   X = geom.X[0,:]
   Y = geom.Y[:,0]
   X_flip = numpy.flipud(X)
   Y_flip = numpy.flipud(Y)

   if geom.cube_face == 0:

      u1_itf_i[-1, 0, :] = (glb_u1_itf_i[1])[1, 0, :]
      u2_itf_i[-1, 0, :] = 2 * Y / ( 1 + Y**2 ) * (glb_u1_itf_i[1])[1, 0, :] + (glb_u2_itf_i[1])[1, 0, :]

      u1_itf_i[0, 1, :]  = (glb_u1_itf_i[3])[-2, 1, :]
      u2_itf_i[0, 1, :]  = -2 * Y / (1 + Y**2 ) * (glb_u1_itf_i[3])[-2, 1, :] + (glb_u2_itf_i[3])[-2, 1, :]

      u1_itf_j[-1, 0, :] = (glb_u1_itf_j[4])[1, 0, :] + 2 * X / ( 1 + X**2) * (glb_u2_itf_j[4])[1, 0, :]
      u2_itf_j[-1, 0, :] = (glb_u2_itf_j[4])[1, 0, :]

      u1_itf_j[0, 1, :]  = (glb_u1_itf_j[5])[-2, 1, :] - 2 * X / ( 1 + X**2) * (glb_u2_itf_j[5])[-2, 1, :]
      u2_itf_j[0, 1, :]  = (glb_u2_itf_j[5])[-2, 1, :]

   elif geom.cube_face == 1:

      u1_itf_i[0, 1, :]  = (glb_u1_itf_i[0])[-2, 1, :]
      u2_itf_i[0, 1, :]  = -2 * Y / (1 + Y**2 ) * (glb_u1_itf_i[0])[-2, 1, :] + (glb_u2_itf_i[0])[-2, 1, :]

      u1_itf_i[-1, 0, :] = (glb_u1_itf_i[2])[1, 0, :]
      u2_itf_i[-1, 0, :] = 2 * Y / ( 1 + Y**2 ) * (glb_u1_itf_i[2])[1, 0, :] + (glb_u2_itf_i[2])[1, 0, :]

      u1_itf_j[-1, 0, :] = -2 * X / ( 1 + X**2 ) * (glb_u1_itf_i[4])[-2, 1, :] + (glb_u2_itf_i[4])[-2, 1, :]
      u2_itf_j[-1, 0, :] = -(glb_u1_itf_i[4])[-2, 1, :]

      u1_itf_j[0, 1, :]  = numpy.flipud( -2 * X_flip / ( 1 + X_flip**2 ) * (glb_u1_itf_i[5])[-2, 1, :] - (glb_u2_itf_i[5])[-2, 1, :] )
      u2_itf_j[0, 1, :]  = numpy.flipud( (glb_u1_itf_i[5])[-2, 1, :] )

   elif geom.cube_face == 2:

      u1_itf_i[0, 1, :]  = (glb_u1_itf_i[1])[-2, 1, :]
      u2_itf_i[0, 1, :]  = -2 * Y / ( 1 + Y**2 ) * (glb_u1_itf_i[1])[-2, 1, :] + (glb_u2_itf_i[1])[-2, 1, :]

      u1_itf_i[-1, 0, :] = (glb_u1_itf_i[3])[1, 0, :]
      u2_itf_i[-1, 0, :] = 2 * Y / ( 1 + Y**2 ) * (glb_u1_itf_i[3])[1, 0, :] + (glb_u2_itf_i[3])[1, 0, :]

      u1_itf_j[-1, 0, :] = numpy.flipud( -(glb_u1_itf_j[4])[-2, 1, :] - 2 * X_flip / ( 1 + X_flip**2 ) * (glb_u2_itf_j[4])[-2, 1, :] )
      u2_itf_j[-1, 0, :] = numpy.flipud( -(glb_u2_itf_j[4])[-2, 1, :] )

      u1_itf_j[0, 1, :]  = numpy.flipud( -(glb_u1_itf_j[5])[1, 0, :] + 2 * X_flip / ( 1 + X_flip**2 ) * (glb_u2_itf_j[5])[1, 0, :] )
      u2_itf_j[0, 1, :]  = numpy.flipud( -(glb_u2_itf_j[5])[1, 0, :] )

   elif geom.cube_face == 3:

      u1_itf_i[0, 1, :]  = (glb_u1_itf_i[2])[-2, 1, :]
      u2_itf_i[0, 1, :]  = -2 * Y / ( 1 + Y**2 ) * (glb_u1_itf_i[2])[-2, 1, :] + (glb_u2_itf_i[2])[-2, 1, :]

      u1_itf_i[-1, 0, :] = (glb_u1_itf_i[0])[1, 0, :]
      u2_itf_i[-1, 0, :] = 2 * Y / ( 1 + Y**2 ) * (glb_u1_itf_i[0])[1, 0, :] + (glb_u2_itf_i[0])[1, 0, :]

      u1_itf_j[-1, 0, :] = numpy.flipud( 2 * X_flip / ( 1 + X_flip**2 ) * (glb_u1_itf_i[4])[1, 0, :] - (glb_u2_itf_i[4])[1, 0, :] )
      u2_itf_j[-1, 0, :] = numpy.flipud( (glb_u1_itf_i[4])[1, 0, :] )

      u1_itf_j[0, 1, :]  = 2 * X / ( 1 + X**2 ) * (glb_u1_itf_i[5])[1, 0, :] + (glb_u2_itf_i[5])[1, 0, :]
      u2_itf_j[0, 1, :]  = -(glb_u1_itf_i[5])[1, 0, :]

   elif geom.cube_face == 4:

      u1_itf_j[0, 1, :] = (glb_u1_itf_j[0])[-2, 1, :] - 2 * X / ( 1 + X**2) * (glb_u2_itf_j[0])[-2, 1, :]
      u2_itf_j[0, 1, :] = (glb_u2_itf_j[0])[-2, 1, :]

      u1_itf_i[-1, 0, :] = - (glb_u2_itf_j[1])[-2, 1, :]
      u2_itf_i[-1, 0, :] = (glb_u1_itf_j[1])[-2, 1, :] - 2 * X / ( 1 + X**2 ) * (glb_u2_itf_j[1])[-2, 1, :]

      u1_itf_j[-1, 0, :] = numpy.flipud( - (glb_u1_itf_j[2])[-2, 1, :] + 2 * X / ( 1 + X**2 ) * (glb_u2_itf_j[2])[-2, 1, :] )
      u2_itf_j[-1, 0, :] = numpy.flipud( - (glb_u2_itf_j[2])[-2, 1, :] )

      u1_itf_i[0, 1, :] = numpy.flipud( (glb_u2_itf_j[3])[-2, 1, :] )
      u2_itf_i[0, 1, :] = numpy.flipud( - (glb_u1_itf_j[3])[-2, 1, :] + 2 * X / ( 1 + X**2 ) * (glb_u2_itf_j[3])[-2, 1, :] )

   elif geom.cube_face == 5:

      u1_itf_j[-1, 0, :]  = (glb_u1_itf_j[0])[1, 0, :] + 2 * X / ( 1 + X**2 ) * (glb_u2_itf_j[0])[1, 0, :]
      u2_itf_j[-1, 0, :]  = (glb_u2_itf_j[0])[1, 0, :]

      u1_itf_i[-1, 0, :] = numpy.flipud( (glb_u2_itf_j[1])[1, 0, :] )
      u2_itf_i[-1, 0, :] = numpy.flipud( -(glb_u1_itf_j[1])[1, 0, :] -2 * X / ( 1 + X**2 ) * (glb_u2_itf_j[1])[1, 0, :] )

      u1_itf_j[0, 1, :] = numpy.flipud( -(glb_u1_itf_j[2])[1, 0, :] -2 * X / ( 1 + X**2 ) * (glb_u2_itf_j[2])[1, 0, :] )
      u2_itf_j[0, 1, :] = numpy.flipud( -(glb_u2_itf_j[2])[1, 0, :] )

      u1_itf_i[0, 1, :]  = -(glb_u2_itf_j[3])[1, 0, :]
      u2_itf_i[0, 1, :]  = (glb_u1_itf_j[3])[1, 0, :] + 2 * X / ( 1 + X**2 ) * (glb_u2_itf_j[3])[1, 0, :]

   return
