import jax.numpy as jp
from mpi4py import MPI
import mpi4jax
import numpy

from .parallel import DistributedWorld

class JaxDistributedWorld(DistributedWorld):
   def __init__(self):
      super().__init__()
      self.world = MPI.COMM_WORLD.Clone()

   def xchange_simple_vectors(self, X, Y, u1_n, u2_n, u1_s, u2_s, u1_w, u2_w, u1_e, u2_e, u3_n=None, u3_s=None,
                              u3_w=None, u3_e=None, sync=True):
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

      return self.send_recv_neighbors(sendbuf[0], sendbuf[1], sendbuf[2], sendbuf[3], flip_dim, sync=sync)
