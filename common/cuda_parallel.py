import cupy as cp
from mpi4py import MPI

from .definitions import *
from .parallel import DistributedWorld, EulerExchangeRequest, VectorNonBlockingExchangeRequest

# typing
from typing import Self
from numpy.typing import NDArray
from geometry.cubed_sphere import CubedSphere

class CudaDistributedWorld(DistributedWorld):

   def __init__(self: Self):
      super().__init__()

   def send_recv_neighbors(self: Self,
                           north_send: NDArray[cp.float64],
                           south_send: NDArray[cp.float64],
                           west_send: NDArray[cp.float64],
                           east_send: NDArray[cp.float64],
                           flip_dim: None | int | tuple[int, ...],
                           sync: bool = True) \
      -> tuple[MPI.Request, NDArray[cp.float64], NDArray[cp.float64], NDArray[cp.float64], NDArray[cp.float64]]:

      send_buffer = cp.empty((4,) + north_send.shape, dtype=north_send.dtype)
      for do_flip, data, buffer, in zip((self.flip_north, self.flip_south, self.flip_west, self.flip_east),
                                       (north_send, south_send, west_send, east_send),
                                       (send_buffer[0], send_buffer[1], send_buffer[2], send_buffer[3])):
         buffer[:] = cp.flip(data, flip_dim) if do_flip else data
      
      receive_buffer = cp.empty_like(send_buffer)
      cp.cuda.get_current_stream().synchronize()
      request = self.comm_dist_graph.Ineighbor_alltoall(send_buffer, receive_buffer)

      if sync:
         request.Wait()
      return request, receive_buffer[0], receive_buffer[1], receive_buffer[2], receive_buffer[3]
   
   def xchange_Euler_interfaces(self: Self,
                              geom: CubedSphere,
                              variables_itf_i: NDArray[cp.float64],
                              variables_itf_j: NDArray[cp.float64],
                              blocking: bool = True) -> EulerExchangeRequest:
      
      X = cp.asarray(geom.X[0, :])
      Y = cp.asarray(geom.Y[:, 0])
      flip_dim = 1
      id_first_tracer = 5

      init_shape = variables_itf_i.shape
      dtype = variables_itf_i.dtype
      send_buffer = cp.empty((4, init_shape[0], init_shape[1], init_shape[4]), dtype=dtype)
      recv_buffer = cp.empty_like(send_buffer)

      var_n = variables_itf_j[:, :, -2, 1, :]
      var_s = variables_itf_j[:, :, 1, 0, :]
      var_w = variables_itf_i[:, :, 1, 0, :]
      var_e = variables_itf_i[:, :, -2, 1, :]

      # Fill the send buffer, flipping when necessary and converting vector values
      for do_flip, convert, positions, var, buffer in zip(
         (self.flip_north, self.flip_south, self.flip_west, self.flip_east),
         (self.convert_contra_north, self.convert_contra_south, self.convert_contra_west, self.convert_contra_east),
         (X, X, Y, Y),
         (var_n, var_s, var_w, var_e),
         (send_buffer[0], send_buffer[1], send_buffer[2], send_buffer[3])):

         for id in (idx_rho, idx_rho_w, idx_rho_theta):
               buffer[id, :] = cp.flip(var[id], flip_dim) if do_flip else var[id]
         
         tmp1, tmp2 = convert(var[idx_rho_u1], var[idx_rho_u2], positions)
         buffer[idx_rho_u1] = cp.flip(tmp1, flip_dim) if do_flip else tmp1
         buffer[idx_rho_u2] = cp.flip(tmp2, flip_dim) if do_flip else tmp2

         buffer[id_first_tracer:] = cp.flip(var[id_first_tracer:], flip_dim + 1) if do_flip else var[id_first_tracer:]

      # Initiate MPI transfer
      cp.cuda.get_current_stream().synchronize()
      mpi_request = self.comm_dist_graph.Ineighbor_alltoall(send_buffer, recv_buffer)

      # Setup request so that data ends up in the right arrays when the wait() function is called
      var_n_dest = variables_itf_j[:, :, -1, 0, :]
      var_s_dest = variables_itf_j[:, :, 0, 1, :]
      var_w_dest = variables_itf_i[:, :, 0, 1, :]
      var_e_dest = variables_itf_i[:, :, -1, 0, :]

      request = EulerExchangeRequest(recv_buffer, (var_n_dest, var_s_dest, var_w_dest, var_e_dest), mpi_request)

      if blocking:
         request.wait()

      return request
    