from mpi4py import MPI
import numpy
from typing import Optional

from init.initialize import Topo
from common.definitions import idx_h, idx_hu1, idx_hu2, gravity
from common.process_topology import ProcessTopology
from common.device import Device, default_device
from geometry import CubedSphere, DFROperators, Metric3DTopo
from rhs.rhs import RHS

class RhsShallowWater(RHS):

   def __init__(self,
                shape: tuple[int, ...],
                geom: CubedSphere,
                mtrx: DFROperators,
                metric: Metric3DTopo,
                topo: Optional[Topo],
                ptopo: Optional[ProcessTopology],
                nbsolpts: int,
                nb_elements_hori: int,
                device: Device = default_device):
      super().__init__(shape, geom, mtrx, metric, topo, ptopo, nbsolpts, nb_elements_hori, device)

   def __compute_rhs__(self,
                       Q: numpy.ndarray,
                       geom: CubedSphere,
                       mtrx: DFROperators,
                       metric: Metric3DTopo,
                       topo: Optional[Topo],
                       ptopo: Optional[ProcessTopology],
                       nbsolpts: int,
                       nb_elements_hori: int,
                       device: Device = default_device):
      rank = MPI.COMM_WORLD.rank

      xp = device.xp

      type_vec = Q.dtype
      nb_equations = Q.shape[0]
      nb_interfaces_hori = nb_elements_hori + 1

      df1_dx1, df2_dx2, flux_x1, flux_x2 = [xp.empty_like(Q, dtype=type_vec) for _ in range(4)]

      flux_x1_itf_i = xp.zeros((nb_equations, nb_elements_hori+2, nbsolpts*nb_elements_hori, 2), dtype=type_vec)
      flux_x2_itf_j, var_itf_i, var_itf_j= [xp.zeros((nb_equations, nb_elements_hori+2, 2, nbsolpts*nb_elements_hori), dtype=type_vec) for _ in range(3)]

      forcing = xp.zeros_like(Q, dtype=type_vec)

      itf_shape = (nb_equations, nb_elements_hori * (nb_elements_hori+2), 2 * nbsolpts)
      def to_new_itf_i(a):
         if a.ndim == 4:
            tmp_shape = (nb_equations, nb_elements_hori+2, 2, nb_elements_hori, nbsolpts)
            new_shape = itf_shape
            return a.reshape(tmp_shape).transpose(0, 3, 1, 2, 4).reshape(new_shape)
         elif a.ndim == 3:
            tmp_shape = (nb_elements_hori+2, 2, nb_elements_hori, nbsolpts)
            new_shape = itf_shape[1:]
            return a.reshape(tmp_shape).transpose(2, 0, 1, 3).reshape(new_shape)

         raise ValueError


      def to_new_itf_if(a):
         tmp_shape = (nb_equations, nb_elements_hori+2, nb_elements_hori, nbsolpts, 2)
         new_shape = itf_shape
         return a.reshape(tmp_shape).transpose(0, 2, 1, 4, 3).reshape(new_shape)

      def to_new_itf_j(a):
         if a.ndim == 4:
            tmp_shape = (nb_equations, nb_elements_hori+2, 2, nb_elements_hori, nbsolpts)
            new_shape = itf_shape
            return a.reshape(tmp_shape).transpose(0, 1, 3, 2, 4).reshape(new_shape)
         elif a.ndim == 3:
            tmp_shape = (nb_elements_hori+2, 2, nb_elements_hori, nbsolpts)
            new_shape = itf_shape[1:]
            return a.reshape(tmp_shape).transpose(0, 2, 1, 3).reshape(new_shape)
         raise ValueError

      def middle_itf_i(a):
         tmp_shape = (nb_equations, nb_elements_hori, nb_elements_hori + 2, nbsolpts * 2)
         new_shape = (nb_equations, nb_elements_hori * nb_elements_hori, nbsolpts * 2)
         return a.reshape(tmp_shape)[:, :, 1:-1, :].reshape(new_shape)

      def middle_itf_j(a):
         tmp_shape = (nb_equations, nb_elements_hori + 2, nb_elements_hori, nbsolpts * 2)
         new_shape = (nb_equations, nb_elements_hori * nb_elements_hori, nbsolpts * 2)
         return a.reshape(tmp_shape)[:, 1:-1, :, :].reshape(new_shape)

      Q_new = geom._to_new(Q)

      # if MPI.COMM_WORLD.rank == 0:
      #    print(f'old Q = \n{Q}')
      #    print(f'new Q = \n{Q_new}')

      # Offset due to the halo
      offset = 1

      # Unpack dynamical variables
      HH = Q[idx_h] if topo is None else Q[idx_h] + topo.hsurf
      u1 = Q[idx_hu1] / Q[idx_h]
      u2 = Q[idx_hu2] / Q[idx_h]

      Q_new_unpacked = Q_new.copy()
      if topo is not None: Q_new_unpacked[idx_h] += geom._to_new(topo.hsurf)

      var_itf_i_new = xp.zeros(itf_shape, dtype=Q.dtype)
      vi = var_itf_i_new.reshape((nb_equations, nb_elements_hori, nb_elements_hori+2, nbsolpts * 2))
      vi[:, :, 1:-1, :] = (Q_new_unpacked @ mtrx.extrap_x).reshape((nb_equations, nb_elements_hori, nb_elements_hori, nbsolpts * 2))

      var_itf_j_new = xp.zeros(itf_shape, dtype=Q.dtype)
      vj = var_itf_j_new.reshape((nb_equations, nb_elements_hori+2, nb_elements_hori, nbsolpts * 2))
      vj[:, 1:-1, :, :] = (Q_new_unpacked @ mtrx.extrap_y).reshape((nb_equations, nb_elements_hori, nb_elements_hori, nbsolpts * 2))

      Q_new_unpacked[idx_hu1] /= Q_new[idx_h]
      Q_new_unpacked[idx_hu2] /= Q_new[idx_h]

      u1_new = Q_new_unpacked[idx_hu1]
      u2_new = Q_new_unpacked[idx_hu2]

      # Interpolate to the element interface
      for elem in range(nb_elements_hori):
         epais = elem * nbsolpts + xp.arange(nbsolpts)
         pos   = elem + offset

         # --- Direction x1

         var_itf_i[idx_h, pos, 0, :] = HH[:, epais] @ mtrx.extrap_west
         var_itf_i[idx_h, pos, 1, :] = HH[:, epais] @ mtrx.extrap_east

         var_itf_i[1:, pos, 0, :] = Q[1:, :, epais] @ mtrx.extrap_west
         var_itf_i[1:, pos, 1, :] = Q[1:, :, epais] @ mtrx.extrap_east

         # --- Direction x2
         var_itf_j[idx_h, pos, 0, :] = mtrx.extrap_south @ HH[epais, :]
         var_itf_j[idx_h, pos, 1, :] = mtrx.extrap_north @ HH[epais, :]

         var_itf_j[1:, pos, 0, :] = mtrx.extrap_south @ Q[1:, epais, :]
         var_itf_j[1:, pos, 1, :] = mtrx.extrap_north @ Q[1:, epais, :]


      # if rank == 0:
      #    print(f'Old itf i: \n{var_itf_i[0]}')
      #    print(f'Old itf with new shape: \n{to_new_itf_i(var_itf_i)[0]}')
      #    print(f'New itf i: \n{var_itf_i_new[0]}')

      #    print(f'Old itf j: \n{var_itf_j[0]}')
      #    print(f'Old itf with new shape: \n{to_new_itf_j(var_itf_j)[0]}')
      #    print(f'New itf j: \n{var_itf_j_new[0]}')

      diff = var_itf_i_new - to_new_itf_i(var_itf_i)
      diff_norm = xp.linalg.norm(diff) / xp.linalg.norm(var_itf_i)
      if diff_norm > 1e-10:
         print(f'{rank} itf i values are different: \n'
               f'{diff}')
         raise ValueError

      diff = var_itf_j_new - to_new_itf_j(var_itf_j)
      diff_norm = xp.linalg.norm(diff) / xp.linalg.norm(var_itf_j)
      if diff_norm > 1e-10:
         print(f'{rank} itf j values are different: \n'
               f'{diff}')
         raise ValueError

      # Initiate transfers
      request_u = ptopo.start_exchange_vectors(
                                 (var_itf_j[idx_hu1,  1, 0], var_itf_j[idx_hu2,  1, 0]), # South boundary
                                 (var_itf_j[idx_hu1, -2, 1], var_itf_j[idx_hu2, -2, 1]), # North boundary
                                 (var_itf_i[idx_hu1,  1, 0], var_itf_i[idx_hu2,  1, 0]), # West boundary
                                 (var_itf_i[idx_hu1, -2, 1], var_itf_i[idx_hu2, -2, 1]), # East boundary
                                 geom.X[0, :], geom.Y[:, 0])  # Coordinates at the boundary
      request_h = ptopo.start_exchange_scalars(
         var_itf_j[idx_h, 1, 0], var_itf_j[idx_h, -2, 1],var_itf_i[idx_h, 1, 0], var_itf_i[idx_h, -2, 1])

      vj_tmp = var_itf_j_new.reshape((nb_equations, nb_elements_hori+2, nb_elements_hori, 2, nbsolpts))
      vi_tmp = var_itf_i_new.reshape((nb_equations, nb_elements_hori, nb_elements_hori+2, 2, nbsolpts))
      request_u_new = ptopo.start_exchange_vectors(
                           (xp.ravel(vj_tmp[idx_hu1,  1,  :, 0, :]), xp.ravel(vj_tmp[idx_hu2,  1,  :, 0, :])), # South boundary
                           (xp.ravel(vj_tmp[idx_hu1, -2,  :, 1, :]), xp.ravel(vj_tmp[idx_hu2, -2,  :, 1, :])), # North boundary
                           (xp.ravel(vi_tmp[idx_hu1,  :,  1, 0, :]), xp.ravel(vi_tmp[idx_hu2,  :,  1, 0, :])), # West boundary
                           (xp.ravel(vi_tmp[idx_hu1,  :, -2, 1, :]), xp.ravel(vi_tmp[idx_hu2,  :, -2, 1, :])), # East boundary
                           geom.boundary_sn, geom.boundary_we
                           )
      request_h_new = ptopo.start_exchange_scalars(
         xp.ravel(vj_tmp[idx_h, 1, :, 0, :]), xp.ravel(vj_tmp[idx_h, -2, :, 1, :]),
         xp.ravel(vi_tmp[idx_h, :, 1, 0, :]), xp.ravel(vi_tmp[idx_h, :, -2, 1, :])
      )

      # Compute the fluxes
      flux_x1[idx_h] = metric.sqrtG * Q[idx_hu1]
      flux_x2[idx_h] = metric.sqrtG * Q[idx_hu2]

      hsquared = Q[idx_h]**2
      flux_x1[idx_hu1] = metric.sqrtG * ( Q[idx_hu1] * u1 + 0.5 * gravity * metric.H_contra_11 * hsquared )
      flux_x2[idx_hu1] = metric.sqrtG * ( Q[idx_hu1] * u2 + 0.5 * gravity * metric.H_contra_12 * hsquared )

      flux_x1[idx_hu2] = metric.sqrtG * ( Q[idx_hu2] * u1 + 0.5 * gravity * metric.H_contra_21 * hsquared )
      flux_x2[idx_hu2] = metric.sqrtG * ( Q[idx_hu2] * u2 + 0.5 * gravity * metric.H_contra_22 * hsquared )


      flux_x1_new = xp.empty_like(Q_new)
      flux_x2_new = xp.empty_like(Q_new)

      flux_x1_new[idx_h] = metric.sqrtG_new * Q_new[idx_hu1]
      flux_x2_new[idx_h] = metric.sqrtG_new * Q_new[idx_hu2]

      hsquared_new = Q_new[idx_h]**2
      flux_x1_new[idx_hu1] = metric.sqrtG_new * ( Q_new[idx_hu1] * u1_new + 0.5 * gravity * metric.H_contra_11_new * hsquared_new )
      flux_x2_new[idx_hu1] = metric.sqrtG_new * ( Q_new[idx_hu1] * u2_new + 0.5 * gravity * metric.H_contra_12_new * hsquared_new )

      flux_x1_new[idx_hu2] = metric.sqrtG_new * ( Q_new[idx_hu2] * u1_new + 0.5 * gravity * metric.H_contra_21_new * hsquared_new )
      flux_x2_new[idx_hu2] = metric.sqrtG_new * ( Q_new[idx_hu2] * u2_new + 0.5 * gravity * metric.H_contra_22_new * hsquared_new )

      # if rank == 0:
      #    print(f' sqrtG diff = \n{metric.sqrtG_new - geom._to_new(metric.sqrtG)}')
      #    print(f'flux x1 old = \n{flux_x1}')
      #    print(f'flux_x1 old with new shape = \n{geom._to_new(flux_x1)}')
      #    print(f'flux x1 new = \n{flux_x1_new}')

      diff = flux_x1_new - geom._to_new(flux_x1)
      diff_norm = xp.linalg.norm(diff) / xp.linalg.norm(flux_x1)
      if diff_norm > 1e-10:
         print(f'{rank} flux x1 values are different: \n'
               f'{diff}')
         raise ValueError
      diff = flux_x2_new - geom._to_new(flux_x2)
      diff_norm = xp.linalg.norm(diff) / xp.linalg.norm(flux_x2)
      if diff_norm > 1e-10:
         print(f'{rank} flux x2 values are different: \n'
               f'{diff}')
         raise ValueError

      # Interior contribution to the derivatives, corrections for the boundaries will be added later
      for elem in range(nb_elements_hori):
         epais = elem * nbsolpts + xp.arange(nbsolpts)

         # --- Direction x1
         df1_dx1[:,:,epais] = flux_x1[:,:,epais] @ mtrx.diff_solpt_tr

         # --- Direction x2
         df2_dx2[:,epais,:] = mtrx.diff_solpt @ flux_x2[:,epais,:]

      df1_dx1_new = flux_x1_new @ mtrx.derivative_x
      df2_dx2_new = flux_x2_new @ mtrx.derivative_y

      diff = df1_dx1_new - geom._to_new(df1_dx1)
      diff_norm = xp.linalg.norm(diff) / xp.linalg.norm(df1_dx1)
      if diff_norm > 1e-10:
         print(f'{rank} df1_dx1 values are different: \n'
               f'{diff}')
         raise ValueError
      diff = df2_dx2_new - geom._to_new(df2_dx2)
      diff_norm = xp.linalg.norm(diff) / xp.linalg.norm(df2_dx2)
      if diff_norm > 1e-10:
         print(f'{rank} df2_dx2 values are different: \n'
               f'{diff}')
         raise ValueError

      # Finish transfers
      (var_itf_j[idx_hu1, 0, 1], var_itf_j[idx_hu2, 0, 1]), (var_itf_j[idx_hu1, -1, 0], var_itf_j[idx_hu2, -1, 0]), \
      (var_itf_i[idx_hu1, 0, 1], var_itf_i[idx_hu2, 0, 1]), (var_itf_i[idx_hu1, -1, 0], var_itf_i[idx_hu2, -1, 0]) = request_u.wait()
      var_itf_j[idx_h, 0, 1], var_itf_j[idx_h, -1, 0], var_itf_i[idx_h, 0, 1], var_itf_i[idx_h, -1, 0] = request_h.wait()

      # TODO make this simpler
      (s1, s2), (n1, n2), (w1, w2), (e1, e2) = request_u_new.wait()
      vj_tmp[idx_hu1,  0,  :, 1, :] = s1.reshape(nb_elements_hori, nbsolpts)
      vj_tmp[idx_hu2,  0,  :, 1, :] = s2.reshape(nb_elements_hori, nbsolpts)
      vj_tmp[idx_hu1, -1,  :, 0, :] = n1.reshape(nb_elements_hori, nbsolpts)
      vj_tmp[idx_hu2, -1,  :, 0, :] = n2.reshape(nb_elements_hori, nbsolpts)
      vi_tmp[idx_hu1,  :,  0, 1, :] = w1.reshape(nb_elements_hori, nbsolpts)
      vi_tmp[idx_hu2,  :,  0, 1, :] = w2.reshape(nb_elements_hori, nbsolpts)
      vi_tmp[idx_hu1,  :, -1, 0, :] = e1.reshape(nb_elements_hori, nbsolpts)
      vi_tmp[idx_hu2,  :, -1, 0, :] = e2.reshape(nb_elements_hori, nbsolpts)

      s, n, w, e = request_h_new.wait()
      vj_tmp[idx_h, 0, :, 1, :]  = s.reshape(nb_elements_hori, nbsolpts)
      vj_tmp[idx_h, -1, :, 0, :] = n.reshape(nb_elements_hori, nbsolpts)
      vi_tmp[idx_h, :, 0, 1, :]  = w.reshape(nb_elements_hori, nbsolpts)
      vi_tmp[idx_h, :, -1, 0, :] = e.reshape(nb_elements_hori, nbsolpts)

      # if rank == 1:
      #    print(f'Old itf i: \n{var_itf_i[0]}')
      #    print(f'Old itf with new shape: \n{to_new_itf_i(var_itf_i)[0]}')
      #    print(f'New itf i: \n{var_itf_i_new[0]}')
      #    print(f'diff = \n{var_itf_i_new[0] - to_new_itf_i(var_itf_i)[0]}')

      #    print(f'received south = {s}')
      #    print(f'Old itf j: \n{var_itf_j[0]}')
      #    print(f'Old itf with new shape: \n{to_new_itf_j(var_itf_j)[0]}')
      #    print(f'New itf j: \n{var_itf_j_new[0]}')
      #    print(f'diff = \n{var_itf_j_new[0] - to_new_itf_j(var_itf_j)[0]}')

      diff = var_itf_i_new - to_new_itf_i(var_itf_i)
      diff_norm = xp.linalg.norm(diff) / xp.linalg.norm(var_itf_i)
      if diff_norm > 1e-10:
         print(f'{rank} itf i values are different: \n'
               f'{diff}')
         raise ValueError

      diff = var_itf_j_new - to_new_itf_j(var_itf_j)
      diff_norm = xp.linalg.norm(diff) / xp.linalg.norm(var_itf_j)
      if diff_norm > 1e-10:
         print(f'{rank} itf j values are different: \n'
               f'{diff}')
         raise ValueError


      # Substract topo after extrapolation
      if topo is not None:
         var_itf_i[idx_h] -= topo.hsurf_itf_i
         var_itf_j[idx_h] -= topo.hsurf_itf_j

         var_itf_i_new[idx_h] -= to_new_itf_i(topo.hsurf_itf_i)
         var_itf_j_new[idx_h] -= to_new_itf_j(topo.hsurf_itf_j)

      # West and east are defined relative to the elements, *not* to the interface itself.
      # Therefore, a certain interface will be the western interface of its eastern element and vice-versa
      #
      #   western-elem   itf  eastern-elem
      #   ________________|_____________________|
      #                   |
      #   west .  east -->|<-- west  .  east -->
      #                   |
      west = xp.s_[..., 1:,  :nbsolpts]
      east = xp.s_[..., :-1, nbsolpts:]
      south = xp.s_[..., nb_elements_hori:,  :nbsolpts]
      north = xp.s_[..., :-nb_elements_hori, nbsolpts:]

      a = xp.sqrt(gravity * var_itf_i_new[idx_h] * metric.H_contra_11_itf_i_new)   
      m = var_itf_i_new[idx_hu1] / (var_itf_i_new[idx_h] * a)
      m[xp.where(xp.isnan(m))] = 0.0
      # if rank == 0:
      #    print(f'm = \n{m}')

      big_M = 0.25 * ((m[east] + 1.)**2 - (m[west] - 1.)**2)

      flux_x1_itf_new = xp.zeros_like(var_itf_i_new)
      # ------ Advection part
      flux_x1_itf_new[east] = metric.sqrtG_itf_i_new[east] * (
                                             xp.maximum(0., big_M) * a[east] * var_itf_i_new[east] +
                                             xp.minimum(0., big_M) * a[west] * var_itf_i_new[west] )
      # ------ Pressure part
      p11 = metric.sqrtG_itf_i_new * (0.5 * gravity) * metric.H_contra_11_itf_i_new * var_itf_i_new[idx_h]**2
      p21 = metric.sqrtG_itf_i_new * (0.5 * gravity) * metric.H_contra_21_itf_i_new * var_itf_i_new[idx_h]**2
      flux_x1_itf_new[idx_hu1, :-1, nbsolpts:] += 0.5 * ( (1. + m[east]) * p11[east] + (1. - m[west]) * p11[west] )
      flux_x1_itf_new[idx_hu2, :-1, nbsolpts:] += 0.5 * ( (1. + m[east]) * p21[east] + (1. - m[west]) * p21[west] )

      # ------ Copy to west interface of eastern element
      flux_x1_itf_new[west] = flux_x1_itf_new[east]

      pam_old = xp.zeros_like(var_itf_i)
      p21_old = xp.zeros_like(var_itf_i)

      # Common AUSM fluxes
      for itf in range(nb_interfaces_hori):

         elem_L = itf
         elem_R = itf + 1

         ################
         # Direction x1 #
         ################

         # Left state
         p11_L = metric.sqrtG_itf_i[itf, :] * 0.5 * gravity * metric.H_contra_11_itf_i[itf, :] * var_itf_i[idx_h, elem_L, 1, :]**2
         p21_L = metric.sqrtG_itf_i[itf, :] * 0.5 * gravity * metric.H_contra_21_itf_i[itf, :] * var_itf_i[idx_h, elem_L, 1, :]**2
         aL = xp.sqrt( gravity * var_itf_i[idx_h, elem_L, 1, :] * metric.H_contra_11_itf_i[itf, :] )
         mL = var_itf_i[idx_hu1, elem_L, 1, :] / (var_itf_i[idx_h, elem_L, 1, :] * aL)

         # Right state
         p11_R = metric.sqrtG_itf_i[itf, :] * 0.5 * gravity * metric.H_contra_11_itf_i[itf, :] * var_itf_i[idx_h, elem_R, 0, :]**2
         p21_R = metric.sqrtG_itf_i[itf, :] * 0.5 * gravity * metric.H_contra_21_itf_i[itf, :] * var_itf_i[idx_h, elem_R, 0, :]**2
         aR = xp.sqrt( gravity * var_itf_i[idx_h, elem_R, 0, :] * metric.H_contra_11_itf_i[itf, :] )
         mR = var_itf_i[idx_hu1, elem_R, 0, :] / (var_itf_i[idx_h, elem_R, 0, :] * aR)

         pam_old[0, elem_L, 1, :] = p11_L
         p21_old[0, elem_L, 1, :] = p21_L
         pam_old[0, elem_R, 0, :] = p11_R
         p21_old[0, elem_R, 0, :] = p21_R

         pam_old[1, elem_L, 1, :] = aL
         pam_old[1, elem_R, 0, :] = aR
         pam_old[2, elem_L, 1, :] = mL
         pam_old[2, elem_R, 0, :] = mR

         M = 0.25 * ( (mL + 1.)**2 - (mR - 1.)**2 )

         # --- Advection part

         flux_x1_itf_i[:, elem_L, :, 1] = metric.sqrtG_itf_i[itf, :] * ( xp.maximum(0., M) * aL * var_itf_i[:, elem_L, 1, :] +  xp.minimum(0., M) * aR * var_itf_i[:, elem_R, 0, :] )

         # --- Pressure part

         flux_x1_itf_i[idx_hu1, elem_L, :, 1] += 0.5 * ( (1. + mL) * p11_L + (1. - mR) * p11_R )
         flux_x1_itf_i[idx_hu2, elem_L, :, 1] += 0.5 * ( (1. + mL) * p21_L + (1. - mR) * p21_R )

         flux_x1_itf_i[:, elem_R, :, 0] = flux_x1_itf_i[:, elem_L, :, 1]


      diff11 = p11 - to_new_itf_i(pam_old)[0]
      diff21 = p21 - to_new_itf_i(p21_old)[0]
      if xp.linalg.norm(diff11) / xp.linalg.norm(p11) > 1e-10 or \
         xp.linalg.norm(diff21) / xp.linalg.norm(p21) > 1e-10:
         print(f'{rank} p11/21 different: \ndiff11 = \n{diff11}\ndiff21 = \n{diff21}')
         raise ValueError

      diffa = a - to_new_itf_i(pam_old)[1]
      diffm = m - to_new_itf_i(pam_old)[2]
      if xp.linalg.norm(diffa) / xp.linalg.norm(a) > 1e-10 or \
         xp.linalg.norm(diffm) / xp.linalg.norm(m) > 1e-10:
         print(f'a/m different \ndiffa = \n{diffa}\ndiffm = \n{diffm}')
         raise ValueError

      diff = flux_x1_itf_new - to_new_itf_if(flux_x1_itf_i)
      d1 = xp.linalg.norm(diff)
      d2 = xp.linalg.norm(flux_x1_itf_i)
      diff_norm = xp.linalg.norm(diff) / xp.linalg.norm(flux_x1_itf_i)
      if diff_norm > 1e-10:
         if rank == 0: print(f'p11 old: \n{pam_old[0]}\np11 new:\n{p11}\n'
               f'flux x1 old: \n{flux_x1_itf_i[1]}\nflux x1 itf new = \n{flux_x1_itf_new[1]}\n'
               f'{rank} flux itf i values are different ({diff_norm:.3e}, {d1:.2e}/{d2:.2e}): \n'
               f'{diff[1]}')
         raise ValueError

      a = xp.sqrt(gravity * var_itf_j_new[idx_h] * metric.H_contra_22_itf_j_new)
      m = var_itf_j_new[idx_hu2] / (var_itf_j_new[idx_h] * a)
      m[xp.where(xp.isnan(m))] = 0.0
      big_M = 0.25 * ((m[north] + 1.)**2 - (m[south] - 1.)**2)

      flux_x2_itf_new = xp.zeros_like(var_itf_j_new)
      # ------ Advection part
      flux_x2_itf_new[north] = metric.sqrtG_itf_j_new[north] * (
                                 xp.maximum(0., big_M) * a[north] * var_itf_j_new[north] +
                                 xp.minimum(0., big_M) * a[south] * var_itf_j_new[south] )
      # ------ Pressure part
      p12 = metric.sqrtG_itf_j_new * (0.5 * gravity) * metric.H_contra_12_itf_j_new * var_itf_j_new[idx_h]**2
      p22 = metric.sqrtG_itf_j_new * (0.5 * gravity) * metric.H_contra_22_itf_j_new * var_itf_j_new[idx_h]**2
      flux_x2_itf_new[idx_hu1, :-nb_elements_hori, nbsolpts:] += 0.5 * ( (1. + m[north]) * p12[north] + (1. - m[south]) * p12[south] )
      flux_x2_itf_new[idx_hu2, :-nb_elements_hori, nbsolpts:] += 0.5 * ( (1. + m[north]) * p22[north] + (1. - m[south]) * p22[south] )
      # ------ Copy to south interface of northern element
      flux_x2_itf_new[south] = flux_x2_itf_new[north]

      old_a = xp.zeros_like(var_itf_j[0])
      old_m = xp.zeros_like(var_itf_j[0])
      old_bigm = xp.zeros_like(var_itf_j[0])

      for itf in range(nb_interfaces_hori):

         elem_L = itf
         elem_R = itf + 1

         ################
         # Direction x2 #
         ################

         # Left state
         p12_L = metric.sqrtG_itf_j[itf, :] * 0.5 * gravity * metric.H_contra_12_itf_j[itf, :] * var_itf_j[idx_h, elem_L, 1, :]**2
         p22_L = metric.sqrtG_itf_j[itf, :] * 0.5 * gravity * metric.H_contra_22_itf_j[itf, :] * var_itf_j[idx_h, elem_L, 1, :]**2
         aL = xp.sqrt( gravity * var_itf_j[idx_h, elem_L, 1, :] * metric.H_contra_22_itf_j[itf, :] )
         mL = var_itf_j[idx_hu2, elem_L, 1, :] / (var_itf_j[idx_h, elem_L, 1, :] * aL)

         # Right state
         p12_R = metric.sqrtG_itf_j[itf, :] * 0.5 * gravity * metric.H_contra_12_itf_j[itf, :] * var_itf_j[idx_h, elem_R, 0, :]**2
         p22_R = metric.sqrtG_itf_j[itf, :] * 0.5 * gravity * metric.H_contra_22_itf_j[itf, :] * var_itf_j[idx_h, elem_R, 0, :]**2
         aR = xp.sqrt( gravity * var_itf_j[idx_h, elem_R, 0, :] * metric.H_contra_22_itf_j[itf, :] )
         mR = var_itf_j[idx_hu2, elem_R, 0, :] / (var_itf_j[idx_h, elem_R, 0, :] * aR)

         M = 0.25 * ( (mL + 1.)**2 - (mR - 1.)**2 )
         old_a[elem_R, 0, :] = aR
         old_a[elem_L, 1, :] = aL
         old_m[elem_R, 0, :] = mR
         old_m[elem_L, 1, :] = mL

         old_bigm[elem_L, 1, :] = M

         # --- Advection part

         flux_x2_itf_j[:, elem_L, 1, :] = metric.sqrtG_itf_j[itf, :] * ( xp.maximum(0., M) * aL * var_itf_j[:, elem_L, 1, :] + xp.minimum(0., M) * aR * var_itf_j[:, elem_R, 0, :] )

         # --- Pressure part

         flux_x2_itf_j[idx_hu1, elem_L, 1, :] += 0.5 * ( (1. + mL) * p12_L + (1. - mR) * p12_R )
         flux_x2_itf_j[idx_hu2, elem_L, 1, :] += 0.5 * ( (1. + mL) * p22_L + (1. - mR) * p22_R )

         flux_x2_itf_j[:, elem_R, 0, :] = flux_x2_itf_j[:, elem_L, 1, :]


      old_pam = xp.zeros_like(var_itf_j)
      old_pam[0] = old_a
      old_pam[1] = old_m

      # if rank == 0:
      #    print(f'old m = \n{old_m}\nnew M = \n{m}')
      #    print(f'm south = \n{m[south]}\nm north = \n{m[north]}')
      #    print(f'old big M = \n{old_bigm}\nNew big M = \n{big_M}')

      old_a = to_new_itf_j(old_pam)[0]
      old_m = to_new_itf_j(old_pam)[1]

      diffa = a - old_a
      diffm = m - old_m

      if xp.linalg.norm(diffa) > 1e-10 or xp.linalg.norm(diffm) > 1e-10:
         print(f'diff a/m (j): \n{diffa}\n{diffm}')
         raise ValueError

      diff = flux_x2_itf_new - to_new_itf_j(flux_x2_itf_j)
      diff_norm = xp.linalg.norm(diff) / xp.linalg.norm(flux_x2_itf_j)
      if diff_norm > 1e-10:
         print(
            f'flux x2 old: \n{flux_x2_itf_j}\nflux x2 itf new = \n{flux_x2_itf_new}\n'
            f'{rank} flux itf j values are different: \n{diff}')
         raise ValueError

      # if rank == 0:
      #    print(f'shapes: df1dx1 = {df1_dx1_new.shape}, flux x1 = {flux_x1_itf_new.shape},'
      #          f' op = {mtrx.correction_WE.shape}, product = {(flux_x1_itf_new @ mtrx.correction_WE).shape}')

      df1_dx1_new[...] += middle_itf_i(flux_x1_itf_new) @ mtrx.correction_WE
      df2_dx2_new[...] += middle_itf_j(flux_x2_itf_new) @ mtrx.correction_SN

      # Compute the derivatives
      for elem in range(nb_elements_hori):
         epais = elem * nbsolpts + xp.arange(nbsolpts)

         # --- Direction x1

         df1_dx1[:,:,epais] += flux_x1_itf_i[:, elem+offset,:,:] @ mtrx.correction_tr

         # --- Direction x2

         df2_dx2[:,epais,:] += mtrx.correction @ flux_x2_itf_j[:, elem+offset,:,:]

      diff = df1_dx1_new - geom._to_new(df1_dx1)
      diff_norm = xp.linalg.norm(diff) / xp.linalg.norm(df1_dx1)
      if diff_norm > 1e-10:
         print(f'{rank} df1_dx1 values are different (2): \n'
               f'{diff}')
         raise ValueError
      diff = df2_dx2_new - geom._to_new(df2_dx2)
      diff_norm = xp.linalg.norm(diff) / xp.linalg.norm(df2_dx2)
      if diff_norm > 1e-10:
         print(f'{rank} df2_dx2 values are different (2): \n'
               f'{diff}')
         raise ValueError

      if topo is None:
         topo_dzdx1 = xp.zeros_like(metric.H_contra_11)
         topo_dzdx2 = xp.zeros_like(metric.H_contra_11)

         topo_dzdx1_new = xp.zeros_like(metric.H_contra_11_new)
         topo_dzdx2_new = xp.zeros_like(metric.H_contra_11_new)

      else:
         topo_dzdx1 = topo.dzdx1
         topo_dzdx2 = topo.dzdx2

         topo_dzdx1_new = geom._to_new(topo.dzdx1)
         topo_dzdx2_new = geom._to_new(topo.dzdx2)

      # Add coriolis, metric and terms due to varying bottom topography
      # Note: christoffel_1_22 and metric.christoffel_2_11 are zero
      forcing[idx_hu1,:,:] = 2.0 * ( metric.christoffel_1_01 * Q[idx_hu1] + metric.christoffel_1_02 * Q[idx_hu2]) \
            + metric.christoffel_1_11 * Q[idx_hu1] * u1 + 2.0 * metric.christoffel_1_12 * Q[idx_hu1] * u2 \
            + gravity * Q[idx_h] * ( metric.H_contra_11 * topo_dzdx1 + metric.H_contra_12 * topo_dzdx2)

      forcing[idx_hu2,:,:] = 2.0 * (metric.christoffel_2_01 * Q[idx_hu1] + metric.christoffel_2_02 * Q[idx_hu2]) \
            + 2.0 * metric.christoffel_2_12 * Q[idx_hu1] * u2 + metric.christoffel_2_22 * Q[idx_hu2] * u2 \
            + gravity * Q[idx_h] * ( metric.H_contra_21 * topo_dzdx1 + metric.H_contra_22 * topo_dzdx2)

      forcing_new = xp.zeros_like(Q_new)
      forcing_new[idx_hu1] = 2.0 * ( metric.christoffel_1_01_new * Q_new[idx_hu1] + metric.christoffel_1_02_new * Q_new[idx_hu2]) \
            + metric.christoffel_1_11_new * Q_new[idx_hu1] * u1_new + 2.0 * metric.christoffel_1_12_new * Q_new[idx_hu1] * u2_new \
            + gravity * Q_new[idx_h] * ( metric.H_contra_11_new * topo_dzdx1_new + metric.H_contra_12_new * topo_dzdx2_new)
      forcing_new[idx_hu2] = 2.0 * (metric.christoffel_2_01_new * Q_new[idx_hu1] + metric.christoffel_2_02_new * Q_new[idx_hu2]) \
            + 2.0 * metric.christoffel_2_12_new * Q_new[idx_hu1] * u2_new + metric.christoffel_2_22_new * Q_new[idx_hu2] * u2_new \
            + gravity * Q_new[idx_h] * ( metric.H_contra_21_new * topo_dzdx1_new + metric.H_contra_22_new * topo_dzdx2_new)

      diff = forcing_new - geom._to_new(forcing)
      diff_norm = xp.linalg.norm(diff) / xp.linalg.norm(forcing)
      if diff_norm > 1e-10:
         print(f'{rank} different forcings ({diff_norm:.2e}): \n{diff}')
         raise ValueError

      # Assemble the right-hand sides
      rhs = metric.inv_sqrtG * - ( df1_dx1 + df2_dx2 ) - forcing
      rhs_new = metric.inv_sqrtG_new * (-df1_dx1_new - df2_dx2_new) - forcing_new

      diff = rhs_new - geom._to_new(rhs)
      diff_norm = xp.linalg.norm(diff) / xp.linalg.norm(rhs)
      if diff_norm > 1e-10:
         print(f'different rhs!: \n{diff}')
         raise ValueError

      return rhs
