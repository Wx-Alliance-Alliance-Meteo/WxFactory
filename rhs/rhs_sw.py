from numpy.typing import NDArray
from typing import Optional, Tuple

from init.initialize import Topo
from common.definitions import idx_h, idx_hu1, idx_hu2, gravity
from common.device import CudaDevice, Device, default_device
from common.process_topology import ProcessTopology
from geometry import CubedSphere, DFROperators, Metric3DTopo
from rhs.rhs import RHS

class RhsShallowWater(RHS):
   def __init__(self, shape: Tuple[int, ...],
                geom: CubedSphere,
                mtrx: DFROperators,
                metric: Metric3DTopo,
                topo: Optional[Topo],
                ptopo: ProcessTopology,
                nbsolpts: int,
                nb_elements_hori: int,
                device: Device = default_device):
      super().__init__(shape,
                       geom, mtrx, metric, topo, ptopo,
                       nbsolpts, nb_elements_hori,
                       device)
      # TODO : fuse functions here if possible

   def __compute_rhs__(self,
                       Q: NDArray,
                       geom: CubedSphere,
                       mtrx: DFROperators,
                       metric: Metric3DTopo,
                       topo: Optional[Topo],
                       ptopo: ProcessTopology,
                       nbsolpts: int,
                       nb_elements_hori: int,
                       device: Device = default_device):
      '''Evaluate the right-hand side of the two dimensional Shallow Water equations

      '''
      
      xp = device.xp

      type_vec = Q.dtype
      nb_equations = Q.shape[0]
      nb_interfaces_hori = nb_elements_hori + 1

      df1_dx1, df2_dx2, flux_x1, flux_x2 = [xp.empty_like(Q, dtype=type_vec) for _ in range(4)]

      flux_x1_itf_i = xp.empty((nb_equations, nb_elements_hori+2, nbsolpts*nb_elements_hori, 2), dtype=type_vec)
      flux_x2_itf_j, var_itf_i, var_itf_j= [xp.empty((nb_equations, nb_elements_hori+2, 2, nbsolpts*nb_elements_hori), dtype=type_vec) for _ in range(3)]

      forcing = xp.zeros_like(Q, dtype=type_vec)

      # Offset due to the halo
      offset = 1

      # Unpack dynamical variables
      HH = Q[idx_h] if topo is None else Q[idx_h] + topo.hsurf
      u1 = Q[idx_hu1] / Q[idx_h]
      u2 = Q[idx_hu2] / Q[idx_h]

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

      # Initiate transfers
      all_request = ptopo.xchange_sw_interfaces(geom, var_itf_i[idx_h], var_itf_j[idx_h], var_itf_i[idx_hu1], var_itf_i[idx_hu2], var_itf_j[idx_hu1], var_itf_j[idx_hu2], blocking=False, device=device)

      # Compute the fluxes
      flux_x1[idx_h] = metric.sqrtG * Q[idx_hu1]
      flux_x2[idx_h] = metric.sqrtG * Q[idx_hu2]

      hsquared = Q[idx_h]**2
      flux_x1[idx_hu1] = metric.sqrtG * ( Q[idx_hu1] * u1 + 0.5 * gravity * metric.H_contra_11 * hsquared )
      flux_x2[idx_hu1] = metric.sqrtG * ( Q[idx_hu1] * u2 + 0.5 * gravity * metric.H_contra_12 * hsquared )

      flux_x1[idx_hu2] = metric.sqrtG * ( Q[idx_hu2] * u1 + 0.5 * gravity * metric.H_contra_21 * hsquared )
      flux_x2[idx_hu2] = metric.sqrtG * ( Q[idx_hu2] * u2 + 0.5 * gravity * metric.H_contra_22 * hsquared )

      # Interior contribution to the derivatives, corrections for the boundaries will be added later
      for elem in range(nb_elements_hori):
         epais = elem * nbsolpts + xp.arange(nbsolpts)

         # --- Direction x1
         df1_dx1[:,:,epais] = flux_x1[:,:,epais] @ mtrx.diff_solpt_tr

         # --- Direction x2
         df2_dx2[:,epais,:] = mtrx.diff_solpt @ flux_x2[:,epais,:]

      # Finish transfers
      all_request.wait()

      # Substract topo after extrapolation
      if topo is not None:
         var_itf_i[idx_h] -= topo.hsurf_itf_i
         var_itf_j[idx_h] -= topo.hsurf_itf_j

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

         M = 0.25 * ( (mL + 1.)**2 - (mR - 1.)**2 )

         # --- Advection part

         flux_x1_itf_i[:, elem_L, :, 1] = metric.sqrtG_itf_i[itf, :] * ( xp.maximum(0., M) * aL * var_itf_i[:, elem_L, 1, :] +  xp.minimum(0., M) * aR * var_itf_i[:, elem_R, 0, :] )

         # --- Pressure part

         flux_x1_itf_i[idx_hu1, elem_L, :, 1] += 0.5 * ( (1. + mL) * p11_L + (1. - mR) * p11_R )
         flux_x1_itf_i[idx_hu2, elem_L, :, 1] += 0.5 * ( (1. + mL) * p21_L + (1. - mR) * p21_R )

         flux_x1_itf_i[:, elem_R, :, 0] = flux_x1_itf_i[:, elem_L, :, 1]

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

         # --- Advection part

         flux_x2_itf_j[:, elem_L, 1, :] = metric.sqrtG_itf_j[itf, :] * ( xp.maximum(0., M) * aL * var_itf_j[:, elem_L, 1, :] + xp.minimum(0., M) * aR * var_itf_j[:, elem_R, 0, :] )

         # --- Pressure part

         flux_x2_itf_j[idx_hu1, elem_L, 1, :] += 0.5 * ( (1. + mL) * p12_L + (1. - mR) * p12_R )
         flux_x2_itf_j[idx_hu2, elem_L, 1, :] += 0.5 * ( (1. + mL) * p22_L + (1. - mR) * p22_R )

         flux_x2_itf_j[:, elem_R, 0, :] = flux_x2_itf_j[:, elem_L, 1, :]

      # Compute the derivatives
      for elem in range(nb_elements_hori):
         epais = elem * nbsolpts + xp.arange(nbsolpts)

         # --- Direction x1

         df1_dx1[:,:,epais] += flux_x1_itf_i[:, elem+offset,:,:] @ mtrx.correction_tr

         # --- Direction x2

         df2_dx2[:,epais,:] += mtrx.correction @ flux_x2_itf_j[:, elem+offset,:,:]

      if topo is None:
         topo_dzdx1 = xp.zeros_like(metric.H_contra_11)
         topo_dzdx2 = xp.zeros_like(metric.H_contra_11)
      else:
         topo_dzdx1 = topo.dzdx1
         topo_dzdx2 = topo.dzdx2

      # Add coriolis, metric and terms due to varying bottom topography
      # Note: christoffel_1_22 and metric.christoffel_2_11 are zero
      forcing[idx_hu1,:,:] = 2.0 * ( metric.christoffel_1_01 * Q[idx_hu1] + metric.christoffel_1_02 * Q[idx_hu2]) \
            + metric.christoffel_1_11 * Q[idx_hu1] * u1 + 2.0 * metric.christoffel_1_12 * Q[idx_hu1] * u2 \
            + gravity * Q[idx_h] * ( metric.H_contra_11 * topo_dzdx1 + metric.H_contra_12 * topo_dzdx2)

      forcing[idx_hu2,:,:] = 2.0 * (metric.christoffel_2_01 * Q[idx_hu1] + metric.christoffel_2_02 * Q[idx_hu2]) \
            + 2.0 * metric.christoffel_2_12 * Q[idx_hu1] * u2 + metric.christoffel_2_22 * Q[idx_hu2] * u2 \
            + gravity * Q[idx_h] * ( metric.H_contra_21 * topo_dzdx1 + metric.H_contra_22 * topo_dzdx2)

      # Assemble the right-hand sides
      rhs = metric.inv_sqrtG * - ( df1_dx1 + df2_dx2 ) - forcing

      return rhs
