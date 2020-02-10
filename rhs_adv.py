import numpy

from definitions import idx_h, idx_u1, idx_u2
from xchange import xchange_scalars, xchange_vectors

def rhs_adv(Q, geom, mtrx, metric, cube_face, nbsolpts, nb_elements_horiz, α):

   type_vec = type(Q[0, 0, 0])

   nb_interfaces_horiz = nb_elements_horiz + 1

   df1_dx1 = numpy.zeros_like(Q, dtype=type_vec)
   df2_dx2 = numpy.zeros_like(Q, dtype=type_vec)

   flux_Eq1_itf_j = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   h_itf_j        = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   u1_itf_j       = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   u2_itf_j       = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)

   flux_Eq1_itf_i = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   h_itf_i        = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   u1_itf_i       = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   u2_itf_i       = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)

   eig_L          = numpy.zeros(nbsolpts*nb_elements_horiz)
   eig_R          = numpy.zeros(nbsolpts*nb_elements_horiz)
   eig            = numpy.zeros(nbsolpts*nb_elements_horiz)

   flux_L         = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=type_vec)
   flux_R         = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=type_vec)

   # Unpack physical variables
   h        = Q[idx_h, :, :]
   u1       = Q[idx_u1, :, :]
   u2       = Q[idx_u2, :, :]

   # Compute the fluxes
   flux_Eq1_x1 = h * metric.sqrtG * u1
   flux_Eq1_x2 = h * metric.sqrtG * u2

   # Offset due to the halo
   offset = 1

   # Interpolate to the element interface
   for elem in range(nb_elements_horiz):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      pos = elem + offset

      # --- Direction x1
      
      flux_Eq1_itf_i[pos, 0, :] = flux_Eq1_x1[:, epais] @ mtrx.extrap_west
      flux_Eq1_itf_i[pos, 1, :] = flux_Eq1_x1[:, epais] @ mtrx.extrap_east

      h_itf_i[pos, 0, :] = h[:, epais] @ mtrx.extrap_west 
      h_itf_i[pos, 1, :] = h[:, epais] @ mtrx.extrap_east

      u1_itf_i[pos, 0, :] = u1[:, epais] @ mtrx.extrap_west
      u1_itf_i[pos, 1, :] = u1[:, epais] @ mtrx.extrap_east

      u2_itf_i[pos, 0, :] = u2[:, epais] @ mtrx.extrap_west
      u2_itf_i[pos, 1, :] = u2[:, epais] @ mtrx.extrap_east

      # --- Direction x2

      flux_Eq1_itf_j[pos, 0, :] = mtrx.extrap_south @ flux_Eq1_x2[epais, :]
      flux_Eq1_itf_j[pos, 1, :] = mtrx.extrap_north @ flux_Eq1_x2[epais, :]

      h_itf_j[pos, 0, :] = mtrx.extrap_south @ h[epais, :]
      h_itf_j[pos, 1, :] = mtrx.extrap_north @ h[epais, :]

      u1_itf_j[pos, 0, :] = mtrx.extrap_south @ u1[epais, :]
      u1_itf_j[pos, 1, :] = mtrx.extrap_north @ u1[epais, :]

      u2_itf_j[pos, 0, :] = mtrx.extrap_south @ u2[epais, :]
      u2_itf_j[pos, 1, :] = mtrx.extrap_north @ u2[epais, :]

   xchange_scalars(geom, h_itf_i, h_itf_j, cube_face)
   xchange_vectors(geom, u1_itf_i, u2_itf_i, u1_itf_j, u2_itf_j, cube_face)

   # Common Rusanov fluxes
   for itf in range(nb_interfaces_horiz):

      elem_L = itf
      elem_R = itf + 1

      # --- Direction x1

      eig_L[:] = numpy.abs(u1_itf_i[elem_L, 1, :])
      eig_R[:] = numpy.abs(u1_itf_i[elem_R, 0, :])
      eig[:]   = numpy.maximum(eig_L, eig_R)

      flux_L[:] = h_itf_i[elem_L, 1, :] * u1_itf_i[elem_L, 1, :] * metric.sqrtG_itf_i[:, itf]
      flux_R[:] = h_itf_i[elem_R, 0, :] * u1_itf_i[elem_R, 0, :] * metric.sqrtG_itf_i[:, itf]

      flux_Eq1_itf_i[elem_L, 1, :] = 0.5 * ( flux_L  + flux_R - eig * metric.sqrtG_itf_i[:, itf] * ( h_itf_i[elem_R, 0, :]  - h_itf_i[elem_L, 1, :] ) )
      flux_Eq1_itf_i[elem_R, 0, :] = flux_Eq1_itf_i[elem_L, 1, :]

      # --- Direction x2

      eig_L[:] = numpy.abs(u2_itf_j[elem_L, 1, :])
      eig_R[:] = numpy.abs(u2_itf_j[elem_R, 0, :])
      eig[:]   = numpy.maximum(eig_L, eig_R)

      flux_L[:] = h_itf_j[elem_L, 1, :] * u2_itf_j[elem_L, 1, :] * metric.sqrtG_itf_j[itf, :]
      flux_R[:] = h_itf_j[elem_R, 0, :] * u2_itf_j[elem_R, 0, :] * metric.sqrtG_itf_j[itf, :]

      flux_Eq1_itf_j[elem_L, 1, :] = 0.5 * ( flux_L + flux_R - eig * metric.sqrtG_itf_j[itf, :] * ( h_itf_j[elem_R, 0, :] - h_itf_j[elem_L, 1, :] ) )
      flux_Eq1_itf_j[elem_R, 0, :] = flux_Eq1_itf_j[elem_L, 1, :]

   # Compute the derivatives
   for elem in range(nb_elements_horiz):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      # --- Direction x1

      df1_dx1[idx_h,:,epais] = mtrx.diff_solpt @ flux_Eq1_x1[:,epais].T + mtrx.correction @ flux_Eq1_itf_i[elem+offset,:,:] # TODO : éviter la transpose

      # --- Direction x2

      df2_dx2[idx_h,epais,:] = mtrx.diff_solpt @ flux_Eq1_x2[epais,:] + mtrx.correction @ flux_Eq1_itf_j[elem+offset,:,:]
      
   # Assemble the right-hand sides
   rhs = metric.inv_sqrtG * ( - df1_dx1 - df2_dx2 )

   rhs[1:,:,:] = numpy.zeros_like(rhs[1:,:,:])

   return rhs
