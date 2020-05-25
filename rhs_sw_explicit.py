import numpy

from definitions import idx_h, idx_hu1, idx_hu2, idx_u1, idx_u2, gravity
from parallel import xchange_scalars, xchange_vectors

def rhs_sw_explicit(Q, geom, mtrx, metric, topo, comm_dist_graph, nbsolpts, nb_elements_horiz, α, case_number):

   type_vec = type(Q[0, 0, 0])

   shallow_water_equations = ( case_number > 1 )

   nb_interfaces_horiz = nb_elements_horiz + 1

   df1_dx1 = numpy.zeros_like(Q, dtype=type_vec)
   df2_dx2 = numpy.zeros_like(Q, dtype=type_vec)
   forcing = numpy.zeros_like(Q, dtype=type_vec)

   flux_Eq0_itf_j = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   flux_Eq1_itf_j = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   flux_Eq2_itf_j = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   h_itf_j        = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   u1_itf_j       = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   u2_itf_j       = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)

   flux_Eq0_itf_i = numpy.zeros((nb_elements_horiz+2, nbsolpts*nb_elements_horiz, 2), dtype=type_vec)
   flux_Eq1_itf_i = numpy.zeros((nb_elements_horiz+2, nbsolpts*nb_elements_horiz, 2), dtype=type_vec)
   flux_Eq2_itf_i = numpy.zeros((nb_elements_horiz+2, nbsolpts*nb_elements_horiz, 2), dtype=type_vec)
   h_itf_i        = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   u1_itf_i       = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   u2_itf_i       = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)

   eig_L          = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=type_vec)
   eig_R          = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=type_vec)
   eig            = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=type_vec)

   flux_L         = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=type_vec)
   flux_R         = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=type_vec)

   # Unpack physical variables
   h = Q[idx_h, :, :]

   if shallow_water_equations:
      u1 = Q[idx_hu1,:,:] / h
      u2 = Q[idx_hu2,:,:] / h
   else:
      u1 = Q[idx_u1, :, :]
      u2 = Q[idx_u2, :, :]

   # Compute the fluxes
   flux_Eq0_x1 = h * metric.sqrtG * u1
   flux_Eq0_x2 = h * metric.sqrtG * u2

   flux_Eq1_x1 = metric.sqrtG * ( Q[idx_hu1,:,:] * u1 )
   flux_Eq1_x2 = metric.sqrtG * ( Q[idx_hu1,:,:] * u2 )

   flux_Eq2_x1 = metric.sqrtG * ( Q[idx_hu2,:,:] * u1 )
   flux_Eq2_x2 = metric.sqrtG * ( Q[idx_hu2,:,:] * u2 )

   # Offset due to the halo
   offset = 1

   # Interpolate to the element interface
   for elem in range(nb_elements_horiz):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      pos = elem + offset

      # --- Direction x1

      h_itf_i[pos, 0, :] = h[:, epais] @ mtrx.extrap_west
      h_itf_i[pos, 1, :] = h[:, epais] @ mtrx.extrap_east

      u1_itf_i[pos, 0, :] = u1[:, epais] @ mtrx.extrap_west
      u1_itf_i[pos, 1, :] = u1[:, epais] @ mtrx.extrap_east

      u2_itf_i[pos, 0, :] = u2[:, epais] @ mtrx.extrap_west
      u2_itf_i[pos, 1, :] = u2[:, epais] @ mtrx.extrap_east

      # --- Direction x2

      h_itf_j[pos, 0, :] = mtrx.extrap_south @ h[epais, :]
      h_itf_j[pos, 1, :] = mtrx.extrap_north @ h[epais, :]

      u1_itf_j[pos, 0, :] = mtrx.extrap_south @ u1[epais, :]
      u1_itf_j[pos, 1, :] = mtrx.extrap_north @ u1[epais, :]

      u2_itf_j[pos, 0, :] = mtrx.extrap_south @ u2[epais, :]
      u2_itf_j[pos, 1, :] = mtrx.extrap_north @ u2[epais, :]

   xchange_scalars(comm_dist_graph, geom, h_itf_i, h_itf_j)
   xchange_vectors(comm_dist_graph, geom, u1_itf_i, u2_itf_i, u1_itf_j, u2_itf_j)

   # Common Rusanov fluxes
   for itf in range(nb_interfaces_horiz):

      elem_L = itf
      elem_R = itf + 1

      # Direction x1

      eig_L[:] = numpy.abs( u1_itf_i[elem_L, 1, :] )
      eig_R[:] = numpy.abs( u1_itf_i[elem_R, 0, :] )
      eig[:]   = numpy.maximum(eig_L, eig_R)

      # --- Continuity equation

      flux_L[:] = metric.sqrtG_itf_i[:, itf] * h_itf_i[elem_L, 1, :] * u1_itf_i[elem_L, 1, :]
      flux_R[:] = metric.sqrtG_itf_i[:, itf] * h_itf_i[elem_R, 0, :] * u1_itf_i[elem_R, 0, :]

      flux_Eq0_itf_i[elem_L, :, 1] = 0.5 * ( flux_L  + flux_R - eig * metric.sqrtG_itf_i[:, itf] * ( h_itf_i[elem_R, 0, :]  - h_itf_i[elem_L, 1, :] ) )
      flux_Eq0_itf_i[elem_R, :, 0] = flux_Eq0_itf_i[elem_L, :, 1]

      # --- u1 equation

      flux_L[:] = metric.sqrtG_itf_i[:, itf] * ( h_itf_i[elem_L, 1, :] * u1_itf_i[elem_L, 1, :]**2 )
      flux_R[:] = metric.sqrtG_itf_i[:, itf] * ( h_itf_i[elem_R, 0, :] * u1_itf_i[elem_R, 0, :]**2 )

      flux_Eq1_itf_i[elem_L, :, 1] = 0.5 * ( flux_L  + flux_R - eig * metric.sqrtG_itf_i[:, itf] \
            * ( h_itf_i[elem_R, 0, :] * u1_itf_i[elem_R, 0, :]  - h_itf_i[elem_L, 1, :] * u1_itf_i[elem_L, 1, :] ) )
      flux_Eq1_itf_i[elem_R, :, 0] = flux_Eq1_itf_i[elem_L, :, 1]

      # --- u2 equation

      flux_L[:] = metric.sqrtG_itf_i[:, itf] * ( h_itf_i[elem_L, 1, :] * u2_itf_i[elem_L, 1, :] * u1_itf_i[elem_L, 1, :] )
      flux_R[:] = metric.sqrtG_itf_i[:, itf] * ( h_itf_i[elem_R, 0, :] * u2_itf_i[elem_R, 0, :] * u1_itf_i[elem_R, 0, :] )

      flux_Eq2_itf_i[elem_L, :, 1] = 0.5 * ( flux_L  + flux_R - eig * metric.sqrtG_itf_i[:, itf] \
            * ( h_itf_i[elem_R, 0, :] * u2_itf_i[elem_R, 0, :]  - h_itf_i[elem_L, 1, :] * u2_itf_i[elem_L, 1, :] ) )
      flux_Eq2_itf_i[elem_R, :, 0] = flux_Eq2_itf_i[elem_L, :, 1]

      # Direction x2

      eig_L[:] = numpy.abs( u2_itf_j[elem_L, 1, :] )
      eig_R[:] = numpy.abs( u2_itf_j[elem_R, 0, :] )
      eig[:]   = numpy.maximum(eig_L, eig_R)

      # --- Continuity equation

      flux_L[:] = metric.sqrtG_itf_j[itf, :] * h_itf_j[elem_L, 1, :] * u2_itf_j[elem_L, 1, :]
      flux_R[:] = metric.sqrtG_itf_j[itf, :] * h_itf_j[elem_R, 0, :] * u2_itf_j[elem_R, 0, :]

      flux_Eq0_itf_j[elem_L, 1, :] = 0.5 * ( flux_L + flux_R - eig * metric.sqrtG_itf_j[itf, :] * ( h_itf_j[elem_R, 0, :] - h_itf_j[elem_L, 1, :] ) )
      flux_Eq0_itf_j[elem_R, 0, :] = flux_Eq0_itf_j[elem_L, 1, :]

      # --- u1 equation

      flux_L[:] = metric.sqrtG_itf_j[itf, :] * ( h_itf_j[elem_L, 1, :] * u1_itf_j[elem_L, 1, :] * u2_itf_j[elem_L, 1, :] )
      flux_R[:] = metric.sqrtG_itf_j[itf, :] * ( h_itf_j[elem_R, 0, :] * u1_itf_j[elem_R, 0, :] * u2_itf_j[elem_R, 0, :] )

      flux_Eq1_itf_j[elem_L, 1, :] = 0.5 * ( flux_L + flux_R - eig * metric.sqrtG_itf_j[itf, :] \
            * ( h_itf_j[elem_R, 0, :] * u1_itf_j[elem_R, 0, :] - h_itf_j[elem_L, 1, :] * u1_itf_j[elem_L, 1, :]) )
      flux_Eq1_itf_j[elem_R, 0, :] = flux_Eq1_itf_j[elem_L, 1, :]

      # --- u2 equation

      flux_L[:] = metric.sqrtG_itf_j[itf, :] * ( h_itf_j[elem_L, 1, :] * u2_itf_j[elem_L, 1, :]**2 )
      flux_R[:] = metric.sqrtG_itf_j[itf, :] * ( h_itf_j[elem_R, 0, :] * u2_itf_j[elem_R, 0, :]**2  )

      flux_Eq2_itf_j[elem_L, 1, :] = 0.5 * ( flux_L + flux_R - eig * metric.sqrtG_itf_j[itf, :] \
            * ( h_itf_j[elem_R, 0, :] * u2_itf_j[elem_R, 0, :] - h_itf_j[elem_L, 1, :] * u2_itf_j[elem_L, 1, :]) )
      flux_Eq2_itf_j[elem_R, 0, :] = flux_Eq2_itf_j[elem_L, 1, :]

   # Compute the derivatives
   for elem in range(nb_elements_horiz):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      # --- Direction x1

      df1_dx1[idx_h][:,epais]   = ( flux_Eq0_x1[:,epais] @ mtrx.diff_solpt_tr + flux_Eq0_itf_i[elem+offset,:,:] @ mtrx.correction_tr ) * 2.0 / geom.Δx1
      df1_dx1[idx_hu1][:,epais] = ( flux_Eq1_x1[:,epais] @ mtrx.diff_solpt_tr + flux_Eq1_itf_i[elem+offset,:,:] @ mtrx.correction_tr ) * 2.0 / geom.Δx1
      df1_dx1[idx_hu2][:,epais] = ( flux_Eq2_x1[:,epais] @ mtrx.diff_solpt_tr + flux_Eq2_itf_i[elem+offset,:,:] @ mtrx.correction_tr ) * 2.0 / geom.Δx1

      # --- Direction x2

      df2_dx2[idx_h,epais,:]   = ( mtrx.diff_solpt @ flux_Eq0_x2[epais,:] + mtrx.correction @ flux_Eq0_itf_j[elem+offset,:,:] ) * 2.0 / geom.Δx2
      df2_dx2[idx_hu1,epais,:] = ( mtrx.diff_solpt @ flux_Eq1_x2[epais,:] + mtrx.correction @ flux_Eq1_itf_j[elem+offset,:,:] ) * 2.0 / geom.Δx2
      df2_dx2[idx_hu2,epais,:] = ( mtrx.diff_solpt @ flux_Eq2_x2[epais,:] + mtrx.correction @ flux_Eq2_itf_j[elem+offset,:,:] ) * 2.0 / geom.Δx2

   # Add coriolis, metric and terms due to varying bottom topography
   forcing[idx_h,:,:] = 0.0

   forcing[idx_hu1,:,:] = 2.0 * ( metric.christoffel_1_01 * h * u1 + metric.christoffel_1_02 * h * u2) \
         + metric.christoffel_1_11 * h * u1**2 + 2.0 * metric.christoffel_1_12 * h * u1 * u2 

   forcing[idx_hu2,:,:] = 2.0 * (metric.christoffel_2_01 * h * u1 + metric.christoffel_2_02 * h * u2) \
         + 2.0 * metric.christoffel_2_21 * h * u2 * u1 + metric.christoffel_2_22 * h * u2**2 

   # Assemble the right-hand sides
   rhs = metric.inv_sqrtG * ( - df1_dx1 - df2_dx2 ) - forcing

   if not shallow_water_equations:
      rhs[idx_hu1,:,:] = 0.0
      rhs[idx_hu2,:,:] = 0.0

   return rhs