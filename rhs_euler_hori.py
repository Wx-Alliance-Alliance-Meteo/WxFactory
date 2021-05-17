import numpy

from definitions import idx_rho_u1, idx_rho_u2, idx_rho_u3, idx_rho, idx_rho_theta, gravity, p0, Rd, cpd, cvd, heat_capacity_ratio

def rhs_euler_hori(Q, geom, mtrx, metric, topo, ptopo, nbsolpts: int, nb_elements_horiz: int, case_number: int, filter_rhs: bool = False):

   datatype = Q.dtype

   adv_only = (case_number == 11)
   
   nb_equations, nj, ni  = Q.shape
   nb_tracers = nb_equations - 5

   nb_interfaces_horiz = nb_elements_horiz + 1

   df1_dx1 = numpy.zeros_like(Q, dtype=datatype)
   df2_dx2 = numpy.zeros_like(Q, dtype=datatype)
   rhs = numpy.zeros_like(Q, dtype=datatype)

   tracers         = numpy.zeros((nb_tracers, nj, ni), dtype=datatype)
   flux_tracers_x1 = numpy.zeros((nb_tracers, nj, ni), dtype=datatype)
   flux_tracers_x2 = numpy.zeros((nb_tracers, nj, ni), dtype=datatype)

   rho_itf_j      = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=datatype)
   u1_itf_j       = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=datatype)
   u2_itf_j       = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=datatype)

   tracers_itf_j      = numpy.zeros((nb_tracers, nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=datatype)
   flux_tracers_itf_j = numpy.zeros((nb_tracers, nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=datatype)

   rho_itf_i      = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=datatype)
   u1_itf_i       = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=datatype)
   u2_itf_i       = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=datatype)

   tracers_itf_i      = numpy.zeros((nb_tracers, nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=datatype)
   flux_tracers_itf_i = numpy.zeros((nb_tracers, nb_elements_horiz+2, nbsolpts*nb_elements_horiz, 2), dtype=datatype)

   eig_L          = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=datatype)
   eig_R          = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=datatype)
   eig            = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=datatype)

   flux_L         = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=datatype)
   flux_R         = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=datatype)

   # Unpack dynamical variables
   rho     = Q[idx_rho, :, :]
   rho_u1  = Q[idx_rho_u1, :, :]
   rho_u2  = Q[idx_rho_u2, :, :]
   rho_u3  = Q[idx_rho_u3, :, :]
   rho_theta = Q[idx_rho_theta, :, :]

   u1 = rho_u1 / rho
   u2 = rho_u2 / rho

   # Compute the fluxes
   for idx_tr in range(nb_tracers):
      tracers[idx_tr, :, :] = Q[5+idx_tr, :, :] / rho
      flux_tracers_x1[idx_tr, :, :] = metric.sqrtG * Q[5+idx_tr, :, :] * u1
      flux_tracers_x2[idx_tr, :, :] = metric.sqrtG * Q[5+idx_tr, :, :] * u2

   # Offset due to the halo
   offset = 1

   # Interpolate to the element interface
   for elem in range(nb_elements_horiz):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      pos = elem + offset

      # --- Direction x1

      rho_itf_i[pos, 0, :] = rho[:, epais] @ mtrx.extrap_west
      rho_itf_i[pos, 1, :] = rho[:, epais] @ mtrx.extrap_east

      u1_itf_i[pos, 0, :] = u1[:, epais] @ mtrx.extrap_west
      u1_itf_i[pos, 1, :] = u1[:, epais] @ mtrx.extrap_east

      u2_itf_i[pos, 0, :] = u2[:, epais] @ mtrx.extrap_west
      u2_itf_i[pos, 1, :] = u2[:, epais] @ mtrx.extrap_east


      # --- Direction x2

      rho_itf_j[pos, 0, :] = mtrx.extrap_south @ rho[epais, :]
      rho_itf_j[pos, 1, :] = mtrx.extrap_north @ rho[epais, :]

      u1_itf_j[pos, 0, :] = mtrx.extrap_south @ u1[epais, :]
      u1_itf_j[pos, 1, :] = mtrx.extrap_north @ u1[epais, :]

      u2_itf_j[pos, 0, :] = mtrx.extrap_south @ u2[epais, :]
      u2_itf_j[pos, 1, :] = mtrx.extrap_north @ u2[epais, :]

   # TODO : faire une seule fois les échanges ... 
   ptopo.xchange_scalars(geom, rho_itf_i, rho_itf_j)
   ptopo.xchange_vectors(geom, u1_itf_i, u2_itf_i, u1_itf_j, u2_itf_j)

   for idx_tr in range(nb_tracers):
      for elem in range(nb_elements_horiz):
         epais = elem * nbsolpts + numpy.arange(nbsolpts)
   
         pos = elem + offset
   
         # --- Direction x1
   
         tracers_itf_i[idx_tr, pos, 0, :] = tracers[idx_tr][:, epais] @ mtrx.extrap_west
         tracers_itf_i[idx_tr, pos, 1, :] = tracers[idx_tr][:, epais] @ mtrx.extrap_east
   
         # --- Direction x2
   
         tracers_itf_j[idx_tr, pos, 0, :] = mtrx.extrap_south @ tracers[idx_tr, epais, :]
         tracers_itf_j[idx_tr, pos, 1, :] = mtrx.extrap_north @ tracers[idx_tr, epais, :]

      ptopo.xchange_scalars(geom, tracers_itf_i[idx_tr], tracers_itf_j[idx_tr])
  

   # Common Rusanov fluxes
   for itf in range(nb_interfaces_horiz):

      elem_L = itf
      elem_R = itf + 1

      # Direction x1

      eig_L[:] = numpy.abs( u1_itf_i[elem_L, 1, :] )
      eig_R[:] = numpy.abs( u1_itf_i[elem_R, 0, :] )

      eig[:] = numpy.maximum(eig_L, eig_R)

      # --- Advection of tracers

      for idx_tr in range(nb_tracers):
         flux_L[:] = metric.sqrtG_itf_i[:, itf] * rho_itf_i[elem_L, 1, :] * tracers_itf_i[idx_tr, elem_L, 1, :] * u1_itf_i[elem_L, 1, :]
         flux_R[:] = metric.sqrtG_itf_i[:, itf] * rho_itf_i[elem_R, 0, :] * tracers_itf_i[idx_tr, elem_R, 0, :] * u1_itf_i[elem_R, 0, :]

         flux_tracers_itf_i[idx_tr, elem_L, :, 1] = 0.5 * ( flux_L  + flux_R - eig * metric.sqrtG_itf_i[:, itf] * ( rho_itf_i[elem_R, 0, :] *  tracers_itf_i[idx_tr, elem_R, 0, :] - rho_itf_i[elem_L, 1, :] * tracers_itf_i[idx_tr, elem_L, 1, :]) )
         flux_tracers_itf_i[idx_tr, elem_R, :, 0] = flux_tracers_itf_i[idx_tr, elem_L, :, 1]

      # Direction x2

      eig_L[:] = numpy.abs( u2_itf_j[elem_L, 1, :] )
      eig_R[:] = numpy.abs( u2_itf_j[elem_R, 0, :] )

      eig[:] = numpy.maximum(eig_L, eig_R)

      # --- Advection of tracers

      for idx_tr in range(nb_tracers):
         flux_L[:] = metric.sqrtG_itf_j[itf, :] * rho_itf_j[elem_L, 1, :] * tracers_itf_j[idx_tr, elem_L, 1, :] * u2_itf_j[elem_L, 1, :]
         flux_R[:] = metric.sqrtG_itf_j[itf, :] * rho_itf_j[elem_R, 0, :] * tracers_itf_j[idx_tr, elem_R, 0, :] * u2_itf_j[elem_R, 0, :]

         flux_tracers_itf_j[idx_tr, elem_L, 1, :] = 0.5 * ( flux_L + flux_R - eig * metric.sqrtG_itf_j[itf, :] * ( rho_itf_j[elem_R, 0, :] * tracers_itf_j[idx_tr, elem_R, 0, :] - rho_itf_j[elem_L, 1, :] * tracers_itf_j[idx_tr, elem_L, 1, :]) )
         flux_tracers_itf_j[idx_tr, elem_R, 0, :] = flux_tracers_itf_j[idx_tr, elem_L, 1, :]

   # Compute the derivatives
   for idx_tr in range(nb_tracers):
      for elem in range(nb_elements_horiz):
         epais = elem * nbsolpts + numpy.arange(nbsolpts)
      
         # --- Direction x1
         df1_dx1[5+idx_tr][ :,epais] = flux_tracers_x1[idx_tr][:, epais] @ mtrx.diff_solpt_tr + flux_tracers_itf_i[idx_tr][elem+offset, :, :] @ mtrx.correction_tr
   
         # --- Direction x2
         df2_dx2[5+idx_tr][epais, :] = mtrx.diff_solpt @ flux_tracers_x2[idx_tr][epais, :] + mtrx.correction @ flux_tracers_itf_j[idx_tr][elem+offset, :, :]

   # Assemble the right-hand sides
   for var in range(nb_equations):
      rhs[var] = metric.inv_sqrtG * -( df1_dx1[var] + df2_dx2[var] )

   rhs[idx_rho,:,:]       = 0.
   rhs[idx_rho_u1,:,:]    = 0.
   rhs[idx_rho_u2,:,:]    = 0.
   rhs[idx_rho_u3,:,:]    = 0.
   rhs[idx_rho_theta,:,:] = 0.

   return rhs
