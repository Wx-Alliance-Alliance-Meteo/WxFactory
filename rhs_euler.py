import numpy

from definitions import idx_rho_u1, idx_rho_u2, idx_rho_w, idx_rho, idx_rho_theta, gravity
from dgfilter import apply_filter

def rhs_euler(Q, geom, mtrx, metric, topo, ptopo, nbsolpts: int, nb_elements_horiz: int, nb_elements_vert: int, case_number: int, filter_rhs: bool = False):

   type_vec = Q.dtype
   nb_equations = Q.shape[0]
   idx_first_tracer = 5

   nb_interfaces_horiz = nb_elements_horiz + 1
   nb_pts_horiz = nb_elements_horiz * nbsolpts
   nb_vertical_levels = nb_elements_vert * nbsolpts

   df1_dx1 = numpy.zeros_like(Q, dtype=type_vec)
   df2_dx2 = numpy.zeros_like(Q, dtype=type_vec)
   forcing = numpy.zeros_like(Q, dtype=type_vec)
   rhs     = numpy.zeros_like(Q, dtype=type_vec)

   variables_itf_i = numpy.zeros((nb_equations, nb_vertical_levels, nb_elements_horiz + 2, 2, nb_pts_horiz), dtype=type_vec)
   flux_x1_itf_i   = numpy.zeros((nb_equations, nb_vertical_levels, nb_elements_horiz + 2, nb_pts_horiz, 2), dtype=type_vec)

   variables_itf_j = numpy.zeros((nb_equations, nb_vertical_levels, nb_elements_horiz + 2, 2, nb_pts_horiz), dtype=type_vec)
   flux_x2_itf_j   = numpy.zeros((nb_equations, nb_vertical_levels, nb_elements_horiz + 2, 2, nb_pts_horiz), dtype=type_vec)

   # Offset due to the halo
   offset = 1

   # Interpolate to the element interface
   for elem in range(nb_elements_horiz):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)
      pos = elem + offset

      # --- Direction x1
      variables_itf_i[:, :, pos, 0, :] = Q[:, :, :, epais] @ mtrx.extrap_west
      variables_itf_i[:, :, pos, 1, :] = Q[:, :, :, epais] @ mtrx.extrap_east

      # --- Direction x2
      variables_itf_j[:, :, pos, 0, :] = mtrx.extrap_south @ Q[:, :, epais, :]
      variables_itf_j[:, :, pos, 1, :] = mtrx.extrap_north @ Q[:, :, epais, :]

   # Initiate transfers
   all_request = ptopo.xchange_Euler_interfaces(geom, variables_itf_i, variables_itf_j, blocking=False)

   # Unpack dynamical variables
   rho = Q[idx_rho]
   u1  = Q[idx_rho_u1] / rho
   u2  = Q[idx_rho_u2] / rho

   # Compute the fluxes
   flux_x1 = metric.sqrtG * u1 * Q
   flux_x2 = metric.sqrtG * u2 * Q

   # Compute the derivatives, corrections for the boundaries will be added later
   for elem in range(nb_elements_horiz):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      # --- Direction x1

      df1_dx1[:,:,:,epais] = flux_x1[:,:,:,epais] @ mtrx.diff_solpt_tr

      # --- Direction x2

      df2_dx2[:,:,epais,:] = mtrx.diff_solpt @ flux_x2[:,:,epais,:]

   # Finish transfers
   all_request.wait()

   # Common Rusanov fluxes
   for itf in range(nb_interfaces_horiz):

      elem_L = itf
      elem_R = itf + 1

      # Direction x1
      u1_L = variables_itf_i[idx_rho_u1, :, elem_L, 1, :] / variables_itf_i[idx_rho, :, elem_L, 1, :]
      u1_R = variables_itf_i[idx_rho_u1, :, elem_R, 0, :] / variables_itf_i[idx_rho, :, elem_R, 0, :]

      eig_L = numpy.abs( u1_L )
      eig_R = numpy.abs( u1_R )

      eig = numpy.maximum(eig_L, eig_R)

      # --- Continuity equation

      flux_L = metric.sqrtG_itf_i[:, itf] * u1_L * variables_itf_i[idx_first_tracer:, :, elem_L, 1, :]
      flux_R = metric.sqrtG_itf_i[:, itf] * u1_R * variables_itf_i[idx_first_tracer:, :, elem_R, 0, :]

      flux_x1_itf_i[idx_first_tracer:, :, elem_L, :, 1] = 0.5 * ( flux_L  + flux_R - eig * metric.sqrtG_itf_i[:, itf] * ( variables_itf_i[idx_first_tracer:, :, elem_R, 0, :] - variables_itf_i[idx_first_tracer:, :, elem_L, 1, :] ) )
      flux_x1_itf_i[idx_first_tracer:, :, elem_R, :, 0] = flux_x1_itf_i[idx_first_tracer:, :, elem_L, :, 1]

      # Direction x2

      u2_L = variables_itf_j[idx_rho_u2, :, elem_L, 1, :] / variables_itf_j[idx_rho, :, elem_L, 1, :]
      u2_R = variables_itf_j[idx_rho_u2, :, elem_R, 0, :] / variables_itf_j[idx_rho, :, elem_R, 0, :]

      eig_L = numpy.abs( u2_L )
      eig_R = numpy.abs( u2_R )

      eig = numpy.maximum(eig_L, eig_R)

      # --- Continuity equation

      flux_L = metric.sqrtG_itf_j[itf, :] * u2_L * variables_itf_j[idx_first_tracer:, :, elem_L, 1, :]
      flux_R = metric.sqrtG_itf_j[itf, :] * u2_R * variables_itf_j[idx_first_tracer:, :, elem_R, 0, :]

      flux_x2_itf_j[idx_first_tracer:, :, elem_L, 1, :] = 0.5 * ( flux_L + flux_R - eig * metric.sqrtG_itf_j[itf, :] * ( variables_itf_j[idx_first_tracer:, :, elem_R, 0, :] - variables_itf_j[idx_first_tracer:, :, elem_L, 1, :] ) )
      flux_x2_itf_j[idx_first_tracer:, :, elem_R, 0, :] = flux_x2_itf_j[idx_first_tracer:, :, elem_L, 1, :]

   # Add corrections to the derivatives
   for elem in range(nb_elements_horiz):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      # --- Direction x1

      df1_dx1[:,:,:,epais] += flux_x1_itf_i[:, :, elem+offset, :, :] @ mtrx.correction_tr

      # --- Direction x2

      df2_dx2[:,:,epais,:] += mtrx.correction @ flux_x2_itf_j[:, :, elem+offset, :, :]

   # Assemble the right-hand sides
   rhs = - metric.inv_sqrtG * ( df1_dx1 + df2_dx2 )

   # For pure advection problems, we do not update the dynamical variables
   rhs[idx_rho]       = 0.0
   rhs[idx_rho_u1]    = 0.0
   rhs[idx_rho_u2]    = 0.0
   rhs[idx_rho_theta] = 0.0

   return rhs
