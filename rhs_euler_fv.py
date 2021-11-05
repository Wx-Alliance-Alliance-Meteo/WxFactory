import numpy

from definitions import idx_rho_u1, idx_rho_u2, idx_rho_w, idx_rho, idx_rho_theta, gravity
from dgfilter import apply_filter3D

def rhs_euler_fv(Q, geom, mtrx, metric, topo, ptopo, nbsolpts: int, nb_elements_hori: int, nb_elements_vert: int, case_number: int, filter_rhs: bool = False):

   type_vec = Q.dtype
   nb_equations = Q.shape[0]
   idx_first_tracer = 5

   nb_interfaces_hori = nb_elements_hori + 1
   nb_interfaces_vert = nb_elements_vert + 1
   nb_pts_hori = nb_elements_hori * nbsolpts
   nb_vertical_levels = nb_elements_vert * nbsolpts

   df1_dx1 = numpy.zeros_like(Q, dtype=type_vec)
   df2_dx2 = numpy.zeros_like(Q, dtype=type_vec)
   df3_dx3 = numpy.zeros_like(Q, dtype=type_vec)
   forcing = numpy.zeros_like(Q, dtype=type_vec)
   rhs     = numpy.zeros_like(Q, dtype=type_vec)

   variables_itf_i = numpy.zeros((nb_equations, nb_vertical_levels, nb_elements_hori + 2, 2, nb_pts_hori), dtype=type_vec)
   flux_x1_itf_i   = numpy.zeros((nb_equations, nb_vertical_levels, nb_elements_hori + 2, nb_pts_hori, 2), dtype=type_vec)

   variables_itf_j = numpy.zeros((nb_equations, nb_vertical_levels, nb_elements_hori + 2, 2, nb_pts_hori), dtype=type_vec)
   flux_x2_itf_j   = numpy.zeros((nb_equations, nb_vertical_levels, nb_elements_hori + 2, 2, nb_pts_hori), dtype=type_vec)

   variables_itf_k = numpy.zeros((nb_equations, nb_pts_hori, nb_elements_vert + 2, 2, nb_pts_hori), dtype=type_vec)
   flux_x3_itf_k   = numpy.zeros((nb_equations, nb_pts_hori, nb_elements_vert + 2, 2, nb_pts_hori), dtype=type_vec)

   # Offset due to the halo
   offset = 1

   # Interpolate to the element interface
   for elem in range(nb_elements_hori):
      epais = elem
      pos = elem + offset

      # --- Direction x1
      variables_itf_i[:, :, pos, 0, :] = Q[:, :, :, epais]
      variables_itf_i[:, :, pos, 1, :] = Q[:, :, :, epais]

      # --- Direction x2
      variables_itf_j[:, :, pos, 0, :] = Q[:, :, epais, :]
      variables_itf_j[:, :, pos, 1, :] = Q[:, :, epais, :]

   # Initiate transfers
   all_request = ptopo.xchange_Euler_interfaces(geom, variables_itf_i, variables_itf_j, blocking=False)

   # Unpack dynamical variables
   rho = Q[idx_rho]
   u1  = Q[idx_rho_u1] / rho
   u2  = Q[idx_rho_u2] / rho
   w   = Q[idx_rho_w]  / rho # TODO : u3

   # Compute the fluxes
   flux_x1 = metric.sqrtG * u1 * Q
   flux_x2 = metric.sqrtG * u2 * Q
   flux_x3 = metric.sqrtG * w  * Q

   # print(f'diff sol pt: \n{mtrx.diff_solpt_tr}, \n{mtrx.diff_solpt}')

   # Interior contribution to the derivatives, corrections for the boundaries will be added later
   # for elem in range(nb_elements_hori):
   #    epais = elem * nbsolpts + numpy.arange(nbsolpts)

   #    # --- Direction x1
   #    df1_dx1[:, :, :, epais] = flux_x1[:, :, :, epais] @ mtrx.diff_solpt_tr

   #    # --- Direction x2
   #    df2_dx2[:, :, epais, :] = mtrx.diff_solpt @ flux_x2[:, :, epais, :]


   # --- Direction x3

   # Important notice : all the vertical stuff should be done before the synchronization of the horizontal communications.

   for slab in range(nb_pts_hori):
      for elem in range(nb_elements_vert):
         epais = elem * nbsolpts + numpy.arange(nbsolpts)
         pos = elem + offset

         variables_itf_k[:, slab, pos, 0, :] = mtrx.extrap_down @ Q[:, epais, slab, :]
         variables_itf_k[:, slab, pos, 1, :] = mtrx.extrap_up   @ Q[:, epais, slab, :]

   # TODO: temporary hack to avoid division by zero in Riemann solver 
   variables_itf_k[:, :, 0, 1, :] = variables_itf_k[:, :, 1, 0, :]
   variables_itf_k[:, :, 0, 0, :] = variables_itf_k[:, :, 0, 1, :]
   variables_itf_k[:, :, -1, 0, :] = variables_itf_k[:, :, -2, 1, :]
   variables_itf_k[:, :, -1, 1, :] = variables_itf_k[:, :, -1, 0, :]

   for itf in range(nb_interfaces_vert):

      elem_D = itf
      elem_U = itf + 1

      # Direction x3

      w_D = variables_itf_k[idx_rho_w, :, elem_D, 1, :] / variables_itf_k[idx_rho, :, elem_D, 1, :]
      w_U = variables_itf_k[idx_rho_w, :, elem_U, 0, :] / variables_itf_k[idx_rho, :, elem_U, 0, :]

      eig_D = numpy.abs( w_D )
      eig_U = numpy.abs( w_U )

      eig = numpy.maximum(eig_D, eig_U)

      # --- Continuity equation

      flux_D = metric.sqrtG * w_D * variables_itf_k[idx_first_tracer:, :, elem_D, 1, :]
      flux_U = metric.sqrtG * w_U * variables_itf_k[idx_first_tracer:, :, elem_U, 0, :]

      flux_x3_itf_k[idx_first_tracer:, :, elem_D, 1, :] = 0.5 * ( flux_D + flux_U - eig * metric.sqrtG * ( variables_itf_k[idx_first_tracer:, :, elem_U, 0, :] - variables_itf_k[idx_first_tracer:, :, elem_D, 1, :] ) )
      flux_x3_itf_k[idx_first_tracer:, :, elem_U, 0, :] = flux_x3_itf_k[idx_first_tracer:, :, elem_D, 1, :]
   
   # Boundary treatement
   flux_x3_itf_k[idx_first_tracer:, :, 0, 1, :] = 0.
   flux_x3_itf_k[idx_first_tracer:, :, 0, 0, :] = 0.
   flux_x3_itf_k[idx_first_tracer:, :, -1, 0, :] = 0.
   flux_x3_itf_k[idx_first_tracer:, :, -1, 1, :] = 0. 

   for slab in range(nb_pts_hori):
      for elem in range(nb_elements_vert):
         epais = elem * nbsolpts + numpy.arange(nbsolpts)
         # TODO : inclure la transformation vers l'élément de référence dans la vitesse w.
         df3_dx3[:, epais, slab, :] = ( mtrx.diff_solpt @ flux_x3[:, epais, slab, :] + mtrx.correction @ flux_x3_itf_k[:, slab, elem+offset, :, :] ) * 2.0 / geom.Δx3

   # Finish transfers
   all_request.wait()

   # Common Rusanov fluxes
   for itf in range(nb_interfaces_hori):

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
   for elem in range(nb_elements_hori):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      # --- Direction x1

      df1_dx1[:, :, :, epais] = flux_x1_itf_i[:, :, elem+offset, :, :] @ mtrx.correction_tr

      # --- Direction x2

      df2_dx2[:, :, epais, :] = mtrx.correction @ flux_x2_itf_j[:, :, elem+offset, :, :]


   # Assemble the right-hand sides
   rhs = - metric.inv_sqrtG * ( df1_dx1 + df2_dx2 + df3_dx3 )

   if filter_rhs:
      for var in range(nb_equations):
         rhs[var] = apply_filter3D(rhs[var], mtrx, nb_elements_hori, nb_elements_vert, nbsolpts)

   # For pure advection problems, we do not update the dynamical variables
   rhs[idx_rho]       = 0.0
   rhs[idx_rho_u1]    = 0.0
   rhs[idx_rho_u2]    = 0.0
   rhs[idx_rho_w]     = 0.0
   rhs[idx_rho_theta] = 0.0

   return rhs
