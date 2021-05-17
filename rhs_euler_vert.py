import numpy

from definitions import idx_rho_u1, idx_rho_u2, idx_rho_u3, idx_rho, idx_rho_theta, gravity, p0, Rd, cpd, cvd, heat_capacity_ratio

def rhs_euler_vert(Q, sqrtG, mtrx, nbsolpts: int, nb_elements_vert: int, case_number: int, filter_rhs: bool = False): #, geom, mtrx, metric, topo, ptopo, , , ):

   datatype = Q.dtype

   adv_only = (case_number == 11)
   
   nb_equations, nk = Q.shape
   nb_tracers = nb_equations - 5

   nb_interfaces_vert = nb_elements_vert + 1

   df3_dx3 = numpy.zeros_like(Q, dtype=datatype)
   rhs = numpy.zeros_like(Q, dtype=datatype)

   tracers         = numpy.zeros((nb_tracers, nk), dtype=datatype)
   flux_tracers_x3 = numpy.zeros((nb_tracers, nk), dtype=datatype)

   flux_tracers_itf_k = numpy.zeros((nb_tracers, nb_elements_vert+2, 2), dtype=datatype)

   rho_itf_k      = numpy.zeros((nb_elements_vert+2, 2), dtype=datatype)
   u3_itf_k       = numpy.zeros((nb_elements_vert+2, 2), dtype=datatype)
   tracers_itf_k  = numpy.zeros((nb_tracers, nb_elements_vert+2, 2), dtype=datatype)

   # Unpack dynamical variables
   rho = Q[idx_rho, :]

   u3 = Q[idx_rho_u3, :] / rho

   # Fluxes on solution points
   for idx_tr in range(nb_tracers):
      tracers[idx_tr, :] = Q[5+idx_tr, :] / rho
      flux_tracers_x3[idx_tr, :] = sqrtG * Q[5+idx_tr, :] * u3

   # Offset due to the halo
   offset = 1

   # Interpolate to the element interface
   for elem in range(nb_elements_vert):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      pos = elem + offset

      # --- Direction x3

      rho_itf_k[pos, 0] = mtrx.extrap_south @ rho[epais]
      rho_itf_k[pos, 1] = mtrx.extrap_north @ rho[epais]

      u3_itf_k[pos, 0] = mtrx.extrap_south @ u3[epais]
      u3_itf_k[pos, 1] = mtrx.extrap_north @ u3[epais]

   for idx_tr in range(nb_tracers):
      for elem in range(nb_elements_vert):
         epais = elem * nbsolpts + numpy.arange(nbsolpts)
   
         pos = elem + offset
   
         # --- Direction x3
   
         tracers_itf_k[idx_tr, pos, 0] = mtrx.extrap_south @ tracers[idx_tr, epais]
         tracers_itf_k[idx_tr, pos, 1] = mtrx.extrap_north @ tracers[idx_tr, epais]

   # --- Boundary treatements

   # Surface
   u3_itf_k[0, 1] = 0.
   u3_itf_k[1, 0] = 0.

   # Top
   u3_itf_k[-1, 0] = 0.
   u3_itf_k[-2, 1] = 0.

   # Common Rusanov fluxes
   for itf in range(nb_interfaces_vert):

      elem_L = itf
      elem_R = itf + 1

      # Direction x3

      eig_L = numpy.abs( u3_itf_k[elem_L, 1] )
      eig_R = numpy.abs( u3_itf_k[elem_R, 0] )

      eig = numpy.maximum(eig_L, eig_R)

      # --- Advection of tracers

      for idx_tr in range(nb_tracers):
         flux_L = sqrtG * rho_itf_k[elem_L, 1] * tracers_itf_k[idx_tr, elem_L, 1] * u3_itf_k[elem_L, 1]
         flux_R = sqrtG * rho_itf_k[elem_R, 0] * tracers_itf_k[idx_tr, elem_R, 0] * u3_itf_k[elem_R, 0]

         flux_tracers_itf_k[idx_tr, elem_L, 1] = 0.5 * ( flux_L  + flux_R - eig * sqrtG * ( rho_itf_k[elem_R, 0] *  tracers_itf_k[idx_tr, elem_R, 0] - rho_itf_k[elem_L, 1] * tracers_itf_k[idx_tr, elem_L, 1]) )
         flux_tracers_itf_k[idx_tr, elem_R, 0] = flux_tracers_itf_k[idx_tr, elem_L, 1]

   # Compute the derivatives
   for idx_tr in range(nb_tracers):
      for elem in range(nb_elements_vert):
         epais = elem * nbsolpts + numpy.arange(nbsolpts)
      
         # --- Direction x3
         df3_dx3[5+idx_tr][epais] = mtrx.diff_solpt @ flux_tracers_x3[idx_tr][epais] + mtrx.correction @ flux_tracers_itf_k[idx_tr][elem+offset, :]
   
   # Assemble the right-hand sides
   for var in range(nb_equations):
      rhs[var] = -1. / sqrtG * df3_dx3[var]

   rhs[idx_rho,:]       = 0.
   rhs[idx_rho_u1,:]    = 0.
   rhs[idx_rho_u2,:]    = 0.
   rhs[idx_rho_u3,:]    = 0.
   rhs[idx_rho_theta,:] = 0.

   return rhs
