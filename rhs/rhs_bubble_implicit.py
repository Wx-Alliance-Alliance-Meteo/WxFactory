import numpy

from common.definitions import idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta, \
                               Rd, p0, cpd, cvd, gravity, heat_capacity_ratio

def rhs_bubble_implicit(Q, geom, mtrx, nbsolpts, nb_elements_x, nb_elements_z):

   datatype = Q.dtype
   nb_equations = Q.shape[0] # Number of constituent Euler equations.  Probably 6.

   nb_interfaces_x = nb_elements_x + 1
   nb_interfaces_z = nb_elements_z + 1

   rhs = numpy.zeros_like(Q, dtype=datatype)
   flux_x1 = numpy.zeros_like(Q, dtype=datatype)
   flux_x3 = numpy.zeros_like(Q, dtype=datatype)

   df1_dx1 = numpy.zeros_like(Q, dtype=datatype)
   df3_dx3 = numpy.zeros_like(Q, dtype=datatype)

   kfaces_flux = numpy.zeros((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)
   kfaces_var  = numpy.zeros((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)

   ifaces_flux = numpy.zeros((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)
   ifaces_var  = numpy.zeros((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)

   # --- Unpack physical variables
   rho      = Q[idx_2d_rho,:,:]
   uu       = Q[idx_2d_rho_u,:,:] / rho
   ww       = Q[idx_2d_rho_w,:,:] / rho
   theta    = Q[idx_2d_rho_theta,:,:] / rho
   pressure = p0 * (Q[idx_2d_rho_theta,:,:] * Rd / p0)**(cpd / cvd)

   # --- Compute the fluxes
   flux_x1[idx_2d_rho,:,:]       = Q[idx_2d_rho_u,:,:]
   flux_x1[idx_2d_rho_u,:,:]     = pressure

   flux_x3[idx_2d_rho,:,:]       = Q[idx_2d_rho_w,:,:]
   flux_x3[idx_2d_rho_w,:,:]     = pressure

   # --- Interpolate to the element interface
   for elem in range(nb_elements_z):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      kfaces_var[:,elem,0,:] = mtrx.extrap_down @ Q[:,epais,:]
      kfaces_var[:,elem,1,:] = mtrx.extrap_up @ Q[:,epais,:]

   for elem in range(nb_elements_x):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      ifaces_var[:,elem,:,0] = Q[:,:,epais] @ mtrx.extrap_west
      ifaces_var[:,elem,:,1] = Q[:,:,epais] @ mtrx.extrap_east

   # --- Interface pressure
   ifaces_pres = p0 * (ifaces_var[idx_2d_rho_theta] * Rd / p0)**(cpd / cvd)
   kfaces_pres = p0 * (kfaces_var[idx_2d_rho_theta] * Rd / p0)**(cpd / cvd)

   # --- Bondary treatement

   # zeros flux BCs everywhere ...
   kfaces_flux[:,0,0,:]  = 0.0
   kfaces_flux[:,-1,1,:] = 0.0

   ifaces_flux[:, 0,:,0] = 0.0
   ifaces_flux[:,-1,:,1] = 0.0

   # except for momentum eqs where pressure is extrapolated to BCs.
   kfaces_flux[idx_2d_rho_w, 0, 0,:] = kfaces_pres[0,0,:]
   kfaces_flux[idx_2d_rho_w,-1,1,:]  = kfaces_pres[-1,1,:]

   ifaces_flux[idx_2d_rho_u, 0,:,0] = ifaces_pres[0,:,0]  # TODO : pour les cas théoriques seulement ...
   ifaces_flux[idx_2d_rho_u,-1,:,1] = ifaces_pres[-1,:,1]

   # --- Common AUSM fluxes
   for itf in range(1, nb_interfaces_z - 1):

      left  = itf - 1
      right = itf

      # Left state
      a_L   = numpy.sqrt( heat_capacity_ratio * kfaces_pres[left,1,:] / kfaces_var[idx_2d_rho,left,1,:] )
      M_L   = kfaces_var[idx_2d_rho_w,left,1,:] / ( kfaces_var[idx_2d_rho,left,1,:] * a_L )

      # Right state
      a_R   = numpy.sqrt( heat_capacity_ratio * kfaces_pres[right,0,:] / kfaces_var[idx_2d_rho,right,0,:] )
      M_R   = kfaces_var[idx_2d_rho_w,right,0,:] / ( kfaces_var[idx_2d_rho,right,0,:] * a_R )

      M = 0.25 * (( M_L + 1.)**2 - (M_R - 1.)**2)

      kfaces_flux[idx_2d_rho,right,0,:] = (kfaces_var[idx_2d_rho,left,1,:] * numpy.maximum(0., M) * a_L) + (kfaces_var[idx_2d_rho,right,0,:] * numpy.minimum(0., M) * a_R)
      kfaces_flux[idx_2d_rho_w,right,0,:] += 0.5 * ( (1. + M_L) * kfaces_pres[left,1,:] + (1. - M_R) * kfaces_pres[right,0,:] )
   
      kfaces_flux[:,left,1,:] = kfaces_flux[:,right,0,:]

   for itf in range(1, nb_interfaces_x - 1):

      left  = itf - 1
      right = itf

      # Left state
      a_L   = numpy.sqrt( heat_capacity_ratio * ifaces_pres[left,:,1] / ifaces_var[idx_2d_rho,left,:,1] )
      M_L   = ifaces_var[idx_2d_rho_u,left,:,1] / ( ifaces_var[idx_2d_rho,left,:,1] * a_L )

      # Right state
      a_R   = numpy.sqrt( heat_capacity_ratio * ifaces_pres[right,:,0] / ifaces_var[idx_2d_rho,right,:,0] )
      M_R   = ifaces_var[idx_2d_rho_u,right,:,0] / ( ifaces_var[idx_2d_rho,right,:,0] * a_R )

      M = 0.25 * ((M_L + 1.)**2 - (M_R - 1.)**2)

      ifaces_flux[idx_2d_rho,right,:,0] = (ifaces_var[idx_2d_rho,left,:,1] * numpy.maximum(0., M) * a_L) + (ifaces_var[idx_2d_rho,right,:,0] * numpy.minimum(0., M) * a_R)
      ifaces_flux[idx_2d_rho_u,right,:,0] += 0.5 * ( (1. + M_L) * ifaces_pres[left,:,1] + (1. - M_R) * ifaces_pres[right,:,0] )

      ifaces_flux[:,left,:,1] = ifaces_flux[:,right,:,0]

   # --- Compute the derivatives
   for elem in range(nb_elements_z):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      df3_dx3[idx_2d_rho,epais,:] = ( mtrx.diff_solpt @ flux_x3[idx_2d_rho,epais,:] + mtrx.correction @ kfaces_flux[idx_2d_rho,elem,:,:] ) * 2.0/geom.Δx3
      df3_dx3[idx_2d_rho_w,epais,:] = ( mtrx.diff_solpt @ flux_x3[idx_2d_rho_w,epais,:] + mtrx.correction @ kfaces_flux[idx_2d_rho_w,elem,:,:] ) * 2.0/geom.Δx3

   for elem in range(nb_elements_x):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      df1_dx1[:-2,:,epais] = ( flux_x1[:-2,:,epais] @ mtrx.diff_solpt.T + ifaces_flux[:-2,elem,:,:] @ mtrx.correction.T ) * 2.0/geom.Δx1

   # --- Assemble the right-hand sides

   rhs[:-1,:,:] = - ( df1_dx1[:-1,:,:] + df3_dx3[:-1,:,:] )
   rhs[idx_2d_rho_w,:,:] -= Q[idx_2d_rho,:,:] * gravity
   rhs[idx_2d_rho_theta] = - theta * ( df1_dx1[idx_2d_rho] + df3_dx3[idx_2d_rho] )

   return rhs
