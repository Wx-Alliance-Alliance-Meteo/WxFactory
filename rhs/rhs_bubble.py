import numpy
import pdb

from common.definitions import idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta,  \
                               p0, Rd, cpd, cvd, heat_capacity_ratio, gravity

def rhs_bubble(Q, geom, mtrx, nbsolpts, nb_elements_x, nb_elements_z):

   datatype = Q.dtype
   nb_equations = Q.shape[0] # Number of constituent Euler equations.  Probably 6.

   nb_interfaces_x = nb_elements_x + 1
   nb_interfaces_z = nb_elements_z + 1

   flux_x1 = numpy.empty_like(Q, dtype=datatype)
   flux_x3 = numpy.empty_like(Q, dtype=datatype)

   df1_dx1 = numpy.empty_like(Q, dtype=datatype)
   df3_dx3 = numpy.empty_like(Q, dtype=datatype)

   kfaces_flux = numpy.empty((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)
   kfaces_var  = numpy.empty((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)
   # interfaceFlux_k = numpy.empty((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)

   ifaces_flux = numpy.empty((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)
   ifaces_var  = numpy.empty((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)
   interfaceFlux_i = numpy.empty((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)

   # --- Unpack physical variables
   rho      = Q[idx_2d_rho,:,:]
   uu       = Q[idx_2d_rho_u,:,:] / rho
   ww       = Q[idx_2d_rho_w,:,:] / rho
   pressure = p0 * numpy.exp((cpd/cvd) * numpy.log((Rd/p0)*Q[idx_2d_rho_theta, :, :]))

   # --- Compute the fluxes
   flux_x1[idx_2d_rho,:,:]       = Q[idx_2d_rho_u,:,:]
   flux_x1[idx_2d_rho_u,:,:]     = Q[idx_2d_rho_u,:,:] * uu + pressure
   flux_x1[idx_2d_rho_w,:,:]     = Q[idx_2d_rho_u,:,:] * ww
   flux_x1[idx_2d_rho_theta,:,:] = Q[idx_2d_rho_theta,:,:] * uu

   flux_x3[idx_2d_rho,:,:]       = Q[idx_2d_rho_w,:,:]
   flux_x3[idx_2d_rho_u,:,:]     = Q[idx_2d_rho_w,:,:] * uu
   flux_x3[idx_2d_rho_w,:,:]     = Q[idx_2d_rho_w,:,:] * ww + pressure
   flux_x3[idx_2d_rho_theta,:,:] = Q[idx_2d_rho_theta,:,:] * ww

   # --- Interpolate to the element interface
   standard_slice = numpy.arange(nbsolpts)
   for elem in range(nb_elements_z):
      epais = elem * nbsolpts + standard_slice

      kfaces_var[:,elem,0,:] = mtrx.extrap_down @ Q[:,epais,:]
      kfaces_var[:,elem,1,:] = mtrx.extrap_up @ Q[:,epais,:]

   for elem in range(nb_elements_x):
      epais = elem * nbsolpts + standard_slice

      ifaces_var[:,elem,:,0] = Q[:,:,epais] @ mtrx.extrap_west
      ifaces_var[:,elem,:,1] = Q[:,:,epais] @ mtrx.extrap_east
      interfaceFlux_i[:,elem,:,0] = Q[:,:,epais] @ mtrx.extrap_west
      interfaceFlux_i[:,elem,:,1] = Q[:,:,epais] @ mtrx.extrap_east

   # --- Interface pressure
   ifaces_pres = p0 * (ifaces_var[idx_2d_rho_theta] * Rd / p0)**(cpd / cvd)
   kfaces_pres = p0 * (kfaces_var[idx_2d_rho_theta] * Rd / p0)**(cpd / cvd)

   # --- Bondary treatement

   # zeros flux BCs everywhere ...
   kfaces_flux[:,0,0,:]  = 0.0
   kfaces_flux[:,-1,1,:] = 0.0

   # Skip periodic faces
   if not geom.xperiodic:
      ifaces_flux[:, 0,:,0] = 0.0
      ifaces_flux[:,-1,:,1] = 0.0

   # except for momentum eqs where pressure is extrapolated to BCs.
   kfaces_flux[idx_2d_rho_w, 0, 0, :] = kfaces_pres[ 0, 0, :]
   kfaces_flux[idx_2d_rho_w,-1, 1, :] = kfaces_pres[-1, 1, :]

   ifaces_flux[idx_2d_rho_u, 0,:,0] = ifaces_pres[0,:,0]  # TODO : pour les cas théoriques seulement ...
   ifaces_flux[idx_2d_rho_u,-1,:,1] = ifaces_pres[-1,:,1]

   # --- Common AUSM fluxes
   for itf in range(1, nb_interfaces_z - 1):

      left  = itf - 1
      right = itf

      # Left state
      a_L = numpy.sqrt(heat_capacity_ratio * kfaces_pres[left, 1, :] / kfaces_var[idx_2d_rho, left, 1, :])
      M_L = kfaces_var[idx_2d_rho_w, left, 1, :] / (kfaces_var[idx_2d_rho, left, 1, :] * a_L)

      # Right state
      a_R = numpy.sqrt(heat_capacity_ratio * kfaces_pres[right, 0, :] / kfaces_var[idx_2d_rho, right, 0, :])
      M_R = kfaces_var[idx_2d_rho_w, right, 0, :] / (kfaces_var[idx_2d_rho, right, 0, :] * a_R)

      M = 0.25 * (( M_L + 1.)**2 - (M_R - 1.)**2)

      kfaces_flux[:,right,0,:] = (kfaces_var[:,left,1,:] * numpy.maximum(0., M) * a_L) + \
                                 (kfaces_var[:,right,0,:] * numpy.minimum(0., M) * a_R)
      kfaces_flux[idx_2d_rho_w,right,0,:] += 0.5 * ((1. + M_L) * kfaces_pres[left,1,:] + \
                                                    (1. - M_R) * kfaces_pres[right,0,:])

      kfaces_flux[:,left,1,:] = kfaces_flux[:,right,0,:]


   start = 0 if geom.xperiodic else 1

   for itf in range(start, nb_interfaces_x - 1):

      left  = itf - 1
      right = itf

      # Left state
      a_L = numpy.sqrt(heat_capacity_ratio * ifaces_pres[left, :, 1] / ifaces_var[idx_2d_rho, left, :, 1])

      # Right state
      a_R = numpy.sqrt(heat_capacity_ratio * ifaces_pres[right, :, 0] / ifaces_var[idx_2d_rho, right, :, 0])
   
      # Roe-averaged Quantities
      ifaces_u     = ifaces_var[idx_2d_rho_u] / ifaces_var[idx_2d_rho]
      ifaces_w     = ifaces_var[idx_2d_rho_w] / ifaces_var[idx_2d_rho]
      ifaces_theta = ifaces_var[idx_2d_rho_theta] / ifaces_var[idx_2d_rho]
      RT           = numpy.sqrt(ifaces_var[idx_2d_rho, right, :, 0] / ifaces_var[idx_2d_rho, left, :, 1])
      u_hat        = (ifaces_u[left,:,1] + RT * ifaces_u[right,:,0]) / (1+RT)
      w_hat        = (ifaces_w[left,:,1] + RT * ifaces_w[right,:,0]) / (1+RT)
      theta_hat    = (ifaces_theta[left,:,1] + RT * ifaces_theta[right,:,0]) / (1+RT)
      a_hat        = (a_L + RT * a_R) / (1+RT)


      # Difference in conserved variables
      drho         = ifaces_var[idx_2d_rho, right, :, 0] - ifaces_var[idx_2d_rho, left, :, 1]
      drhou        = ifaces_var[idx_2d_rho_u, right, :, 0] - ifaces_var[idx_2d_rho_u, left, :, 1]
      drhow        = ifaces_var[idx_2d_rho_w, right, :, 0] - ifaces_var[idx_2d_rho_w, left, :, 1]
      drhotheta    = ifaces_var[idx_2d_rho_theta, right, :, 0] - ifaces_var[idx_2d_rho_theta, left, :, 1]

      # Wave strength (Characteristic Variables)
      dV           = numpy.zeros((4,nbsolpts*nb_elements_z))
      dV[0]        = drho - (drhotheta/theta_hat)
      dV[1]        = drhow - (w_hat*drhotheta)/theta_hat
      dV[2]        = drhotheta/2 + (theta_hat*drhou)/(2*a_hat) - (u_hat*theta_hat*drho)/(2*a_hat) 
      dV[3]        = drhotheta/2 - (theta_hat*drhou)/(2*a_hat) + (u_hat*theta_hat*drho)/(2*a_hat)

      # Absolute values of the wave speeds (Eigenvalues)
      ws           = numpy.zeros((4,nbsolpts*nb_elements_z))
      ws[0]        = numpy.abs(u_hat)
      ws[1]        = numpy.abs(u_hat)
      ws[2]        = numpy.abs(u_hat + a_hat)
      ws[3]        = numpy.abs(u_hat - a_hat)

      # Right Eigenvectors
      R        = numpy.zeros((4,4,nbsolpts*nb_elements_z))
      R[0,0,:] = 1; R[1,0,:] = u_hat; R[2,0,:] = 0; R[3,0,:] = 0
      R[0,1,:] = 0; R[1,1,:] = 0; R[2,1,:] = 1; R[3,1,:] = 0
      R[0,2,:] = 1/theta_hat; R[1,2,:] = (u_hat+a_hat) / theta_hat; R[2,2,:] = w_hat/theta_hat; R[3,2,:] = 1
      R[0,3,:] = 1/theta_hat; R[1,3,:] = (u_hat-a_hat) / theta_hat; R[2,3,:] = w_hat/theta_hat; R[3,3,:] = 1

      # Compute the average flux 
      Roe = numpy.zeros((4,nbsolpts*nb_elements_z))
      Roe = 0.5 * ( interfaceFlux_i[:, left, :, 1] + interfaceFlux_i[:, right, :, 0] )
      
      # Add the matrix dissipation term to complete the Roe flux.
      for j in range(4):
         for k in range(4):
            print((0.5 * ws[k] * dV[k] * R[j, k,:]).max())
            Roe[j] -= 0.5 * ws[k] * dV[k] * R[j, k,:]

      
      ifaces_flux[:,right,:,0] = Roe
      ifaces_flux[:,left,:,1] = ifaces_flux[:,right,:,0]
      
   
   if geom.xperiodic:
      ifaces_flux[:, 0, :, 0] = ifaces_flux[:, -1, :, 1]

   # --- Compute the derivatives
   for elem in range(nb_elements_z):
      epais = elem * nbsolpts + standard_slice
      factor = 2.0 / geom.Δx3
      if elem < geom.nb_elements_relief_layer:
         factor = 2.0 / geom.relief_layer_delta

      df3_dx3[:, epais, :] = \
         (mtrx.diff_solpt @ flux_x3[:, epais, :] + mtrx.correction @ kfaces_flux[:, elem, :, :]) * factor

   for elem in range(nb_elements_x):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      df1_dx1[:,:,epais] = (flux_x1[:,:,epais] @ mtrx.diff_solpt.T + ifaces_flux[:,elem,:,:] @ mtrx.correction.T) * \
                           2.0/geom.Δx1

   # --- Assemble the right-hand sides
   rhs = - ( df1_dx1 + df3_dx3 )

   rhs[idx_2d_rho_w,:,:] -= Q[idx_2d_rho,:,:] * gravity

   # TODO : Add sources terms for Brikman penalization
   # It may be better to do this elementwise...
   if geom.nb_elements_relief_layer > 1:

      end = geom.nb_elements_relief_layer * nbsolpts
      etac = 1.0 # 1e-1

      normal_flux = numpy.where( \
            geom.relief_boundary_mask,
            geom.normals_x * df1_dx1[idx_2d_rho_u, :end, :] + geom.normals_z * df3_dx3[idx_2d_rho_w, :end, :],
            0.0)

      rhs[idx_2d_rho_u, :end, :] = numpy.where( \
            geom.relief_mask, -(1.0 / etac) * normal_flux * geom.normals_x, rhs[idx_2d_rho_u, :end, :])
      rhs[idx_2d_rho_w, :end, :] = numpy.where( \
            geom.relief_mask, -(1.0 / etac) * normal_flux * geom.normals_z, rhs[idx_2d_rho_w, :end, :])

   pdb.set_trace()
   return rhs
