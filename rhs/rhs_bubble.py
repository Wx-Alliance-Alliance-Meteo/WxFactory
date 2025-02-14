import numpy
import pdb
import sympy as sp

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

   kfaces_flux     = numpy.zeros((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)
   kfaces_var      = numpy.zeros((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)
   kfaces_pres     = numpy.zeros((nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)
   kfaces_enthalpy = numpy.zeros((nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)
   kfaces_height   = numpy.zeros((nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)

   ifaces_flux     = numpy.zeros((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)
   ifaces_var      = numpy.zeros((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)
   ifaces_pres     = numpy.zeros((nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)
   ifaces_enthalpy = numpy.zeros((nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)
   ifaces_height   = numpy.zeros((nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)


   # --- Unpack physical variables
   rho      = Q[idx_2d_rho,:,:]
   uu       = Q[idx_2d_rho_u,:,:] / rho
   ww       = Q[idx_2d_rho_w,:,:] / rho
   ee       = Q[idx_2d_rho_theta,:,:] / rho
   height   = geom.X3

   pressure = (heat_capacity_ratio-1) * (Q[idx_2d_rho_theta, :, :] - 0.5*rho*(uu**2 + ww**2) - rho*gravity*geom.X3)
   enthalpy = (heat_capacity_ratio/(heat_capacity_ratio-1))*(pressure/rho) + 0.5*(uu**2 + ww**2) + gravity*geom.X3

   # --- Compute the fluxes
   flux_x1[idx_2d_rho,:,:]       = Q[idx_2d_rho_u,:,:]
   flux_x1[idx_2d_rho_u,:,:]     = Q[idx_2d_rho_u,:,:] * uu + pressure
   flux_x1[idx_2d_rho_w,:,:]     = Q[idx_2d_rho_u,:,:] * ww
   flux_x1[idx_2d_rho_theta,:,:] = (Q[idx_2d_rho_theta,:,:] + pressure) * uu 
   

   flux_x3[idx_2d_rho,:,:]       = Q[idx_2d_rho_w,:,:]
   flux_x3[idx_2d_rho_u,:,:]     = Q[idx_2d_rho_w,:,:] * uu
   flux_x3[idx_2d_rho_w,:,:]     = Q[idx_2d_rho_w,:,:] * ww + pressure
   flux_x3[idx_2d_rho_theta,:,:] = (Q[idx_2d_rho_theta,:,:] + pressure) * ww


   # --- Interpolate to the element interface
   standard_slice = numpy.arange(nbsolpts)
   for elem in range(nb_elements_z):
      epais = elem * nbsolpts + standard_slice

      kfaces_var[:,elem,0,:]    = mtrx.extrap_down @ Q[:,epais,:]
      kfaces_var[:,elem,1,:]    = mtrx.extrap_up @ Q[:,epais,:]
      kfaces_pres[elem,0,:]     = mtrx.extrap_down @ pressure[epais,:]
      kfaces_pres[elem,1,:]     = mtrx.extrap_up @ pressure[epais,:]
      kfaces_height[elem,0,:]   = mtrx.extrap_down @ height[epais,:]
      kfaces_height[elem,1,:]   = mtrx.extrap_up @ height[epais,:]
      kfaces_enthalpy[elem,0,:] = mtrx.extrap_down @ enthalpy[epais,:]
      kfaces_enthalpy[elem,1,:] = mtrx.extrap_up @ enthalpy[epais,:]


   for elem in range(nb_elements_x):
      epais = elem * nbsolpts + standard_slice

      ifaces_var[:,elem,:,0]    = Q[:,:,epais] @ mtrx.extrap_west
      ifaces_var[:,elem,:,1]    = Q[:,:,epais] @ mtrx.extrap_east
      ifaces_pres[elem,:,0]     = pressure[:,epais] @ mtrx.extrap_west
      ifaces_pres[elem,:,1]     = pressure[:,epais] @ mtrx.extrap_east
      ifaces_height[elem,:,0]   = height[:,epais] @ mtrx.extrap_west
      ifaces_height[elem,:,1]   = height[:,epais] @ mtrx.extrap_east
      ifaces_enthalpy[elem,:,0] = enthalpy[:,epais] @ mtrx.extrap_west
      ifaces_enthalpy[elem,:,1] = enthalpy[:,epais] @ mtrx.extrap_east

   
   # --- Bondary treatement

   # zeros flux BCs everywhere ...
   kfaces_flux[:,0,0,:]  = 0.0
   kfaces_flux[:,-1,1,:] = 0.0

   # Skip periodic faces
   # if not geom.zperiodic:
   #    kfaces_flux[:,0,0,:]  = 0.0
   #    kfaces_flux[:,-1,1,:] = 0.0

   if not geom.xperiodic:
      ifaces_flux[:, 0,:,0] = 0.0
      ifaces_flux[:,-1,:,1] = 0.0

   # except for momentum eqs where pressure is extrapolated to BCs.
   kfaces_flux[idx_2d_rho_w, 0, 0, :] = kfaces_pres[ 0, 0, :]
   kfaces_flux[idx_2d_rho_w,-1, 1, :] = kfaces_pres[-1, 1, :]

   ifaces_flux[idx_2d_rho_u, 0,:,0] = ifaces_pres[0,:,0]  # TODO : pour les cas théoriques seulement ...
   ifaces_flux[idx_2d_rho_u,-1,:,1] = ifaces_pres[-1,:,1]


   # Compute ifaces u and w
   ifaces_u   = ifaces_var[idx_2d_rho_u] / ifaces_var[idx_2d_rho]
   ifaces_w   = ifaces_var[idx_2d_rho_w] / ifaces_var[idx_2d_rho] 
   kfaces_u   = kfaces_var[idx_2d_rho_u] / kfaces_var[idx_2d_rho]
   kfaces_w   = kfaces_var[idx_2d_rho_w] / kfaces_var[idx_2d_rho]

   
   # --- Common Roe fluxes
   # start = 0 if geom.zperiodic else 1
   start = 1
   for itf in range(start, nb_interfaces_z - 1):

      left  = itf - 1
      right = itf

      
      # Compute left and right flux
      flux_L  = kfaces_var[:, left, 1, :] * kfaces_w[left, 1, :]
      flux_L[idx_2d_rho_theta] += kfaces_pres[left, 1, :] * kfaces_w[left, 1, :]
      flux_R  = kfaces_var[:, right, 0, :] * kfaces_w[right, 0, :]
      flux_R[idx_2d_rho_theta]  += kfaces_pres[right, 0, :] * kfaces_w[right, 0, :]
      # Add pressure with rho*u*u
      flux_L[idx_2d_rho_w] += kfaces_pres[left, 1, :]
      flux_R[idx_2d_rho_w] += kfaces_pres[right, 0, :]

      # Compute Roe-Averages
      RT         = numpy.sqrt( kfaces_var[idx_2d_rho, right, 0, :] / kfaces_var[idx_2d_rho, left, 1, :] )
      rho_avg    = RT * kfaces_var[idx_2d_rho, left, 1, :]
      u_avg      = (RT * kfaces_u[right, 0, :] + kfaces_u[left, 1, :]) / (RT+1)
      w_avg      = (RT * kfaces_w[right, 0, :] + kfaces_w[left, 1, :]) / (RT+1)
      H_avg      = (RT * kfaces_enthalpy[right, 0, :] + kfaces_enthalpy[left, 1, :]) / (RT+1)
      height_avg = 0.5 * (kfaces_height[right, 0, :] + kfaces_height[left, 1, :])
      c          = numpy.sqrt((heat_capacity_ratio-1)*(H_avg - 0.5*(u_avg**2 + w_avg**2) - gravity*height_avg))

      # Compute local Mach number M, numrically defined the average of both sides of the interface
      a_L = numpy.sqrt(heat_capacity_ratio * kfaces_pres[left, 1, :] / kfaces_var[idx_2d_rho, left, 1, :])
      M_L = numpy.sqrt(kfaces_u[left, 1, :]**2 + kfaces_w[left, 1, :]**2) / a_L #kfaces_w[left, 1, :] / a_L
      a_R = numpy.sqrt(heat_capacity_ratio * kfaces_pres[right, 0, :] / kfaces_var[idx_2d_rho, right, 0, :])
      M_R = numpy.sqrt(kfaces_u[right, 0, :]**2 + kfaces_w[right, 0, :]**2) / a_R #kfaces_w[right, 0, :] / a_R
      M = 0.5 * (M_L + M_R)

      # Compute preconditioning parameter delta 
      mu       = numpy.minimum(1, numpy.maximum(M, 1E-7))     # Assumed M_cut = 1E-7
      delta    = 1/mu - 1

      # Auxiliary Variables to compute eigenvectors and its inverse
      alph1   = 0.5*(u_avg**2 + w_avg**2)         # alpha
      psi     = heat_capacity_ratio-1
      omega   = delta / (1 + delta**2)
      tau     = numpy.sqrt(c**2*(1+delta**2) - delta**2*w_avg**2)
      lamb3   = w_avg + tau
      lamb4   = w_avg - tau
      S1      = -(rho_avg / (2*c*tau)) * ((c + omega*lamb3)*numpy.abs(lamb4) - (c + omega*lamb4)*numpy.abs(lamb3))
      S2      = - (1 / (2*rho_avg*c*tau)) * ((c - omega*lamb3)*numpy.abs(lamb4) - (c - omega*lamb4)*numpy.abs(lamb3))
      S3      = (numpy.abs(lamb3)+numpy.abs(lamb4)) / (2*(1+delta**2)) + (omega*delta*u_avg*(numpy.abs(lamb3)-numpy.abs(lamb4))) / (2*tau)
      zeta1   = (S3 - numpy.abs(w_avg)) / c**2
      zeta2   = S2*rho_avg + w_avg*zeta1
      zeta3   = (S3*rho_avg + S1*w_avg) / rho_avg
      zeta4   = (S3/psi) + alph1*zeta1 + S2*rho_avg*w_avg
      zeta5   = ( S3*rho_avg*w_avg + ((S1*c**2)/psi) + S1*alph1 ) / rho_avg
      
      # Compute vector ΔU
      ΔU      = kfaces_var[:, right, 0, :] - kfaces_var[:, left, 1, :]

      # Compute preconditioned dissipation matrix
      D = numpy.zeros((nbsolpts*nb_elements_x, nb_equations, nb_equations))
      D[:,0,0] = numpy.abs(w_avg) - ((S1*w_avg)/rho_avg) + alph1*psi*zeta1; D[:,0,1] = -psi*u_avg*zeta1; D[:,0,2] = (S1/rho_avg) - psi*w_avg*zeta1; D[:,0,3] = psi*zeta1
      D[:,1,0] = alph1*psi*u_avg*zeta1 - (S1*u_avg*w_avg)/rho_avg; D[:,1,1] = numpy.abs(w_avg) - psi*u_avg**2*zeta1; D[:,1,2] = (S1*u_avg)/rho_avg - psi*u_avg*w_avg*zeta1; D[:,1,3] = psi*u_avg*zeta1
      D[:,2,0] = w_avg*numpy.abs(w_avg) + alph1*psi*zeta2 - w_avg*zeta3; D[:,2,1] = -psi*u_avg*zeta2; D[:,2,2] = -w_avg*psi*zeta2 + zeta3; D[:,2,3] = psi*zeta2
      D[:,3,0] = alph1*numpy.abs(w_avg) - w_avg*zeta5 - u_avg**2*numpy.abs(w_avg) + alph1*psi*zeta4; D[:,3,1] =u_avg*numpy.abs(w_avg) - psi*u_avg*zeta4; D[:,3,2] =  zeta5 - psi*w_avg*zeta4; D[:,3,3] = psi*zeta4

      # Compute P*|lambda|*Pinv*ΔU
      phi = numpy.array([numpy.dot(D[i], ΔU[:,i]) for i in range(D.shape[0])])

      # Final common flux
      kfaces_flux[:,right,0,:] = 0.5*(flux_L + flux_R) - 0.5*phi.T

      kfaces_flux[:,left,1,:] = kfaces_flux[:,right,0,:]


   # if geom.zperiodic:
   #    kfaces_flux[:, 0, 0, :] = kfaces_flux[:, -1, 1, :]



   # ifaces flux
   start      = 0 if geom.xperiodic else 1
   for itf in range(start, nb_interfaces_x - 1):

      left    = itf - 1
      right   = itf

      # Compute left and right flux
      flux_L  = ifaces_var[:, left, :, 1] * ifaces_u[left, :, 1]
      flux_L[idx_2d_rho_theta] += ifaces_pres[left, :, 1] * ifaces_u[left, :, 1]
      flux_R  = ifaces_var[:, right, :, 0] * ifaces_u[right, :, 0]
      flux_R[idx_2d_rho_theta]  += ifaces_pres[right, :, 0] * ifaces_u[right, :, 0]
      # Add pressure with rho*u*u
      flux_L[idx_2d_rho_u] += ifaces_pres[left,:,1]
      flux_R[idx_2d_rho_u] += ifaces_pres[right,:,0]


      # Compute Roe-Averages
      RT         = numpy.sqrt( ifaces_var[idx_2d_rho, right, :, 0] / ifaces_var[idx_2d_rho, left, :, 1] )
      rho_avg    = RT * ifaces_var[idx_2d_rho, left, :, 1]
      u_avg      = (RT * ifaces_u[right, :, 0] + ifaces_u[left, :, 1]) / (RT+1)
      w_avg      = (RT * ifaces_w[right, :, 0] + ifaces_w[left, :, 1]) / (RT+1)
      H_avg      = (RT * ifaces_enthalpy[right, :, 0] + ifaces_enthalpy[left, :, 1]) / (RT+1)
      height_avg = 0.5 * (ifaces_height[right, :, 0] + ifaces_height[left, :, 1])
      c          = numpy.sqrt((heat_capacity_ratio-1)*(H_avg - 0.5*(u_avg**2 + w_avg**2) - gravity*height_avg))  # Speed of sound c

      # Compute local Mach number M, numrically defined the average of both sides of the interface
      a_L = numpy.sqrt(heat_capacity_ratio * ifaces_pres[left, :, 1] / ifaces_var[idx_2d_rho, left, :, 1])
      M_L = numpy.sqrt(ifaces_u[left, :, 1]**2 + ifaces_w[left, :, 1]**2) / a_L  #ifaces_u[left, :, 1] / a_L
      a_R = numpy.sqrt(heat_capacity_ratio * ifaces_pres[right, :, 0] / ifaces_var[idx_2d_rho, right, :, 0])
      M_R = numpy.sqrt(ifaces_u[right, :, 0]**2 + ifaces_w[right, :, 0]**2) / a_R  #ifaces_u[right, :, 0] / a_R
      M   = 0.5*(M_L + M_R)

      # Compute preconditioning parameter delta 
      mu       = numpy.minimum(1, numpy.maximum(M, 1E-7))     # Assumed M_cut = 0
      delta    = 1/mu -1
   

      # Auxiliary Variables to compute preconditioned dissipation matrix
      alph1   = 0.5*(u_avg**2 + w_avg**2)
      psi     = heat_capacity_ratio-1
      omega   = delta / (1 + delta**2)
      tau     = numpy.sqrt(c**2*(1+delta**2) - delta**2*u_avg**2)
      lamb3   = u_avg + tau
      lamb4   = u_avg - tau
      S1      = -(rho_avg / (2*c*tau)) * ((c + omega*lamb3)*numpy.abs(lamb4) - (c + omega*lamb4)*numpy.abs(lamb3))
      S2      = - (1 / (2*rho_avg*c*tau)) * ((c - omega*lamb3)*numpy.abs(lamb4) - (c - omega*lamb4)*numpy.abs(lamb3))
      S3      = (numpy.abs(lamb3)+numpy.abs(lamb4)) / (2*(1+delta**2)) + (omega*delta*u_avg*(numpy.abs(lamb3)-numpy.abs(lamb4))) / (2*tau)
      zeta1   = (S3 - numpy.abs(u_avg)) / c**2
      zeta2   = S2*rho_avg + u_avg*zeta1
      zeta3   = (S3*rho_avg + S1*u_avg) / rho_avg
      zeta4   = (S3/psi) + alph1*zeta1 + S2*rho_avg*u_avg
      zeta5   = ( S3*rho_avg*u_avg + ((S1*c**2)/psi) + S1*alph1 ) / rho_avg


      # Compute vector ΔU
      ΔU      = ifaces_var[:, right, :, 0] - ifaces_var[:, left, :, 1]


      # Compute preconditioned dissipation matrix
      D = numpy.zeros((nbsolpts*nb_elements_z, nb_equations, nb_equations))
      D[:,0,0] = numpy.abs(u_avg) - ((S1*u_avg)/rho_avg) + alph1*psi*zeta1; D[:,0,1] = (S1/rho_avg) - psi*u_avg*zeta1; D[:,0,2] = -psi*w_avg*zeta1; D[:,0,3] = psi*zeta1
      D[:,1,0] = u_avg*numpy.abs(u_avg) + alph1*psi*zeta2 - u_avg*zeta3; D[:,1,1] = -u_avg*psi*zeta2 + zeta3; D[:,1,2] = -psi*w_avg*zeta2; D[:,1,3] = psi*zeta2
      D[:,2,0] = alph1*psi*w_avg*zeta1 - (S1*u_avg*w_avg)/rho_avg; D[:,2,1] = (S1*w_avg)/rho_avg - psi*u_avg*w_avg*zeta1; D[:,2,2] = numpy.abs(u_avg) - psi*w_avg**2*zeta1; D[:,2,3] = psi*w_avg*zeta1
      D[:,3,0] = alph1*numpy.abs(u_avg) - u_avg*zeta5 - w_avg**2*numpy.abs(u_avg) + alph1*psi*zeta4; D[:,3,1] = zeta5 - psi*u_avg*zeta4; D[:,3,2] = w_avg*numpy.abs(u_avg) - psi*w_avg*zeta4; D[:,3,3] = psi*zeta4

      # Compute P*|lambda|*Pinv*ΔU
      phi = numpy.array([numpy.dot(D[i], ΔU[:,i]) for i in range(D.shape[0])])

      # Final common flux
      ifaces_flux[:,right,:,0] = 0.5*(flux_L + flux_R) - 0.5*phi.T

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

   
   return rhs
