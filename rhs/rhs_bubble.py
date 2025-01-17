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

   kfaces_flux     = numpy.empty((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)
   kfaces_var      = numpy.empty((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)
   kfaces_pres     = numpy.empty((nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)
   kfaces_enthalpy = numpy.empty((nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)

   ifaces_flux = numpy.empty((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)
   ifaces_var  = numpy.empty((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)
   ifaces_pres = numpy.empty((nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)
   ifaces_pres = numpy.empty((nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)


   # --- Unpack physical variables
   rho      = Q[idx_2d_rho,:,:]
   uu       = Q[idx_2d_rho_u,:,:] / rho
   ww       = Q[idx_2d_rho_w,:,:] / rho
   ee       = Q[idx_2d_rho_w,:,:] / rho

   pressure = (heat_capacity_ratio-1) * (Q[idx_2d_rho_theta, :, :] - 0.5*rho*(uu**2 + ww**2))


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

      kfaces_var[:,elem,0,:] = mtrx.extrap_down @ Q[:,epais,:]
      kfaces_var[:,elem,1,:] = mtrx.extrap_up @ Q[:,epais,:]
      kfaces_pres[elem,0,:] = mtrx.extrap_down @ pressure[epais,:]
      kfaces_pres[elem,1,:] = mtrx.extrap_up @ pressure[epais,:]


   for elem in range(nb_elements_x):
      epais = elem * nbsolpts + standard_slice

      ifaces_var[:,elem,:,0] = Q[:,:,epais] @ mtrx.extrap_west
      ifaces_var[:,elem,:,1] = Q[:,:,epais] @ mtrx.extrap_east
      ifaces_pres[elem,:,0] = pressure[:,epais] @ mtrx.extrap_west
      ifaces_pres[elem,:,1] = pressure[:,epais] @ mtrx.extrap_east

   
   # --- Bondary treatement

   # zeros flux BCs everywhere ...

   # Skip periodic faces
   if not geom.zperiodic:
      kfaces_flux[:,0,0,:]  = 0.0
      kfaces_flux[:,-1,1,:] = 0.0

   if not geom.xperiodic:
      ifaces_flux[:, 0,:,0] = 0.0
      ifaces_flux[:,-1,:,1] = 0.0

   # except for momentum eqs where pressure is extrapolated to BCs.
   kfaces_flux[idx_2d_rho_w, 0, 0, :] = kfaces_pres[ 0, 0, :]
   kfaces_flux[idx_2d_rho_w,-1, 1, :] = kfaces_pres[-1, 1, :]

   ifaces_flux[idx_2d_rho_u, 0,:,0] = ifaces_pres[0,:,0]  # TODO : pour les cas théoriques seulement ...
   ifaces_flux[idx_2d_rho_u,-1,:,1] = ifaces_pres[-1,:,1]
   
   # --- Common AUSM fluxes
   start = 0 if geom.zperiodic else 1
   for itf in range(start, nb_interfaces_z - 1):

      left  = itf - 1
      right = itf

      # Left state
      a_L = numpy.sqrt(heat_capacity_ratio * kfaces_pres[left, 1, :] / kfaces_var[idx_2d_rho, left, 1, :])
      M_L = kfaces_var[idx_2d_rho_w, left, 1, :] / (kfaces_var[idx_2d_rho, left, 1, :] * a_L)

      # Right state
      a_R = numpy.sqrt(heat_capacity_ratio * kfaces_pres[right, 0, :] / kfaces_var[idx_2d_rho, right, 0, :])
      M_R = kfaces_var[idx_2d_rho_w, right, 0, :] / (kfaces_var[idx_2d_rho, right, 0, :] * a_R)
      
      M = 0.25 * (( M_L + 1.)**2 - (M_R - 1.)**2)

      kfaces_flux[0:3,right,0,:] = (kfaces_var[0:3,left,1,:] * numpy.maximum(0., M) * a_L) + \
                                 (kfaces_var[0:3,right,0,:] * numpy.minimum(0., M) * a_R)
      kfaces_flux[3,right,0,:] = ((kfaces_var[3,left,1,:] + kfaces_pres[left,1,:]) * numpy.maximum(0., M) * a_L) + \
                                 ((kfaces_var[3,right,0,:] + kfaces_pres[right,0,:]) * numpy.minimum(0., M) * a_R)
      

      kfaces_flux[idx_2d_rho_w,right,0,:] += 0.5 * ((1. + M_L) * kfaces_pres[left,1,:] + \
                                                    (1. - M_R) * kfaces_pres[right,0,:])

      kfaces_flux[:,left,1,:] = kfaces_flux[:,right,0,:]

   if geom.zperiodic:
      kfaces_flux[:, 0, 0, :] = kfaces_flux[:, -1, 1, :]

   # Compute ifaces u and w
   ifaces_u   = ifaces_var[idx_2d_rho_u] / ifaces_var[idx_2d_rho]
   ifaces_w   = ifaces_var[idx_2d_rho_w] / ifaces_var[idx_2d_rho] 

   # --- Interface total enthalpy
   #ifaces_enthalpy = (ifaces_var[idx_2d_rho_theta] + ifaces_pres) / ifaces_var[idx_2d_rho]
   ifaces_enthalpy = (1.4/0.4)*(ifaces_pres/ifaces_var[idx_2d_rho]) + 0.5*(ifaces_u**2 + ifaces_w**2)
   kfaces_enthalpy = (kfaces_var[idx_2d_rho_theta] + kfaces_pres) / kfaces_var[idx_2d_rho]


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
      RT      = numpy.sqrt( ifaces_var[idx_2d_rho, right, :, 0] / ifaces_var[idx_2d_rho, left, :, 1] )
      rho_avg = RT * ifaces_var[idx_2d_rho, left, :, 1]
      u_avg   = (RT * ifaces_u[right, :, 0] + ifaces_u[left, :, 1]) / (RT+1)
      w_avg   = (RT * ifaces_w[right, :, 0] + ifaces_w[left, :, 1]) / (RT+1)
      H_avg   = (RT * ifaces_enthalpy[right, :, 0] + ifaces_enthalpy[left, :, 1]) / (RT+1)
      a       = numpy.sqrt((heat_capacity_ratio-1)*(H_avg - 0.5*(u_avg**2 + w_avg**2)))

      # Auxiliary Variables to compute eigenvectors and its inverse
      alph1   = 0.5*(u_avg**2 + w_avg**2)
      alph2   = (heat_capacity_ratio-1) / a**2
      alph3   = 0.5 * (heat_capacity_ratio-1)
      alph4   = 1 / (heat_capacity_ratio-1)

      # Compute vector ΔU
      ΔU      = ifaces_var[:, right, :, 0] - ifaces_var[:, left, :, 1]


      # Compute matrix of eigenvector
      P = numpy.zeros((nbsolpts*nb_elements_z, nb_equations, nb_equations))
      P[:,0,0] = 1; P[:,0,1] = 0; P[:,0,2] = 1/a**2; P[:,0,3] = 1/a**2
      P[:,1,0] = u_avg; P[:,1,1] = 0; P[:,1,2] = (u_avg/a**2) + (1/a); P[:,1,3] = (u_avg/a**2) - (1/a)
      P[:,2,0] = w_avg; P[:,2,1] = rho_avg; P[:,2,2] = w_avg/a**2; P[:,2,3] = w_avg/a**2
      P[:,3,0] = alph1; P[:,3,1] = rho_avg*w_avg; P[:,3,2] = alph4+(alph1/a**2)+(u_avg/a); P[:,3,3] = alph4+(alph1/a**2)-(u_avg/a)


      # Compute matrix of inverse eigenvector
      Pinv = numpy.zeros((nbsolpts*nb_elements_z, nb_equations, nb_equations))
      Pinv[:, 0, 0] = 1 - (alph1*alph2); Pinv[:, 0, 1] = u_avg*alph2; Pinv[:, 0, 2] = w_avg*alph2; Pinv[:, 0, 3] = -alph2
      Pinv[:, 1, 0] = -(w_avg/rho_avg); Pinv[:, 1, 1] = 0; Pinv[:, 1, 2] = 1/rho_avg; Pinv[:, 1, 3] = 0 
      Pinv[:, 2, 0] = alph1*alph3 - 0.5*a*u_avg; Pinv[:, 2, 1] = a/2 - u_avg*alph3; Pinv[:, 2, 2] = -w_avg*alph3; Pinv[:, 2, 3] = alph3
      Pinv[:, 3, 0] = alph1*alph3 + 0.5*a*u_avg; Pinv[:, 3, 1] = -a/2 - u_avg*alph3; Pinv[:, 3, 2] = -w_avg*alph3; Pinv[:, 3, 3] = alph3
      

      # Compute matrix of eigenvalues
      # Initialize the lamb matrix with shape (30, 4, 4)
      lamb = numpy.zeros((nbsolpts*nb_elements_z, nb_equations, nb_equations))
      lamb[:, 0, 0] = numpy.abs(u_avg); lamb[:, 1, 1] = numpy.abs(u_avg); lamb[:, 2, 2] = numpy.abs(u_avg + a); lamb[:, 3, 3] = numpy.abs(u_avg - a)


      # Compute Roe matrix
      A = numpy.array([numpy.dot(P[i], lamb[i]) for i in range(P.shape[0])])
      B = numpy.array([numpy.dot(A[i], Pinv[i]) for i in range(A.shape[0])])

      # Compute P*|lambda|*Pinv*-w_avg*ΔU
      phi = numpy.array([numpy.dot(B[i], ΔU[:,i]) for i in range(B.shape[0])])

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

   # rhs[idx_2d_rho_w,:,:] -= Q[idx_2d_rho,:,:] * gravity

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
