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
   kfaces_pres = numpy.empty((nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)

   ifaces_flux = numpy.empty((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)
   ifaces_var  = numpy.empty((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)
   ifaces_pres = numpy.empty((nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)

   # --- Unpack physical variables
   rho      = Q[idx_2d_rho,:,:]
   uu       = Q[idx_2d_rho_u,:,:] / rho
   ww       = Q[idx_2d_rho_w,:,:] / rho
   rho_E    = Q[idx_2d_rho_theta,:,:]
   # pressure = p0 * numpy.exp((cpd/cvd) * numpy.log((Rd/p0)*Q[idx_2d_rho_theta, :, :])) # in general
   # pressure = Rd * Q[idx_2d_rho_theta, :, :] # For vortex problem
   pressure = 0.4 * (Q[idx_2d_rho_theta, :, :] - 0.5*rho*(uu**2 + ww**2))
   Q[idx_2d_rho_theta] += pressure

   

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
      kfaces_pres[elem,0,:] = mtrx.extrap_down @ pressure[epais,:]
      kfaces_pres[elem,1,:] = mtrx.extrap_up @ pressure[epais,:]


   for elem in range(nb_elements_x):
      epais = elem * nbsolpts + standard_slice

      ifaces_var[:,elem,:,0] = Q[:,:,epais] @ mtrx.extrap_west
      ifaces_var[:,elem,:,1] = Q[:,:,epais] @ mtrx.extrap_east
      ifaces_pres[elem,:,0] = pressure[:,epais] @ mtrx.extrap_west
      ifaces_pres[elem,:,1] = pressure[:,epais] @ mtrx.extrap_east

   # --- Interface pressure
   # ifaces_pres = 0.4 * (ifaces_rho_E - (0.5 / ifaces_var[idx_2d_rho])*(ifaces_var[idx_2d_rho_u]**2 + ifaces_var[idx_2d_rho_w]**2))
   # kfaces_pres = 0.4 * (kfaces_rho_E - (0.5 / kfaces_var[idx_2d_rho])*(kfaces_var[idx_2d_rho_u]**2 + kfaces_var[idx_2d_rho_w]**2))

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

      kfaces_flux[:,right,0,:] = (kfaces_var[:,left,1,:] * numpy.maximum(0., M) * a_L) + \
                                 (kfaces_var[:,right,0,:] * numpy.minimum(0., M) * a_R)
      kfaces_flux[idx_2d_rho_w,right,0,:] += 0.5 * ((1. + M_L) * kfaces_pres[left,1,:] + \
                                                    (1. - M_R) * kfaces_pres[right,0,:])

      kfaces_flux[:,left,1,:] = kfaces_flux[:,right,0,:]

   if geom.zperiodic:
      kfaces_flux[:, 0, 0, :] = kfaces_flux[:, -1, 1, :]


   start = 0 if geom.xperiodic else 1
   for itf in range(start, nb_interfaces_x - 1):

      left  = itf - 1
      right = itf

      # Left state
      a_L = numpy.sqrt(heat_capacity_ratio * ifaces_pres[left, :, 1] / ifaces_var[idx_2d_rho, left, :, 1])
      M_L = ifaces_var[idx_2d_rho_u, left, :, 1] / (ifaces_var[idx_2d_rho, left, :, 1] * a_L)

      # Right state
      a_R = numpy.sqrt(heat_capacity_ratio * ifaces_pres[right, :, 0] / ifaces_var[idx_2d_rho, right, :, 0])
      M_R = ifaces_var[idx_2d_rho_u, right, :, 0] / ( ifaces_var[idx_2d_rho, right, :, 0] * a_R)

      M = 0.25 * ((M_L + 1.)**2 - (M_R - 1.)**2)

      ifaces_flux[:,right,:,0] = (ifaces_var[:,left,:,1] * numpy.maximum(0., M) * a_L) + \
                                 (ifaces_var[:,right,:,0] * numpy.minimum(0., M) * a_R)
      ifaces_flux[idx_2d_rho_u,right,:,0] += 0.5 * ((1. + M_L) * ifaces_pres[left,:,1] + \
                                                    (1. - M_R) * ifaces_pres[right,:,0])

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
