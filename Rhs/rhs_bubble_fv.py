import numpy

from constants import *

def rhs_bubble_fv(Q, geom, mtrx, nbsolpts, nb_elements_x, nb_elements_z):

   datatype = Q.dtype

   nb_interfaces_x = nb_elements_x + 1
   nb_interfaces_z = nb_elements_z + 1

   # flux_x1 = numpy.empty_like(Q, dtype=datatype)
   # flux_x3 = numpy.empty_like(Q, dtype=datatype)

   df1_dx = numpy.empty_like(Q, dtype=datatype)
   df3_dz = numpy.empty_like(Q, dtype=datatype)

   kfaces_flux = numpy.empty((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)
   kfaces_var  = numpy.empty((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)

   ifaces_flux = numpy.empty((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)
   ifaces_var  = numpy.empty((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)

   kfaces_pres = numpy.empty((nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)
   ifaces_pres = numpy.empty((nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)

   # --- Unpack physical variables
   rho      = Q[RHO,:,:]
   uu       = Q[RHO_U,:,:] / rho
   ww       = Q[RHO_W,:,:] / rho
   pressure = P0 * (Q[RHO_THETA,:,:] * Rd / P0)**(cpd / cvd)

   # --- Compute the fluxes
   # flux_x1[RHO,:,:]       = Q[RHO_U,:,:]
   # flux_x1[RHO_U,:,:]     = Q[RHO_U,:,:] * uu + pressure
   # flux_x1[RHO_W,:,:]     = Q[RHO_U,:,:] * ww
   # flux_x1[RHO_THETA,:,:] = Q[RHO_THETA,:,:] * uu

   # flux_x3[RHO,:,:]       = Q[RHO_W,:,:]
   # flux_x3[RHO_U,:,:]     = Q[RHO_W,:,:] * uu
   # flux_x3[RHO_W,:,:]     = Q[RHO_W,:,:] * ww + pressure
   # flux_x3[RHO_THETA,:,:] = Q[RHO_THETA,:,:] * ww

   # --- Interpolate to the element interface
   for elem in range(nb_elements_z):
      epais = elem

      kfaces_var[:,elem,0,:] = Q[:,epais,:]
      kfaces_var[:,elem,1,:] = Q[:,epais,:]

   for elem in range(nb_elements_x):
      epais = elem

      ifaces_var[:,elem,:,0] = Q[:,:,epais]
      ifaces_var[:,elem,:,1] = Q[:,:,epais]

   # --- Interface pressure
   ifaces_pres = P0 * (ifaces_var[RHO_THETA] * Rd / P0)**(cpd / cvd)
   kfaces_pres = P0 * (kfaces_var[RHO_THETA] * Rd / P0)**(cpd / cvd)

   # --- Bondary treatement

   # zeros flux BCs everywhere ...
   kfaces_flux[:,0,0,:]  = 0.0
   kfaces_flux[:,-1,1,:] = 0.0

   ifaces_flux[:, 0,:,0] = 0.0
   ifaces_flux[:,-1,:,1] = 0.0

   # except for momentum eqs where pressure is extrapolated to BCs.
   kfaces_flux[RHO_W, 0, 0,:] = kfaces_pres[0,0,:]
   kfaces_flux[RHO_W,-1,1,:]  = kfaces_pres[-1,1,:]

   ifaces_flux[RHO_U, 0,:,0] = ifaces_pres[0,:,0]  # TODO : pour les cas théoriques seulement ...
   ifaces_flux[RHO_U,-1,:,1] = ifaces_pres[-1,:,1]

   # --- Common AUSM fluxes
   for itf in range(1, nb_interfaces_z - 1):

      left  = itf - 1
      right = itf

      # Left state
      a_L   = numpy.sqrt( heat_capacity_ratio * kfaces_pres[left,1,:] / kfaces_var[RHO,left,1,:] )
      M_L   = kfaces_var[RHO_W,left,1,:] / ( kfaces_var[RHO,left,1,:] * a_L )

      # Right state
      a_R   = numpy.sqrt( heat_capacity_ratio * kfaces_pres[right,0,:] / kfaces_var[RHO,right,0,:] )
      M_R   = kfaces_var[RHO_W,right,0,:] / ( kfaces_var[RHO,right,0,:] * a_R )

      M = 0.25 * (( M_L + 1.)**2 - (M_R - 1.)**2)

      kfaces_flux[:,right,0,:] = (kfaces_var[:,left,1,:] * numpy.maximum(0., M) * a_L) + (kfaces_var[:,right,0,:] * numpy.minimum(0., M) * a_R)
      kfaces_flux[RHO_W,right,0,:] += 0.5 * ( (1. + M_L) * kfaces_pres[left,1,:] + (1. - M_R) * kfaces_pres[right,0,:] )
   
      kfaces_flux[:,left,1,:] = kfaces_flux[:,right,0,:]

   for itf in range(1, nb_interfaces_x - 1):

      left  = itf - 1
      right = itf

      # Left state
      a_L   = numpy.sqrt( heat_capacity_ratio * ifaces_pres[left,:,1] / ifaces_var[RHO,left,:,1] )
      M_L   = ifaces_var[RHO_U,left,:,1] / ( ifaces_var[RHO,left,:,1] * a_L )

      # Right state
      a_R   = numpy.sqrt( heat_capacity_ratio * ifaces_pres[right,:,0] / ifaces_var[RHO,right,:,0] )
      M_R   = ifaces_var[RHO_U,right,:,0] / ( ifaces_var[RHO,right,:,0] * a_R )

      M = 0.25 * ((M_L + 1.)**2 - (M_R - 1.)**2)

      ifaces_flux[:,right,:,0] = (ifaces_var[:,left,:,1] * numpy.maximum(0., M) * a_L) + (ifaces_var[:,right,:,0] * numpy.minimum(0., M) * a_R)
      ifaces_flux[RHO_U,right,:,0] += 0.5 * ( (1. + M_L) * ifaces_pres[left,:,1] + (1. - M_R) * ifaces_pres[right,:,0] )

      ifaces_flux[:,left,:,1] = ifaces_flux[:,right,:,0]

   # --- Compute the derivatives
   for elem in range(nb_elements_z):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      # if elem == 0:
      #    print(f'correction: {mtrx.correction}')
      #    print(f'shape: {kfaces_flux[:, elem, :, :].shape}')
      #    print(f'flux \n{kfaces_flux[:, elem, :, :]}')
      #    print(f'product: \n{mtrx.correction @ kfaces_flux[:,elem,:,:]}')
      #    print(f'product2: \n{0.5 * (kfaces_flux[:,elem,1,:] - kfaces_flux[:,elem,0,:])}')


      # df3_dz[:,epais,:] = ( mtrx.correction @ kfaces_flux[:,elem,:,:] ) * 2.0/geom.Δz
      df3_dz[:, elem, :] = (kfaces_flux[:, elem, 1, :] - kfaces_flux[:, elem, 0, :]) / geom.Δz

   for elem in range(nb_elements_x):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      # df1_dx[:,:,epais] = ( ifaces_flux[:,elem,:,:] @ mtrx.correction.T ) * 2.0/geom.Δx
      df1_dx[:,:,elem] = (ifaces_flux[:,elem,:,1] - ifaces_flux[:,elem,:,0]) / geom.Δx

   # --- Assemble the right-hand sides
   rhs = - ( df1_dx + df3_dz )

   rhs[RHO_W,:,:] -= Q[RHO,:,:] * gravity

   return rhs
