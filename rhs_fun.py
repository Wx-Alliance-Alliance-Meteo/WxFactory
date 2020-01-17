import numpy

from constants import *

def rhs_fun(Q, geom, mtrx, nbsolpts, nb_elements_x, nb_elements_z, α):

   type_vec = type(Q[1,1,1])

   nb_interfaces_x = nb_elements_x - 1
   nb_interfaces_z = nb_elements_z - 1

   flux_x1          = numpy.zeros_like(Q, dtype=type_vec)
   flux_x3          = numpy.zeros_like(Q, dtype=type_vec)

   df1_dx           = numpy.zeros_like(Q, dtype=type_vec)
   df3_dz           = numpy.zeros_like(Q, dtype=type_vec)
   dQ_dx            = numpy.zeros_like(Q, dtype=type_vec)
   dQ_dz            = numpy.zeros_like(Q, dtype=type_vec)
   Q_xx             = numpy.zeros_like(Q, dtype=type_vec)
   Q_zz             = numpy.zeros_like(Q, dtype=type_vec)
   upwind_diffusion = numpy.zeros_like(Q, dtype=type_vec)

   kfaces_flux      = numpy.zeros((2, nbsolpts*nb_elements_x, nb_equations, nb_elements_z), dtype=type_vec)
   kfaces_var       = numpy.zeros((2, nbsolpts*nb_elements_x, nb_equations, nb_elements_z), dtype=type_vec)
   kfaces_diffusion = numpy.zeros((2, nbsolpts*nb_elements_x, nb_equations, nb_elements_z), dtype=type_vec)

   ifaces_flux      = numpy.zeros((nbsolpts*nb_elements_z, 2, nb_equations, nb_elements_x), dtype=type_vec)
   ifaces_var       = numpy.zeros((nbsolpts*nb_elements_z, 2, nb_equations, nb_elements_x), dtype=type_vec)
   ifaces_diffusion = numpy.zeros((nbsolpts*nb_elements_z, 2, nb_equations, nb_elements_x), dtype=type_vec)

   kfaces_pres = numpy.zeros((2, nbsolpts*nb_elements_x, nb_elements_z), dtype=type_vec)
   ifaces_pres = numpy.zeros((nbsolpts*nb_elements_z, 2, nb_elements_x), dtype=type_vec)

   # Unpack physical variables
   uu       = Q[:,:,RHO_U] / Q[:,:,RHO]
   ww       = Q[:,:,RHO_W] / Q[:,:,RHO]
   pressure = P0 * (Q[:,:,RHO_THETA] * Rd / P0)**(cpd / cvd)

   # Compute the fluxes
   flux_x1[:,:,0] = Q[:,:,RHO_U]
   flux_x1[:,:,1] = Q[:,:,RHO_U] * uu + pressure
   flux_x1[:,:,2] = Q[:,:,RHO_U] * ww
   flux_x1[:,:,3] = Q[:,:,RHO_THETA] * uu

   flux_x3[:,:,0] = Q[:,:,RHO_W]
   flux_x3[:,:,1] = Q[:,:,RHO_W] * uu
   flux_x3[:,:,2] = Q[:,:,RHO_W] * ww + pressure
   flux_x3[:,:,3] = Q[:,:,RHO_THETA] * ww

   # Interpolate to the element interface
   for elem in range(nb_elements_z):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      for eq in range(nb_equations):
         kfaces_flux[0,:,eq,elem] = mtrx.lcoef @ flux_x3[epais,:,eq]
         kfaces_flux[1,:,eq,elem] = mtrx.rcoef @ flux_x3[epais,:,eq]

         kfaces_var[0,:,eq,elem] = mtrx.lcoef @ Q[epais,:,eq]
         kfaces_var[1,:,eq,elem] = mtrx.rcoef @ Q[epais,:,eq]

         kfaces_pres[0,:,elem] = mtrx.lcoef @ pressure[epais,:]
         kfaces_pres[1,:,elem] = mtrx.rcoef @ pressure[epais,:]


   for elem in range(nb_elements_x):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      for eq in range(nb_equations):
         ifaces_flux[:,0,eq,elem] = flux_x1[:,epais,eq] @ mtrx.lcoef
         ifaces_flux[:,1,eq,elem] = flux_x1[:,epais,eq] @ mtrx.rcoef

         ifaces_var[:,0,eq,elem] = Q[:,epais,eq] @ mtrx.lcoef
         ifaces_var[:,1,eq,elem] = Q[:,epais,eq] @ mtrx.rcoef

         ifaces_pres[:,0,elem] = pressure[:,epais] @ mtrx.lcoef
         ifaces_pres[:,1,elem] = pressure[:,epais] @ mtrx.rcoef

   # Bondary treatement
   # zeros flux BCs everywhere ...
   kfaces_flux[0,:,:,0]  = 0.0
   kfaces_flux[1,:,:,-1] = 0.0

   ifaces_flux[:,0,:,0]  = 0.0
   ifaces_flux[:,1,:,-1] = 0.0

   kfaces_diffusion[0,:,:,0]  = 0.0
   kfaces_diffusion[1,:,:,-1] = 0.0

   ifaces_diffusion[:,0,:,0]  = 0.0
   ifaces_diffusion[:,1,:,-1] = 0.0

   # ... except for momentum eqs where pressure is extrapolated to BCs.
   kfaces_flux[0,:,RHO_W,0]  = kfaces_pres[0,:,0]
   kfaces_flux[1,:,RHO_W,-1] = kfaces_pres[1,:,-1]

   ifaces_flux[:,0,RHO_U,0]   = ifaces_pres[:,0,0]  # TODO : theo seulement ...
   ifaces_flux[:,1,RHO_U,-1] = ifaces_pres[:,1,-1]
#
   # Common Rusanov fluxes
   for itf in range(nb_interfaces_z):

      eig_L = numpy.abs(kfaces_var[1,:,RHO_W,itf] / kfaces_var[1,:,RHO,itf]) \
            + numpy.sqrt( heat_capacity_ratio * kfaces_pres[1,:,itf] / kfaces_var[1,:,RHO,itf]  )

      eig_R = numpy.abs(kfaces_var[0,:,RHO_W,itf+1] / kfaces_var[0,:,RHO,itf+1]) \
            + numpy.sqrt( heat_capacity_ratio * kfaces_pres[0,:,itf+1] / kfaces_var[0,:,RHO,itf+1])

      for eq in range(nb_equations):
         kfaces_flux[0,:,eq,itf+1] = 0.5 * ( kfaces_flux[1,:,eq,itf] + kfaces_flux[0,:,eq,itf+1] \
               - numpy.maximum(numpy.abs(eig_L), numpy.abs(eig_R)) * ( kfaces_var[0,:,eq,itf+1] - kfaces_var[1,:,eq,itf] ) )

      kfaces_flux[1,:,:,itf] = kfaces_flux[0,:,:,itf+1]

      kfaces_diffusion[0,:,:,itf+1] = 0.5 * ( kfaces_var[1,:,:,itf] + kfaces_var[0,:,:,itf+1] )
      kfaces_diffusion[1,:,:,itf]   = kfaces_diffusion[0,:,:,itf+1]

   for itf in range(nb_interfaces_x):

      eig_L = numpy.abs(ifaces_var[:,1,RHO_U,itf] / ifaces_var[:,1,RHO,itf]) \
            + numpy.sqrt( heat_capacity_ratio * ifaces_pres[:,1,itf] / ifaces_var[:,1,RHO,itf] )
      eig_R = numpy.abs(ifaces_var[:,0,RHO_U,itf+1] / ifaces_var[:,0,RHO,itf+1]) \
            + numpy.sqrt( heat_capacity_ratio * ifaces_pres[:,0,itf+1] / ifaces_var[:,0,RHO,itf+1] )

      for eq in range(nb_equations):
         ifaces_flux[:,0,eq,itf+1] = 0.5 * ( ifaces_flux[:,1,eq,itf] + ifaces_flux[:,0,eq,itf+1] \
               - numpy.maximum(numpy.abs(eig_L), numpy.abs(eig_R)) * ( ifaces_var[:,0,eq,itf+1] - ifaces_var[:,1,eq,itf] ) )

      ifaces_flux[:,1,:,itf] = ifaces_flux[:,0,:,itf+1]

      ifaces_diffusion[:,0,:,itf+1] = 0.5 * ( ifaces_var[:,1,:,itf] + ifaces_var[:,0,:,itf+1] )
      ifaces_diffusion[:,1,:,itf]   = ifaces_diffusion[:,0,:,itf+1]

   # Compute the derivatives
   for elem in range(nb_elements_z):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      for eq in range(nb_equations):
         df3_dz[epais,:,eq] = ( mtrx.diff_solpt @ flux_x3[epais,:,eq] + mtrx.correction @ kfaces_flux[:,:,eq,elem] ) * 2.0/geom.Δz
         dQ_dz[epais,:,eq]  = ( mtrx.diff_solpt @ Q[epais,:,eq] + mtrx.correction @ kfaces_diffusion[:,:,eq,elem] ) * 2.0/geom.Δz

   for elem in range(nb_elements_x):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      for eq in range(nb_equations):
         df1_dx[:,epais,eq] = ( flux_x1[:,epais,eq] @ mtrx.diff_solpt.T + ifaces_flux[:,:,eq,elem] @ mtrx.correction.T ) * 2.0/geom.Δx
         dQ_dx[:,epais,eq]  = ( Q[:,epais,eq] @ mtrx.diff_solpt.T + ifaces_diffusion[:,:,eq,elem] @ mtrx.correction.T ) * 2.0/geom.Δx

   # Interpolate first derivative of diffusion to the interface
   for elem in range(nb_elements_z):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      for eq in range(nb_equations):
         kfaces_var[0,:,eq,elem] = mtrx.lcoef @ dQ_dz[epais,:,eq]
         kfaces_var[1,:,eq,elem] = mtrx.rcoef @ dQ_dz[epais,:,eq]

   for elem in range(nb_elements_x):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      for eq in range(nb_equations):
         ifaces_var[:,0,eq,elem] = dQ_dx[:,epais,eq] @ mtrx.lcoef
         ifaces_var[:,1,eq,elem] = dQ_dx[:,epais,eq] @ mtrx.rcoef

   # Communication at cell interfaces with central flux
   for itf in range(nb_interfaces_z):
      kfaces_diffusion[0,:,:,itf+1] = 0.5 * ( kfaces_var[1,:,:,itf] + kfaces_var[0,:,:,itf+1] )
      kfaces_diffusion[1,:,:,itf]   = kfaces_diffusion[0,:,:,itf+1]

   for itf in range(nb_interfaces_x):
      ifaces_diffusion[:,0,:,itf+1] = 0.5 * ( ifaces_var[:,1,:,itf] + ifaces_var[:,0,:,itf+1] )
      ifaces_diffusion[:,1,:,itf]   = ifaces_diffusion[:,0,:,itf+1]

   # Bondary treatement (this should be equivalent to a zero diffusion coefficient at the boundary)
   kfaces_diffusion[0,:,:,0]  = 0.0
   kfaces_diffusion[1,:,:,-1] = 0.0

   ifaces_diffusion[:,0,:,0]  = 0.0
   ifaces_diffusion[:,1,:,-1] = 0.0
#
   # Finalize the diffusion operator
   for elem in range(nb_elements_z):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      for eq in range(nb_equations):
         Q_zz[epais,:,eq] = ( mtrx.diff_solpt @ dQ_dz[epais,:,eq] + mtrx.correction @ kfaces_diffusion[:,:,eq,elem] ) * 2.0/geom.Δz

   for elem in range(nb_elements_x):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      for eq in range(nb_equations):
         Q_xx[:,epais,eq] = ( dQ_dx[:,epais,eq] @ mtrx.diff_solpt.T + ifaces_diffusion[:,:,eq,elem] @ mtrx.correction.T ) * 2.0/geom.Δx

   for eq in range(nb_equations):
      upwind_diffusion[:,:,eq] = (α * 0.5 * geom.Δx / nbsolpts) * numpy.abs(uu) * Q_xx[:,:,eq] + (α * 0.5 * geom.Δz / nbsolpts) * numpy.abs(ww) * Q_zz[:,:,eq]

   # Assemble the right-hand sides
   rhs = - df1_dx - df3_dz + upwind_diffusion

   rhs[:,:,RHO_W] = rhs[:,:,RHO_W] - Q[:,:,RHO] * gravity

   return rhs
