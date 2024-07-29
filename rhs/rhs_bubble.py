import numpy

num_it = 0

from common.definitions import idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta,  \
                               p0, Rd, cpd, cvd, heat_capacity_ratio, gravity

@profile
def rhs_bubble(Q, geom, mtrx, nbsolpts, nb_elements_x, nb_elements_z):

   global num_it
   # print(f'****************************************************************************')
   # print(f'********************* RHS BUBBLE it = {num_it} *****************************')
   # print(f'****************************************************************************')

   nb_equations = Q.shape[0] # Number of constituent Euler equations.  Probably 6.

   bulk_shape = (nb_equations, nb_elements_x * nb_elements_z, nbsolpts, nbsolpts)
   itf_shape = (nb_equations, nb_elements_x * nb_elements_z, 4, nbsolpts)

   # print(f'Q in (shape {Q.shape}) it {num_it}: \n{Q.reshape(bulk_shape)}')

   nb_interfaces_x = nb_elements_x + 1
   nb_interfaces_z = nb_elements_z + 1

   flux_x1 = numpy.empty_like(Q)
   flux_x3 = numpy.empty_like(Q)

#   df1_dx1 = numpy.empty_like(Q)
#   df3_dx3 = numpy.empty_like(Q)

   # --- Unpack physical variables
   rho      = Q[idx_2d_rho,:,:]
   uu       = Q[idx_2d_rho_u,:,:] / rho
   ww       = Q[idx_2d_rho_w,:,:] / rho
   # print(numpy.min(Q[idx_2d_rho_theta, :, :]), numpy.max(Q[idx_2d_rho_theta, :, :]))
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
   var_itf = Q @ mtrx.extrap
   # print(f'extrap (shape {mtrx.extrap.shape})= \n{mtrx.extrap}')
   # print(f'********** ITF VAR **********')
   # print(f'var itf = (shape {var_itf.shape})\n{var_itf.reshape(itf_shape)}')

   # --- Interface pressure
   pressure_itf = p0 * (var_itf[idx_2d_rho_theta] * Rd / p0)**(cpd / cvd)
   sound_itf = numpy.sqrt(heat_capacity_ratio * pressure_itf / var_itf[idx_2d_rho])
   mach_itf = var_itf[idx_2d_rho_w] / (var_itf[idx_2d_rho] * sound_itf)

   common_flux = numpy.zeros_like(var_itf)
#   print(common_flux.shape, nb_elements_x*nb_elements_z, 4*nbsolpts) # TODO : tempo

   # print(f'********** PRESSURE **********')
   # print(f'itf pressure = \n{pressure_itf.reshape(itf_shape[1:])}')

   # --- Common AUSM fluxes
   bot_itf  = numpy.arange(nbsolpts)
   top_itf = numpy.arange(nbsolpts, 2*nbsolpts)
   west_itf  = numpy.arange(2*nbsolpts, 3*nbsolpts)
   east_itf = numpy.arange(3*nbsolpts, 4*nbsolpts)
   for itf in range(1, nb_interfaces_z - 1):
      for ei in range(nb_elements_x):
         ek = itf
         elem_top = ei + nb_elements_x * ek
         elem_bot = ei + nb_elements_x * (ek-1)

         a_T = sound_itf[elem_top, bot_itf]
         a_B = sound_itf[elem_bot, top_itf]

         M_T = mach_itf[elem_top, bot_itf]
         M_B = mach_itf[elem_bot, top_itf]

         M = 0.25 * ((M_B + 1.)**2 - (M_T - 1.)**2)

         common_flux[:, elem_top, bot_itf] = (var_itf[:, elem_bot, top_itf] * numpy.maximum(0., M) * a_B) + \
                                             (var_itf[:, elem_top, bot_itf] * numpy.minimum(0., M) * a_T)

         common_flux[idx_2d_rho_w, elem_top, bot_itf] = 0.5 * ((1. + M_B) * pressure_itf[elem_bot, top_itf] + \
                                                               (1. - M_T) * pressure_itf[elem_top, bot_itf])

         common_flux[:, elem_bot, top_itf] = common_flux[:, elem_top, bot_itf]

   # print(f'********** FLUX Z MID ONLY **********')
   # print(f'common flux = \n{common_flux.reshape(itf_shape)}')

   for ei in range(nb_elements_x):
      # zeros flux BCs everywhere ...
      common_flux[:, ei, bot_itf] = 0.
      common_flux[:, ei + nb_elements_x*(nb_elements_z-1), top_itf] = 0.

      # except for momentum eqs where pressure is extrapolated to BCs.
      common_flux[idx_2d_rho_w, ei, bot_itf] = pressure_itf[ei, bot_itf]
      common_flux[idx_2d_rho_w, ei + nb_elements_x*(nb_elements_z-1), top_itf] = pressure_itf[ei + nb_elements_x*(nb_elements_z-1), top_itf]

   for ei in range(0, nb_elements_z * nb_elements_x, nb_elements_x):
      # print(f'fx{ei}')
      common_flux[idx_2d_rho_u, ei, west_itf] = pressure_itf[ei, west_itf]

   for ei in range(nb_elements_x - 1, nb_elements_z * nb_elements_x, nb_elements_x):
      # print(f'fx{ei}')
      common_flux[idx_2d_rho_u, ei, east_itf] = pressure_itf[ei, east_itf]

   # ifaces_flux[idx_2d_rho_u, 0,:,0] = ifaces_pres[0,:,0]  # TODO : pour les cas théoriques seulement ...
   # ifaces_flux[idx_2d_rho_u,-1,:,1] = ifaces_pres[-1,:,1]

   mach_itf = var_itf[idx_2d_rho_u] / (var_itf[idx_2d_rho] * sound_itf)
   for itf in range(1, nb_interfaces_x - 1):
      for ek in range(nb_elements_z):
         ei = itf
         elem_west = (ei-1) + nb_elements_x * ek
         elem_east = ei     + nb_elements_x * ek

         a_W = sound_itf[elem_west, east_itf]
         a_E = sound_itf[elem_east, west_itf]
         # print(f'Sounds: {a_W}, {a_E}')

         M_W = mach_itf[elem_west, east_itf]
         M_E = mach_itf[elem_east, west_itf]
         # print(f'Machs:  {M_W}, {M_E}')

         M = 0.25 * ((M_W + 1.)**2 - (M_E - 1.)**2)
         # print(f'M = {M}')

         common_flux[:, elem_east, west_itf] = (var_itf[:, elem_west, east_itf] * numpy.maximum(0., M) * a_W) + \
                                               (var_itf[:, elem_east, west_itf] * numpy.minimum(0., M) * a_E)

         common_flux[idx_2d_rho_u, elem_east, west_itf] = 0.5 * ((1. + M_W) * pressure_itf[elem_west, east_itf] + \
                                                                 (1. - M_E) * pressure_itf[elem_east, west_itf])

         common_flux[:, elem_west, east_itf] = common_flux[:, elem_east, west_itf]



   # --- Bondary treatement

   # zeros flux BCs everywhere ...
#   kfaces_flux[:,0,0,:]  = 0.0
#   kfaces_flux[:,-1,1,:] = 0.0
#
   # Skip periodic faces
#   if not geom.xperiodic:
#      ifaces_flux[:, 0,:,0] = 0.0
#      ifaces_flux[:,-1,:,1] = 0.0
#
   # except for momentum eqs where pressure is extrapolated to BCs.
#   kfaces_flux[idx_2d_rho_w, 0, 0, :] = kfaces_pres[ 0, 0, :]
#   kfaces_flux[idx_2d_rho_w,-1, 1, :] = kfaces_pres[-1, 1, :]
#
#   ifaces_flux[idx_2d_rho_u, 0,:,0] = ifaces_pres[0,:,0]  # TODO : pour les cas théoriques seulement ...
#   ifaces_flux[idx_2d_rho_u,-1,:,1] = ifaces_pres[-1,:,1]
#
   # --- Common AUSM fluxes
#   for itf in range(1, nb_interfaces_z - 1):
#
#      left  = itf - 1
#      right = itf
#
      # Left state
#      a_L = numpy.sqrt(heat_capacity_ratio * kfaces_pres[left, 1, :] / kfaces_var[idx_2d_rho, left, 1, :])
#      M_L = kfaces_var[idx_2d_rho_w, left, 1, :] / (kfaces_var[idx_2d_rho, left, 1, :] * a_L)
#
      # Right state
#      a_R = numpy.sqrt(heat_capacity_ratio * kfaces_pres[right, 0, :] / kfaces_var[idx_2d_rho, right, 0, :])
#      M_R = kfaces_var[idx_2d_rho_w, right, 0, :] / (kfaces_var[idx_2d_rho, right, 0, :] * a_R)
#
#      M = 0.25 * (( M_L + 1.)**2 - (M_R - 1.)**2)
#
#      kfaces_flux[:,right,0,:] = (kfaces_var[:,left,1,:] * numpy.maximum(0., M) * a_L) + \
#                                 (kfaces_var[:,right,0,:] * numpy.minimum(0., M) * a_R)
#      kfaces_flux[idx_2d_rho_w,right,0,:] += 0.5 * ((1. + M_L) * kfaces_pres[left,1,:] + \
#                                                    (1. - M_R) * kfaces_pres[right,0,:])
#
#      kfaces_flux[:,left,1,:] = kfaces_flux[:,right,0,:]
#
#
#   start = 0 if geom.xperiodic else 1
#   for itf in range(start, nb_interfaces_x - 1):
#
#      left  = itf - 1
#      right = itf
#
      # Left state
#      a_L = numpy.sqrt(heat_capacity_ratio * ifaces_pres[left, :, 1] / ifaces_var[idx_2d_rho, left, :, 1])
#      M_L = ifaces_var[idx_2d_rho_u, left, :, 1] / (ifaces_var[idx_2d_rho, left, :, 1] * a_L)
#
      # Right state
#      a_R = numpy.sqrt(heat_capacity_ratio * ifaces_pres[right, :, 0] / ifaces_var[idx_2d_rho, right, :, 0])
#      M_R = ifaces_var[idx_2d_rho_u, right, :, 0] / ( ifaces_var[idx_2d_rho, right, :, 0] * a_R)
#
#      M = 0.25 * ((M_L + 1.)**2 - (M_R - 1.)**2)
#
#      ifaces_flux[:,right,:,0] = (ifaces_var[:,left,:,1] * numpy.maximum(0., M) * a_L) + \
#                                 (ifaces_var[:,right,:,0] * numpy.minimum(0., M) * a_R)
#      ifaces_flux[idx_2d_rho_u,right,:,0] += 0.5 * ((1. + M_L) * ifaces_pres[left,:,1] + \
#                                                    (1. - M_R) * ifaces_pres[right,:,0])
#
#      ifaces_flux[:,left,:,1] = ifaces_flux[:,right,:,0]
#
#   if geom.xperiodic:
#      ifaces_flux[:, 0, :, 0] = ifaces_flux[:, -1, :, 1]

   # print(f'**************************')
   # print(f'********** FLUX **********')
   # print(f'**************************')
   # print(f'common flux shape = {common_flux.shape}, correction WE shape = {mtrx.correction_WE.shape}')
   # print(f'common flux: \n{common_flux.reshape(itf_shape)}')

   # print(f'**********************************')
   # print(f'********** CORRECTION X **********')
   # print(f'**********************************')
   # print(f'correction WE: \n{mtrx.correction_WE}')
   # print(f'Deriv op = \n{mtrx.derivative_x}')
   # print(f'FLUX X1 = \n{flux_x1.reshape(bulk_shape)}')

   # print(f'f only = \n{(flux_x1 @ mtrx.derivative_x).reshape(bulk_shape)}')
   # print(f'Contribution = \n{(common_flux @ mtrx.correction_WE).reshape(bulk_shape)}')

   df1_dx1 = (flux_x1 @ mtrx.derivative_x + common_flux @ mtrx.correction_WE) * (2.0/geom.Δx1)
   # print(f'********** DF1 DX1 **********')
   # print(f'df1 dx1 (shape {df1_dx1.shape}) = \n{df1_dx1}')
#   print('youpi!')
#   exit(0)

   # print(f'**********************************')
   # print(f'********** CORRECTION Z **********')
   # print(f'**********************************')
   # print(f'correction DU: \n{mtrx.correction_DU}')
   # print(f'Deriv op = \n{mtrx.derivative_z}')
   # print(f'FLUX X3 = \n{flux_x3.reshape(bulk_shape)}')

   # print(f'f only = \n{(flux_x3 @ mtrx.derivative_z).reshape(bulk_shape)}')
   # print(f'Contribution = \n{(common_flux @ mtrx.correction_DU).reshape(bulk_shape)}')

   df3_dx3 = (flux_x3 @ mtrx.derivative_z + common_flux @ mtrx.correction_DU) * (2.0 / geom.Δx3)
   # print(f'********** DF3 DX3 **********')
   # print(f'********** DF3 DX3 **********')
   # print(f'df3 dx3 (shape {df3_dx3.shape}) = \n{df3_dx3.reshape(bulk_shape)}')

   # --- Compute the derivatives
#   for elem in range(nb_elements_z):
#      epais = elem * nbsolpts + standard_slice
#      factor = 2.0 / geom.Δx3
#      if elem < geom.nb_elements_relief_layer:
#         factor = 2.0 / geom.relief_layer_delta
#
#      df3_dx3[:, epais, :] = \
#         (mtrx.diff_solpt @ flux_x3[:, epais, :] + mtrx.correction @ kfaces_flux[:, elem, :, :]) * factor
#
#   for elem in range(nb_elements_x):
#      epais = elem * nbsolpts + numpy.arange(nbsolpts)
#
#      df1_dx1[:,:,epais] = (flux_x1[:,:,epais] @ mtrx.diff_solpt.T + ifaces_flux[:,elem,:,:] @ mtrx.correction.T) * \
#                           2.0/geom.Δx1

   # --- Assemble the right-hand sides
   rhs = - ( df1_dx1 + df3_dx3 )

   rhs[idx_2d_rho_w,:,:] -= Q[idx_2d_rho,:,:] * gravity

   # print(f'rhs out: \n{rhs.reshape(bulk_shape)}')
   # if num_it >= 3:
   #    raise ValueError(f'Early interruption at it = {num_it}')

   num_it += 1

   return rhs
