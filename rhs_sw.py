import numpy

from definitions import idx_h, idx_hu1, idx_hu2, idx_u1, idx_u2, gravity
from dgfilter import apply_filter

def rhs_sw(Q, geom, mtrx, metric, topo, ptopo, nbsolpts: int, nb_elements_horiz: int, case_number: int, filter_rhs: bool = False):

   type_vec = Q.dtype

   shallow_water_equations = ( case_number > 1 )

   nb_interfaces_horiz = nb_elements_horiz + 1

   df1_dx1 = numpy.zeros_like(Q, dtype=type_vec)
   df2_dx2 = numpy.zeros_like(Q, dtype=type_vec)
   forcing = numpy.zeros_like(Q, dtype=type_vec)
   rhs = numpy.zeros_like(Q, dtype=type_vec)

   flux_Eq0_itf_j = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   flux_Eq1_itf_j = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   flux_Eq2_itf_j = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   h_itf_j        = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   u1_itf_j       = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   u2_itf_j       = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)

   flux_Eq0_itf_i = numpy.zeros((nb_elements_horiz+2, nbsolpts*nb_elements_horiz, 2), dtype=type_vec)
   flux_Eq1_itf_i = numpy.zeros((nb_elements_horiz+2, nbsolpts*nb_elements_horiz, 2), dtype=type_vec)
   flux_Eq2_itf_i = numpy.zeros((nb_elements_horiz+2, nbsolpts*nb_elements_horiz, 2), dtype=type_vec)
   h_itf_i        = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   u1_itf_i       = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   u2_itf_i       = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)

   eig_L          = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=type_vec)
   eig_R          = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=type_vec)
   eig            = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=type_vec)

   # TODO : enlever
   flux_L         = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=type_vec)
   flux_R         = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=type_vec)

   flux_P         = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=type_vec)
   flux_M         = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=type_vec)

   # Unpack dynamical variables
   h = Q[idx_h, :, :]
   hsquared = Q[idx_h, :, :]**2

   if shallow_water_equations:
      u1 = Q[idx_hu1,:,:] / h
      u2 = Q[idx_hu2,:,:] / h
   else:
      u1 = Q[idx_u1, :, :]
      u2 = Q[idx_u2, :, :]

   # Compute the fluxes
   flux_Eq0_x1 = h * metric.sqrtG * u1
   flux_Eq0_x2 = h * metric.sqrtG * u2

   flux_Eq1_x1 = metric.sqrtG * ( Q[idx_hu1,:,:] * u1 + 0.5 * gravity * metric.H_contra_11 * hsquared )
   flux_Eq1_x2 = metric.sqrtG * ( Q[idx_hu1,:,:] * u2 + 0.5 * gravity * metric.H_contra_12 * hsquared )

   flux_Eq2_x1 = metric.sqrtG * ( Q[idx_hu2,:,:] * u1 + 0.5 * gravity * metric.H_contra_21 * hsquared )
   flux_Eq2_x2 = metric.sqrtG * ( Q[idx_hu2,:,:] * u2 + 0.5 * gravity * metric.H_contra_22 * hsquared )

   # Offset due to the halo
   offset = 1

   HH = h + topo.hsurf

   # Interpolate to the element interface
   for elem in range(nb_elements_horiz):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      pos = elem + offset

      # --- Direction x1

      h_itf_i[pos, 0, :] = HH[:, epais] @ mtrx.extrap_west
      h_itf_i[pos, 1, :] = HH[:, epais] @ mtrx.extrap_east

      u1_itf_i[pos, 0, :] = u1[:, epais] @ mtrx.extrap_west
      u1_itf_i[pos, 1, :] = u1[:, epais] @ mtrx.extrap_east

      u2_itf_i[pos, 0, :] = u2[:, epais] @ mtrx.extrap_west
      u2_itf_i[pos, 1, :] = u2[:, epais] @ mtrx.extrap_east

      # --- Direction x2

      h_itf_j[pos, 0, :] = mtrx.extrap_south @ HH[epais, :]
      h_itf_j[pos, 1, :] = mtrx.extrap_north @ HH[epais, :]

      u1_itf_j[pos, 0, :] = mtrx.extrap_south @ u1[epais, :]
      u1_itf_j[pos, 1, :] = mtrx.extrap_north @ u1[epais, :]

      u2_itf_j[pos, 0, :] = mtrx.extrap_south @ u2[epais, :]
      u2_itf_j[pos, 1, :] = mtrx.extrap_north @ u2[epais, :]

   ptopo.xchange_scalars(geom, h_itf_i, h_itf_j)
   ptopo.xchange_vectors(geom, u1_itf_i, u2_itf_i, u1_itf_j, u2_itf_j)

   # Common Rusanov fluxes
   for itf in range(nb_interfaces_horiz):

      elem_L = itf
      elem_R = itf + 1

      h_itf_i[elem_L, 1, :] -= topo.hsurf_itf_i[elem_L, :, 1]
      h_itf_i[elem_R, 0, :] -= topo.hsurf_itf_i[elem_R, :, 0]

      h_itf_j[elem_L, 1, :] -= topo.hsurf_itf_j[elem_L, 1, :]
      h_itf_j[elem_R, 0, :] -= topo.hsurf_itf_j[elem_R, 0, :]

      ################
      # Direction x1 #
      ################

      # TODO : adv seul

      # Left state
      p11_L = metric.sqrtG_itf_i[:, itf] * 0.5 * gravity * metric.H_contra_11_itf_i[:, itf] * h_itf_i[elem_L, 1, :]**2
      p21_L = metric.sqrtG_itf_i[:, itf] * 0.5 * gravity * metric.H_contra_21_itf_i[:, itf] * h_itf_i[elem_L, 1, :]**2
      aL = numpy.sqrt( gravity * h_itf_i[elem_L, 1, :] * metric.H_contra_11_itf_i[:, itf] )
      mL = u1_itf_i[elem_L, 1, :] / aL

      # Right state
      p11_R = metric.sqrtG_itf_i[:, itf] * 0.5 * gravity * metric.H_contra_11_itf_i[:, itf] * h_itf_i[elem_R, 0, :]**2
      p21_R = metric.sqrtG_itf_i[:, itf] * 0.5 * gravity * metric.H_contra_21_itf_i[:, itf] * h_itf_i[elem_R, 0, :]**2
      aR = numpy.sqrt( gravity * h_itf_i[elem_R, 0, :] * metric.H_contra_11_itf_i[:, itf] )
      mR = u1_itf_i[elem_R, 0, :] / aR

      # In the following, positive part of flux is evaluated in the left element 
      # and negative part of flux is evaluated in the right element 

      # Positive part
      poly_11 = 0.25 * p11_L * (1. + mL)**2 * (2. - mL)
      p11_P = numpy.where(mL <= -1, 0., p11_L)
      p11_P = numpy.where(mL < 1., poly_11, p11_P)

      poly_21 = 0.25 * p21_L * (1. + mL)**2 * (2. - mL)
      p21_P = numpy.where(mL <= -1, 0., p21_L)
      p21_P = numpy.where(mL < 1., poly_21, p21_P)

      poly_m = 0.25 * (mL + 1.)**2
      m_P = numpy.where(mL <= -1, 0., mL)
      m_P = numpy.where(mL < 1., poly_m, m_P)

      # Negative part
      poly_11 = 0.25 * p11_R * (1. - mR)**2 * (2. + mR)
      p11_M = numpy.where(mR <= -1., p11_R, 0.)
      p11_M = numpy.where(mR < 1., poly_11, p11_M)

      poly_21 = 0.25 * p21_R * (1. - mR)**2 * (2. + mR)
      p21_M = numpy.where(mR <= -1., p21_R, 0.)
      p21_M = numpy.where(mR < 1., poly_21, p21_M)

      poly_m = -0.25 * (mR - 1.)**2
      m_M = numpy.where(mR <= -1., mR, 0.)
      m_M = numpy.where(mR < 1., poly_m, m_M)

      M = m_P + m_M
      
      # --- Continuity equation
   
      flux_P[:] = metric.sqrtG_itf_i[:, itf] * numpy.maximum(0., M) * aL * h_itf_i[elem_L, 1, :]
      flux_M[:] = metric.sqrtG_itf_i[:, itf] * numpy.minimum(0., M) * aR * h_itf_i[elem_R, 0, :]
      
      flux_Eq0_itf_i[elem_L, :, 1] = flux_P + flux_M
      flux_Eq0_itf_i[elem_R, :, 0] = flux_Eq0_itf_i[elem_L, :, 1]

      # --- u1 equation

      flux_P[:] = metric.sqrtG_itf_i[:, itf] * numpy.maximum(0., M) * aL * h_itf_i[elem_L, 1, :] * u1_itf_i[elem_L, 1, :] + p11_P
      flux_M[:] = metric.sqrtG_itf_i[:, itf] * numpy.minimum(0., M) * aR * h_itf_i[elem_R, 0, :] * u1_itf_i[elem_R, 0, :] + p11_M

      flux_Eq1_itf_i[elem_L, :, 1] = flux_P + flux_M
      flux_Eq1_itf_i[elem_R, :, 0] = flux_Eq1_itf_i[elem_L, :, 1]

      # --- u2 equation

      flux_P[:] = metric.sqrtG_itf_i[:, itf] * numpy.maximum(0., M) * aL * h_itf_i[elem_L, 1, :] * u2_itf_i[elem_L, 1, :] + p21_P
      flux_M[:] = metric.sqrtG_itf_i[:, itf] * numpy.minimum(0., M) * aR * h_itf_i[elem_R, 0, :] * u2_itf_i[elem_R, 0, :] + p21_M

      flux_Eq2_itf_i[elem_L, :, 1] = flux_P + flux_M
      flux_Eq2_itf_i[elem_R, :, 0] = flux_Eq2_itf_i[elem_L, :, 1]

      ################
      # Direction x2 #
      ################

      # Left state
      p12_L = metric.sqrtG_itf_j[itf, :] * 0.5 * gravity * metric.H_contra_12_itf_j[itf, :] * h_itf_j[elem_L, 1, :]**2
      p22_L = metric.sqrtG_itf_j[itf, :] * 0.5 * gravity * metric.H_contra_22_itf_j[itf, :] * h_itf_j[elem_L, 1, :]**2
      aL = numpy.sqrt( gravity * h_itf_j[elem_L, 1, :] * metric.H_contra_22_itf_j[itf, :] )
      mL = u2_itf_j[elem_L, 1, :] / aL

      # Right state
      p12_R = metric.sqrtG_itf_j[itf, :] * 0.5 * gravity * metric.H_contra_12_itf_j[itf, :] * h_itf_j[elem_R, 0, :]**2
      p22_R = metric.sqrtG_itf_j[itf, :] * 0.5 * gravity * metric.H_contra_22_itf_j[itf, :] * h_itf_j[elem_R, 0, :]**2
      aR = numpy.sqrt( gravity * h_itf_j[elem_R, 0, :] * metric.H_contra_22_itf_j[itf, :] )
      mR = u2_itf_j[elem_R, 0, :] / aR

      # Again, positive part of flux is evaluated in the left element 
      # and negative part of flux is evaluated in the right element 

      # Positive part
      poly_12 = 0.25 * p12_L * (1. + mL)**2 * (2. - mL)
      p12_P = numpy.where(mL <= -1, 0., p12_L)
      p12_P = numpy.where(mL < 1., poly_12, p12_P)

      poly_22 = 0.25 * p22_L * (1. + mL)**2 * (2. - mL)
      p22_P = numpy.where(mL <= -1, 0., p22_L)
      p22_P = numpy.where(mL < 1., poly_22, p22_P)

      poly_m = 0.25 * (mL + 1.)**2
      m_P = numpy.where(mL <= -1, 0., mL)
      m_P = numpy.where(mL < 1., poly_m, m_P)

      # Negative part
      poly_12 = 0.25 * p12_R * (1. - mR)**2 * (2. + mR)
      p12_M = numpy.where(mR <= -1., p12_R, 0.)
      p12_M = numpy.where(mR < 1., poly_12, p12_M)

      poly_22 = 0.25 * p22_R * (1. - mR)**2 * (2. + mR)
      p22_M = numpy.where(mR <= -1., p22_R, 0.)
      p22_M = numpy.where(mR < 1., poly_22, p22_M)

      poly_m = -0.25 * (mR - 1.)**2
      m_M = numpy.where(mR <= -1., mR, 0.)
      m_M = numpy.where(mR < 1., poly_m, m_M)

      M = m_P + m_M

      # --- Continuity equation

      flux_P[:] = metric.sqrtG_itf_j[itf, :] * numpy.maximum(0., M) * aL * h_itf_j[elem_L, 1, :]
      flux_M[:] = metric.sqrtG_itf_j[itf, :] * numpy.minimum(0., M) * aR * h_itf_j[elem_R, 0, :]
      
      flux_Eq0_itf_j[elem_L, 1, :] = flux_P + flux_M
      flux_Eq0_itf_j[elem_R, 0, :] = flux_Eq0_itf_j[elem_L, 1, :]

      # --- u1 equation

      flux_P[:] = metric.sqrtG_itf_j[itf, :] * numpy.maximum(0., M) * aL * h_itf_j[elem_L, 1, :] * u1_itf_j[elem_L, 1, :] + p12_P
      flux_M[:] = metric.sqrtG_itf_j[itf, :] * numpy.minimum(0., M) * aR * h_itf_j[elem_R, 0, :] * u1_itf_j[elem_R, 0, :] + p12_M

      flux_Eq1_itf_j[elem_L, 1, :] = flux_P + flux_M
      flux_Eq1_itf_j[elem_R, 0, :] = flux_Eq1_itf_j[elem_L, 1, :]

      # --- u2 equation

      flux_P[:] = metric.sqrtG_itf_j[itf, :] * numpy.maximum(0., M) * aL * h_itf_j[elem_L, 1, :] * u2_itf_j[elem_L, 1, :] + p22_P
      flux_M[:] = metric.sqrtG_itf_j[itf, :] * numpy.minimum(0., M) * aR * h_itf_j[elem_R, 0, :] * u2_itf_j[elem_R, 0, :] + p22_M

      flux_Eq2_itf_j[elem_L, 1, :] = flux_P + flux_M
      flux_Eq2_itf_j[elem_R, 0, :] = flux_Eq2_itf_j[elem_L, 1, :]

####################################################################################################################

   # Compute the derivatives
   for elem in range(nb_elements_horiz):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      # --- Direction x1

      df1_dx1[idx_h][:,epais]   = flux_Eq0_x1[:,epais] @ mtrx.diff_solpt_tr + flux_Eq0_itf_i[elem+offset,:,:] @ mtrx.correction_tr
      df1_dx1[idx_hu1][:,epais] = flux_Eq1_x1[:,epais] @ mtrx.diff_solpt_tr + flux_Eq1_itf_i[elem+offset,:,:] @ mtrx.correction_tr
      df1_dx1[idx_hu2][:,epais] = flux_Eq2_x1[:,epais] @ mtrx.diff_solpt_tr + flux_Eq2_itf_i[elem+offset,:,:] @ mtrx.correction_tr

      # --- Direction x2

      df2_dx2[idx_h,epais,:]   = mtrx.diff_solpt @ flux_Eq0_x2[epais,:] + mtrx.correction @ flux_Eq0_itf_j[elem+offset,:,:]
      df2_dx2[idx_hu1,epais,:] = mtrx.diff_solpt @ flux_Eq1_x2[epais,:] + mtrx.correction @ flux_Eq1_itf_j[elem+offset,:,:]
      df2_dx2[idx_hu2,epais,:] = mtrx.diff_solpt @ flux_Eq2_x2[epais,:] + mtrx.correction @ flux_Eq2_itf_j[elem+offset,:,:]

   # Add coriolis, metric and terms due to varying bottom topography
   forcing[idx_h,:,:] = 0.0

   # Note: christoffel_1_22 is zero
   forcing[idx_hu1,:,:] = 2.0 * ( metric.christoffel_1_01 * h * u1 + metric.christoffel_1_02 * h * u2) \
         + metric.christoffel_1_11 * h * u1**2 + 2.0 * metric.christoffel_1_12 * h * u1 * u2 \
         + gravity * h * ( metric.H_contra_11 * topo.dzdx1 + metric.H_contra_12 * topo.dzdx2)

   # Note: metric.christoffel_2_11 is zero
   forcing[idx_hu2,:,:] = 2.0 * (metric.christoffel_2_01 * h * u1 + metric.christoffel_2_02 * h * u2) \
         + 2.0 * metric.christoffel_2_12 * h * u1 * u2 + metric.christoffel_2_22 * h * u2**2 \
         + gravity * h * ( metric.H_contra_21 * topo.dzdx1 + metric.H_contra_22 * topo.dzdx2)

   # Assemble the right-hand sides
   for var in range(3):
      rhs[var] = metric.inv_sqrtG * -( df1_dx1[var] + df2_dx2[var] ) - forcing[var]

   if not shallow_water_equations:
      rhs[idx_hu1,:,:] = 0.0
      rhs[idx_hu2,:,:] = 0.0

   if filter_rhs:
      for var in range(3):
         rhs[var,:,:] = apply_filter(rhs[var,:,:], mtrx, nb_elements_horiz, nbsolpts)

   return rhs
