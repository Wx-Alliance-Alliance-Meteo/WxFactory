import numpy
import math

from cubed_sphere import cubed_sphere
from matrices import DFR_operators

class Metric_3d_topo:
   def __init__(self, geom : cubed_sphere, matrix: DFR_operators):
      # Token initialization: store geometry and matrix objects.  Defer construction of the metric itself,
      # so that initialization can take place after topography is defined inside the 'geom' object

      self.geom = geom
      self.matrix = matrix
   
   def build_metric(self):
      # Construct the metric terms, with the assurance that topography is now defined.  This defines full, 3D arrays
      # for the metric and Christoffel symbols.

      # Retrieve geometry and matrix objects
      geom = self.geom
      matrix = self.matrix

      # Gnomonic coordinates in element interiors
      X_int = geom.coordVec_gnom[0,:,:,:]
      Y_int = geom.coordVec_gnom[1,:,:,:]
      R_int = geom.coordVec_gnom[2,:,:,:] + geom.earth_radius
      self.R_int = R_int

      # Gnomonic coordinates at i-interface
      X_itf_i = geom.coordVec_gnom_itf_i[0,:,:,:]
      Y_itf_i = geom.coordVec_gnom_itf_i[1,:,:,:]
      R_itf_i = geom.coordVec_gnom_itf_i[2,:,:,:] + geom.earth_radius

      # At j-interface
      X_itf_j = geom.coordVec_gnom_itf_j[0,:,:,:]
      Y_itf_j = geom.coordVec_gnom_itf_j[1,:,:,:]
      R_itf_j = geom.coordVec_gnom_itf_j[2,:,:,:] + geom.earth_radius

      # Gnomonic coordinates at i-interface
      X_itf_k = geom.coordVec_gnom_itf_k[0,:,:,:]
      Y_itf_k = geom.coordVec_gnom_itf_k[1,:,:,:]
      R_itf_k = geom.coordVec_gnom_itf_k[2,:,:,:] + geom.earth_radius

      # Grid scaling factors, necessary to define the metric and Christoffel symbols
      # with respect to the standard element
      Δx = geom.Δx1
      Δy = geom.Δx2
      Δeta = geom.Δeta

      # Grid rotation terms
      alpha = geom.angle_p # Clockwise rotation about the Z axis
      phi = geom.lat_p # Counterclockwise rotation about the X axis, sending [0,0,1] to a particular latitude
      lam = geom.lon_p # Counterclockwise rotation about the Z axis, sending [0,-1,0] to a particular longitude
      salp = numpy.sin(alpha); calp = numpy.cos(alpha)
      sphi = numpy.sin(phi); cphi = numpy.cos(phi)
      slam = numpy.sin(lam); clam = numpy.cos(lam)

      ## Compute partial derivatives of R

      # First, we won't use R.  R = H + (radius), and we can improve numerical conditioning by removing that
      # DC offset term.  dR/d(stuff) = dH/d(stuff)
      height_int = geom.coordVec_gnom[2,:,:,:] 
      height_itf_i = geom.coordVec_gnom_itf_i[2,:,:,:]
      height_itf_j = geom.coordVec_gnom_itf_j[2,:,:,:]
      height_itf_k = geom.coordVec_gnom_itf_k[2,:,:,:]

      # Build the boundary-extensions of h, based on the interface boundaries
      height_ext_i = numpy.zeros((geom.nk, geom.nj, geom.nb_elements_x1, 2))
      height_ext_j = numpy.zeros((geom.nk, geom.nb_elements_x2, 2, geom.ni))
      height_ext_k = numpy.zeros((geom.nb_elements_x3, 2, geom.nj, geom.ni))

      height_ext_i[:,:,:,0] = height_itf_i[:,:,:-1] # Assign the left ("west") boundary of each element
      height_ext_i[:,:,:,1] = height_itf_i[:,:,1:] # Assign the right ("east") boundary of each element
      height_ext_j[:,:,0,:] = height_itf_j[:,:-1,:] # Downest ("south") boundary of each element
      height_ext_j[:,:,1,:] = height_itf_j[:,1:,:] # Uppest ("north") boundary of each element
      height_ext_k[:,0,:,:] = height_itf_k[:-1,:,:] # Bottom boundary of each element
      height_ext_k[:,1,:,:] = height_itf_k[1:,:,:] # Top boundary of each element

      #dRdx1_int = 0*R_int
      #dRdx2_int = 0*R_int
      #dRdeta_int = (geom.ztop - 0) + 0*R_int

      dRdx1_int = matrix.comma_i(height_int,height_ext_i,geom)*2/Δx
      dRdx2_int = matrix.comma_j(height_int,height_ext_j,geom)*2/Δy
      dRdeta_int = matrix.comma_k(height_int,height_ext_k,geom)*2/Δeta
      
      debug_col = geom.nj//2

      # With the derivative performed, define the averaged interface values
      dRdx1_itf_i = numpy.empty_like(R_itf_i)
      dRdx2_itf_i = numpy.empty_like(R_itf_i)
      dRdeta_itf_i = numpy.empty_like(R_itf_i)
      dRdx1_itf_j = numpy.empty_like(R_itf_j)
      dRdx2_itf_j = numpy.empty_like(R_itf_j)
      dRdeta_itf_j = numpy.empty_like(R_itf_j)
      dRdx1_itf_k = numpy.empty_like(R_itf_k)
      dRdx2_itf_k = numpy.empty_like(R_itf_k)
      dRdeta_itf_k = numpy.empty_like(R_itf_k)
      for (d_int, d_itf_i, d_itf_j, d_itf_k) in zip((dRdx1_int, dRdx2_int, dRdeta_int),
                                                    (dRdx1_itf_i, dRdx2_itf_i, dRdeta_itf_i),
                                                    (dRdx1_itf_j, dRdx2_itf_j, dRdeta_itf_j),
                                                    (dRdx1_itf_k, dRdx2_itf_k, dRdeta_itf_k)):
         # Extrapolate the interior values to each edge
         d_extrap_i = matrix.extrapolate_i(d_int, geom)
         d_extrap_j = matrix.extrapolate_j(d_int, geom)
         d_extrap_k = matrix.extrapolate_k(d_int, geom)

         # Assign absolute minimum/maximum interface values based on the one-sided extrapolation
         d_itf_i[:,:,0] = d_extrap_i[:,:,0,0]
         d_itf_i[:,:,-1] = d_extrap_i[:,:,-1,1]
         d_itf_j[:,0,:] = d_extrap_j[:,0,0,:]
         d_itf_j[:,-1,:] = d_extrap_j[:,-1,1,:]
         d_itf_k[0,:,:] = d_extrap_k[0,0,:,:]
         d_itf_k[-1,:,:] = d_extrap_k[-1,1,:,:]

         # Assign interior values based on the average of the bordering extrapolations
         d_itf_i[:,:,1:-1] = 0.5*(d_extrap_i[:,:,1:,0] + d_extrap_i[:,:,:-1,1])
         d_itf_j[:,1:-1,:] = 0.5*(d_extrap_j[:,1:,0,:] + d_extrap_j[:,:-1,1,:])
         d_itf_k[1:-1,:,:] = 0.5*(d_extrap_k[1:,0,:,:] + d_extrap_k[:-1,1,:,:])

      # FIXME
      # The i/j interface values should now be "fixed up" with a boundary exchange.  However, the vector
      # exchange code is specialized for u^i*e_i = u^i(δ_i), but here we essentially have covariant components
      # with respect to x1/x2/eta (not X/Y/eta).  This will require a scalar exchange and manual processing of
      # the boundaries.

      # Initialize metric arrays

      # Covariant space-only metric
      H_cov = numpy.empty((3,3) + X_int.shape,dtype=numpy.double)
      H_cov_itf_i = numpy.empty((3,3) + X_itf_i.shape,dtype=numpy.double)
      H_cov_itf_j = numpy.empty((3,3) + X_itf_j.shape,dtype=numpy.double)
      H_cov_itf_k = numpy.empty((3,3) + X_itf_k.shape,dtype=numpy.double)

      H_contra = numpy.empty((3,3) + X_int.shape,dtype=numpy.double)
      H_contra_itf_i = numpy.empty((3,3) + X_itf_i.shape,dtype=numpy.double)
      H_contra_itf_j = numpy.empty((3,3) + X_itf_j.shape,dtype=numpy.double)
      H_contra_itf_k = numpy.empty((3,3) + X_itf_k.shape,dtype=numpy.double)

      sqrtG = numpy.empty_like(X_int)
      sqrtG_itf_i = numpy.empty_like(X_itf_i)
      sqrtG_itf_j = numpy.empty_like(X_itf_j)
      sqrtG_itf_k = numpy.empty_like(X_itf_k)

      # Loop over the interior and interface variants of the fields, computing the metric terms
      for (Hcov, Hcontra, rootG, X, Y, R, dRdx1, dRdx2, dRdeta) in \
                  zip( (H_cov,H_cov_itf_i,H_cov_itf_j,H_cov_itf_k),
                       (H_contra,H_contra_itf_i,H_contra_itf_j,H_contra_itf_k),
                       (sqrtG,sqrtG_itf_i,sqrtG_itf_j,sqrtG_itf_k),
                       (X_int, X_itf_i, X_itf_j, X_itf_k),
                       (Y_int, Y_itf_i, Y_itf_j, Y_itf_k),
                       (R_int, R_itf_i, R_itf_j, R_itf_k),
                       (dRdx1_int, dRdx1_itf_i, dRdx1_itf_j, dRdx1_itf_k),
                       (dRdx2_int, dRdx2_itf_i, dRdx2_itf_j, dRdx2_itf_k),
                       (dRdeta_int, dRdeta_itf_i, dRdeta_itf_j, dRdeta_itf_k)):
         delsq = 1 + X**2 + Y**2 # δ², per Charron May 2022
         del4 = delsq**2
         Hcov[0,0,:] = (Δx**2/4)*(R**2/del4*(1+X**2)**2*(1+Y**2) + dRdx1**2) # g_11
         
         Hcov[0,1,:] = (Δx*Δy/4)*(-R**2/del4*X*Y*(1+X**2)*(1+Y**2) + dRdx1*dRdx2) # g_12
         Hcov[1,0,:] = Hcov[0,1,:] # g_21 (by symmetry)
         
         Hcov[0,2,:] = Δeta*Δx/4*dRdx1*dRdeta # g_13
         Hcov[2,0,:] = Hcov[0,2,:] # g_31 by symmetry

         Hcov[1,1,:] = Δy**2/4*(R**2/del4*(1+X**2)*(1+Y**2)**2 + dRdx2**2) # g_22
         
         Hcov[1,2,:] = Δeta*Δy/4*dRdx2*dRdeta # g_23
         Hcov[2,1,:] = Hcov[1,2,:] # g_32 by symmetry

         Hcov[2,2,:] = (Δeta**2/4)*dRdeta**2 # g_33

         Hcontra[0,0,:] = (4/Δx**2)*(delsq/(R**2*(1+X**2))) # h^11

         Hcontra[0,1,:] = (4/Δx/Δy)*(X*Y*delsq/(R**2*(1+X**2)*(1+Y**2))) # h^12
         Hcontra[1,0,:] = Hcontra[0,1,:] # h^21 by symmetry

         Hcontra[0,2,:] = (4/Δx/Δeta)*(-(dRdx1*delsq/(R**2*(1+X**2)) + dRdx2*delsq*X*Y/(R**2*(1+X**2)*(1+Y**2)))/(dRdeta)) # h^13
         Hcontra[2,0,:] = Hcontra[0,2,:] # h^31 by symmetry

         Hcontra[1,1,:] = (4/Δy**2)*(delsq/(R**2*(1+Y**2))) # h^22

         Hcontra[1,2,:] = (4/Δy/Δeta)*(-(dRdx1*X*Y*delsq/(R**2*(1+X**2)*(1+Y**2)) + dRdx2*delsq/(R**2*(1+Y**2)))/dRdeta) # h^23
         Hcontra[2,1,:] = Hcontra[1,2,:] # h^32 by symmetry

         Hcontra[2,2,:] = (4/Δeta**2)*(1 + dRdx1**2*delsq/(R**2*(1+X**2)) + \
                                       2*dRdx1*dRdx2*X*Y*delsq/(R**2*(1+X**2)*(1+Y**2)) + \
                                       dRdx2**2*delsq/(R**2*(1+Y**2)))/dRdeta**2

         rootG[:] = (Δx/2)*(Δy/2)*(Δeta/2)*R**2*(1+X**2)*(1+Y**2)*numpy.abs(dRdeta)/delsq**(1.5)  

      # Christoffel symbols, including time as the 0th index
      # Note that we only need the Γ^(1..3)_ab terms.
      # Christoffel = numpy.empty((4,4) +  X.shape,dtype=numpy.double)

      # Rotation terms that appear throughout the symbol definitions:
      rot1 = sphi - X_int*cphi*salp + Y_int*cphi*calp
      rot2 = (1+X_int**2)*cphi*calp - Y_int*sphi + X_int*Y_int*cphi*salp
      rot3 = (1+Y_int**2)*cphi*salp + X_int*sphi + X_int*Y_int*cphi*calp

      deltasq = (1+X_int**2+Y_int**2)
      Omega = geom.rotation_speed

      # Γ^1_ab, a≤b
      Christoffel_1_01 = Omega*X_int*Y_int/deltasq*rot1 + dRdx1_int*Omega/(R_int*(1+X_int**2))*rot2
      Christoffel_1_02 = -Omega*(-(1+Y_int**2)/deltasq)*rot1 + dRdx2_int*Omega/(R_int*(1+X_int**2))*rot2
      Christoffel_1_03 = dRdeta_int*Omega/(R_int*(1+X_int**2))*rot2

      Christoffel_1_11 = 2*X_int*Y_int**2/deltasq + dRdx1_int*2/R_int
      Christoffel_1_12 = -Y_int*(1+Y_int**2)/deltasq + dRdx2_int/R_int
      Christoffel_1_13 = dRdeta_int/R_int

      Christoffel_1_22 = 0
      Christoffel_1_23 = 0

      Christoffel_1_33 = 0

      # Γ^2_ab, a≤b
      Christoffel_2_01 = Omega*(1+X_int**2)/deltasq*rot1 + dRdx1_int*Omega/(R_int*(1+Y_int**2))*rot3
      Christoffel_2_02 = -Omega*X_int*Y_int/deltasq*rot2 + dRdx2_int*Omega/(R_int*(1+Y_int**2))*rot3
      Christoffel_2_03 = dRdeta_int*Omega/(R_int*(1+Y_int**2))*rot3

      Christoffel_2_11 = 0
      Christoffel_2_12 = -X_int*(1+X_int**2)/deltasq + dRdx1_int/R_int
      Christoffel_2_13 = 0

      Christoffel_2_22 = 2*X_int**2*Y_int/deltasq + dRdx2_int*2/R_int
      Christoffel_2_23 = dRdeta_int/R_int

      Christoffel_2_33 = 0

      # Γ^3_ab, a≤b
      # For this set of terms, we need the second derivatives of R with respect to x1, x1, and η

      # Build the extensions of R_(i,j,k) to the element boundaries, using the previously-found itf_(i,k,k) values
      # Because we assume the quality of mixed partial derivatives (d^2f/dadb = d^2f/dbda), we need to extend _(i,j,k)
      # for x1, _(j,k) for x2, and only _k for eta.

      dRdx1_ext_i = numpy.zeros((geom.nk, geom.nj, geom.nb_elements_x1, 2))
      dRdx1_ext_i[:,:,:,0] = dRdx1_itf_i[:,:,:-1] # Assign min-i boundary
      dRdx1_ext_i[:,:,:,1] = dRdx1_itf_i[:,:,1:]  # Assign max-i boundary
      dRdx1_ext_j = numpy.zeros((geom.nk, geom.nb_elements_x2, 2, geom.ni))
      dRdx1_ext_j[:,:,0,:] = dRdx1_itf_j[:,:-1,:] # Assign min-j boundary
      dRdx1_ext_j[:,:,1,:] = dRdx1_itf_j[:,1:,:]  # Assign max-j boundary
      dRdx1_ext_k = numpy.zeros((geom.nb_elements_x3, 2, geom.nj, geom.ni))
      dRdx1_ext_k[:,0,:,:] = dRdx1_itf_k[:-1,:,:] # Assign min-k boundary
      dRdx1_ext_k[:,1,:,:] = dRdx1_itf_k[1:,:,:]  # Assign max-k boundary
      
      dRdx2_ext_j = numpy.zeros((geom.nk, geom.nb_elements_x2, 2, geom.ni))
      dRdx2_ext_j[:,:,0,:] = dRdx2_itf_j[:,:-1,:] # Assign min-j boundary
      dRdx2_ext_j[:,:,1,:] = dRdx2_itf_j[:,1:,:]  # Assign max-j boundary
      dRdx2_ext_k = numpy.zeros((geom.nb_elements_x3, 2, geom.nj, geom.ni))
      dRdx2_ext_k[:,0,:,:] = dRdx2_itf_k[:-1,:,:] # Assign min-k boundary
      dRdx2_ext_k[:,1,:,:] = dRdx2_itf_k[1:,:,:]  # Assign max-k boundary

      dRdeta_ext_k = numpy.zeros((geom.nb_elements_x3, 2, geom.nj, geom.ni))
      dRdeta_ext_k[:,0,:,:] = dRdeta_itf_k[:-1,:,:] # Assign min-k boundary
      dRdeta_ext_k[:,1,:,:] = dRdeta_itf_k[1:,:,:]  # Assign max-k boundary

      # With the extension information, compute the partial derivatives.  We do not need any parallel
      # synchronization here because we only use the Christoffel symbols at element-interior points.
      d2Rdx1x1 = matrix.comma_i(dRdx1_int,dRdx1_ext_i,geom)*2/Δx
      d2Rdx1x2 = matrix.comma_j(dRdx1_int,dRdx1_ext_j,geom)*2/Δy
      d2Rdx1eta = matrix.comma_k(dRdx1_int,dRdx1_ext_k,geom)*2/Δeta

      d2Rdx2x2 = matrix.comma_j(dRdx2_int,dRdx2_ext_j,geom)*2/Δy
      d2Rdx2eta = matrix.comma_k(dRdx2_int,dRdx2_ext_k,geom)*2/Δeta

      d2Rdetaeta = matrix.comma_k(dRdeta_int,dRdeta_ext_k,geom)*2/Δeta

      Christoffel_3_01 = -(dRdeta_int**-1)*(dRdx1_int*Christoffel_1_01 + dRdx2_int*Christoffel_2_01 + \
                     R_int/deltasq*Omega*(1+X_int**2)*(cphi*calp - Y_int*sphi))
      Christoffel_3_02 = -(dRdeta_int**-1)*(dRdx1_int*Christoffel_1_02 + dRdx2_int*Christoffel_2_02 + \
                     R_int/deltasq*Omega*(1+Y_int**2)*(cphi*salp + X_int*sphi))
      Christoffel_3_03 = -dRdx1_int*Omega/(R_int*(1+X_int**2))*rot2 - dRdx2_int*Omega/(R_int*(1+Y_int**2))*rot3

      Christoffel_3_11 = (dRdeta_int**-1)*(d2Rdx1x1 - dRdx1_int*Christoffel_1_11 - R_int/deltasq**2*(1+X_int**2)**2*(1+Y_int**2))
      Christoffel_3_12 = (dRdeta_int**-1)*(d2Rdx1x2 - dRdx1_int*Christoffel_1_12 - dRdx2_int*Christoffel_2_12 + \
                                             R_int/deltasq**2*X_int*Y_int*(1+X_int**2)*(1+Y_int**2))
      Christoffel_3_13 = (dRdeta_int**-1)*d2Rdx1eta - dRdx1_int/R_int

      Christoffel_3_22 = (dRdeta_int**-1)*(d2Rdx2x2 - dRdx2_int*Christoffel_2_22 - R_int/deltasq**2*(1+X_int**2)*(1+Y_int**2)**2)
      Christoffel_3_23 = (dRdeta_int**-1)*d2Rdx2eta - dRdx2_int/R_int

      Christoffel_3_33 = (dRdeta_int**-1)*d2Rdetaeta

      # Now, normalize the Christoffel symbols by the appropriate grid scaling factor.  To this point, the symbols have
      # been defined in terms of x1, x2, and η, but for compatibility with the numerical differentiation we need to define
      # these symbols in terms of i, j, and k by applying the scaling factors Δx/2, Δy/2, and Δeta/2 respectively.

      # The Christoffel symbol is definitionally Γ^a_bc = (∂e_b/δx^c) · ẽ^a, so we apply the scaling factor for the b and c
      # indices and the inverse scaling factor for the a index.

      Christoffel_1_01 *=   (2/Δx) *      (1) *   (Δx/2); self.christoffel_1_01 = Christoffel_1_01
      Christoffel_1_02 *=   (2/Δx) *      (1) *   (Δy/2); self.christoffel_1_02 = Christoffel_1_02
      Christoffel_1_03 *=   (2/Δx) *      (1) * (Δeta/2); self.christoffel_1_03 = Christoffel_1_03

      Christoffel_1_11 *=   (2/Δx) *   (Δx/2) *   (Δx/2); self.christoffel_1_11 = Christoffel_1_11
      Christoffel_1_12 *=   (2/Δx) *   (Δx/2) *   (Δy/2); self.christoffel_1_12 = Christoffel_1_12
      Christoffel_1_13 *=   (2/Δx) *   (Δx/2) * (Δeta/2); self.christoffel_1_13 = Christoffel_1_13

      Christoffel_1_22 *=   (2/Δx) *   (Δy/2) *   (Δy/2); self.christoffel_1_22 = Christoffel_1_22
      Christoffel_1_23 *=   (2/Δx) *   (Δy/2) * (Δeta/2); self.christoffel_1_23 = Christoffel_1_23

      Christoffel_1_33 *=   (2/Δx) * (Δeta/2) * (Δeta/2); self.christoffel_1_33 = Christoffel_1_33

      Christoffel_2_01 *=   (2/Δy) *      (1) *   (Δx/2); self.christoffel_2_01 = Christoffel_2_01
      Christoffel_2_02 *=   (2/Δy) *      (1) *   (Δy/2); self.christoffel_2_02 = Christoffel_2_02
      Christoffel_2_03 *=   (2/Δy) *      (1) * (Δeta/2); self.christoffel_2_03 = Christoffel_2_03

      Christoffel_2_11 *=   (2/Δy) *   (Δx/2) *   (Δx/2); self.christoffel_2_11 = Christoffel_2_11
      Christoffel_2_12 *=   (2/Δy) *   (Δx/2) *   (Δy/2); self.christoffel_2_12 = Christoffel_2_12
      Christoffel_2_13 *=   (2/Δy) *   (Δx/2) * (Δeta/2); self.christoffel_2_13 = Christoffel_2_13

      Christoffel_2_22 *=   (2/Δy) *   (Δy/2) *   (Δy/2); self.christoffel_2_22 = Christoffel_2_22
      Christoffel_2_23 *=   (2/Δy) *   (Δy/2) * (Δeta/2); self.christoffel_2_23 = Christoffel_2_23

      Christoffel_2_33 *=   (2/Δy) * (Δeta/2) * (Δeta/2); self.christoffel_2_33 = Christoffel_2_33

      Christoffel_3_01 *= (2/Δeta) *      (1) *   (Δx/2); self.christoffel_3_01 = Christoffel_3_01
      Christoffel_3_02 *= (2/Δeta) *      (1) *   (Δy/2); self.christoffel_3_02 = Christoffel_3_02
      Christoffel_3_03 *= (2/Δeta) *      (1) * (Δeta/2); self.christoffel_3_03 = Christoffel_3_03

      Christoffel_3_11 *= (2/Δeta) *   (Δx/2) *   (Δx/2); self.christoffel_3_11 = Christoffel_3_11
      Christoffel_3_12 *= (2/Δeta) *   (Δx/2) *   (Δy/2); self.christoffel_3_12 = Christoffel_3_12
      Christoffel_3_13 *= (2/Δeta) *   (Δx/2) * (Δeta/2); self.christoffel_3_13 = Christoffel_3_13

      Christoffel_3_22 *= (2/Δeta) *   (Δy/2) *   (Δy/2); self.christoffel_3_22 = Christoffel_3_22
      Christoffel_3_23 *= (2/Δeta) *   (Δy/2) * (Δeta/2); self.christoffel_3_23 = Christoffel_3_23

      Christoffel_3_33 *= (2/Δeta) * (Δeta/2) * (Δeta/2); self.christoffel_3_33 = Christoffel_3_33

      # Assign H_cov and its elements to the object
      self.H_cov = H_cov
      self.H_cov_11 = H_cov[0,0,:,:,:]
      self.H_cov_12 = H_cov[0,1,:,:,:]
      self.H_cov_13 = H_cov[0,2,:,:,:]
      self.H_cov_21 = H_cov[1,0,:,:,:]
      self.H_cov_22 = H_cov[1,1,:,:,:]
      self.H_cov_23 = H_cov[1,2,:,:,:]
      self.H_cov_31 = H_cov[2,0,:,:,:]
      self.H_cov_32 = H_cov[2,1,:,:,:]
      self.H_cov_33 = H_cov[2,2,:,:,:]

      self.H_cov_itf_i = H_cov_itf_i
      self.H_cov_11_itf_i = H_cov_itf_i[0,0,:,:,:]
      self.H_cov_12_itf_i = H_cov_itf_i[0,1,:,:,:]
      self.H_cov_13_itf_i = H_cov_itf_i[0,2,:,:,:]
      self.H_cov_21_itf_i = H_cov_itf_i[1,0,:,:,:]
      self.H_cov_22_itf_i = H_cov_itf_i[1,1,:,:,:]
      self.H_cov_23_itf_i = H_cov_itf_i[1,2,:,:,:]
      self.H_cov_31_itf_i = H_cov_itf_i[2,0,:,:,:]
      self.H_cov_32_itf_i = H_cov_itf_i[2,1,:,:,:]
      self.H_cov_33_itf_i = H_cov_itf_i[2,2,:,:,:]

      self.H_cov_itf_j = H_cov_itf_j
      self.H_cov_11_itf_j = H_cov_itf_j[0,0,:,:,:]
      self.H_cov_12_itf_j = H_cov_itf_j[0,1,:,:,:]
      self.H_cov_13_itf_j = H_cov_itf_j[0,2,:,:,:]
      self.H_cov_21_itf_j = H_cov_itf_j[1,0,:,:,:]
      self.H_cov_22_itf_j = H_cov_itf_j[1,1,:,:,:]
      self.H_cov_23_itf_j = H_cov_itf_j[1,2,:,:,:]
      self.H_cov_31_itf_j = H_cov_itf_j[2,0,:,:,:]
      self.H_cov_32_itf_j = H_cov_itf_j[2,1,:,:,:]
      self.H_cov_33_itf_j = H_cov_itf_j[2,2,:,:,:]

      self.H_cov_itf_k = H_cov_itf_k
      self.H_cov_11_itf_k = H_cov_itf_k[0,0,:,:,:]
      self.H_cov_12_itf_k = H_cov_itf_k[0,1,:,:,:]
      self.H_cov_13_itf_k = H_cov_itf_k[0,2,:,:,:]
      self.H_cov_21_itf_k = H_cov_itf_k[1,0,:,:,:]
      self.H_cov_22_itf_k = H_cov_itf_k[1,1,:,:,:]
      self.H_cov_23_itf_k = H_cov_itf_k[1,2,:,:,:]
      self.H_cov_31_itf_k = H_cov_itf_k[2,0,:,:,:]
      self.H_cov_32_itf_k = H_cov_itf_k[2,1,:,:,:]
      self.H_cov_33_itf_k = H_cov_itf_k[2,2,:,:,:]

      # Assign H_contra and its elements to the object
      self.H_contra = H_contra
      self.H_contra_11 = H_contra[0,0,:,:,:]
      self.H_contra_12 = H_contra[0,1,:,:,:]
      self.H_contra_13 = H_contra[0,2,:,:,:]
      self.H_contra_21 = H_contra[1,0,:,:,:]
      self.H_contra_22 = H_contra[1,1,:,:,:]
      self.H_contra_23 = H_contra[1,2,:,:,:]
      self.H_contra_31 = H_contra[2,0,:,:,:]
      self.H_contra_32 = H_contra[2,1,:,:,:]
      self.H_contra_33 = H_contra[2,2,:,:,:]

      self.H_contra_itf_i = H_contra_itf_i
      self.H_contra_11_itf_i = H_contra_itf_i[0,0,:,:,:]
      self.H_contra_12_itf_i = H_contra_itf_i[0,1,:,:,:]
      self.H_contra_13_itf_i = H_contra_itf_i[0,2,:,:,:]
      self.H_contra_21_itf_i = H_contra_itf_i[1,0,:,:,:]
      self.H_contra_22_itf_i = H_contra_itf_i[1,1,:,:,:]
      self.H_contra_23_itf_i = H_contra_itf_i[1,2,:,:,:]
      self.H_contra_31_itf_i = H_contra_itf_i[2,0,:,:,:]
      self.H_contra_32_itf_i = H_contra_itf_i[2,1,:,:,:]
      self.H_contra_33_itf_i = H_contra_itf_i[2,2,:,:,:]

      self.H_contra_itf_j = H_contra_itf_j
      self.H_contra_11_itf_j = H_contra_itf_j[0,0,:,:,:]
      self.H_contra_12_itf_j = H_contra_itf_j[0,1,:,:,:]
      self.H_contra_13_itf_j = H_contra_itf_j[0,2,:,:,:]
      self.H_contra_21_itf_j = H_contra_itf_j[1,0,:,:,:]
      self.H_contra_22_itf_j = H_contra_itf_j[1,1,:,:,:]
      self.H_contra_23_itf_j = H_contra_itf_j[1,2,:,:,:]
      self.H_contra_31_itf_j = H_contra_itf_j[2,0,:,:,:]
      self.H_contra_32_itf_j = H_contra_itf_j[2,1,:,:,:]
      self.H_contra_33_itf_j = H_contra_itf_j[2,2,:,:,:]

      self.H_contra_itf_k = H_contra_itf_k
      self.H_contra_11_itf_k = H_contra_itf_k[0,0,:,:,:]
      self.H_contra_12_itf_k = H_contra_itf_k[0,1,:,:,:]
      self.H_contra_13_itf_k = H_contra_itf_k[0,2,:,:,:]
      self.H_contra_21_itf_k = H_contra_itf_k[1,0,:,:,:]
      self.H_contra_22_itf_k = H_contra_itf_k[1,1,:,:,:]
      self.H_contra_23_itf_k = H_contra_itf_k[1,2,:,:,:]
      self.H_contra_31_itf_k = H_contra_itf_k[2,0,:,:,:]
      self.H_contra_32_itf_k = H_contra_itf_k[2,1,:,:,:]
      self.H_contra_33_itf_k = H_contra_itf_k[2,2,:,:,:]

      self.sqrtG = sqrtG
      self.sqrtG_itf_i = sqrtG_itf_i
      self.sqrtG_itf_j = sqrtG_itf_j
      self.sqrtG_itf_k = sqrtG_itf_k
      self.inv_sqrtG = 1/sqrtG

      self.coriolis_f = 2 * geom.rotation_speed / geom.delta * ( math.sin(geom.lat_p) - geom.X * math.cos(geom.lat_p) * math.sin(geom.angle_p) + geom.Y * math.cos(geom.lat_p) * math.cos(geom.angle_p))

      self.inv_dzdeta = 1/dRdeta_int * 2/Δeta


class Metric:
   '''Metric for a smooth, three-dimensional earthlike cubed-sphere with the shallow atmosphere approximation'''
   def __init__(self, geom : cubed_sphere):
      # 3D Jacobian, for the cubed-sphere mapping
      # Note that with no topography, ∂z/∂η=1; the model top is included
      # inside the geometry definition, and η=x3

      self.sqrtG       = geom.earth_radius**2 * (1.0 + geom.X**2) * (1.0 + geom.Y**2) / ( geom.delta2 * geom.delta )
      self.sqrtG_itf_i = geom.earth_radius**2 * (1.0 + geom.X_itf_i**2) * (1.0 + geom.Y_itf_i**2) / ( geom.delta2_itf_i * geom.delta_itf_i )
      self.sqrtG_itf_j = geom.earth_radius**2 * (1.0 + geom.X_itf_j**2) * (1.0 + geom.Y_itf_j**2) / ( geom.delta2_itf_j * geom.delta_itf_j )

      self.inv_sqrtG   = 1.0 / self.sqrtG

      # 3D contravariant metric

      self.H_contra_11 = geom.delta2 / ( geom.earth_radius**2 * (1.0 + geom.X**2) )
      self.H_contra_12 = geom.delta2 * geom.X * geom.Y / ( geom.earth_radius**2 * (1.0 + geom.X**2) * (1.0 + geom.Y**2) )
      self.H_contra_13 = 0

      self.H_contra_21 = self.H_contra_12.copy()
      self.H_contra_22 = geom.delta2 / ( geom.earth_radius**2 * (1.0 + geom.Y**2) )
      self.H_contra_23 = 0

      self.H_contra_31 = 0
      self.H_contra_32 = 0
      self.H_contra_33 = 1

      zero_itf_i = numpy.zeros(geom.delta2_itf_i.shape)
      one_itf_i  = numpy.ones(geom.delta2_itf_i.shape)
      zero_itf_j = numpy.zeros(geom.delta2_itf_j.shape)
      one_itf_j  = numpy.ones(geom.delta2_itf_j.shape)

      # Metric at interfaces
      self.H_contra_11_itf_i = geom.delta2_itf_i / ( geom.earth_radius**2 * (1.0 + geom.X_itf_i**2) )
      self.H_contra_12_itf_i = geom.delta2_itf_i * geom.X_itf_i * geom.Y_itf_i / ( geom.earth_radius**2 * (1.0 + geom.X_itf_i**2) * (1.0 + geom.Y_itf_i**2) )
      self.H_contra_13_itf_i = zero_itf_i.copy()

      self.H_contra_21_itf_i = self.H_contra_12_itf_i.copy()
      self.H_contra_22_itf_i = geom.delta2_itf_i / ( geom.earth_radius**2 * (1.0 + geom.Y_itf_i**2) )
      self.H_contra_23_itf_i = zero_itf_i.copy()

      self.H_contra_11_itf_j = geom.delta2_itf_j / ( geom.earth_radius**2 * (1.0 + geom.X_itf_j**2) )
      self.H_contra_12_itf_j = geom.delta2_itf_j * geom.X_itf_j * geom.Y_itf_j / ( geom.earth_radius**2 * (1.0 + geom.X_itf_j**2) * (1.0 + geom.Y_itf_j**2) )
      self.H_contra_13_itf_j = zero_itf_j.copy()

      self.H_contra_21_itf_j = self.H_contra_12_itf_j.copy()
      self.H_contra_22_itf_j = geom.delta2_itf_j / ( geom.earth_radius**2 * (1.0 + geom.Y_itf_j**2) )
      self.H_contra_23_itf_j = zero_itf_j.copy()

      self.H_contra_31_itf_i = zero_itf_i.copy()
      self.H_contra_32_itf_i = zero_itf_i.copy()
      self.H_contra_33_itf_i = one_itf_i.copy()

      self.H_contra_31_itf_j = zero_itf_j.copy()
      self.H_contra_32_itf_j = zero_itf_j.copy()
      self.H_contra_33_itf_j = one_itf_j.copy()

      # 2D covariant metric

      fact = geom.earth_radius**2 / geom.delta**4
      self.H_cov_11 = fact * (1 + geom.X**2)**2 * (1 + geom.Y**2)
      self.H_cov_12 = - fact * geom.X * geom.Y * (1 + geom.X**2) * (1 + geom.Y**2)
      self.H_cov_13 = 0
      
      self.H_cov_21 = self.H_cov_12.copy()
      self.H_cov_22 = fact * (1 + geom.X**2) * (1 + geom.Y**2)**2
      self.H_cov_23 = 0

      self.H_cov_31 = 0
      self.H_cov_32 = 0
      self.H_cov_33 = 1

      # z/eta relationship.  Initially, z and η/x3 are identical
      self.inv_dzdeta = 1

      # Christoffel symbols

      gridrot = math.sin(geom.lat_p) - geom.X * math.cos(geom.lat_p) * math.sin(geom.angle_p) + geom.Y * math.cos(geom.lat_p) * math.cos(geom.angle_p)

      self.christoffel_1_01 = geom.rotation_speed * geom.X * geom.Y / geom.delta2 * gridrot
      self.christoffel_1_10 = self.christoffel_1_01.copy()

      self.christoffel_1_02 = -geom.rotation_speed * (1.0 + geom.Y**2) / geom.delta2 * gridrot
      self.christoffel_1_20 = self.christoffel_1_02.copy()

      self.christoffel_1_03 = 0
      self.christoffel_1_30 = 0

      self.christoffel_2_01 = geom.rotation_speed * (1.0 + geom.X**2) / geom.delta2 * gridrot
      self.christoffel_2_10 = self.christoffel_2_01.copy()

      self.christoffel_2_02 =-geom.rotation_speed * geom.X * geom.Y / geom.delta2 * gridrot
      self.christoffel_2_20 = self.christoffel_2_02.copy()

      self.christoffel_2_03 = 0
      self.christoffel_2_30 = 0

      self.christoffel_1_11 = 2 * geom.X * geom.Y**2 / geom.delta2

      self.christoffel_1_12 = - (geom.Y + geom.Y**3) / geom.delta2
      self.christoffel_1_21 = self.christoffel_1_12.copy()

      self.christoffel_1_22 = 0

      self.christoffel_1_31 = 0
      self.christoffel_1_13 = 0

      self.christoffel_1_32 = 0
      self.christoffel_1_23 = 0

      self.christoffel_1_33 = 0

      self.christoffel_2_11 = 0

      self.christoffel_2_12 = -geom.X * (1.0 + geom.X**2) / geom.delta2
      self.christoffel_2_21 = self.christoffel_2_12.copy()

      self.christoffel_2_13 = 0
      self.christoffel_2_31 = 0

      self.christoffel_2_22 = 2.0 * geom.X**2 * geom.Y / geom.delta2

      self.christoffel_2_23 = 0
      self.christoffel_2_32 = 0

      self.christoffel_2_33 = 0

      # Γ^3_ab is zero without coordinate mapping
      self.christoffel_3_01 = 0
      self.christoffel_3_10 = 0
      self.christoffel_3_02 = 0
      self.christoffel_3_20 = 0
      self.christoffel_3_03 = 0
      self.christoffel_3_30 = 0

      self.christoffel_3_11 = 0

      self.christoffel_3_12 = 0
      self.christoffel_3_21 = 0

      self.christoffel_3_13 = 0
      self.christoffel_3_31 = 0

      self.christoffel_3_22 = 0

      self.christoffel_3_23 = 0
      self.christoffel_3_32 = 0

      self.christoffel_3_33 = 0

      # Coriolis parameter
      self.coriolis_f = 2 * geom.rotation_speed / geom.delta * ( math.sin(geom.lat_p) - geom.X * math.cos(geom.lat_p) * math.sin(geom.angle_p) + geom.Y * math.cos(geom.lat_p) * math.cos(geom.angle_p))

      # Now, apply the conversion to the reference element.  In geometric coordinates,
      # x1 and x2 run from -pi/4 to pi/4 and x3 runs from 0 to htop, but inside each elemenet
      # (from the perspective of the differentiation matrices) each coordinate runs from -1 to 1.

      self.sqrtG *= geom.Δx1 * geom.Δx2 * geom.Δx3 / 8.
      self.sqrtG_itf_i *= geom.Δx1 * geom.Δx2 * geom.Δx3 / 8.
      self.sqrtG_itf_j *= geom.Δx1 * geom.Δx2 * geom.Δx3 / 8.

      self.inv_sqrtG   = 1.0 / self.sqrtG

      # Modify dz/deta.  Within an element, η now runs from -1 to 1 but covers Δx3 height
      self.inv_dzdeta *= 2/geom.Δx3

      self.H_cov_11 *= geom.Δx1**2 / 4.
      self.H_cov_12 *= geom.Δx1 * geom.Δx2 / 4.
      self.H_cov_13 *= geom.Δx1 * geom.Δx3 / 4.
      self.H_cov_21 *= geom.Δx2 * geom.Δx1 / 4.
      self.H_cov_22 *= geom.Δx2**2 / 4.
      self.H_cov_23 *= geom.Δx2 * geom.Δx3 / 4.
      self.H_cov_33 *= geom.Δx3**2 / 4.

      self.H_contra_11 *= 4. / (geom.Δx1**2)
      self.H_contra_12 *= 4. / (geom.Δx1 * geom.Δx2)
      self.H_contra_13 *= 4. / (geom.Δx1 * geom.Δx3)

      self.H_contra_21 *= 4. / (geom.Δx1 * geom.Δx2)
      self.H_contra_22 *= 4. / (geom.Δx2**2)
      self.H_contra_23 *= 4. / (geom.Δx3 * geom.Δx2)

      self.H_contra_31 *= 4. / (geom.Δx1 * geom.Δx3)
      self.H_contra_32 *= 4. / (geom.Δx2 * geom.Δx3)
      self.H_contra_33 *= 4. / (geom.Δx3**2)

      self.H_contra_11_itf_i *= 4. / (geom.Δx1**2)
      self.H_contra_12_itf_i *= 4. / (geom.Δx1 * geom.Δx2)
      self.H_contra_13_itf_i *= 4. / (geom.Δx1 * geom.Δx3)

      self.H_contra_21_itf_i *= 4. / (geom.Δx1 * geom.Δx2)
      self.H_contra_22_itf_i *= 4. / (geom.Δx2**2)
      self.H_contra_23_itf_i *= 4. / (geom.Δx2 * geom.Δx3)

      self.H_contra_31_itf_i *= 4. / (geom.Δx1 * geom.Δx3)
      self.H_contra_32_itf_i *= 4. / (geom.Δx2 * geom.Δx3)
      self.H_contra_33_itf_i *= 4. / (geom.Δx3**2)

      self.H_contra_11_itf_j *= 4. / (geom.Δx1**2)
      self.H_contra_12_itf_j *= 4. / (geom.Δx1 * geom.Δx2)
      self.H_contra_13_itf_j *= 4. / (geom.Δx1 * geom.Δx3)

      self.H_contra_21_itf_j *= 4. / (geom.Δx1 * geom.Δx2)
      self.H_contra_22_itf_j *= 4. / (geom.Δx2**2)
      self.H_contra_23_itf_j *= 4. / (geom.Δx2 * geom.Δx3)
      
      self.H_contra_31_itf_j *= 4. / (geom.Δx1 * geom.Δx3)
      self.H_contra_32_itf_j *= 4. / (geom.Δx2 * geom.Δx3)
      self.H_contra_33_itf_j *= 4. / (geom.Δx3**2)

      self.christoffel_1_11 *= 0.5 * geom.Δx1
      self.christoffel_1_12 *= 0.5 * geom.Δx1
      self.christoffel_1_13 *= 0.5 * geom.Δx1
      self.christoffel_1_21 *= 0.5 * geom.Δx1
      self.christoffel_1_22 *= 0.5 * geom.Δx1
      self.christoffel_1_23 *= 0.5 * geom.Δx1
      self.christoffel_1_31 *= 0.5 * geom.Δx1
      self.christoffel_1_32 *= 0.5 * geom.Δx1
      self.christoffel_1_33 *= 0.5 * geom.Δx1

      self.christoffel_2_11 *= 0.5 * geom.Δx2
      self.christoffel_2_12 *= 0.5 * geom.Δx2
      self.christoffel_2_13 *= 0.5 * geom.Δx2
      self.christoffel_2_21 *= 0.5 * geom.Δx2
      self.christoffel_2_22 *= 0.5 * geom.Δx2
      self.christoffel_2_23 *= 0.5 * geom.Δx2
      self.christoffel_2_31 *= 0.5 * geom.Δx2
      self.christoffel_2_32 *= 0.5 * geom.Δx2
      self.christoffel_2_33 *= 0.5 * geom.Δx2

      self.christoffel_3_11 *= 0.5 * geom.Δx3
      self.christoffel_3_12 *= 0.5 * geom.Δx3
      self.christoffel_3_13 *= 0.5 * geom.Δx3
      self.christoffel_3_21 *= 0.5 * geom.Δx3
      self.christoffel_3_22 *= 0.5 * geom.Δx3
      self.christoffel_3_23 *= 0.5 * geom.Δx3
      self.christoffel_3_31 *= 0.5 * geom.Δx3
      self.christoffel_3_32 *= 0.5 * geom.Δx3
      self.christoffel_3_33 *= 0.5 * geom.Δx3


