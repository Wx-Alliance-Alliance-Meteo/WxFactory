import numpy
import numpy.linalg
import math
import sympy
from typing import Optional

from common.definitions     import idx_2d_rho_w
from common.program_options import Configuration
from .cartesian_2d_mesh     import Cartesian2D
from .cubed_sphere          import CubedSphere
from .geometry              import Geometry

class DFROperators:
   '''Set of operators used for Direct Flux Reconstruction.

   The relevant internal matrices are:
      * The extrapolation matrices, `extrap_west`, `extrap_east`, `extrap_south`, `extrap_north`, `extrap_down`, and
        `extrap_up`. They are used to compute values at the boundaries of an element.
      * Differentiation matrices: `diff`, `diff_ext`, `diff_tr`, `diff_solpt`, `diff_solpt_tr`
      * Correction matrices: `correction`, `correction_tr`.
   '''

   def __init__(self, grd: Geometry, param: Configuration):
      '''Initialize the Direct Flux Reconstruction operators (matrices) based on input grid parameters.

      Parameters
      ----------
      grd : Geometry
         Underlying grid, which must define `solutionPoints`, `solutionPoints_sym`, `extension`, `extension_sym` and
         `nbsolpts` as member variables
      filter_apply : bool
         Whether to apply an exponential filter in defininng the differential operators
      filter_order : int
         If applied, what order of exponential to use for the filter
      filter_cutoff : float
         If applied, at what relative wavenumber (0 < cutoff < 1) to begin applying the filter
      '''

      # Build Vandermonde matrix to transform the modal representation to the (interior)
      # nodal representation
      V = numpy.polynomial.legendre.legvander(grd.solutionPoints,grd.nbsolpts-1)
      # Invert the matrix to transform from interior nodes to modes
      invV = numpy.linalg.inv(V)

      # Build the negative and positive-side extrapolation matrices by:
      # *) transforming interior nodes to modes
      # *) evaluating the modes at ± 1

      # Note that extrap_neg and extrap_pos should be vectors, not a one-row matrix; numpy
      # treats the two differently.
      extrap_neg = (numpy.polynomial.legendre.legvander([-1],grd.nbsolpts-1) @ invV).reshape((-1,))
      extrap_pos = (numpy.polynomial.legendre.legvander([+1],grd.nbsolpts-1) @ invV).reshape((-1,))

      self.extrap_west = extrap_neg
      self.extrap_east = extrap_pos
      self.extrap_south = extrap_neg
      self.extrap_north = extrap_pos
      self.extrap_down = extrap_neg
      self.extrap_up = extrap_pos


      # self.extrap_west = lagrangeEval(grd.solutionPoints_sym, -1)
      # self.extrap_east = lagrangeEval(grd.solutionPoints_sym,  1)

      # self.extrap_south = lagrangeEval(grd.solutionPoints_sym, -1)
      # self.extrap_north = lagrangeEval(grd.solutionPoints_sym,  1)

      # self.extrap_down = lagrangeEval(grd.solutionPoints_sym, -1)
      # self.extrap_up   = lagrangeEval(grd.solutionPoints_sym,  1)

      # Create highest-mode filter
      V = numpy.polynomial.legendre.legvander(grd.solutionPoints,grd.nbsolpts-1) # Transform mode space to grid space
      invV = inv(V) # Transform grid space to mode space
      feye = numpy.eye(grd.nbsolpts) # Suppress high mode
      feye[-1,-1] = 0
      self.highfilter = V @ (feye @ invV)

      diff = diffmat(grd.extension_sym)

      if param.filter_apply:
         self.V = vandermonde(grd.extension)
         self.invV = inv(self.V)
         N = len(grd.extension)-1
         Nc = math.floor(param.filter_cutoff * N)
         self.filter = filter_exponential(N, Nc, param.filter_order, self.V, self.invV)
         self.diff_ext = ( self.filter @ diff ).astype(float)
         self.diff_ext[numpy.abs(self.diff_ext) < 1e-20] = 0.

      else:
         self.diff_ext = diff

      self.expfilter_apply = param.expfilter_apply
      if param.expfilter_apply:
         if not isinstance(grd, CubedSphere):
            raise TypeError(f'The 3D filter can only be applied on a CubedSphere geometry')
         self.expfilter = self.make_filter(param.expfilter_strength, param.expfilter_order, param.expfilter_cutoff,
                                           grd)

      # Create sponge layer (if desired)
      self.apply_sponge = param.apply_sponge
      if param.apply_sponge:
         if not isinstance(grd, Cartesian2D):
            raise TypeError(f'The sponge can only be applied on a Cartesian2D geometry')
         nk, ni = grd.X1.shape
         zs = param.z1 - param.sponge_zscale  # zs is bottom of layer
         self.beta= numpy.zeros_like(grd.X1)  # used as our damping profile
         # Loop over points
         for k in range(nk):
            for i in range(ni):
               if (grd.X3[k,i] >= zs):
                  self.beta[k,i] = self.beta[k,i] + (1.0 / param.sponge_tscale) * \
                                   numpy.sin((0.5*numpy.pi) * (grd.X3[k,i] - zs) / (param.z1 - zs))**2

      # if check_skewcentrosymmetry(self.diff_ext) is False:
      #    print('Something horribly wrong has happened in the creation of the differentiation matrix')
      #    exit(1)

      # Force matrices to be in C-contiguous order
      self.diff_solpt = numpy.ascontiguousarray( self.diff_ext[1:-1, 1:-1] )
      self.correction = numpy.ascontiguousarray( numpy.column_stack((self.diff_ext[1:-1,0], self.diff_ext[1:-1,-1])) )

      self.diff_solpt_tr = self.diff_solpt.T.copy()
      self.correction_tr = self.correction.T.copy()

      # Ordinary differentiation matrices (used only in diagnostic calculations)
      self.diff = diffmat(grd.solutionPoints)
      self.diff_tr = self.diff.T

      self.quad_weights = numpy.outer(grd.glweights, grd.glweights)

   def make_filter(self, alpha: float, order: int, cutoff: float, geom: Geometry) -> numpy.ndarray:
      '''Build an exponential modal filter as described in Warburton, eqn 5.16.'''

      # Scaled mode numbers
      modes = numpy.arange(geom.nbsolpts) / (geom.nbsolpts-1)
      Nc = cutoff

      # After applying the filter, each mode is reduced in proportion to the filter order
      # and the mode number relative to nbsolpts, with modes below the cutoff limit untouched

      residual_modes = numpy.ones_like(modes)
      residual_modes[modes > Nc] = numpy.exp(-alpha*((modes[modes > Nc] - cutoff)/(1 - cutoff))**order)

      # Now, use a Vandermonde matrix to transform this modal filter into a nodal form

      # mode-to-node operator
      vander = numpy.polynomial.legendre.legvander(geom.solutionPoints,geom.nbsolpts-1)

      # node-to-mode operator
      ivander = numpy.linalg.inv(vander)

      return vander @ numpy.diag(residual_modes) @ ivander

   def apply_filters(self, Q: numpy.ndarray, geom: Geometry, metric, dt: float):
      '''Apply the filters that have been activated on the given state vector.'''

      if self.expfilter_apply:
         Q = self.apply_filter_3d(Q, geom, metric)

      # Apply Sponge (if desired)
      if self.apply_sponge:
         nk, ni = geom.X1.shape
         # Loop over points
         for k in range(nk):
            for i in range(ni):
               # !!!!!!!!
               # !!! Important Note:
               #     TODO: For 3D, we want to rotate radially, apply sponge, rotate back
               # !!!!!!!!
               ww = (1.0 / (1.0 + self.beta[k,i] * dt)) * Q[idx_2d_rho_w, k, i]
               Q[idx_2d_rho_w, k, i] = ww

      return Q

   def apply_filter_3d(self,Q : numpy.ndarray, geom : CubedSphere, metric):
      '''Apply the exponential filter precomputed in expfilter to input fields \sqrt(g)*Q, and return the
      filtered array.'''
      
      if len(Q.shape) > 3:
         nbvars = Q.shape[0]
      else:
         nbvars = 1

      ni = geom.ni
      nj = geom.nj
      nk = geom.nk
      np = geom.nbsolpts

      result = metric.sqrtG * Q

      # for nof in range (40):
         
      # Filter in i
      result.shape = (nbvars*nk*nj*(ni//np),np)
      result = result @ self.expfilter.T 
      #result = (self.expfilter @ result.T).T
      #result = (self.expfilter @ result.transpose((0,2,1))).transpose((0,2,1))

      # Filter in j
      result.shape = (nbvars*nk*(nj//np), np, ni)
      result = (self.expfilter @ result)

      # Filter in k
      result.shape = (nbvars*(nk//np), np, nj * ni)
      result = (self.expfilter @ result)

      result.shape = Q.shape

      result *= metric.inv_sqrtG

      return result

   def comma_i(self, field_interior : numpy.ndarray, border_i : numpy.ndarray, grid: CubedSphere) -> numpy.ndarray :
      '''Take a partial derivative along the i-index

      This method takes the partial derivative of an input field, potentially consisting of several
      variables, along the `i` index.  This derivative is performed with respect to the canonical element,
      so it contains no corrections for the problem geometry.

      Parameters
      ----------
      field_interior : numpy.ndarray
         The element-interior values of the variable(s) to be differentiated.  This should have
         a shape of `(numvars,npts_z,npts_y,npts_x)`, respecting the prevailing parallel decomposition.
      border_i : numpy.array
         The element-boundary values of the fields to be differentiated, along the i-axis.  This should
         have a shape of `(numvars,npts_z,npts_y,nels_x,2)`, with [:,0] being the leftmost boundary
         (minimal `i`), and [:,1] being the rightmost boundary (maximal `i`)
      grid : Geometry
         Grid-defining class, used here solely to provide the canonical definition of the local
         computational region.
      '''
      # Create an empty array for output
      output = numpy.empty_like(field_interior)

      # Create views of the input arrays for reshaping, in order to express the differentiation as
      # a set of matrix multiplications

      field_view = field_interior.view()
      border_i_view = border_i.view()

      # Reshape to a flat view.  Assigning to array.shape will raise an exception if the new shape would
      # require a memory copy; this implicitly ensures that the input arrays are fully contiguous in
      # memory.
      field_view.shape = (-1, grid.nbsolpts)
      border_i_view.shape = (-1,2)
      output.shape = field_view.shape

      # Perform the matrix transposition
      numpy.dot(field_view, self.diff_solpt_tr,out=output)
      #output[:] = field_view @ self.diff_solpt_tr# + border_i_view @ self.correction_tr
      output[:] += border_i_view @ self.correction_tr
      # print(grid.ptopo.rank, field_view[:2,:], '\n', border_i_view[:2,:],'\n',output[:2,:])

      # Reshape the output array back to its canonical extents
      output.shape = field_interior.shape

      return output

   def extrapolate_i(self, field_interior : numpy.ndarray, grid : CubedSphere) -> numpy.ndarray:
      '''Compute the i-border values along each element of field_interior

      This method extrapolates the variables in `field_interior` to the boundary along
      the i-dimension (last index), using the `extrap_west` and `extrap_east` matrices.

      Parameters
      ----------
      field_interior : numpy.ndarray
         The element-interior values of the variable(s) to be differentiated.  This should have
         a shape of `(numvars,npts_z,npts_y,npts_x)`, respecting the prevailing parallel decomposition.
      grid : Geometry
         Grid-defining class, used here solely to provide the canonical definition of the local
         computational region.'''

      # Array shape for the i-border of a single variable, based on the grid decomposition
      border_shape = (grid.nb_elements_x1,2)
      # Number of variables we're extending
      #nbvars = numpy.prod(field_interior.shape) // (grid.ni)
      nbvars = field_interior.size // grid.ni

      # Create an array for the output
      border = numpy.empty((nbvars,) + border_shape, dtype=field_interior.dtype)
      # Reshape to the from required for matrix multiplication
      border.shape = (-1,2)

      # Create an array view of the interior, reshaped for matrix multiplication
      field_interior_view = field_interior.view()
      field_interior_view.shape = (-1,grid.nbsolpts)

      # Perform the extrapolations via matrix multiplication
      border[:,0] = field_interior_view @ self.extrap_west
      border[:,1] = field_interior_view @ self.extrap_east

      border.shape = tuple(field_interior.shape[0:-1]) + border_shape
      return border

   def comma_j(self, field_interior : numpy.ndarray, border_j : numpy.ndarray, grid : CubedSphere) -> numpy.ndarray:
      '''Take a partial derivative along the j-index

      This method takes the partial derivative of an input field, potentially consisting of several
      variables, along the `j` index.  This derivative is performed with respect to the canonical element,
      so it contains no corrections for the problem geometry.

      Parameters
      ----------
      field_interior : numpy.ndarray
         The element-interior values of the variable(s) to be differentiated.  This should have
         a shape of `(numvars,npts_z,npts_y,npts_x)`, respecting the prevailing parallel decomposition.
      border_j : numpy.ndarray
         The element-boundary values of the fields to be differentiated, along the i-axis.  This should
         have a shape of `(numvars,npts_z,nels_y,2,npts_x)`, with [:,0,:] being the southmost boundary
         (minimal `j`), and [:,1,:] being the north boundary (maximal `j`)
      grid : Geometry
         Grid-defining class, used here solely to provide the canonical definition of the local
         computational region.
      '''
      # Create an empty array for output
      output = numpy.empty_like(field_interior)
      # Compute the number of variables we're differentiating, including number of levels
      #nbvars = numpy.prod(output.shape) // (grid.ni * grid.nj)
      nbvars = output.size // (grid.ni * grid.nj)

      # Create views of the input arrays for reshaping, in order to express the differentiation as
      # a set of matrix multiplications

      field_view = field_interior.view()
      border_j_view = border_j.view()

      # Reshape to a flat view.  Assigning to array.shape will raise an exception if the new shape would
      # require a memory copy; this implicitly ensures that the input arrays are fully contiguous in
      # memory.
      field_view.shape = (nbvars*grid.nb_elements_x2,grid.nbsolpts,grid.ni)
      border_j_view.shape = (nbvars*grid.nb_elements_x2,2,grid.ni)
      output.shape = field_view.shape

      # Perform the matrix transposition
      output[:] = self.diff_solpt @ field_view + self.correction @ border_j_view

      # Reshape the output array back to its canonical extents
      output.shape = field_interior.shape

      return output

   def extrapolate_j(self, field_interior : numpy.ndarray, grid : CubedSphere) -> numpy.ndarray :
      '''Compute the j-border values along each element of field_interior

      This method extrapolates the variables in `field_interior` to the boundary along
      the j-dimension (second last index), using the `extrap_south` and `extrap_north` matrices.

      Parameters
      ----------
      field_interior : numpy.ndarray
         The element-interior values of the variable(s) to be differentiated.  This should have
         a shape of `(numvars,npts_z,npts_y,npts_x)`, respecting the prevailing parallel decomposition.
         To allow for differentiation of 2D objects, npts_z can be one.
      grid : Geometry
         Grid-defining class, used here solely to provide the canonical definition of the local
         computational region.'''

      # Array shape for the i-border of a single variable, based on the grid decomposition
      border_shape = (grid.nb_elements_x2,2,
                      grid.ni)
      # Number of variables times number of vertical levels we're extending
      #nbvars = numpy.prod(field_interior.shape) // (grid.ni * grid.nj)
      nbvars = field_interior.size // (grid.ni * grid.nj)

      # Create an array for the output
      border = numpy.empty((nbvars,) + border_shape, dtype=field_interior.dtype)
      # Reshape to the from required for matrix multiplication
      border.shape = (-1,2,grid.ni)

      # Create an array view of the interior, reshaped for matrix multiplication
      field_interior_view = field_interior.view()
      field_interior_view.shape = (-1,grid.nbsolpts,grid.ni)

      # Perform the extrapolations via matrix multiplication
      # print(border[:,0,:].shape, field_interior_view.shape, self.extrap_south.T.shape)
      border[:,0,:] = (self.extrap_south @ field_interior_view)
      border[:,1,:] = (self.extrap_north @ field_interior_view)

      # field_interior.shape[0:-2] is (nbvars,nk) for many 3D fields, (nbvars,) for many 2D fields,
      # (nk) for a single 3D field, and () for a single 2D field.
      border.shape = tuple(field_interior.shape[0:-2]) + border_shape
      return border

   def comma_k(self, field_interior : numpy.ndarray, border_k : numpy.ndarray, grid : CubedSphere) -> numpy.ndarray:
      '''Take a partial derivative along the k-index

      This method takes the partial derivative of an input field, potentially consisting of several
      variables, along the `k` index.  This derivative is performed with respect to the canonical element,
      so it contains no corrections for the problem geometry.

      Parameters
      ----------
      field_interior : numpy.ndarray
         The element-interior values of the variable(s) to be differentiated.  This should have
         a shape of `(numvars,npts_z,npts_y,npts_x)`, respecting the prevailing parallel decomposition.
      border_k : numpy.array
         The element-boundary values of the fields to be differentiated, along the i-axis.  This should
         have a shape of `(numvars,nels_z,2,npts_y,npts_x)`, with [:,0,:] being the downmost boundary
         (minimal `k`), and [:,1,:] being the upmost boundary (maximal `k`)
      grid : Geometry
         Grid-defining class, used here solely to provide the canonical definition of the local
         computational region.
      '''
      # Create an empty array for output
      output = numpy.empty_like(field_interior)
      # Compute the number of variables we're differentiating
      #nbvars = numpy.prod(output.shape) // (grid.ni * grid.nj * grid.nk)
      nbvars = output.size // (grid.ni * grid.nj * grid.nk)

      # Create views of the input arrays for reshaping, in order to express the differentiation as
      # a set of matrix multiplications

      field_view = field_interior.view()
      border_k_view = border_k.view()

      # Reshape to a flat view.  Assigning to array.shape will raise an exception if the new shape would
      # require a memory copy; this implicitly ensures that the input arrays are fully contiguous in
      # memory.
      field_view.shape = (nbvars*grid.nb_elements_x3,grid.nbsolpts,grid.ni*grid.nj)
      border_k_view.shape = (nbvars*grid.nb_elements_x3,2,grid.ni*grid.nj)
      output.shape = field_view.shape

      # Perform the matrix transposition
      output[:] = self.diff_solpt @ field_view + self.correction @ border_k_view

      # Reshape the output array back to its canonical extents
      output.shape = field_interior.shape

      return output

   def filter_k(self, field_interior, grid : CubedSphere) -> numpy.ndarray:
      '''Apply a modal filter to remove the highest mode of fiield_interior along the k-dimension

      This method applies the pre-computed 'highfilter' matrix to the field-interior points along
      the vertical (k) dimension, independently of other directions.  The typical use case is to
      filter out the highest element mode to avoid an inconsistency in the gravity term of the
      vertical-only Euler equations, where w_t is proportional to rho*g but rho_t is proportional
      to w_x.

      Parameters:
      -----------
      field_interior : numpy.ndarray
         The element-interior values of the variable(s) to be differentiated.  This should have
         a shape of `(numvars,npts_z,npts_y,npts_x)`, respecting the prevailing parallel decomposition.
      grid : Geometry
         Grid-defining class, used here solely to provide the canonical definition of the local
         computational region.'''

      # Number of variables we're extending
      #nbvars = numpy.prod(field_interior.shape) // (grid.ni * grid.nj * grid.nk)
      nbvars = field_interior.size // (grid.ni * grid.nj * grid.nk)

      # Output array
      filtered = numpy.empty( (nbvars*grid.nb_elements_x3,grid.nbsolpts,grid.ni*grid.nj), dtype=field_interior.dtype)

      # Create an array view of the interior, reshaped for matrix multiplication
      field_interior_view = field_interior.view()
      field_interior_view.shape = (nbvars*grid.nb_elements_x3,grid.nbsolpts,grid.ni*grid.nj)

      filtered[:] = self.highfilter @ field_interior_view
      filtered.shape = field_interior.shape

      return filtered

   def extrapolate_k(self, field_interior : numpy.ndarray, grid : CubedSphere) -> numpy.ndarray:
      '''Compute the k-border values along each element of field_interior

      This method extrapolates the variables in `field_interior` to the boundary along
      the k-dimension (third last index), using the `extrap_down` and `extrap_up` matrices.

      Parameters
      ----------
      field_interior : numpy.ndarray
         The element-interior values of the variable(s) to be differentiated.  This should have
         a shape of `(numvars,npts_z,npts_y,npts_x)`, respecting the prevailing parallel decomposition.
      grid : Geometry
         Grid-defining class, used here solely to provide the canonical definition of the local
         computational region.'''

      # Array shape for the i-border of a single variable, based on the grid decomposition
      border_shape = (grid.nb_elements_x3,2,
                      grid.nj,
                      grid.ni)
      # Number of variables we're extending
      #nbvars = numpy.prod(field_interior.shape) // (grid.ni * grid.nj * grid.nk)
      nbvars = field_interior.size // (grid.ni * grid.nj * grid.nk)

      # Create an array for the output
      border = numpy.empty((nbvars,) + border_shape, dtype=field_interior.dtype)
      # Reshape to the from required for matrix multiplication
      border.shape = (-1,2,grid.ni*grid.nj)

      # Create an array view of the interior, reshaped for matrix multiplication
      field_interior_view = field_interior.view()
      field_interior_view.shape = (-1,grid.nbsolpts,grid.ni*grid.nj)

      # Perform the extrapolations via matrix multiplication
      border[:,0,:] = (self.extrap_down @ field_interior_view)
      border[:,1,:] = (self.extrap_up @ field_interior_view)

      if nbvars > 1:
         border.shape = (nbvars,) + border_shape
      else:
         border.shape = border_shape
      return border

   # Take the gradient of one or more variables, with output shape [3,nvars,ni,nj,nk]
   def grad(self, field : numpy.ndarray,
                  itf_i : numpy.ndarray,
                  itf_j : numpy.ndarray,
                  itf_k : numpy.ndarray,
                  geom : CubedSphere) -> numpy.ndarray:
      ''' Take the gradient of one or more variables, given interface values (not element extensions)

      This function takes the gradient (covariant derivative) along i, j, and k of each of the input
      variables.  Unlike comma_{i,j,k}, this function builds the extended element view internally
      based on the provided interface arrays, making the implicit assumption that the field is
      continuous.

      Parameters:
      -----------
      field: numpy.ndarray (shape [neqs,nk,nj,ni] or [nk,nj,ni])
         Input variable, on element-internal nodal points in the conventional lexical order.  If this
         field is a four-dimensional array, the first dimension is the one separating equations.
      itf_i : numpy.ndarray (shape [...,nk,nj,nel_i])
         Values along the i-interface
      itf_j : numpy.ndarray (shape [...,nk,nel_j,ni])
         Values along the j-interface
      itf_k : numpy.ndarray (shape [...,nel_k,nj,ni])
         Values along the k-interface
      geom : Geometry
         Geometry object

      Returns:
      -------
      grad : numpy.ndarray, shape [3,...]
         Gradiant (covariant derivatives) of the input field
      '''
      (nk, nj, ni) = field.shape[-3:]
      ff = field.view()
      ff.shape = (-1,nk,nj,ni)

      nvar = ff.shape[0]
      nel_i = itf_i.shape[-1]-1
      nel_j = itf_j.shape[-2]-1
      nel_k = itf_k.shape[-3]-1

      iti = itf_i.view()
      iti.shape = (nvar,nk,nj,nel_i+1)

      itj = itf_j.view()
      itj.shape = (nvar,nk,nel_j+1,ni)

      itk = itf_k.view()
      itk.shape = (nvar,nel_k+1,nj,ni)

      ext_i = numpy.zeros((nvar,nk,nj,nel_i,2))
      ext_j = numpy.zeros((nvar,nk,nel_j,2,ni))
      ext_k = numpy.zeros((nvar,nel_k,2,nj,ni))

      ext_i[:,:,:,:,0] = iti[:,:,:,0:-1]
      ext_i[:,:,:,:,1] = iti[:,:,:,1:]

      ext_j[:,:,:,0,:] = itj[:,:,0:-1,:]
      ext_j[:,:,:,1,:] = itj[:,:,1:,:]

      ext_k[:,:,0,:,:] = itk[:,0:-1,:,:]
      ext_k[:,:,1,:,:] = itk[:,1:,:,:]

      output = numpy.zeros((3,nvar,nk,nj,ni))
      output[0,:,:,:,:] = self.comma_i(ff,ext_i,geom)
      output[1,:,:,:,:] = self.comma_j(ff,ext_j,geom)
      output[2,:,:,:,:] = self.comma_k(ff,ext_k,geom)

      output.shape = (3,)+field.shape

      return output


def lagrange_eval(points, newPt):
   '''Evaluate the Lagrange polynomial of a set of points at a specific point.'''
   M = len(points)
   x = sympy.symbols('x')
   l = numpy.zeros_like(points)
   if M == 1:
      l[0] = 1 # Constant
   else:
      for i in range(M):
         l[i] = lagrange_poly(x, M-1, i, points).evalf(subs={x: newPt}, n=20)
   return l.astype(float)

def diffmat(points) -> numpy.ndarray:
   '''Create a 2D differentiation matrix for the given set of points.'''
   M = len(points)
   D = numpy.zeros((M,M))

   x = sympy.symbols('x')
   for i in range(M):
      dL = sympy.diff( lagrange_poly(x, M-1, i, points) )
      for j in range(M):
         if i != j:
            D[j,i] = dL.subs(x, points[j])
      D[i, i] = dL.subs(x, points[i])

   return D

def lagrange_poly(x: sympy.Symbol, order: int, i: int, xi):
   '''Create a symbolic Lagrange polynomial basis function.'''
   index = list(range(order+1))
   index.pop(i)
   return sympy.prod([(x-xi[j])/(xi[i]-xi[j]) for j in index])

def lebesgue(points):
   '''Symbolically compute the integral of the Lagrange polynomial that corresponds to the given points.'''
   M = len(points)
   eval_set = numpy.linspace(-1,1,M)
   x = sympy.symbols('x')
   l = 0
   for i in range(M):
      l = l + sympy.Abs( lagrange_poly(x,M-1,i,points) )
   return [l.subs(x, eval_set[i]) for i in range(M)]

def vandermonde(x: numpy.ndarray):
   r"""Initialize the 1D Vandermonde matrix, \(\mathcal{V}_{ij}=P_j(x_i)\)."""
   N = len(x)

   V = numpy.zeros((N, N), dtype=object)
   y = sympy.symbols('y')
   for j in range(N):
      for i in range(N):
         V[i, j] = sympy.legendre(j, y).evalf(subs={y: x[i]}, n=30, chop=True)

   return V


def remesh_operator(src_points: numpy.ndarray, target_points: numpy.ndarray) -> numpy.ndarray:
   '''Create an element operator to reduce/prolong a grid.'''
   src_nbsolpts = len(src_points)
   target_nbsolpts = len(target_points)

   # Projection
   inv_V_src = inv(vandermonde(src_points))
   V_target = vandermonde(target_points)

   modes = numpy.zeros((target_nbsolpts, src_nbsolpts))
   for i in range(min(src_nbsolpts,target_nbsolpts)):
      modes[i,i] = 1.
   modes[i,i] = 0.5  # damp the highest mode

   return ( V_target @ modes @ inv_V_src ).astype(float)


def filter_exponential(N, Nc, s, V, invV):
   r"""
   Create an exponential filter matrix that can be used to filter out
   high-frequency noise.

   The filter matrix \(\mathcal{F}\) is defined as \(\mathcal{F}=
   \mathcal{V}\Lambda\mathcal{V}^{-1}\) where the diagonal matrix,
   \(\Lambda\) has the entries \(\Lambda_{ii}=\sigma(i-1)\) for
   \(i=1,\ldots,n+1\) and the filter function, \(\sigma(i)\) has the form
   \[
      \sigma(i) =
         \begin{cases}
            1 & 0\le i\le n_c \\
            e^{-\alpha\left (\frac{i-n_c}{n-n_c}\right )^s} & n_c<i\le n.
      \end{cases}
   \]
   Here \(\alpha=-\log(\epsilon_M)\), where \(\epsilon_M\) is the machine
   precision in working precision, \(n\) is the order of the element,
   \(n_c\) is a cutoff, below which the low modes are left untouched and
   \(s\) (has to be even) is the order of the filter.

   Inputs:
      N : The order of the element.
      Nc : The cutoff, below which the low modes are left untouched.
      s : The order of the filter.
      V : The Vandermonde matrix, \(\mathcal{V}\).
      invV : The inverse of the Vandermonde matric, \(\mathcal{V}^{-1}\).

   Outputs:
      F: The return value is the filter matrix, \(\mathcal{F}\).
   """

   n_digit = 30

   alpha = -sympy.log(sympy.Float(numpy.finfo(float).eps, n_digit))

   F = numpy.identity(N+1, dtype=object)
   for i in range(Nc, N+1):
      t = sympy.Rational((i-Nc), (N-Nc))
      F[i,i] = sympy.exp(-alpha*t**s)

   F = V @ F @ invV

   return F


def check_skewcentrosymmetry(m: numpy.ndarray) -> bool:
   '''Verify that the given matrix is skew-centrosymmetric'''
   if m.ndim != 2:
      raise numpy.linalg.LinAlgError(f'Input matrix is not 2-dimensional!')

   n, _ = m.shape
   middle_row = 0

   if n % 2 == 0:
      middle_row = int(n / 2)
   else:
      middle_row = int(n / 2 + 1)

      if m[middle_row-1, middle_row-1] != 0.:
         print()
         print(f'When the order is odd, the central entry of a skew-centrosymmetric matrix must be zero.\n'
               f'Actual value is {m[middle_row-1, middle_row-1]}')
         return False

   for i in range(middle_row):
      for j in range(n):
         if (m[i, j] != -m[n-i-1, n-j-1]):
            print('Non skew-centrosymmetric entries detected:', (m[i, j], m[n-i-1, n-j-1]))
            return False

   return True


# Borrowed from Galois:
# https://github.com/mhostetter/galois
def inv(A: numpy.ndarray) -> numpy.ndarray:
   '''Compute the inverse of a matrix.'''
   if not (A.ndim == 2 and A.shape[0] == A.shape[1]):
      raise numpy.linalg.LinAlgError(f"Argument `A` must be square, not {A.shape}.")
   n = A.shape[0]
   I = numpy.eye(n, dtype=A.dtype)

   # Concatenate A and I to get the matrix AI = [A | I]
   AI = numpy.concatenate((A, I), axis=-1)

   # Perform Gaussian elimination to get the reduced row echelon form AI_rre = [I | A^-1]
   AI_rre = row_reduce(AI, ncols=n)

   # The rank is the number of non-zero rows of the row reduced echelon form
   rank = numpy.sum(~numpy.all(AI_rre[:,0:n] == 0, axis=1))
   if not rank == n:
      raise numpy.linalg.LinAlgError(f"Argument `A` is singular and not invertible because it does not"
                                     f" have full rank of {n}, but rank of {rank}.")

   A_inv = AI_rre[:,-n:]

   return A_inv


def row_reduce(A: numpy.ndarray, ncols: Optional[int] = None) -> numpy.ndarray:
   '''Perform Gaussian elimination using row operations.'''
   if not A.ndim == 2:
      raise ValueError(f"Only 2-D matrices can be converted to reduced row echelon form, not {A.ndim}-D.")

   ncols = A.shape[1] if ncols is None else ncols
   A_rre = A.copy()
   p = 0  # The pivot

   for j in range(ncols):
      # Find a pivot in column `j` at or below row `p`
      idxs = numpy.nonzero(A_rre[p:,j])[0]
      if idxs.size == 0:
         continue
      i = p + idxs[0]  # Row with a pivot

      # Swap row `p` and `i`. The pivot is now located at row `p`.
      A_rre[[p,i],:] = A_rre[[i,p],:]

      # Force pivot value to be 1
      A_rre[p,:] /= A_rre[p,j]

      # Force zeros above and below the pivot
      idxs = numpy.nonzero(A_rre[:,j])[0].tolist()
      idxs.remove(p)
      A_rre[idxs,:] -= numpy.multiply.outer(A_rre[idxs,j], A_rre[p,:])

      p += 1
      if p == A_rre.shape[0]:
         break

   return A_rre
