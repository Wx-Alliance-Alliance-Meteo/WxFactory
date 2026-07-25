from ..common.matmul import kron
import torch
import numpy
import numpy.linalg
import math
import sympy
from typing import Optional
from typing import Self, TypeVar

from numpy.typing import NDArray, DTypeLike

from ..device import Device

from .cubed_sphere_3d import CubedSphere3D
from .geometry import Geometry

T = TypeVar("T", bound=numpy.generic)


class DFROperators:
    """Set of operators used for Direct Flux Reconstruction.

    The relevant internal matrices are:
       * The extrapolation matrices, `extrap_west`, `extrap_east`, `extrap_south`, `extrap_north`, `extrap_down`, and
         `extrap_up`. They are used to compute values at the boundaries of an element.
       * Differentiation matrices: `diff`, `diff_ext`, `diff_tr`, `diff_solpt`, `diff_solpt_tr`
       * Correction matrices: `correction`, `correction_tr`.
    """

    def __init__(self, grd: Geometry, device: Device, dtype: DTypeLike = None):
        """Initialize the Direct Flux Reconstruction operators (matrices) based on input grid parameters.

        Parameters
        ----------
        grd : Geometry
           Underlying grid, which must define `solutionPoints`, `solutionPoints_sym`, `extension`, `extension_sym` and
           `num_solpts` as member variables
        device : Device
           Device on which the operator tensors are created.
        dtype : DTypeLike, optional
           Tensor dtype. Defaults to the device's working real dtype.
        """

        self.dtype = device.real_dtype if dtype is None else dtype

        # Build Vandermonde matrix to transform the modal representation to the (interior)
        # nodal representation
        V = legvander(grd.solutionPoints, grd.num_solpts - 1).astype(self.dtype)
        # Invert the matrix to transform from interior nodes to modes
        invV = torch.linalg.inv(V)

        # Build the negative and positive-side extrapolation matrices by:
        # *) transforming interior nodes to modes
        # *) evaluating the modes at ± 1

        # Note that extrap_neg and extrap_pos should be vectors, not a one-row matrix; numpy
        # treats the two differently.
        extrap_neg = (legvander(torch.tensor([-1.0], dtype=self.dtype), grd.num_solpts - 1) @ invV).reshape((-1,))
        extrap_pos = (legvander(torch.tensor([+1.0], dtype=self.dtype), grd.num_solpts - 1) @ invV).reshape((-1,))

        assert extrap_neg.dtype == self.dtype
        assert extrap_pos.dtype == self.dtype

        self.extrap_west = extrap_neg
        self.extrap_east = extrap_pos
        self.extrap_south = extrap_neg
        self.extrap_north = extrap_pos
        self.extrap_down = extrap_neg
        self.extrap_up = extrap_pos

        V = legvander(grd.solutionPoints, grd.num_solpts - 1).astype(self.dtype)
        invV = torch.linalg.inv(V)
        feye = torch.eye(grd.num_solpts, dtype=self.dtype)
        feye[-1, -1] = 0.0
        self.highfilter = V @ (feye @ invV)

        self.highfilter_k = kron(
            self.highfilter.T, torch.eye(grd.num_solpts**2, dtype=self.dtype)
        )  # Only valid in 3D (hence the **2)

        diff = diffmat(grd.extension_sym)
        self.diff_ext = torch.asarray(diff).astype(self.dtype)

        assert self.diff_ext.dtype == self.dtype

        if check_skewcentrosymmetry(self.diff_ext) is False:
            raise ValueError("Something horribly wrong has happened in the creation of the differentiation matrix")

        # Force matrices to be in C-contiguous order
        self.diff_solpt = self.diff_ext[1:-1, 1:-1].copy()
        self.correction = torch.column_stack((self.diff_ext[1:-1, 0], self.diff_ext[1:-1, -1]))

        self.diff_solpt_tr = self.diff_solpt.T.copy()
        self.correction_tr = self.correction.T.copy()

        # Ordinary differentiation matrices (used only in diagnostic calculations)
        self.diff = diffmat(grd.solutionPoints)
        self.diff = torch.asarray(self.diff).astype(self.dtype)
        self.diff_tr = self.diff.T.copy()

        self.quad_weights = torch.outer(grd.glweights, grd.glweights).astype(self.dtype)

        assert self.diff_solpt.dtype == self.dtype
        assert self.correction.dtype == self.dtype
        assert self.diff_solpt_tr.dtype == self.dtype
        assert self.correction_tr.dtype == self.dtype
        assert self.diff.dtype == self.dtype
        assert self.diff_tr.dtype == self.dtype
        assert self.quad_weights.dtype == self.dtype

        if getattr(grd, "is_3d_euler_grid", False):
            I2 = torch.eye(grd.num_solpts, dtype=V.dtype)
            I3 = torch.eye(grd.num_solpts**2, dtype=V.dtype)

            self.extrap_x = torch.vstack((kron(I3, self.extrap_west), kron(I3, self.extrap_east))).T.copy()
            self.extrap_y = torch.vstack(
                (kron(I2, kron(self.extrap_south, I2)), kron(I2, kron(self.extrap_north, I2)))
            ).T.copy()
            self.extrap_z = torch.vstack((kron(self.extrap_down, I3), kron(self.extrap_up, I3))).T.copy()

            self.derivative_x = kron(I3, self.diff_solpt).T.copy()
            self.derivative_y = kron(I2, kron(self.diff_solpt, I2)).T.copy()
            self.derivative_z = kron(self.diff_solpt, I3).T.copy()

            corr_west = self.diff_ext[1:-1, 0]
            corr_east = self.diff_ext[1:-1, -1]
            corr_south = corr_west
            corr_north = corr_east
            corr_down = corr_west
            corr_up = corr_east

            self.correction_WE = torch.vstack((kron(I3, corr_west), kron(I3, corr_east)))
            self.correction_SN = torch.vstack((kron(I2, kron(corr_south, I2)), kron(I2, kron(corr_north, I2))))
            self.correction_DU = torch.vstack((kron(corr_down, I3), kron(corr_up, I3)))

        else:
            ident = torch.eye(grd.num_solpts, dtype=self.dtype)
            self.extrap_x = torch.vstack((kron(ident, self.extrap_west), kron(ident, self.extrap_east))).T.copy()
            self.extrap_y = torch.vstack((kron(self.extrap_south, ident), kron(self.extrap_north, ident))).T.copy()
            self.extrap_z = torch.vstack((kron(self.extrap_down, ident), kron(self.extrap_up, ident))).T.copy()

            self.derivative_x = kron(ident, self.diff_solpt).T.copy()
            self.derivative_y = kron(self.diff_solpt, ident).T.copy()
            self.derivative_z = kron(self.diff_solpt, ident).T.copy()

            corr_down = self.diff_ext[1:-1, 0]
            corr_up = self.diff_ext[1:-1, -1]
            self.correction_DU = torch.vstack((kron(corr_down, ident), kron(corr_up, ident)))

            self.correction_SN = torch.vstack((kron(corr_down, ident), kron(corr_up, ident)))

            corr_west = self.diff_ext[1:-1, 0]
            corr_east = self.diff_ext[1:-1, -1]
            self.correction_WE = torch.vstack((kron(ident, corr_west), kron(ident, corr_east)))
        assert self.extrap_x.dtype == self.dtype
        assert self.extrap_y.dtype == self.dtype
        assert self.extrap_z.dtype == self.dtype
        assert self.derivative_x.dtype == self.dtype
        assert self.derivative_y.dtype == self.dtype
        assert self.derivative_z.dtype == self.dtype
        assert self.correction_DU.dtype == self.dtype
        assert self.correction_SN.dtype == self.dtype
        assert self.correction_WE.dtype == self.dtype

        # Complex128 variants of operators for mixed-type matmul (complex128 @ complex128)
        # These avoid runtime upcasting when the input array is complex128
        self.extrap_x_complex = self.extrap_x.astype(torch.complex128)
        self.extrap_y_complex = self.extrap_y.astype(torch.complex128)
        self.extrap_z_complex = self.extrap_z.astype(torch.complex128)
        self.derivative_x_complex = self.derivative_x.astype(torch.complex128)
        self.derivative_y_complex = self.derivative_y.astype(torch.complex128)
        self.derivative_z_complex = self.derivative_z.astype(torch.complex128)
        self.correction_WE_complex = self.correction_WE.astype(torch.complex128)
        self.correction_SN_complex = self.correction_SN.astype(torch.complex128)
        self.correction_DU_complex = self.correction_DU.astype(torch.complex128)

    def comma_i(
        self: Self, field_interior: NDArray[T], border_i: NDArray[T], grid: CubedSphere3D, out: NDArray[T] | None = None
    ) -> NDArray[T]:
        """Take a partial derivative along the i-index

        This method takes the partial derivative of an input field, potentially consisting of several
        variables, along the `i` index.  This derivative is performed with respect to the canonical element,
        so it contains no corrections for the problem geometry.

        Parameters
        ----------
        field_interior : NDArray
           The element-interior values of the variable(s) to be differentiated.  This should have
           a shape of `(numvars,npts_z,npts_y,npts_x)`, respecting the prevailing parallel decomposition.
        border_i : NDArray
           The element-boundary values of the fields to be differentiated, along the i-axis.  This should
           have a shape of `(numvars,npts_z,npts_y,nels_x,2)`, with [:,0] being the leftmost boundary
           (minimal `i`), and [:,1] being the rightmost boundary (maximal `i`)
        grid : Geometry
           Grid-defining class, used here solely to provide the canonical definition of the local
           computational region.
        out : NDArray | None
           Destination array for operation. If provided, should be a C-contiguous array with the same shape
           as `field_interior`.
        """
        output = torch.empty_like(field_interior) if out is None else out.reshape(field_interior.shape)

        # Create views of the input arrays for reshaping, in order to express the differentiation as
        # a set of matrix multiplications

        field_view = field_interior.reshape((-1, grid.num_solpts))
        border_i_view = border_i.reshape((-1, 2))

        # Reshape to a flat view.  Assigning to array.shape will raise an exception if the new shape would
        # require a memory copy; this implicitly ensures that the input arrays are fully contiguous in
        # memory.
        output = output.reshape((-1, grid.num_solpts))

        # Perform the matrix transposition
        torch.matmul(field_view, self.diff_solpt_tr, out=output)
        output[:] += border_i_view @ self.correction_tr

        # Reshape the output array back to its canonical extents
        output = output.reshape(field_interior.shape)
        if out is not None:
            out[...] = output[...]

        return output

    def extrapolate_i(
        self: Self, field_interior: NDArray[T], grid: CubedSphere3D, out: NDArray[T] | None = None
    ) -> NDArray[T]:
        """Compute the i-border values along each element of field_interior

        This method extrapolates the variables in `field_interior` to the boundary along
        the i-dimension (last index), using the `extrap_west` and `extrap_east` matrices.

        Parameters
        ----------
        field_interior : numpy.ndarray
           The element-interior values of the variable(s) to be differentiated.  This should have
           a shape of `(numvars,npts_z,npts_y,npts_x)`, respecting the prevailing parallel decomposition.
        grid : Geometry
           Grid-defining class, used here solely to provide the canonical definition of the local
           computational region.
        out : NDArray | None
           Destination array for operation. If provided, should be a C-contiguous array
           with shape (numvars, npts_z, npts_y, nels_x, 2).
        """
        # Array shape for the i-border of a single variable, based on the grid decomposition
        border_shape = (grid.num_elements_x1, 2)
        # Number of variables we're extending
        nbvars = math.prod(field_interior.shape) // (grid.ni)

        if out is None:
            # Create an array for the output
            border = torch.empty((nbvars,) + border_shape, dtype=field_interior.dtype)
        else:
            # Create a view of the output so that the shape of the original is not modified
            border = out

        # Reshape to the from required for matrix multiplication
        border = border.reshape((-1, 2))

        # Create an array view of the interior, reshaped for matrix multiplication
        field_interior_view = field_interior.reshape((-1, grid.num_solpts))

        # Perform the extrapolations via matrix multiplication
        border[:, 0] = field_interior_view @ self.extrap_west
        border[:, 1] = field_interior_view @ self.extrap_east

        border = border.reshape(tuple(field_interior.shape[0:-1]) + border_shape)
        if out is not None:
            out[...] = border

        return border

    def comma_j(
        self: Self, field_interior: NDArray[T], border_j: NDArray[T], grid: CubedSphere3D, out: NDArray[T] | None = None
    ) -> NDArray[T]:
        """Take a partial derivative along the j-index

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
        out : NDArray | None
           Destination array for operation. If provided, should be a C-contiguous array with the same shape
           as `field_interior`.
        """

        output = torch.empty_like(field_interior) if out is None else out.reshape(field_interior.shape)

        # Compute the number of variables we're differentiating, including number of levels
        nbvars = math.prod(output.shape) // (grid.ni * grid.nj)

        # Create views of the input arrays for reshaping, in order to express the differentiation as
        # a set of matrix multiplications

        field_view = field_interior.reshape((nbvars * grid.num_elements_x2, grid.num_solpts, grid.ni))
        border_j_view = border_j.reshape((nbvars * grid.num_elements_x2, 2, grid.ni))

        # Reshape to a flat view.  Assigning to array.shape will raise an exception if the new shape would
        # require a memory copy; this implicitly ensures that the input arrays are fully contiguous in
        # memory.
        output = output.reshape(field_view.shape)

        # Perform the matrix transposition
        output[:] = self.diff_solpt @ field_view + self.correction @ border_j_view

        # Reshape the output array back to its canonical extents
        output = output.reshape(field_interior.shape)
        if out is not None:
            out[...] = output[...]

        return output

    def extrapolate_j(
        self: Self, field_interior: NDArray[T], grid: CubedSphere3D, out: NDArray[T] | None = None
    ) -> NDArray[T]:
        """Compute the j-border values along each element of field_interior

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
           computational region.
        out : NDArray | None
           Destination array for operation. If provided, should be a C-contiguous array with
           shape (numvars, npts_z, nels_y, 2, npts_x).
        """
        # Array shape for the i-border of a single variable, based on the grid decomposition
        border_shape = (grid.num_elements_x2, 2, grid.ni)
        # Number of variables times number of vertical levels we're extending
        nbvars = math.prod(field_interior.shape) // (grid.ni * grid.nj)

        if out is None:
            # Create an array for the output
            border = torch.empty((nbvars,) + border_shape, dtype=field_interior.dtype)
        else:
            # Create a view of the output so that the shape of the original is not modified
            border = out
        # Reshape to the from required for matrix multiplication
        border = border.reshape((-1, 2, grid.ni))

        # Create an array view of the interior, reshaped for matrix multiplication
        field_interior_view = field_interior.reshape((-1, grid.num_solpts, grid.ni))

        # Perform the extrapolations via matrix multiplication
        border[:, 0, :] = self.extrap_south @ field_interior_view
        border[:, 1, :] = self.extrap_north @ field_interior_view

        # field_interior.shape[0:-2] is (nbvars,nk) for many 3D fields, (nbvars,) for many 2D fields,
        # (nk) for a single 3D field, and () for a single 2D field.

        border = border.reshape(tuple(field_interior.shape[0:-2]) + border_shape)
        if out is not None:
            out[...] = border

        return border

    def comma_k(
        self: Self, field_interior: NDArray[T], border_k: NDArray[T], grid: CubedSphere3D, out: NDArray[T] | None = None
    ) -> NDArray[T]:
        """Take a partial derivative along the k-index

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
        out : NDArray | None
           Destination array for operation. If provided, should be a C-contiguous array with the same shape
           as `field_interior`.
        """
        output = torch.empty_like(field_interior) if out is None else out.reshape(field_interior.shape)

        # Compute the number of variables we're differentiating
        nbvars = math.prod(output.shape) // (grid.ni * grid.nj * grid.nk)

        # Create views of the input arrays for reshaping, in order to express the differentiation as
        # a set of matrix multiplications

        field_view = field_interior.reshape((nbvars * grid.num_elements_x3, grid.num_solpts, grid.ni * grid.nj))
        border_k_view = border_k.reshape((nbvars * grid.num_elements_x3, 2, grid.ni * grid.nj))

        # Reshape to a flat view.  Assigning to array.shape will raise an exception if the new shape would
        # require a memory copy; this implicitly ensures that the input arrays are fully contiguous in
        # memory.
        output = output.reshape(field_view.shape)

        # Perform the matrix transposition
        output[:] = self.diff_solpt @ field_view + self.correction @ border_k_view

        # Reshape the output array back to its canonical extents
        output = output.reshape(field_interior.shape)
        if out is not None:
            out[...] = output[...]

        return output

    def filter_k(
        self: Self, field_interior: NDArray[T], grid: CubedSphere3D, out: NDArray[T] | None = None
    ) -> NDArray[T]:
        """Apply a modal filter to remove the highest mode of fiield_interior along the k-dimension

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
           computational region.
        out : NDArray | None
           Destination array for operation. If provided, should be a C-contiguous array with the same shape
           as `field_interior`.
        """

        # Number of variables we're extending
        nbvars = field_interior.size // (grid.ni * grid.nj * grid.nk)

        if out is None:
            # Output array
            filtered = numpy.empty(
                (nbvars * grid.num_elements_x3, grid.num_solpts, grid.ni * grid.nj),
                dtype=field_interior.dtype,
                like=field_interior,
            )
        else:
            # Create a view of the output so that the shape of the original is not modified
            filtered = out.view()
            filtered.shape = (nbvars * grid.num_elements_x3, grid.num_solpts, grid.ni * grid.nj)

        # Create an array view of the interior, reshaped for matrix multiplication
        field_interior_view = field_interior.view()
        field_interior_view.shape = (nbvars * grid.num_elements_x3, grid.num_solpts, grid.ni * grid.nj)

        filtered[:] = self.highfilter @ field_interior_view
        filtered.shape = field_interior.shape

        return filtered

    def extrapolate_k(
        self: Self, field_interior: NDArray[T], grid: CubedSphere3D, out: NDArray[T] | None = None
    ) -> NDArray[T]:
        """Compute the k-border values along each element of field_interior

        This method extrapolates the variables in `field_interior` to the boundary along
        the k-dimension (third last index), using the `extrap_down` and `extrap_up` matrices.

        Parameters
        ----------
        field_interior : numpy.ndarray
           The element-interior values of the variable(s) to be differentiated.  This should have
           a shape of `(numvars,npts_z,npts_y,npts_x)`, respecting the prevailing parallel decomposition.
        grid : Geometry
           Grid-defining class, used here solely to provide the canonical definition of the local
           computational region.
        out : NDArray | None
           Destination array for operation. If provided, should be a C-contiguous array
           with shape (numvars, nels_z, 2, npts_y, npts_x).
        """
        # Array shape for the i-border of a single variable, based on the grid decomposition
        border_shape = (grid.num_elements_x3, 2, grid.nj, grid.ni)
        # Number of variables we're extending
        nbvars = math.prod(field_interior.shape) // (grid.ni * grid.nj * grid.nk)

        if out is None:
            # Create an array for the output
            border = torch.empty((nbvars,) + border_shape, dtype=field_interior.dtype)
        else:
            # Create a view of the output so that the shape of the original is not modified
            border = out
        # Reshape to the from required for matrix multiplication
        border = border.reshape((-1, 2, grid.ni * grid.nj))

        # Create an array view of the interior, reshaped for matrix multiplication
        field_interior_view = field_interior.reshape((-1, grid.num_solpts, grid.ni * grid.nj))

        # Perform the extrapolations via matrix multiplication
        border[:, 0, :] = self.extrap_down @ field_interior_view
        border[:, 1, :] = self.extrap_up @ field_interior_view

        if nbvars > 1:
            border = border.reshape((nbvars,) + border_shape)
        else:
            border = border.reshape(border_shape)

        if out is not None:
            out[...] = border
        return border

    # Take the gradient of one or more variables, with output shape [3,nvars,ni,nj,nk]
    def grad(
        self: Self,
        field: NDArray[T],
        itf_i: NDArray[T],
        itf_j: NDArray[T],
        itf_k: NDArray[T],
        geom: CubedSphere3D,
        out: NDArray[T] | None = None,
    ) -> NDArray[T]:
        """Take the gradient of one or more variables, given interface values (not element extensions)

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
        out : NDArray | None
           Destination array for operation. If provided, should be a C-contiguous array
           with shape (3, neqs, nk, nj, ni).

        Returns:
        -------
        grad : numpy.ndarray, shape [3,...]
           Gradiant (covariant derivatives) of the input field
        """
        nk, nj, ni = field.shape[-3:]
        ff = field.reshape((-1, nk, nj, ni))

        nvar = ff.shape[0]
        nel_i = itf_i.shape[-1] - 1
        nel_j = itf_j.shape[-2] - 1
        nel_k = itf_k.shape[-3] - 1

        iti = itf_i.reshape((nvar, nk, nj, nel_i + 1))

        itj = itf_j.reshape((nvar, nk, nel_j + 1, ni))

        itk = itf_k.reshape((nvar, nel_k + 1, nj, ni))

        # shape: (nvar, nk, nj, nel_i, 2)
        ext_i = torch.stack((iti[:, :, :, :-1], iti[:, :, :, 1:]), dim=-1)
        # shape: (nvar, nk, nel_j, 2, ni)
        ext_j = torch.stack((itj[:, :, :-1, :], itj[:, :, 1:, :]), dim=-2)
        # shape: (nvar, nel_k, 2, nj, ni)
        ext_k = torch.stack((itk[:, :-1, :, :], itk[:, 1:, :, :]), dim=-3)

        if out is None:
            output = torch.zeros((3, nvar, nk, nj, ni), dtype=field.dtype)
        else:
            output = out.reshape((3, nvar, nk, nj, ni))

        self.comma_i(ff, ext_i, geom, out=output[0])
        self.comma_j(ff, ext_j, geom, out=output[1])
        self.comma_k(ff, ext_k, geom, out=output[2])

        output = output.reshape((3,) + field.shape)
        if out is not None:
            out[...] = output

        return output


def lagrange_eval(points, newPt):
    """Evaluate the Lagrange polynomial of a set of points at a specific point."""
    M = len(points)
    x = sympy.symbols("x")
    l = numpy.zeros_like(points)
    if M == 1:
        l[0] = 1  # Constant
    else:
        for i in range(M):
            l[i] = lagrange_poly(x, M - 1, i, points).evalf(subs={x: newPt}, n=20)
    return l.astype(float)


def diffmat(points) -> numpy.ndarray:
    """Create a 2D differentiation matrix for the given set of points."""
    M = len(points)
    D = numpy.zeros((M, M))

    x = sympy.symbols("x")
    for i in range(M):
        dL = sympy.diff(lagrange_poly(x, M - 1, i, points))
        for j in range(M):
            if i != j:
                D[j, i] = dL.subs(x, points[j])
        D[i, i] = dL.subs(x, points[i])

    return D


def lagrange_poly(x: sympy.Symbol, order: int, i: int, xi):
    """Create a symbolic Lagrange polynomial basis function."""
    index = list(range(order + 1))
    index.pop(i)
    return sympy.prod([(x - xi[j]) / (xi[i] - xi[j]) for j in index])


def lebesgue(points):
    """Symbolically compute the integral of the Lagrange polynomial that corresponds to the given points."""
    M = len(points)
    eval_set = numpy.linspace(-1, 1, M)
    x = sympy.symbols("x")
    l = 0
    for i in range(M):
        l = l + sympy.Abs(lagrange_poly(x, M - 1, i, points))
    return [l.subs(x, eval_set[i]) for i in range(M)]


def vandermonde(x: numpy.ndarray):
    r"""Initialize the 1D Vandermonde matrix, \(\mathcal{V}_{ij}=P_j(x_i)\)."""
    N = len(x)

    V = numpy.zeros((N, N), dtype=object)
    y = sympy.symbols("y")
    for j in range(N):
        for i in range(N):
            V[i, j] = sympy.legendre(j, y).evalf(subs={y: x[i]}, n=30, chop=True)

    return V


def remesh_operator(src_points: numpy.ndarray, target_points: numpy.ndarray) -> numpy.ndarray:
    """Create an element operator to reduce/prolong a grid."""
    src_num_solpts = len(src_points)
    target_num_solpts = len(target_points)

    # Projection
    inv_V_src = inv(vandermonde(src_points))
    V_target = vandermonde(target_points)

    modes = numpy.zeros((target_num_solpts, src_num_solpts))
    for i in range(min(src_num_solpts, target_num_solpts)):
        modes[i, i] = 1.0
    modes[i, i] = 0.5  # damp the highest mode

    return (V_target @ modes @ inv_V_src).astype(float)


def check_skewcentrosymmetry(m: numpy.ndarray) -> bool:
    """Verify that the given matrix is skew-centrosymmetric"""
    if m.ndim != 2:
        raise numpy.linalg.LinAlgError(f"Input matrix is not 2-dimensional!")

    n, _ = m.shape
    middle_row = 0

    if n % 2 == 0:
        middle_row = int(n / 2)
    else:
        middle_row = int(n / 2 + 1)

        if m[middle_row - 1, middle_row - 1] != 0.0:
            print()
            print(
                f"When the order is odd, the central entry of a skew-centrosymmetric matrix must be zero.\n"
                f"Actual value is {m[middle_row-1, middle_row-1]}"
            )
            return False

    for i in range(middle_row):
        for j in range(n):
            if m[i, j] != -m[n - i - 1, n - j - 1]:
                print("Non skew-centrosymmetric entries detected:", (m[i, j], m[n - i - 1, n - j - 1]))
                return False

    return True


# Borrowed from Galois:
# https://github.com/mhostetter/galois
def inv(A: numpy.ndarray) -> numpy.ndarray:
    """Compute the inverse of a matrix."""
    if not (A.ndim == 2 and A.shape[0] == A.shape[1]):
        raise numpy.linalg.LinAlgError(f"Argument `A` must be square, not {A.shape}.")
    n = A.shape[0]
    I = numpy.eye(n, dtype=A.dtype)

    # Concatenate A and I to get the matrix AI = [A | I]
    AI = numpy.concatenate((A, I), axis=-1)

    # Perform Gaussian elimination to get the reduced row echelon form AI_rre = [I | A^-1]
    AI_rre = row_reduce(AI, ncols=n)

    # The rank is the number of non-zero rows of the row reduced echelon form
    rank = numpy.sum(~numpy.all(AI_rre[:, 0:n] == 0, axis=1))
    if not rank == n:
        raise numpy.linalg.LinAlgError(
            f"Argument `A` is singular and not invertible because it does not"
            f" have full rank of {n}, but rank of {rank}."
        )

    A_inv = AI_rre[:, -n:]

    return A_inv


def row_reduce(A: numpy.ndarray, ncols: Optional[int] = None) -> numpy.ndarray:
    """Perform Gaussian elimination using row operations."""
    if not A.ndim == 2:
        raise ValueError(f"Only 2-D matrices can be converted to reduced row echelon form, not {A.ndim}-D.")

    ncols = A.shape[1] if ncols is None else ncols
    A_rre = A.copy()
    p = 0  # The pivot

    for j in range(ncols):
        # Find a pivot in column `j` at or below row `p`
        idxs = numpy.nonzero(A_rre[p:, j])[0]
        if idxs.size == 0:
            continue
        i = p + idxs[0]  # Row with a pivot

        # Swap row `p` and `i`. The pivot is now located at row `p`.
        A_rre[[p, i], :] = A_rre[[i, p], :]

        # Force pivot value to be 1
        A_rre[p, :] /= A_rre[p, j]

        # Force zeros above and below the pivot
        idxs = numpy.nonzero(A_rre[:, j])[0].tolist()
        idxs.remove(p)
        A_rre[idxs, :] -= numpy.multiply.outer(A_rre[idxs, j], A_rre[p, :])

        p += 1
        if p == A_rre.shape[0]:
            break

    return A_rre


def legvander(x: NDArray[numpy.float64], deg: int) -> NDArray[numpy.float64]:
    """
    NumPy's legvander, slightly modified to work with any array type.

    See: https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.legvander.html#numpy-polynomial-legendre-legvander
    """

    dims = (deg + 1,) + x.shape
    v = torch.empty(dims, dtype=x.dtype)

    v[0] = 1
    if deg > 0:
        v[1] = x
        for i in range(2, deg + 1):
            v[i] = (v[i - 1] * x * (2 * i - 1) - v[i - 2] * (i - 1)) / i
    return torch.moveaxis(v, 0, -1)
