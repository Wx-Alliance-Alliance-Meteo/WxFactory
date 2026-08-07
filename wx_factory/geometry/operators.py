import math
from typing import TYPE_CHECKING, Self, TypeVar

import numpy
import numpy.linalg
import sympy
import torch
from numpy.typing import NDArray
from torch import Tensor

from ..common.matmul import kron
from ..context import Context
from .cubed_sphere_3d import CubedSphere3D
from .geometry import Geometry

if TYPE_CHECKING:
    from .metric3d import Metric3DTopo

T = TypeVar("T", bound=numpy.generic)


class DFROperators:
    """Set of operators used for Direct Flux Reconstruction.

    The relevant internal matrices are:
       * The extrapolation matrices, `extrap_west`, `extrap_east`, `extrap_south`, `extrap_north`, `extrap_down`, and
         `extrap_up`. They are used to compute values at the boundaries of an element.
       * Differentiation matrices: `diff`, `diff_ext`, `diff_tr`, `diff_solpt`, `diff_solpt_tr`
       * Correction matrices: `correction`, `correction_tr`.
    """

    def __init__(self, grd: Geometry, context: Context, dtype: torch.dtype | None = None):
        """Initialize the Direct Flux Reconstruction operators (matrices) based on input grid parameters.

        Parameters
        ----------
        grd : Geometry
           Underlying grid, which must define `solutionPoints`, `extension`, `extension_sym` and
           `num_solpts` as member variables
        context : Context
           Context object containing the device, MPI communicator and other configuration information.
        dtype : DTypeLike, optional
           Tensor dtype. Defaults to the context's working real dtype.
        """

        self.context = context
        self.dtype = context.real_dtype if dtype is None else dtype
        build_dtype = torch.float64

        # Build Vandermonde matrix to transform the modal representation to the (interior)
        # nodal representation.  Always construct the operators in double precision, then cast the
        # completed matrices once to the configured working precision.  In particular, this keeps
        # roundoff from the Vandermonde inversion out of the stored float32 coefficients.
        V = legvander(grd.solutionPoints, grd.num_solpts - 1).to(build_dtype)
        # Invert the matrix to transform from interior nodes to modes
        invV = torch.linalg.inv(V)

        # Build the negative and positive-side extrapolation matrices by:
        # *) transforming interior nodes to modes
        # *) evaluating the modes at ± 1

        # Note that extrap_neg and extrap_pos should be vectors, not a one-row matrix; numpy
        # treats the two differently.
        extrap_neg = (legvander(torch.tensor([-1.0], dtype=build_dtype), grd.num_solpts - 1) @ invV).reshape((-1,))
        extrap_pos = (legvander(torch.tensor([+1.0], dtype=build_dtype), grd.num_solpts - 1) @ invV).reshape((-1,))

        assert extrap_neg.dtype == build_dtype
        assert extrap_pos.dtype == build_dtype

        self.extrap_west = extrap_neg
        self.extrap_east = extrap_pos
        self.extrap_south = extrap_neg
        self.extrap_north = extrap_pos
        self.extrap_down = extrap_neg
        self.extrap_up = extrap_pos

        V = legvander(grd.solutionPoints, grd.num_solpts - 1).to(build_dtype)
        invV = torch.linalg.inv(V)
        feye = torch.eye(grd.num_solpts, dtype=build_dtype)
        feye[-1, -1] = 0.0
        self.highfilter = V @ (feye @ invV)

        self.highfilter_k = kron(
            self.highfilter.T, torch.eye(grd.num_solpts**2, dtype=build_dtype)
        )  # Only valid in 3D (hence the **2)

        diff = diffmat(grd.extension_sym)
        self.diff_ext = torch.asarray(diff).to(build_dtype)

        assert self.diff_ext.dtype == build_dtype

        if check_skewcentrosymmetry(self.diff_ext) is False:
            raise ValueError("Something horribly wrong has happened in the creation of the differentiation matrix")

        # Force matrices to be in C-contiguous order
        self.diff_solpt = self.diff_ext[1:-1, 1:-1].clone()
        self.correction = torch.column_stack((self.diff_ext[1:-1, 0], self.diff_ext[1:-1, -1]))

        self.diff_solpt_tr = self.diff_solpt.T.clone()
        self.correction_tr = self.correction.T.clone()

        # Ordinary differentiation matrices (used only in diagnostic calculations)
        self.diff = diffmat(grd.solutionPoints)
        self.diff = torch.asarray(self.diff).to(build_dtype)
        self.diff_tr = self.diff.T.clone()

        self.quad_weights = torch.outer(grd.glweights, grd.glweights).to(build_dtype)

        assert self.diff_solpt.dtype == build_dtype
        assert self.correction.dtype == build_dtype
        assert self.diff_solpt_tr.dtype == build_dtype
        assert self.correction_tr.dtype == build_dtype
        assert self.diff.dtype == build_dtype
        assert self.diff_tr.dtype == build_dtype
        assert self.quad_weights.dtype == build_dtype

        if getattr(grd, "is_3d_euler_grid", False):
            I2 = torch.eye(grd.num_solpts, dtype=V.dtype)
            I3 = torch.eye(grd.num_solpts**2, dtype=V.dtype)

            self.extrap_x = torch.vstack((kron(I3, self.extrap_west), kron(I3, self.extrap_east))).T.clone()
            self.extrap_y = torch.vstack(
                (kron(I2, kron(self.extrap_south, I2)), kron(I2, kron(self.extrap_north, I2)))
            ).T.clone()
            self.extrap_z = torch.vstack((kron(self.extrap_down, I3), kron(self.extrap_up, I3))).T.clone()

            self.derivative_x = kron(I3, self.diff_solpt).T.clone()
            self.derivative_y = kron(I2, kron(self.diff_solpt, I2)).T.clone()
            self.derivative_z = kron(self.diff_solpt, I3).T.clone()

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
            ident = torch.eye(grd.num_solpts, dtype=build_dtype)
            self.extrap_x = torch.vstack((kron(ident, self.extrap_west), kron(ident, self.extrap_east))).T.clone()
            self.extrap_y = torch.vstack((kron(self.extrap_south, ident), kron(self.extrap_north, ident))).T.clone()
            self.extrap_z = torch.vstack((kron(self.extrap_down, ident), kron(self.extrap_up, ident))).T.clone()

            self.derivative_x = kron(ident, self.diff_solpt).T.clone()
            self.derivative_y = kron(self.diff_solpt, ident).T.clone()
            self.derivative_z = kron(self.diff_solpt, ident).T.clone()

            corr_down = self.diff_ext[1:-1, 0]
            corr_up = self.diff_ext[1:-1, -1]
            self.correction_DU = torch.vstack((kron(corr_down, ident), kron(corr_up, ident)))

            self.correction_SN = torch.vstack((kron(corr_down, ident), kron(corr_up, ident)))

            corr_west = self.diff_ext[1:-1, 0]
            corr_east = self.diff_ext[1:-1, -1]
            self.correction_WE = torch.vstack((kron(ident, corr_west), kron(ident, corr_east)))
        # Runtime matrix products require the operator and state dtypes to match.  Cast only after
        # every real operator has been derived in float64, so single-precision cases retain their
        # existing memory/performance characteristics with more accurately rounded coefficients.
        if self.dtype != build_dtype:
            for name, value in vars(self).items():
                if hasattr(value, "dtype") and value.dtype == build_dtype:
                    setattr(self, name, value.to(self.dtype))

        if check_skewcentrosymmetry(self.diff_ext) is False:
            raise ValueError("The stored differentiation matrix lost skew-centrosymmetry during precision conversion")

        assert self.extrap_x.dtype == self.dtype
        assert self.extrap_y.dtype == self.dtype
        assert self.extrap_z.dtype == self.dtype
        assert self.derivative_x.dtype == self.dtype
        assert self.derivative_y.dtype == self.dtype
        assert self.derivative_z.dtype == self.dtype
        assert self.correction_DU.dtype == self.dtype
        assert self.correction_SN.dtype == self.dtype
        assert self.correction_WE.dtype == self.dtype

    def make_filter_3d(self, strength: float, order: int, cutoff: float, geom: Geometry):
        """Build an isotropic exponential modal filter for a three-dimensional element."""
        if not getattr(geom, "is_3d_euler_grid", False):
            raise TypeError("The 3D exponential filter requires a three-dimensional Euler geometry")
        if geom.num_solpts < 2:
            raise ValueError("The 3D exponential filter requires at least two solution points")
        if strength < 0.0:
            raise ValueError("The exponential-filter strength must be non-negative")
        if order <= 0 or order % 2:
            raise ValueError("The exponential-filter order must be a positive even integer")
        if not 0.0 <= cutoff <= 1.0:
            raise ValueError("The exponential-filter cutoff must lie in [0, 1]")

        modes = torch.arange(geom.num_solpts, dtype=self.dtype) / (geom.num_solpts - 1)
        attenuation = torch.ones_like(modes)
        filtered = modes > cutoff
        attenuation[filtered] = torch.exp(-strength * ((modes[filtered] - cutoff) / (1.0 - cutoff)) ** order)

        vandermonde = legvander(geom.solutionPoints, geom.num_solpts - 1).to(self.dtype)
        filter_1d = vandermonde @ torch.diag(attenuation) @ torch.linalg.inv(vandermonde)
        identity_1d = torch.eye(geom.num_solpts, dtype=self.dtype)
        identity_2d = torch.eye(geom.num_solpts**2, dtype=self.dtype)
        filter_x = kron(identity_2d, filter_1d).T
        filter_y = kron(identity_1d, kron(filter_1d, identity_1d)).T
        filter_z = kron(filter_1d, identity_2d).T
        return (filter_x @ filter_y) @ filter_z

    @staticmethod
    def apply_filter_3d(Q: NDArray, metric: "Metric3DTopo", filter_matrix: NDArray):
        r"""Filter the metric-weighted conservative state \(\sqrt{G}Q\) element by element."""
        return ((metric.sqrtG_new * Q) @ filter_matrix) * metric.inv_sqrtG_new

    def comma_i(
        self: Self, field_interior: Tensor, border_i: Tensor, grid: CubedSphere3D, out: Tensor | None = None
    ) -> Tensor:
        """Take a partial derivative along the i-index

        This method takes the partial derivative of an input field, potentially consisting of several
        variables, along the `i` index.  This derivative is performed with respect to the canonical element,
        so it contains no corrections for the problem geometry.

        Parameters
        ----------
        field_interior : Tensor
           The element-interior values of the variable(s) to be differentiated.  This should have
           a shape of `(numvars,npts_z,npts_y,npts_x)`, respecting the prevailing parallel decomposition.
        border_i : Tensor
           The element-boundary values of the fields to be differentiated, along the i-axis.  This should
           have a shape of `(numvars,npts_z,npts_y,nels_x,2)`, with [:,0] being the leftmost boundary
           (minimal `i`), and [:,1] being the rightmost boundary (maximal `i`)
        grid : Geometry
           Grid-defining class, used here solely to provide the canonical definition of the local
           computational region.
        out : Tensor | None
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

    def extrapolate_i(self: Self, field_interior: Tensor, grid: CubedSphere3D, out: Tensor | None = None) -> Tensor:
        """Compute the i-border values along each element of field_interior

        This method extrapolates the variables in `field_interior` to the boundary along
        the i-dimension (last index), using the `extrap_west` and `extrap_east` matrices.

        Parameters
        ----------
        field_interior : Tensor
           The element-interior values of the variable(s) to be differentiated.  This should have
           a shape of `(numvars,npts_z,npts_y,npts_x)`, respecting the prevailing parallel decomposition.
        grid : Geometry
           Grid-defining class, used here solely to provide the canonical definition of the local
           computational region.
        out : Tensor | None
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
        self: Self, field_interior: Tensor, border_j: Tensor, grid: CubedSphere3D, out: Tensor | None = None
    ) -> Tensor:
        """Take a partial derivative along the j-index

        This method takes the partial derivative of an input field, potentially consisting of several
        variables, along the `j` index.  This derivative is performed with respect to the canonical element,
        so it contains no corrections for the problem geometry.

        Parameters
        ----------
        field_interior : Tensor
           The element-interior values of the variable(s) to be differentiated.  This should have
           a shape of `(numvars,npts_z,npts_y,npts_x)`, respecting the prevailing parallel decomposition.
        border_j : Tensor
           The element-boundary values of the fields to be differentiated, along the i-axis.  This should
           have a shape of `(numvars,npts_z,nels_y,2,npts_x)`, with [:,0,:] being the southmost boundary
           (minimal `j`), and [:,1,:] being the north boundary (maximal `j`)
        grid : Geometry
           Grid-defining class, used here solely to provide the canonical definition of the local
           computational region.
        out : Tensor | None
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

    def extrapolate_j(self: Self, field_interior: Tensor, grid: CubedSphere3D, out: Tensor | None = None) -> Tensor:
        """Compute the j-border values along each element of field_interior

        This method extrapolates the variables in `field_interior` to the boundary along
        the j-dimension (second last index), using the `extrap_south` and `extrap_north` matrices.

        Parameters
        ----------
        field_interior : Tensor
           The element-interior values of the variable(s) to be differentiated.  This should have
           a shape of `(numvars,npts_z,npts_y,npts_x)`, respecting the prevailing parallel decomposition.
           To allow for differentiation of 2D objects, npts_z can be one.
        grid : Geometry
           Grid-defining class, used here solely to provide the canonical definition of the local
           computational region.
        out : Tensor | None
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
        self: Self, field_interior: Tensor, border_k: Tensor, grid: CubedSphere3D, out: Tensor | None = None
    ) -> Tensor:
        """Take a partial derivative along the k-index

        This method takes the partial derivative of an input field, potentially consisting of several
        variables, along the `k` index.  This derivative is performed with respect to the canonical element,
        so it contains no corrections for the problem geometry.

        Parameters
        ----------
        field_interior : Tensor
           The element-interior values of the variable(s) to be differentiated.  This should have
           a shape of `(numvars,npts_z,npts_y,npts_x)`, respecting the prevailing parallel decomposition.
        border_k : Tensor
           The element-boundary values of the fields to be differentiated, along the i-axis.  This should
           have a shape of `(numvars,nels_z,2,npts_y,npts_x)`, with [:,0,:] being the downmost boundary
           (minimal `k`), and [:,1,:] being the upmost boundary (maximal `k`)
        grid : Geometry
           Grid-defining class, used here solely to provide the canonical definition of the local
           computational region.
        out : Tensor | None
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

    def extrapolate_k(self: Self, field_interior: Tensor, grid: CubedSphere3D, out: Tensor | None = None) -> Tensor:
        """Compute the k-border values along each element of field_interior

        This method extrapolates the variables in `field_interior` to the boundary along
        the k-dimension (third last index), using the `extrap_down` and `extrap_up` matrices.

        Parameters
        ----------
        field_interior : Tensor
           The element-interior values of the variable(s) to be differentiated.  This should have
           a shape of `(numvars,npts_z,npts_y,npts_x)`, respecting the prevailing parallel decomposition.
        grid : Geometry
           Grid-defining class, used here solely to provide the canonical definition of the local
           computational region.
        out : Tensor | None
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
        field: Tensor,
        itf_i: Tensor,
        itf_j: Tensor,
        itf_k: Tensor,
        geom: CubedSphere3D,
        out: Tensor | None = None,
    ) -> Tensor:
        """Take the gradient of one or more variables, given interface values (not element extensions)

        This function takes the gradient (covariant derivative) along i, j, and k of each of the input
        variables.  Unlike comma_{i,j,k}, this function builds the extended element view internally
        based on the provided interface arrays, making the implicit assumption that the field is
        continuous.

        Parameters:
        -----------
        field: torch.Tensor (shape [neqs,nk,nj,ni] or [nk,nj,ni])
           Input variable, on element-internal nodal points in the conventional lexical order.  If this
           field is a four-dimensional array, the first dimension is the one separating equations.
        itf_i : torch.Tensor (shape [...,nk,nj,nel_i])
           Values along the i-interface
        itf_j : torch.Tensor (shape [...,nk,nel_j,ni])
           Values along the j-interface
        itf_k : torch.Tensor (shape [...,nel_k,nj,ni])
           Values along the k-interface
        geom : Geometry
           Geometry object
        out : torch.Tensor | None
           Destination array for operation. If provided, should be a C-contiguous array
           with shape (3, neqs, nk, nj, ni).

        Returns:
        -------
        grad : torch.Tensor, shape [3,...]
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


def check_skewcentrosymmetry(m: Tensor) -> bool:
    """Verify that the given matrix is skew-centrosymmetric"""
    if m.ndim != 2:
        raise numpy.linalg.LinAlgError("Input matrix is not 2-dimensional!")

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
                f"Actual value is {m[middle_row - 1, middle_row - 1]}"
            )
            return False

    for i in range(middle_row):
        for j in range(n):
            if m[i, j] != -m[n - i - 1, n - j - 1]:
                print("Non skew-centrosymmetric entries detected:", (m[i, j], m[n - i - 1, n - j - 1]))
                return False

    return True


def row_reduce(A: numpy.ndarray, ncols: int | None = None) -> numpy.ndarray:
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


def legvander(x: Tensor, deg: int) -> Tensor:
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
