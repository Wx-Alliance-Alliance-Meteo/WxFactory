import numpy
from math import sqrt, copysign
from warnings import warn
import logging
from scipy.integrate._ivp.common import (
    validate_max_step, validate_first_step, warn_extraneous)
from scipy.integrate._ivp.base import OdeSolver, DenseOutput


MIN_FACTOR = 0.2
MAX_FACTOR = 4.0
MAX_FACTOR0 = 10
NFS = numpy.array(0)                                         # failed step counter


def validate_tol(rtol, atol, y):
    """Validate tolerance values. atol can be scalar or array-like, rtol a
    scalar. The bound values are from RKSuite. These differ from those in
    scipy. Bounds are applied without warning.
    """
    atol = numpy.asarray(atol)
    if atol.ndim > 0 and atol.shape != (y.size, ):
        raise ValueError("`atol` has wrong shape.")
    if numpy.any(atol < 0):
        raise ValueError("`atol` must be positive.")
    if not isinstance(rtol, float):
        raise ValueError("`rtol` must be a float.")
    if rtol < 0:
        raise ValueError("`rtol` must be positive.")

    # atol cannot be exactly zero.
    # For double precision float: sqrt(tiny) ~ 1.5e-154
    tiny = numpy.finfo(y.dtype).tiny
    atol = numpy.maximum(atol, sqrt(tiny))

    # rtol is bounded from both sides.
    # The lower bound is lower than in scipy.
    epsneg = numpy.finfo(y.dtype).epsneg
    rtol = numpy.minimum(numpy.maximum(rtol, 10 * epsneg), 0.1)
    return rtol, atol


def calculate_scale(atol, rtol, y, y_new, _mean=False):
    """calculate a scaling vector for the error estimate"""
    if _mean:
        return atol + rtol * 0.5*(numpy.abs(y) + numpy.abs(y_new))
    return atol + rtol * numpy.maximum(numpy.abs(y), numpy.abs(y_new))


def norm(x):
    """Compute RMS norm."""
    return (numpy.real(x @ x.conjugate()) / x.size) ** 0.5


class RungeKutta(OdeSolver):
    """Base class for explicit runge kutta methods.

    This implementation mainly follows the scipy implementation. The current
    differences are:
      - Conventional (non FSAL) methods are detected and failed steps cost
        one function evaluation less than with the scipy implementation.
      - A different, more elaborate estimate for the size of the first step
        is used.
      - Horner's rule is used for dense output calculation.
      - A failed step counter is added.
      - The stepsize near the end of the integration is different:
        - look ahead to prevent too small step sizes
      - the min_step accounts for the distance between C-values
      - a different tolerance validation is used.
      - stiffness detection is added, can be turned off
      - second order stepsize control is added.
    """

    # effective number of stages
    n_stages: int = NotImplemented

    # order of the main method
    order: int = NotImplemented

    # order of the secondary embedded method
    error_estimator_order: int = NotImplemented

    # runge kutta coefficient matrix
    A: numpy.ndarray = NotImplemented              # shape: [n_stages, n_stages]

    # output coefficients (weights)
    B: numpy.ndarray = NotImplemented              # shape: [n_stages]

    # time fraction coefficients (nodes)
    C: numpy.ndarray = NotImplemented              # shape: [n_stages]

    # error coefficients (weights Bh - B); for non-FSAL methods E[-1] == 0.
    E: numpy.ndarray = NotImplemented              # shape: [n_stages + 1]

    # dense output interpolation coefficients, optional
    P: numpy.ndarray = NotImplemented              # shape: [n_stages + 1,
    #                                                     order_polynomial]

    # Parameters for stiffness detection, optional
    stbrad: float = NotImplemented              # radius of the arc
    tanang: float = NotImplemented              # tan(valid angle < pi/2)

    # Parameters for stepsize control, optional
    sc_params = "standard"                      # tuple, or str

    max_factor = MAX_FACTOR0                    # initially
    min_factor = MIN_FACTOR

    def _init_min_step_parameters(self):
        """Define the parameters h_min_a and h_min_b for the min_step rule:
            min_step = max(h_min_a * abs(t), h_min_b)
        from RKSuite.
        """

        # minimum difference between distinct C-values
        cdiff = 1.
        for c1 in self.C:
            for c2 in self.C:
                diff = abs(c1 - c2)
                if diff:
                    cdiff = min(cdiff, diff)
        if cdiff < 1e-3:
            cdiff = 1e-3
            logging.warning(
                'Some C-values of this Runge Kutta method are nearly the '
                'same but not identical. This limits the minimum stepsize'
                'You may want to check the implementation of this method.')

        # determine min_step parameters
        epsneg = numpy.finfo(self.y.dtype).epsneg
        tiny = numpy.finfo(self.y.dtype).tiny
        h_min_a = 10 * epsneg / cdiff
        h_min_b = sqrt(tiny)
        return h_min_a, h_min_b

    def _init_stiffness_detection(self, nfev_stiff_detect):
        if not (isinstance(nfev_stiff_detect, int) and nfev_stiff_detect >= 0):
            raise ValueError(
                "`nfev_stiff_detect` must be a non-negative integer.")
        self.nfev_stiff_detect = nfev_stiff_detect
        if NotImplemented in (self.stbrad, self.tanang):
            # disable stiffness detection if not implemented
            if nfev_stiff_detect not in (5000, 0):
                warn("This method does not implement stiffness detection. "
                     "Changing the value of nfev_stiff_detect does nothing.")
            self.nfev_stiff_detect = 0
        self.jflstp = 0                         # failed step counter, last 40
        if self.nfev_stiff_detect:
            self.okstp = 0                      # successful step counter
            self.havg = 0.0                     # average stepsize

    def _init_sc_control(self, sc_params):
        coefs = {"G": (0.7, -0.4, 0, 0.9),
                 "S": (0.6, -0.2, 0, 0.9),
                 "W": (2, -1, -1, 0.8),
                 "standard": (1, 0, 0, 0.9)}
        # use default controller of method if not specified otherwise
        sc_params = sc_params or self.sc_params
        if (isinstance(sc_params, str) and sc_params in coefs):
            kb1, kb2, a, g = coefs[sc_params]
        elif isinstance(sc_params, tuple) and len(sc_params) == 4:
            kb1, kb2, a, g = sc_params
        else:
            raise ValueError('sc_params should be a tuple of length 3 or one '
                             'of the strings "G", "S", "W" or "standard"')
        # set all parameters
        self.minbeta1 = kb1 * self.error_exponent
        self.minbeta2 = kb2 * self.error_exponent
        self.minalpha = -a
        self.safety = g
        self.safety_sc = g ** (kb1 + kb2)
        self.standard_sc = True                                # for first step

    def __init__(self, fun, t0, y0, t_bound, max_step=numpy.inf, rtol=1e-3,
                 atol=1e-6, vectorized=False, first_step=None,
                 nfev_stiff_detect=5000, sc_params=None, **extraneous):
        warn_extraneous(extraneous)
        super(RungeKutta, self).__init__(fun, t0, y0, t_bound, vectorized,
                                         support_complex=True)
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.y)
        self.f = self.fun(self.t, self.y)
        if self.f.dtype != self.y.dtype:
            raise TypeError('dtypes of solution and derivative do not match')
        self.error_exponent = -1 / (self.error_estimator_order + 1)
        self._init_stiffness_detection(nfev_stiff_detect)
        self.h_min_a, self.h_min_b = self._init_min_step_parameters()
        self.tiny_err = self.h_min_b
        self._init_sc_control(sc_params)

        # size of first step:
        if first_step is None:
            b = self.t + self.direction * min(
                abs(self.t_bound - self.t), self.max_step)
            self.h_abs = abs(select_initial_step(
                self.fun, self.t, b, self.y, self.f,
                self.error_estimator_order, self.rtol, self.atol))
        else:
            self.h_abs = validate_first_step(first_step, t0, t_bound)

        self.K = numpy.empty((self.n_stages + 1, self.n), self.y.dtype)
        self.FSAL = 1 if self.E[self.n_stages] else 0
        self.h_previous = None
        self.y_old = None
        NFS[()] = 0                                # global failed step counter

    def _step_impl(self):
        # mostly follows the scipy implementation of scipy's RungeKutta
        t = self.t
        y = self.y

        h_abs, min_step = self._reassess_stepsize(t, y)

        # loop until the step is accepted
        step_accepted = False
        step_rejected = False
        while not step_accepted:

            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP
            h = h_abs * self.direction
            t_new = t + h

            # calculate stages needed for output
            self.K[0] = self.f
            for i in range(1, self.n_stages):
                self._rk_stage(h, i)

            # calculate error norm and solution
            y_new, error_norm = self._comp_sol_err(y, h)

            # evaluate error
            if error_norm < 1:
                step_accepted = True

                if error_norm < self.tiny_err:
                    factor = self.max_factor
                    self.standard_sc = True

                elif self.standard_sc:
                    factor = self.safety * error_norm ** self.error_exponent
                    self.standard_sc = False

                else:
                    # use second order SC controller
                    h_ratio = h / self.h_previous
                    factor = self.safety_sc * (
                        error_norm ** self.minbeta1 *
                        self.error_norm_old ** self.minbeta2 *
                        h_ratio ** self.minalpha)
                    factor = min(self.max_factor, max(self.min_factor, factor))

                if step_rejected:
                    factor = min(1, factor)

                h_abs *= factor

                if factor < MAX_FACTOR:
                    # reduce max_factor when on scale.
                    self.max_factor = MAX_FACTOR

            else:
                step_rejected = True
                h_abs *= max(self.min_factor,
                             self.safety * error_norm ** self.error_exponent)

                NFS[()] += 1
                self.jflstp += 1                      # for stiffness detection

                if numpy.isnan(error_norm) or numpy.isinf(error_norm):
                    return False, "Overflow or underflow encountered."

        if not self.FSAL:
            # evaluate ouput point for interpolation and next step
            self.K[self.n_stages] = self.fun(t + h, y_new)

        # store for next step, interpolation and stepsize control
        self.h_previous = h
        self.y_old = y
        self.h_abs = h_abs
        self.f_old = self.f
        self.f = self.K[self.n_stages].copy()
        self.error_norm_old = error_norm

        # output
        self.t = t_new
        self.y = y_new

        return True, None

    def _reassess_stepsize(self, t, y):
        # limit step size
        h_abs = self.h_abs
        min_step = max(self.h_min_a * (abs(t) + h_abs), self.h_min_b)
        if h_abs < min_step or h_abs > self.max_step:
            h_abs = min(self.max_step, max(min_step, h_abs))
            self.standard_sc = True

        # handle final integration steps
        d = abs(self.t_bound - t)                     # remaining interval
        if d < 2 * h_abs:
            if d > h_abs:
                # h_abs < d < 2 * h_abs: "look ahead".
                # split d over last two steps. This reduces the chance of a
                # very small last step.
                h_abs = max(0.5 * d, min_step)
                self.standard_sc = True
            else:
                # d <= h_abs: Don't step over t_bound
                h_abs = d

        return h_abs, min_step

    def _estimate_error(self, K, h):
        # exclude K[-1] if not FSAL. It could contain nan or inf
        return h * (K[:self.n_stages + self.FSAL].T @
                    self.E[:self.n_stages + self.FSAL])

    def _estimate_error_norm(self, K, h, scale):
        return norm(self._estimate_error(K, h) / scale)

    def _comp_sol_err(self, y, h):
        """Compute solution and error.
        The calculation of `scale` differs from scipy: The average instead of
        the maximum of abs(y) of the current and previous steps is used.
        """
        y_new = y + h * (self.K[:self.n_stages].T @ self.B)
        scale = calculate_scale(self.atol, self.rtol, y, y_new)

        if self.FSAL:
            # do FSAL evaluation if needed for error estimate
            self.K[self.n_stages, :] = self.fun(self.t + h, y_new)

        error_norm = self._estimate_error_norm(self.K, h, scale)
        return y_new, error_norm

    def _rk_stage(self, h, i):
        """compute a single RK stage"""
        dy = h * (self.K[:i, :].T @ self.A[i, :i])
        self.K[i] = self.fun(self.t + self.C[i] * h, self.y + dy)

    def _dense_output_impl(self):
        """return denseOutput, detect if step was extrapolated linearly"""

        if isinstance(self.P, numpy.ndarray):
            # normal output
            Q = self.K.T @ self.P
            return HornerDenseOutput(self.t_old, self.t, self.y_old, Q)

        # if no interpolant is implemented
        return CubicDenseOutput(self.t_old, self.t, self.y_old, self.y,
                                self.f_old, self.f)

def select_initial_step(df, a, b, y, yprime, morder, rtol, atol):
    """ select_initial_step computes a starting step size to be used in solving initial
    value problems in ordinary differential equations.

    This method is developed by H.A. Watts and described in [1]_. This function
    is a Python translation of the Fortran source code [2]_. The two main
    modifications are:
        using the RMS norm from scipy.integrate
        allowing for complex valued input

    Parameters
    ----------
    df : callable
        Right-hand side of the system. The calling signature is fun(t, y).
        Here t is a scalar. The ndarray y has has shape (n,) and fun must
        return array_like with the same shape (n,).
    a : float
        This is the initial point of integration.
    b : float
        This is a value of the independent variable used to define the
        direction of integration. A reasonable choice is to set `b` to the
        first point at which a solution is desired. You can also use `b , if
        necessary, to restrict the length of the first integration step because
        the algorithm will not compute a starting step length which is bigger
        than abs(b-a), unless `b` has been chosen too close to `a`. (it is
        presumed that select_initial_step has been called with `b` different from `a` on
        the machine being used.
    y : array_like, shape (n,)
        This is the vector of initial values of the n solution components at
        the initial point `a`.
    yprime : array_like, shape (n,)
        This is the vector of derivatives of the n solution components at the
        initial point `a`.  (defined by the differential equations in
        subroutine `df`)
    morder : int
        This is the order of the formula which will be used by the initial
        value method for taking the first integration step.
    rtol : float
        Relative tolereance used by the differential equation method.
    atol : float or array_like
        Absolute tolereance used by the differential equation method.

    Returns
    -------
    float
        An appropriate starting step size to be attempted by the differential
        equation method.

    References
    ----------
    .. [1] H.A. Watts, "Starting step size for an ODE solver", Journal of
           Computational and Applied Mathematics, Vol. 9, No. 2, 1983,
           pp. 177-191, ISSN 0377-0427.
           https://doi.org/10.1016/0377-0427(83)90040-7
    .. [2] Slatec Fortran code dstrt.f.
           https://www.netlib.org/slatec/src/
    """

    # needed to pass scipy unit test:
    if y.size == 0:
        return numpy.inf

    # compensate for modified call list
    neq = y.size
    spy = numpy.empty_like(y)
    pv = numpy.empty_like(y)
    etol = atol + rtol * numpy.abs(y)

    # `small` is a small positive machine dependent constant which is used for
    # protecting against computations with numbers which are too small relative
    # to the precision of floating point arithmetic. `small` should be set to
    # (approximately) the smallest positive DOUBLE PRECISION number such that
    # (1. + small) > 1.  on the machine being used. The quantity small**(3/8)
    # is used in computing increments of variables for approximating
    # derivatives by differences.  Also the algorithm will not compute a
    # starting step length which is smaller than 100*small*ABS(A).
    # `big` is a large positive machine dependent constant which is used for
    # preventing machine overflows. A reasonable choice is to set big to
    # (approximately) the square root of the largest DOUBLE PRECISION number
    # which can be held in the machine.
    big = sqrt(numpy.finfo(y.dtype).max)
    small = numpy.nextafter(numpy.finfo(y.dtype).epsneg, 1.0)

    # following dhstrt.f from here
    dx = b - a
    absdx = abs(dx)
    relper = small**0.375

    # compute an approximate bound (dfdxb) on the partial derivative of the
    # equation with respect to the independent variable.  protect against an
    # overflow.  also compute a bound (fbnd) on the first derivative locally.
    da = copysign(max(min(relper * abs(a), absdx), 100.0 * small * abs(a)), dx)
    da = da or relper * dx
    sf = df(a + da, y)                                               # evaluate
    yp = sf - yprime
    delf = norm(yp)
    dfdxb = big
    if delf < big * abs(da):
        dfdxb = delf / abs(da)
    fbnd = norm(sf)

    # compute an estimate (dfdub) of the local lipschitz constant for the
    # system of differential equations. this also represents an estimate of the
    # norm of the jacobian locally.  three iterations (two when neq=1) are used
    # to estimate the lipschitz constant by numerical differences.  the first
    # perturbation vector is based on the initial derivatives and direction of
    # integration.  the second perturbation vector is formed using another
    # evaluation of the differential equation.  the third perturbation vector
    # is formed using perturbations based only on the initial values.
    # components that are zero are always changed to non-zero values (except
    # on the first iteration).  when information is available, care is taken to
    # ensure that components of the perturbation vector have signs which are
    # consistent with the slopes of local solution curves.  also choose the
    # largest bound (fbnd) for the first derivative.

    # perturbation vector size is held constant for all iterations.  compute
    # this change from the size of the vector of initial values.
    dely = relper * norm(y)
    dely = dely or relper
    dely = copysign(dely, dx)
    delf = norm(yprime)
    fbnd = max(fbnd, delf)

    if delf:
        # use initial derivatives for first perturbation
        spy[:] = yprime
        yp[:] = yprime
    else:
        # cannot have a null perturbation vector
        spy[:] = 0.0
        yp[:] = 1.0
        delf = norm(yp)

    dfdub = 0.0
    lk = min(neq + 1, 3)
    for k in range(1, lk + 1):

        # define perturbed vector of initial values
        pv[:] = y + dely / delf * yp

        if k == 2:
            # use a shifted value of the independent variable in computing
            # one estimate
            yp[:] = df(a + da, pv)                                   # evaluate
            pv[:] = yp - sf

        else:
            # evaluate derivatives associated with perturbed vector and
            # compute corresponding differences
            yp[:] = df(a, pv)                                        # evaluate
            pv[:] = yp - yprime

        # choose largest bounds on the first derivative and a local lipschitz
        # constant
        fbnd = max(fbnd, norm(yp))
        delf = norm(pv)
        if delf >= big * abs(dely):
            # protect against an overflow
            dfdub = big
            break
        dfdub = max(dfdub, delf / abs(dely))

        if k == lk:
            break

        # choose next perturbation vector
        delf = delf or 1.0
        if k == 2:
            dy = y.copy()                                                 # vec
            dy[:] = numpy.where(dy, dy, dely / relper)
        else:
            dy = pv.copy()                              # abs removed (complex)
            dy[:] = numpy.where(dy, dy, delf)
        spy[:] = numpy.where(spy, spy, yp)

        # use correct direction if possible.
        yp[:] = numpy.where(spy, numpy.copysign(dy.real, spy.real), dy.real)
        if numpy.issubdtype(y.dtype, numpy.complexfloating):
            yp[:] += 1j*numpy.where(spy, numpy.copysign(dy.imag, spy.imag), dy.imag)
        delf = norm(yp)

    # compute a bound (ydpb) on the norm of the second derivative
    ydpb = dfdxb + dfdub * fbnd

    # define the tolerance parameter upon which the starting step size is to be
    # based.  a value in the middle of the error tolerance range is selected.
    tolexp = numpy.log10(etol)
    tolsum = tolexp.sum()
    tolmin = min(tolexp.min(), big)
    tolp = 10.0 ** (0.5 * (tolsum / neq + tolmin) / (morder + 1))

    # compute a starting step size based on the above first and second
    # derivative information

    # restrict the step length to be not bigger than abs(b-a).
    # (unless b is too close to a)
    h = absdx
    if ydpb == 0.0 and fbnd == 0.0:
        # both first derivative term (fbnd) and second derivative term (ydpb)
        # are zero
        if tolp < 1.0:
            h = absdx * tolp
    elif ydpb == 0.0:
        #  only second derivative term (ydpb) is zero
        if tolp < fbnd * absdx:
            h = tolp / fbnd
    else:
        # second derivative term (ydpb) is non-zero
        srydpb = sqrt(0.5 * ydpb)
        if tolp < srydpb * absdx:
            h = tolp / srydpb

    # further restrict the step length to be not bigger than  1/dfdub
    if dfdub:                                              # `if` added (div 0)
        h = min(h, 1.0 / dfdub)

    # finally, restrict the step length to be not smaller than
    # 100*small*abs(a).  however, if a=0. and the computed h underflowed to
    # zero, the algorithm returns small*abs(b) for the step length.
    h = max(h, 100.0 * small * abs(a))
    h = h or small * abs(b)

    # now set direction of integration
    h = copysign(h, dx)
    return h


class HornerDenseOutput(DenseOutput):
    """use Horner's rule for the evaluation of the dense output polynomials.
    """
    def __init__(self, t_old, t, y_old, Q):
        super(HornerDenseOutput, self).__init__(t_old, t)
        self.h = t - t_old
        self.Q = Q * self.h
        self.y_old = y_old

    def _call_impl(self, t):

        # scaled time
        x = (t - self.t_old) / self.h

        # Horner's rule:
        y = self.Q.T[-1, :, numpy.newaxis] * x
        for q in reversed(self.Q.T[:-1]):
            y += q[:, numpy.newaxis]
            y *= x
        y += self.y_old[:, numpy.newaxis]

        if t.shape:
            return y
        else:
            return y[:, 0]


class CubicDenseOutput(DenseOutput):
    """Cubic, C1 continuous interpolator
    """
    def __init__(self, t_old, t, y_old, y, f_old, f):
        super(CubicDenseOutput, self).__init__(t_old, t)
        self.h = t - t_old
        self.y_old = y_old
        self.f_old = f_old
        self.y = y
        self.f = f

    def _call_impl(self, t):
        # scaled time
        x = (t - self.t_old) / self.h

        # qubic hermite spline:
        h00 = (1.0 + 2.0*x) * (1.0 - x)**2
        h10 = x * (1.0 - x)**2 * self.h
        h01 = x**2 * (3.0 - 2.0*x)
        h11 = x**2 * (x - 1.0) * self.h

        # output
        y = (h00 * self.y_old[:, numpy.newaxis] + h10 * self.f_old[:, numpy.newaxis]
             + h01 * self.y[:, numpy.newaxis] + h11 * self.f[:, numpy.newaxis])

        if t.shape:
            return y
        else:
            return y[:, 0]
