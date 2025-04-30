"""
Explicit Runge-Kutta integrators
"""

import math
import logging
from typing import Callable, Optional, Tuple, Union, Literal
import numpy
from solvers.global_operations import global_inf_norm
from mpi4py import MPI

# Constants
BIG_FACTOR = 4.0


def limiter(u: float, kappa: float) -> float:
    """
    Step-size ratio limiter.

    Applies an arctangent limiter parametrized by KAPPA to reduce
    large changes in step size.

    Args:
        u: The unrestricted step-size ratio
        kappa: The limiter parameter (larger values allow larger changes)

    Returns:
        The limited step-size ratio
    """
    return 1 + kappa * math.atan((u - 1) / kappa)


class RungeKutta:
    """
    Base class for explicit Runge-Kutta methods for solving ODEs.

    This class implements the core functionality of explicit Runge-Kutta methods with
    adaptive step size control. Specific methods should subclass this and define
    the Butcher tableau (A, B, C) and error estimation coefficients (E).

    Attributes:
        n_stages (int): Number of stages in the Runge-Kutta method
        order (int): Order of accuracy of the main method
        error_estimator_order (int): Order of accuracy of the embedded method
        A (numpy.ndarray): Coefficient matrix with shape [n_stages, n_stages], lower triangular
        B (numpy.ndarray): Output coefficients (weights) with shape [n_stages]
        C (numpy.ndarray): Time fraction coefficients (nodes) with shape [n_stages]
        E (numpy.ndarray): Error coefficients with shape [n_stages + 1]
        controller (str): Default controller for step size adaptation
    """

    # Subclasses must override these attributes
    n_stages: int = NotImplemented
    order: int = NotImplemented
    error_estimator_order: int = NotImplemented
    A: numpy.ndarray = NotImplemented  # shape: [n_stages, n_stages], lower triangular
    B: numpy.ndarray = NotImplemented  # shape: [n_stages]
    C: numpy.ndarray = NotImplemented  # shape: [n_stages]
    E: numpy.ndarray = NotImplemented  # shape: [n_stages + 1]
    controller: Union[str, Tuple[float, float, float, float]] = "deadbeat"

    def __init__(
        self,
        fun: Callable[[float, numpy.ndarray], numpy.ndarray],
        t0: float,
        y0: numpy.ndarray,
        t_bound: float,
        max_step: float = numpy.inf,
        rtol: float = 1e-3,
        atol: float = 1e-6,
        first_step: Optional[float] = None,
        controller: Union[str, Tuple[float, float, float, float], None] = None,
    ):
        """
        Initialize the Runge-Kutta solver.

        Args:
            fun: Function that computes the right-hand side of the ODE system dy/dt = f(t, y)
            t0: Initial time
            y0: Initial state vector
            t_bound: Final time to integrate to
            max_step: Maximum allowed step size
            rtol: Relative tolerance for error control
            atol: Absolute tolerance for error control
            first_step: Initial step size (must be provided)
            controller: Step size controller type or parameters
                        Choices: "deadbeat", "PI3040", "PI4020", "H211PI", "H110", "H211D", "H211B",
                        or a tuple (kb1, kb2, a, g) specifying the controller parameters

        Raises:
            TypeError: If dtypes of solution and derivative do not match
            RuntimeError: If first_step is not defined
            ValueError: If first_step is negative or exceeds bounds, or if controller is invalid
            ValueError: If Butcher tableau coefficients are invalid or inconsistent
        """
        # Validate Butcher tableau coefficients before proceeding
        self._validate_butcher_tableau()

        self.t_old = None
        self.t = t0
        self._fun, self.y = fun, y0
        self.t_bound = t_bound

        def self_fun(t: float, y: numpy.ndarray) -> numpy.ndarray:
            """Wrapper around the ODE function that counts evaluations."""
            self.nfev += 1
            return self._fun(t, y)

        self.fun = self_fun

        self.n = self.y.size
        self.status = "running"

        self.nfev = 0
        self.njev = 0
        self.nlu = 0

        self.max_step = max_step
        self.rtol, self.atol = rtol, atol
        self.f = self.fun(self.t, self.y)

        if self.f.dtype != self.y.dtype:
            raise TypeError("dtypes of solution and derivative do not match")

        self.error_exponent = -1 / (self.error_estimator_order + 1)
        self.error_norm_old = None
        self.h_min_a, self.h_min_b = self._init_min_step_parameters()
        self.tiny_err = self.h_min_b
        self.error_estimation = 0

        self._init_control(controller)

        # Validate and set first step
        if first_step is None:
            raise RuntimeError("`first_step` must be provided")

        if first_step <= 0:
            raise ValueError("`first_step` must be positive")

        if first_step > abs(t_bound - t0):
            raise ValueError("`first_step` exceeds integration bounds")

        self.h = first_step

        # Initialize stage vectors
        self.K = numpy.empty((self.n_stages + 1, self.n), self.y.dtype)
        self.FSAL = 1 if self.E[self.n_stages] else 0
        self.h_previous = None
        self.y_old = None
        self.failed_steps = 0  # failed step counter
        self.num_of_steps = 0  # keep track of the total number of steps

    def _validate_butcher_tableau(self) -> None:
        """
        Validate the Butcher tableau coefficients for consistency and correctness.

        This method checks:
        1. All required coefficients are provided (not NotImplemented)
        2. Dimensions of A, B, C, E are consistent with n_stages
        3. A is strictly lower triangular (explicit method)
        4. C values are between 0 and 1 (standard for time fractions)
        5. B has appropriate values (sums to 1 for consistency)
        6. E has correct length (n_stages + 1)

        Raises:
            ValueError: If any of the Butcher tableau coefficients are invalid
        """
        # Check if all required attributes are implemented
        for attr in ["n_stages", "order", "error_estimator_order", "A", "B", "C", "E"]:
            if getattr(self, attr) is NotImplemented:
                raise ValueError(f"Required attribute '{attr}' is not implemented in this Runge-Kutta method")

        # Validate n_stages
        if not isinstance(self.n_stages, int) or self.n_stages <= 0:
            raise ValueError(f"n_stages must be a positive integer, got {self.n_stages}")

        # Check dimensions of A, B, C, E
        if not isinstance(self.A, numpy.ndarray) or self.A.shape != (self.n_stages, self.n_stages):
            raise ValueError(
                f"A must be a numpy.ndarray with shape ({self.n_stages}, {self.n_stages}), "
                f"got {type(self.A)} with shape {getattr(self.A, 'shape', None)}"
            )

        if not isinstance(self.B, numpy.ndarray) or self.B.shape != (self.n_stages,):
            raise ValueError(
                f"B must be a numpy.ndarray with shape ({self.n_stages},), "
                f"got {type(self.B)} with shape {getattr(self.B, 'shape', None)}"
            )

        if not isinstance(self.C, numpy.ndarray) or self.C.shape != (self.n_stages,):
            raise ValueError(
                f"C must be a numpy.ndarray with shape ({self.n_stages},), "
                f"got {type(self.C)} with shape {getattr(self.C, 'shape', None)}"
            )

        if not isinstance(self.E, numpy.ndarray) or self.E.shape != (self.n_stages + 1,):
            raise ValueError(
                f"E must be a numpy.ndarray with shape ({self.n_stages + 1},), "
                f"got {type(self.E)} with shape {getattr(self.E, 'shape', None)}"
            )

        # Validate that A is strictly lower triangular (explicit method)
        for i in range(self.n_stages):
            for j in range(i, self.n_stages):
                if abs(self.A[i, j]) > numpy.finfo(float).eps:
                    raise ValueError(
                        f"A must be strictly lower triangular for explicit methods, "
                        f"but A[{i},{j}] = {self.A[i,j]} != 0"
                    )

        # Validate that C values are in [0, 1] (standard for time fractions)
        if numpy.any((self.C < 0) | (self.C > 1)):
            invalid_indices = numpy.where((self.C < 0) | (self.C > 1))[0]
            raise ValueError(f"C values must be in [0, 1], but C[{invalid_indices}] = {self.C[invalid_indices]}")

        # Validate that B sums approximately to 1 (consistency condition)
        b_sum = numpy.sum(self.B)
        if abs(b_sum - 1.0) > 1e-10:
            logging.warning(
                f"Sum of B coefficients should be 1.0 for consistency, but got {b_sum}. "
                "This may affect conservation properties."
            )

        # Validate that C values match row sums of A (internal consistency condition)
        for i in range(self.n_stages):
            row_sum = numpy.sum(self.A[i, :])
            if i > 0 and abs(row_sum - self.C[i]) > 1e-10:
                logging.warning(
                    f"For stage {i}, sum of A coefficients ({row_sum}) should match C[{i}] ({self.C[i]}). "
                    "This may indicate an inconsistent Butcher tableau."
                )

        # Validate that order and error_estimator_order are reasonable
        if not isinstance(self.order, int) or self.order <= 0:
            raise ValueError(f"order must be a positive integer, got {self.order}")

        if not isinstance(self.error_estimator_order, int) or self.error_estimator_order <= 0:
            raise ValueError(f"error_estimator_order must be a positive integer, got {self.error_estimator_order}")

        if self.error_estimator_order >= self.order:
            logging.warning(
                f"error_estimator_order ({self.error_estimator_order}) should typically be "
                f"less than order ({self.order}) of the main method."
            )

    def _init_control(self, controller: Union[str, Tuple[float, float, float, float], None]) -> None:
        """
        Initialize step size controller parameters.

        Args:
            controller: Controller type (string) or parameters (tuple)
                        If None, use the class default controller

        Raises:
            ValueError: If the controller type is invalid
        """
        coefs = {
            "DEADBEAT": (1, 0, 0, 0.9),  # elementary controller (I)
            "PI3040": (0.7, -0.4, 0, 0.8),  # PI controller (Gustafsson)
            "PI4020": (0.6, -0.2, 0, 0.8),  # PI controller for nonstiff methods
            "H211PI": (1 / 6, 1 / 6, 0, 0.8),  # LP filter of PI structure
            "H110": (1 / 3, 0, 0, 0.8),  # I controller (convolution filter)
            "H211D": (1 / 2, 1 / 2, 1 / 2, 0.8),  # LP filter with gain = 1/2
            "H211B": (1 / 4, 1 / 4, 1 / 4, 0.8),  # general purpose LP filter
        }

        if self.controller == NotImplemented:
            # use standard controller if not specified otherwise
            controller = controller or "DEADBEAT"
        else:
            # use default controller of method if not specified otherwise
            controller_str = (
                controller.upper()
                if isinstance(controller, str)
                else (self.controller.upper() if controller is None else None)
            )
            controller = controller_str or self.controller.upper()

        if isinstance(controller, str) and controller in coefs:
            kb1, kb2, a, g = coefs[controller]
        elif isinstance(controller, tuple) and len(controller) == 4:
            kb1, kb2, a, g = controller
        else:
            raise ValueError(f"Invalid controller: {controller}. Must be one of {list(coefs.keys())} or a 4-tuple")

        # set all parameters
        self.minbeta1 = kb1 * self.error_exponent
        self.minbeta2 = kb2 * self.error_exponent
        self.minalpha = -a
        self.safety = g
        self.safety_sc = g ** (kb1 + kb2)
        self.standard_sc = True  # for first step

    def step(self) -> None:
        """
        Advance the integration by one step.

        This method advances the integration from the current time t to t + h,
        where h is the current step size. The step size is adapted based on error estimates.

        Raises:
            RuntimeError: If attempting to step on a failed or finished solver
        """
        if self.status != "running":
            raise RuntimeError("Attempt to step on a failed or finished solver")

        if self.n == 0 or self.t == self.t_bound:
            # Handle corner cases of empty solver or no integration
            self.t_old = self.t
            self.t = self.t_bound
            self.status = "finished"
        else:
            t = self.t
            success = self._step_impl()

            if not success:
                self.status = "failed"
                logging.error(f"Integration failed at t={self.t}")
            else:
                self.t_old = t
                if self.t - self.t_bound >= 0:
                    self.status = "finished"

    def _step_impl(self) -> bool:
        """
        Implementation of the adaptive Runge-Kutta step.

        This internal method implements the actual RK step computation with
        error estimation and step size adaptation.

        Returns:
            bool: True if the step was successful, False otherwise
        """
        t = self.t
        y = self.y

        h, min_step = self._reassess_stepsize(t)

        # loop until the step is accepted
        step_accepted = False
        step_rejected = False
        # store for next step, interpolation and stepsize control
        self.h_previous = h

        while not step_accepted:
            if h < min_step:
                logging.error(f"Step size {h} below minimum {min_step} at t={t}")
                return False

            t_new = t + h

            # calculate RK stages
            self.K[0] = self.f
            for i in range(1, self.n_stages):
                dy = h * (self.K[:i, :].T @ self.A[i, :i])
                self.K[i] = self.fun(self.t + self.C[i] * h, self.y + dy)

            # Update solution
            y_new = y + h * (self.K[: self.n_stages].T @ self.B)

            # calculate error norm
            if self.FSAL:
                # do FSAL evaluation if needed for error estimate
                self.K[self.n_stages, :] = self.fun(self.t + h, y_new)

            scale = self.atol + numpy.maximum(numpy.abs(y), numpy.abs(y_new)) * self.rtol

            # exclude K[-1] if not FSAL. It could contain nan or inf
            err_estimate = h * (self.K[: self.n_stages + self.FSAL].T @ self.E[: self.n_stages + self.FSAL])
            error_norm = global_inf_norm(err_estimate / scale)

            # evaluate error
            if error_norm < 1:
                step_accepted = True
                # Debug logging if needed
                # logging.debug(f"Step {self.num_of_steps} accepted: t={t_new}, h={h}, error={error_norm}")

                if error_norm < self.tiny_err:
                    factor = BIG_FACTOR
                    self.standard_sc = True

                elif self.standard_sc:
                    factor = self.safety * error_norm**self.error_exponent
                    self.standard_sc = False

                else:
                    # use second order SC controller
                    h_ratio = h / self.h_previous

                    factor = self.safety_sc * (
                        error_norm**self.minbeta1 * self.error_norm_old**self.minbeta2 * h_ratio**self.minalpha
                    )

                if step_rejected:
                    factor = min(1, factor)

                h *= limiter(factor, 2)
                # keep track of the number of steps
                self.num_of_steps += 1

            else:
                step_rejected = True
                # Debug logging if needed
                # logging.debug(f"Step {self.num_of_steps} rejected: t={t_new}, h={h}, error={error_norm}")

                h *= limiter(self.safety * error_norm**self.error_exponent, 2)

                if h < 1e-12:
                    logging.error(f"Unable to achieve desired tolerance at t={t}. Step size too small: {h}")
                    return False

                self.failed_steps += 1
                # keep track of the number of steps
                self.num_of_steps += 1

                if numpy.isnan(error_norm) or numpy.isinf(error_norm):
                    logging.error(f"Overflow or underflow encountered at t={t}")
                    return False

        if not self.FSAL:
            # evaluate output point for the next step
            self.K[self.n_stages] = self.fun(t + h, y_new)

        # store for next step, interpolation and stepsize control
        self.y_old = y
        self.h = h
        self.f = self.K[self.n_stages].copy()
        self.error_norm_old = error_norm
        self.error_estimation = global_inf_norm(err_estimate)

        # output
        self.t = t_new
        self.y = y_new

        return True

    def _init_min_step_parameters(self) -> Tuple[float, float]:
        """
        Define parameters for the minimum step size.

        This method calculates the h_min_a and h_min_b parameters for the min_step rule:
            min_step = max(h_min_a * abs(t), h_min_b)
        based on method properties and machine precision.

        Returns:
            Tuple[float, float]: (h_min_a, h_min_b) parameters
        """
        # minimum difference between distinct C-values
        cdiff = 1.0
        for c1 in self.C:
            for c2 in self.C:
                diff = abs(c1 - c2)
                if diff:
                    cdiff = min(cdiff, diff)

        if cdiff < 1e-3:
            cdiff = 1e-3
            logging.warning(
                "Some C-values of this Runge Kutta method are nearly the "
                "same but not identical. This limits the minimum stepsize. "
                "You may want to check the implementation of this method."
            )

        # determine min_step parameters
        epsneg = numpy.finfo(self.y.dtype).epsneg
        tiny = numpy.finfo(self.y.dtype).tiny
        h_min_a = 10 * epsneg / cdiff
        h_min_b = math.sqrt(tiny)

        return h_min_a, h_min_b

    def _reassess_stepsize(self, t: float) -> Tuple[float, float]:
        """
        Reassess the step size based on constraints and integration bounds.

        This method ensures the step size is within acceptable bounds and handles
        the final steps of integration to avoid overshooting the integration bound.

        Args:
            t: Current time

        Returns:
            Tuple[float, float]: (adjusted step size, minimum step size)
        """
        # Calculate minimum step size
        h = self.h
        min_step = max(self.h_min_a * (abs(t) + h), self.h_min_b)

        # Limit step size
        if h < min_step or h > self.max_step:
            h = min(self.max_step, max(min_step, h))
            self.standard_sc = True

        # Handle final integration steps
        d = abs(self.t_bound - t)  # remaining interval
        if d < 2 * h:
            if d > h:
                # h < d < 2 * h: "look ahead".
                # split d over last two steps. This reduces the chance of a
                # very small last step.
                h = max(0.5 * d, min_step)
                self.standard_sc = True
            else:
                # d <= h: Don't step over t_bound
                h = d

        return h, min_step
