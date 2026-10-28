"""Adaptive explicit Runge-Kutta methods."""

import logging
import math
from collections.abc import Callable
from functools import cache
from time import perf_counter

import numpy
import torch
from mpi4py import MPI

from ..context import Context
from ..solvers.global_operations import global_inf_norm

BIG_FACTOR = 4.0


def limiter(u: float, kappa: float) -> float:
    """Smoothly limit the proposed step-size ratio."""
    return 1 + kappa * math.atan((u - 1) / kappa)


_logger = logging.getLogger(__name__)


def _stage_state(state, combination, step):
    return state + step * combination


def _error_vectors(state, next_state, combination, step, rtol, atol):
    error = step * combination
    scale = atol + torch.maximum(state.abs(), next_state.abs()) * rtol
    # Square in float64 so the norm neither overflows nor underflows in single precision.
    normalized = (error / scale).double()
    return error, torch.sum(normalized * normalized)


def global_rms_norm(local_sum_of_squares, local_size: int, comm: MPI.Comm) -> float:
    """Root-mean-square norm from the local sums of squares of all PEs."""
    totals = numpy.array([local_sum_of_squares.item(), local_size], dtype=numpy.float64)
    comm.Allreduce(MPI.IN_PLACE, totals, op=MPI.SUM)
    return math.sqrt(totals[0] / totals[1])


def _compile_vector_graph(graph, inputs, *, options):
    announce = MPI.COMM_WORLD.rank == 0
    if announce:
        print("Compiling Runge-Kutta vector operations...", end=" ", flush=True)
    start = perf_counter()
    compiled = torch._inductor.compile(graph, inputs, options=options)
    if announce:
        print(f"done. ({perf_counter() - start:.3f} s)", flush=True)
    return compiled


@cache
def _compiled_vector_operations():
    options = {
        "compile_threads": 1,
        # Preserve the rounding of the eager pointwise operations.
        "emulate_precision_casts": True,
        "eager_numerics.division_rounding": True,
    }
    return tuple(
        torch.compile(operation, backend=_compile_vector_graph, dynamic=False, options=options)
        for operation in (_stage_state, _error_vectors)
    )


class RungeKutta:
    """Embedded explicit Runge-Kutta integration with distributed error control.

    Subclasses supply the NumPy tableau A, B, C and the embedded-minus-main
    error weights E, including an optional endpoint derivative weight.
    Controllers are selected by name or by (kb1, kb2, a, safety) coefficients.
    """

    n_stages: int = NotImplemented
    order: int = NotImplemented
    error_estimator_order: int = NotImplemented
    A: numpy.ndarray = NotImplemented
    B: numpy.ndarray = NotImplemented
    C: numpy.ndarray = NotImplemented
    E: numpy.ndarray = NotImplemented
    controller: str | tuple[float, float, float, float] = "deadbeat"

    def __init__(
        self,
        fun: Callable[[float, torch.Tensor], torch.Tensor],
        t0: float,
        y0: torch.Tensor,
        t_bound: float,
        max_step: float = numpy.inf,
        rtol: float = 1e-3,
        atol: float = 1e-6,
        first_step: float | None = None,
        controller: str | tuple[float, float, float, float] | None = None,
        context: Context | None = None,
    ):
        """Use scalar tolerances and a prescribed first step on a forward time interval."""
        self._validate_butcher_tableau()

        self.comm = context.comm if context is not None else MPI.COMM_WORLD
        self.t = t0
        self._fun, self.y = fun, y0
        self.t_bound = t_bound
        self._A = torch.as_tensor(self.A, dtype=y0.dtype, device=y0.device)
        self._B = torch.as_tensor(self.B, dtype=y0.dtype, device=y0.device)
        self._E = torch.as_tensor(self.E, dtype=y0.dtype, device=y0.device)

        self.n = self.y.numel()
        self.status = "running"

        self.nfev = 0

        if not math.isfinite(t0) or not math.isfinite(t_bound) or t_bound < t0:
            raise ValueError("Runge-Kutta requires a finite forward time interval")
        if math.isnan(max_step) or max_step <= 0:
            raise ValueError("max_step must be positive")
        if not math.isfinite(rtol) or rtol < 0 or not math.isfinite(atol) or atol <= 0:
            raise ValueError("rtol must be finite and nonnegative; atol must be finite and positive")
        self.max_step = max_step
        self.rtol, self.atol = rtol, atol
        self.f = self.fun(self.t, self.y)

        if self.f.dtype != self.y.dtype:
            raise TypeError("dtypes of solution and derivative do not match")

        self.error_exponent = -1 / (self.error_estimator_order + 1)
        self.error_norm_old = None
        self.h_min_a, self.h_min_b = self._init_min_step_parameters()
        self.tiny_err = self.h_min_b
        self._error_estimate = None

        self._init_control(controller)

        if first_step is None:
            raise RuntimeError("`first_step` must be provided")

        if not math.isfinite(first_step) or first_step <= 0:
            raise ValueError("`first_step` must be positive")

        if first_step > abs(t_bound - t0):
            raise ValueError("`first_step` exceeds integration bounds")

        self.h = first_step

        self.K = y0.new_empty((self.n_stages + 1, self.n))
        self._step_tensor = y0.new_empty(()) if y0.is_cuda else None
        self._stage_state, self._error_vectors = (
            _compiled_vector_operations() if y0.is_cuda else (_stage_state, _error_vectors)
        )
        self.endpoint_error_stage = 1 if self.E[self.n_stages] else 0
        self.h_previous = None
        self.failed_steps = 0

    def fun(self, t: float, y):
        self.nfev += 1
        return self._fun(t, y)

    def _validate_butcher_tableau(self) -> None:
        for attr in ["n_stages", "order", "error_estimator_order", "A", "B", "C", "E"]:
            if getattr(self, attr) is NotImplemented:
                raise ValueError(f"Required attribute '{attr}' is not implemented in this Runge-Kutta method")

        if not isinstance(self.n_stages, int) or self.n_stages <= 0:
            raise ValueError(f"n_stages must be a positive integer, got {self.n_stages}")

        # The final column of an explicit tableau is zero and may be omitted.
        shapes = ((self.n_stages, self.n_stages), (self.n_stages, self.n_stages - 1))
        if not isinstance(self.A, numpy.ndarray) or self.A.shape not in shapes:
            raise ValueError(
                f"A must be a numpy.ndarray with shape in {shapes}, "
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

        for i in range(self.n_stages):
            for j in range(i, self.A.shape[1]):
                if abs(self.A[i, j]) > numpy.finfo(float).eps:
                    raise ValueError(
                        f"A must be strictly lower triangular for explicit methods, "
                        f"but A[{i},{j}] = {self.A[i, j]} != 0"
                    )

        if numpy.any((self.C < 0) | (self.C > 1)):
            invalid_indices = numpy.where((self.C < 0) | (self.C > 1))[0]
            raise ValueError(f"C values must be in [0, 1], but C[{invalid_indices}] = {self.C[invalid_indices]}")

        b_sum = numpy.sum(self.B)
        if abs(b_sum - 1.0) > 1e-10:
            _logger.warning(
                f"Sum of B coefficients should be 1.0 for consistency, but got {b_sum}. "
                "This may affect conservation properties."
            )

        for i in range(self.n_stages):
            row_sum = numpy.sum(self.A[i, :])
            if i > 0 and abs(row_sum - self.C[i]) > 1e-10:
                _logger.warning(
                    f"For stage {i}, sum of A coefficients ({row_sum}) should match C[{i}] ({self.C[i]}). "
                    "This may indicate an inconsistent Butcher tableau."
                )

        if not isinstance(self.order, int) or self.order <= 0:
            raise ValueError(f"order must be a positive integer, got {self.order}")

        if not isinstance(self.error_estimator_order, int) or self.error_estimator_order <= 0:
            raise ValueError(f"error_estimator_order must be a positive integer, got {self.error_estimator_order}")

        if self.error_estimator_order >= self.order:
            _logger.warning(
                f"error_estimator_order ({self.error_estimator_order}) should typically be "
                f"less than order ({self.order}) of the main method."
            )

    def _init_control(self, controller: str | tuple[float, float, float, float] | None) -> None:
        coefs = {
            "DEADBEAT": (1, 0, 0, 0.9),  # elementary controller (I)
            "PI3040": (0.7, -0.4, 0, 0.8),  # PI controller (Gustafsson)
            "PI4020": (0.6, -0.2, 0, 0.8),  # PI controller for nonstiff methods
            "H211PI": (1 / 6, 1 / 6, 0, 0.8),  # LP filter of PI structure
            "H110": (1 / 3, 0, 0, 0.8),  # I controller (convolution filter)
            "H211D": (1 / 2, 1 / 2, 1 / 2, 0.8),  # LP filter with gain = 1/2
            "H211B": (1 / 4, 1 / 4, 1 / 4, 0.8),  # general purpose LP filter
        }

        if controller is None:
            controller = self.controller if self.controller is not NotImplemented else "deadbeat"
        if isinstance(controller, str):
            controller = controller.upper()

        if isinstance(controller, str) and controller in coefs:
            kb1, kb2, a, g = coefs[controller]
        elif isinstance(controller, tuple) and len(controller) == 4:
            kb1, kb2, a, g = controller
        else:
            raise ValueError(f"Invalid controller: {controller}. Must be one of {list(coefs.keys())} or a 4-tuple")

        if not all(math.isfinite(value) for value in (kb1, kb2, a, g)) or not 0 < g < 1:
            raise ValueError("Controller coefficients must be finite, with safety strictly between zero and one")
        self.minbeta1 = kb1 * self.error_exponent
        self.minbeta2 = kb2 * self.error_exponent
        self.minalpha = -a
        self.safety = g
        self.safety_sc = g ** (kb1 + kb2)
        self.standard_sc = True

    def step(self) -> None:
        """Advance one accepted step, retrying rejected trials."""
        if self.status != "running":
            raise RuntimeError("Attempt to step on a failed or finished solver")

        if self.n == 0 or self.t == self.t_bound:
            self.t = self.t_bound
            self.status = "finished"
        else:
            success = self._step_impl()

            if not success:
                self.status = "failed"
                _logger.error(f"Integration failed at t={self.t}")
            else:
                if self.t - self.t_bound >= 0:
                    self.status = "finished"

    def _step_impl(self) -> bool:
        t = self.t
        y = self.y

        h, min_step = self._reassess_stepsize(t)

        step_accepted = False
        step_rejected = False

        while not step_accepted:
            if h < min_step or t + h == t:
                _logger.error(f"Step size {h} below minimum {min_step} at t={t}")
                return False

            t_new = t + h
            # A device scalar avoids recompilation when adaptivity changes h.
            step = self._step_tensor.fill_(h) if self._step_tensor is not None else h

            self.K[0] = self.f
            for i in range(1, self.n_stages):
                stage = self._stage_state(y, self.K[:i].T @ self._A[i, :i], step)
                self.K[i] = self.fun(t + self.C[i] * h, stage)

            y_new = self._stage_state(y, self.K[: self.n_stages].T @ self._B, step)

            if self.endpoint_error_stage:
                self.K[self.n_stages, :] = self.fun(self.t + h, y_new)

            n_error_stages = self.n_stages + self.endpoint_error_stage
            err_estimate, error_sum_of_squares = self._error_vectors(
                y, y_new, self.K[:n_error_stages].T @ self._E[:n_error_stages], step, self.rtol, self.atol
            )
            error_norm = global_rms_norm(error_sum_of_squares, self.n, self.comm)

            if error_norm < 1:
                step_accepted = True
                factor = self._step_size_factor(error_norm, h)

                if step_rejected:
                    factor = min(1, factor)

                next_step = h * limiter(factor, 2)

            else:
                step_rejected = True
                if math.isfinite(error_norm):
                    h *= limiter(self.safety * error_norm**self.error_exponent, 2)
                else:
                    h *= 0.25
                    self.standard_sc = True

                self.failed_steps += 1

        if not self.endpoint_error_stage:
            self.K[self.n_stages] = self.fun(t_new, y_new)

        self.h_previous = h
        self.h = next_step
        self.f = self.K[self.n_stages].clone()
        self.error_norm_old = error_norm
        self._error_estimate = err_estimate

        self.t = t_new
        self.y = y_new

        return True

    def estimate_error(self) -> float:
        """Reduce the last accepted error estimate over the solver communicator."""
        if self._error_estimate is None:
            return 0.0
        return global_inf_norm(self._error_estimate, self.comm)

    def _step_size_factor(self, error_norm: float, step: float) -> float:
        if error_norm < self.tiny_err:
            self.standard_sc = True
            return BIG_FACTOR
        if self.standard_sc:
            self.standard_sc = False
            return self.safety * error_norm**self.error_exponent

        step_ratio = step / self.h_previous
        return self.safety_sc * (
            error_norm**self.minbeta1 * self.error_norm_old**self.minbeta2 * step_ratio**self.minalpha
        )

    def _init_min_step_parameters(self) -> tuple[float, float]:
        """Resolve distinct stage times using the working precision."""
        # minimum difference between distinct C-values
        cdiff = 1.0
        for c1 in self.C:
            for c2 in self.C:
                diff = abs(c1 - c2)
                if diff:
                    cdiff = min(cdiff, diff)

        if cdiff < 1e-3:
            cdiff = 1e-3
            _logger.warning(
                "Some C-values of this Runge Kutta method are nearly the "
                "same but not identical. This limits the minimum stepsize. "
                "You may want to check the implementation of this method."
            )

        # Map the Torch dtype to its NumPy equivalent because numpy.finfo provides epsneg.
        np_dtype = numpy.dtype(str(self.y.dtype).rsplit(".", 1)[-1])
        epsneg = numpy.finfo(np_dtype).epsneg
        tiny = numpy.finfo(np_dtype).tiny
        h_min_a = 10 * epsneg / cdiff
        h_min_b = math.sqrt(tiny)

        return h_min_a, h_min_b

    def _reassess_stepsize(self, t: float) -> tuple[float, float]:
        """Respect step bounds and avoid an unnecessarily short final step."""
        h = self.h
        min_step = max(self.h_min_a * (abs(t) + h), self.h_min_b)

        if h < min_step or h > self.max_step:
            h = min(self.max_step, max(min_step, h))
            self.standard_sc = True

        d = abs(self.t_bound - t)  # remaining interval
        if d < 2 * h:
            if d > h:
                # h < d < 2 * h: "look ahead".
                # split d over last two steps. This reduces the chance of a
                # very small last step.
                h = max(0.5 * d, min_step)
                self.standard_sc = True
            else:
                h = d

        return h, min_step
