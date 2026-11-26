"""Analytic actions of the static-geometry Euler spatial Jacobians.

Upwind branches are frozen at the base state, including
the nonlinear RHS's left-state convention at zero normal velocity.
"""

from functools import cache, partial
from time import perf_counter

import torch
from mpi4py import MPI

from ..common.definitions import heat_capacity_ratio
from ..common.matmul import apply_op, sparse_dfr_derivative
from ..common.tensor_math import spatial_matrix_vector
from ..pde.euler_interfaces import acoustic_interface, interface_trace, low_mach_scales, physical_mach_number, pressure


def _mach_gradient(state, inverse_spatial, sound_speed, mach):
    """Differentiate sqrt(g^{ij} v_i v_j)/c with respect to (rho, v_i, rho theta)."""
    safe_mach = torch.where(mach > 0, mach, torch.ones_like(mach))
    velocity_gradient = spatial_matrix_vector(inverse_spatial, state[1:4]) / (sound_speed.square() * safe_mach)
    return torch.cat(
        ((mach / (2 * state[0]))[None], velocity_gradient, (-heat_capacity_ratio * mach / (2 * state[4]))[None]),
        dim=0,
    )


def _compile_jacobian_graph(graph, inputs, *, options, component):
    announce = MPI.COMM_WORLD.rank == 0
    if announce:
        print(f"Compiling {component} Jacobian...", flush=True, end=" ")
    start = perf_counter()
    compiled = torch._inductor.compile(graph, inputs, options=options)
    if announce:
        print(f"done. ({perf_counter() - start:.3f} s)", flush=True)
    return compiled


def _compile_action(action, component):
    return torch.compile(
        action,
        backend=partial(_compile_jacobian_graph, component=component),
        # Allow eager fallback when varied layouts exhaust the compiler cache.
        fullgraph=False,
        dynamic=False,
        options={
            # Avoid a compiler-worker pool in every MPI rank.
            "compile_threads": 1,
            # Preserve eager rounding for division and multiply-add operations.
            "emulate_precision_casts": True,
            "eager_numerics.division_rounding": True,
        },
    )


def _aligned_compiled_input(value):
    """Share one aligned CUDA input across compiled actions in inference mode."""
    # Inductor otherwise copies misaligned inputs separately for each action.
    if value.is_cuda and torch.is_inference_mode_enabled() and value.data_ptr() % 16:
        return value.clone()
    return value


@cache
def _compiled_acoustic_action():
    return _compile_action(AcousticLinearization._apply, "interface")


@cache
def _compiled_volume_actions():
    actions = (
        _prepare_volume,
        _prepare_volume_5var,
        _finalize_trace_5var,
        _direction_action,
        _vertical_rho_theta,
        _momentum_action,
    )
    return tuple(_compile_action(action, "volume") for action in actions)


@cache
def _compiled_dfr_derivative():
    # Three directions and scalar/vector layouts exceed the default eight specializations.
    compiled = _compile_action(_apply_dfr_derivative, "DFR")
    return torch._dynamo.config.patch(recompile_limit=32)(compiled)


def _apply_dfr_derivative(operator, volume, faces):
    # Eager fallback must not materialize the expanded stencil arrays.
    if not torch.compiler.is_compiling():
        return operator.dense(volume, faces)
    return sparse_dfr_derivative(
        volume, faces, operator.indices, operator.weights, operator.face_indices, operator.face_weights
    )


def _apply_dfr_lifting(operator, faces):
    """Apply only the face part of the DFR derivative."""
    if not torch.compiler.is_compiling():
        return apply_op(faces, operator.face_operator)
    return (faces[..., operator.face_indices] * operator.face_weights).sum(dim=-1)


def _prepare_volume(inverse_spatial, delta_state, pressure, state, velocity):
    delta_velocity = spatial_matrix_vector(inverse_spatial, delta_state[1:4])
    delta_pressure = heat_capacity_ratio * pressure * delta_state[4] / state[4]
    delta_energy = (velocity * delta_state[1:4]).sum(dim=0)
    return delta_velocity, delta_pressure, delta_energy


def _prepare_volume_5var(inverse_spatial, volume_jacobian, delta_state, pressure, state, velocity, theta):
    """Prepare nodal variations and face extrapolation inputs for Euler without tracers."""
    delta_velocity = spatial_matrix_vector(inverse_spatial, delta_state[1:4])
    delta_pressure = heat_capacity_ratio * pressure * delta_state[4] / state[4]
    delta_energy = (velocity * delta_state[1:4]).sum(dim=0)
    delta_scalars = torch.stack((delta_state[0], delta_state[4]))
    extrap_input = torch.stack(
        (
            delta_state[0] / state[0],
            delta_state[1],
            delta_state[2],
            delta_state[3],
            delta_state[4] / state[4],
        )
    )
    work_flux = volume_jacobian * delta_velocity
    delta_theta = (delta_state[4] - theta * delta_state[0]) / state[0]
    return delta_velocity, delta_pressure, delta_energy, delta_scalars, extrap_input, work_flux, delta_theta


def _finalize_trace_5var(raw_trace, base_trace, work_flux_trace, face_jacobian, inverse_row, direction: int):
    """Apply linearized log-scalar scaling and pressure-work velocity matching in one pass."""
    work_velocity = work_flux_trace / face_jacobian
    traced_velocity = (inverse_row * raw_trace[1:4]).sum(dim=0)
    correction = (work_velocity - traced_velocity) / inverse_row[direction]
    v1 = raw_trace[1] + correction if direction == 0 else raw_trace[1]
    v2 = raw_trace[2] + correction if direction == 1 else raw_trace[2]
    v3 = raw_trace[3] + correction if direction == 2 else raw_trace[3]
    return torch.stack((base_trace[0] * raw_trace[0], v1, v2, v3, base_trace[4] * raw_trace[4]))


def _direction_action(
    operator,
    volume_jacobian,
    delta_velocity,
    scalars,
    velocity,
    delta_scalars,
    face_jacobian,
    face_flux,
    delta_pressure,
    face_pressure,
    delta_energy,
    face_energy,
    direction: int,
):
    """Compute scalar tendency, pressure gradient, and energy gradient for one coordinate direction."""
    flux = volume_jacobian * (delta_velocity * scalars + velocity * delta_scalars)
    scalar_trace = interface_trace(face_jacobian * face_flux, direction)
    scalar_change = -_apply_dfr_derivative(operator, flux, scalar_trace) / volume_jacobian
    pressure_grad = _apply_dfr_derivative(operator, delta_pressure, interface_trace(face_pressure, direction))
    energy_grad = _apply_dfr_derivative(operator, delta_energy, interface_trace(face_energy, direction))
    if direction == 2:
        return scalar_change, pressure_grad, energy_grad, scalar_trace[0]
    return scalar_change, pressure_grad, energy_grad, None


def _vertical_rho_theta(
    operator,
    delta_state,
    delta_theta,
    delta_element_trace,
    delta_left_rho,
    delta_left_rho_theta,
    delta_right_rho,
    delta_right_rho_theta,
    upwind,
    upwind_theta,
    upwind_rho,
    delta_mass_tendency,
    delta_velocity,
    delta_mass_flux_trace,
    *,
    rho,
    theta,
    velocity,
    mass_tendency,
    theta_gradient,
    mass_flux_trace,
    theta_jump,
    volume_jacobian,
):
    """Linearize vertical theta advection and its face-mass-flux penalty."""
    delta_rho = torch.where(upwind, delta_left_rho, delta_right_rho)
    delta_rho_theta = torch.where(upwind, delta_left_rho_theta, delta_right_rho_theta)
    delta_theta_trace = interface_trace((delta_rho_theta - upwind_theta * delta_rho) / upwind_rho, 2)
    delta_penalty = delta_mass_flux_trace * theta_jump + mass_flux_trace * (delta_theta_trace - delta_element_trace)
    return (
        delta_theta * mass_tendency
        + theta * delta_mass_tendency
        - (delta_state[0] * velocity + rho * delta_velocity) * theta_gradient
        - rho * velocity * _apply_dfr_derivative(operator, delta_theta, delta_element_trace)
        - _apply_dfr_lifting(operator, delta_penalty) / volume_jacobian
    )


def _momentum_action(
    scalar_changes,
    delta_state,
    rho,
    pressure_gradients,
    energy_gradients,
    pressure_acceleration,
    derivatives,
    velocity,
    base_curl,
    frame_curl,
    delta_velocity,
    damping,
    *,
    horizontal_v3_gradient=None,
    horizontal_frame_gradient=None,
    scale: float = 1.0,
):
    """Assemble the full or complementary Euler Jacobian action functionally."""
    delta_rho_over_rho = delta_state[0] / rho
    m0 = -pressure_gradients[0] / rho - energy_gradients[0] - pressure_acceleration[0] * delta_rho_over_rho
    m1 = -pressure_gradients[1] / rho - energy_gradients[1] - pressure_acceleration[1] * delta_rho_over_rho
    if len(pressure_gradients) == 3:
        m2 = -pressure_gradients[2] / rho - energy_gradients[2] - pressure_acceleration[2] * delta_rho_over_rho
    else:
        m2 = torch.zeros_like(m0)

    dv2_dx1, dv3_dx1, dv1_dx2, dv3_dx2, dv1_dx3, dv2_dx3 = derivatives
    curl_12 = dv2_dx1 - dv1_dx2
    curl_13 = dv3_dx1 - dv1_dx3
    curl_23 = dv3_dx2 - dv2_dx3
    m0 = m0 + velocity[1] * curl_12 + velocity[2] * curl_13
    m1 = m1 - velocity[0] * curl_12 + velocity[2] * curl_23
    if horizontal_v3_gradient is None:
        m2 = m2 - velocity[0] * curl_13 - velocity[1] * curl_23
    else:
        m2 = m2 - velocity[0] * dv3_dx1 - velocity[1] * dv3_dx2

    m0 = (
        m0
        + delta_velocity[1] * (base_curl[0, 1] + frame_curl[0, 1])
        + delta_velocity[2] * (base_curl[0, 2] + frame_curl[0, 2])
    )
    m1 = (
        m1
        - delta_velocity[0] * (base_curl[0, 1] + frame_curl[0, 1])
        + delta_velocity[2] * (base_curl[1, 2] + frame_curl[1, 2])
    )
    if horizontal_v3_gradient is None:
        m2 = (
            m2
            - delta_velocity[0] * (base_curl[0, 2] + frame_curl[0, 2])
            - delta_velocity[1] * (base_curl[1, 2] + frame_curl[1, 2])
        )
    else:
        m2 = (
            m2
            - delta_velocity[0] * (horizontal_v3_gradient[0] + horizontal_frame_gradient[0])
            - delta_velocity[1] * (horizontal_v3_gradient[1] + horizontal_frame_gradient[1])
        )

    if damping is not None:
        m0 = m0 - damping * delta_state[1]
        m1 = m1 - damping * delta_state[2]
        m2 = m2 - damping * delta_state[3]

    scalar_tendency = scalar_changes[0] + scalar_changes[1]
    if len(scalar_changes) == 3:
        scalar_tendency = scalar_tendency + scalar_changes[2]

    if delta_state.shape[0] == 5:
        result = torch.stack((scalar_tendency[0], m0, m1, m2, scalar_tendency[1]))
    else:
        result = torch.cat((scalar_tendency[0:1], torch.stack((m0, m1, m2)), scalar_tendency[1:]), dim=0)
    if scale != 1.0:
        result = result * scale
    return result


class AcousticLinearization:
    """Cache an acoustic face state and apply its linearized interface solver."""

    def __init__(self, left, right, geometry, direction, *, mach_cutoff=0.3, scalar_indices=None, scalar_index=None):
        self.scalar_indices = [0, 4] if scalar_indices is None else scalar_indices
        self.scalar_index = (
            torch.tensor(self.scalar_indices, dtype=torch.long, device=left.device)
            if scalar_index is None
            else scalar_index
        )
        self.direction = direction
        self.inverse_spatial = geometry.inverse_spatial
        self.normal_scale = geometry.normal_scale
        self.normal_row = geometry.normal_row
        self.wall = geometry.wall
        self.left, self.right = left, right
        self.pressure_left, self.pressure_right = pressure(left[4]), pressure(right[4])
        self.normal_wind_left = (self.normal_row * left[1:4]).sum(dim=0)
        self.normal_wind_right = (self.normal_row * right[1:4]).sum(dim=0)
        self.sound_speed_left = torch.sqrt(heat_capacity_ratio * self.pressure_left / left[0])
        self.sound_speed_right = torch.sqrt(heat_capacity_ratio * self.pressure_right / right[0])
        self.impedance = (left[0] * self.sound_speed_left + right[0] * self.sound_speed_right) / 2
        mach_left = physical_mach_number(left[1:4], self.inverse_spatial, self.sound_speed_left)
        mach_right = physical_mach_number(right[1:4], self.inverse_spatial, self.sound_speed_right)
        interface_mach, self.pressure_scale = low_mach_scales(mach_left, mach_right, mach_cutoff)
        self.mach_from_left = mach_left >= mach_right
        mach_state = torch.where(self.mach_from_left[None], left, right)
        mach_sound_speed = torch.where(self.mach_from_left, self.sound_speed_left, self.sound_speed_right)
        self.mach_gradient = _mach_gradient(mach_state, self.inverse_spatial, mach_sound_speed, interface_mach)
        self.pressure_scale_active = (interface_mach > mach_cutoff) & (interface_mach < 1)
        self.pressure_scale_derivative = 2 * (1 - torch.clamp(interface_mach, min=mach_cutoff, max=1.0))
        self.pressure_impedance = self.pressure_scale * self.impedance
        normal_velocity, self.common_pressure, _, self.common_covariant_velocity = acoustic_interface(
            left,
            right,
            self.inverse_spatial,
            direction,
            geometry.geopotential,
            wall=self.wall,
            return_velocity=True,
            normal_geometry=(self.normal_scale, self.normal_row),
            mach_cutoff=mach_cutoff,
        )
        self.normal_velocity = normal_velocity
        self.upwind = normal_velocity >= 0
        self.upwind_scalars = torch.where(self.upwind[None], left[self.scalar_index], right[self.scalar_index])
        self.rho_theta_position = self.scalar_indices.index(4)
        self.upwind_theta = self.upwind_scalars[self.rho_theta_position] / self.upwind_scalars[0]
        self.common_contravariant_velocity = spatial_matrix_vector(self.inverse_spatial, self.common_covariant_velocity)

    def __call__(self, delta_left, delta_right, *, velocity_only=False):
        if delta_left.is_cuda:
            return _compiled_acoustic_action()(self, delta_left, delta_right, velocity_only=velocity_only)
        return self._apply(delta_left, delta_right, velocity_only=velocity_only)

    def delta_upwind_theta(self, delta_left, delta_right):
        """Linearize the potential-temperature trace with the upwind side held fixed."""
        delta_rho = torch.where(self.upwind, delta_left[0], delta_right[0])
        delta_rho_theta = torch.where(self.upwind, delta_left[4], delta_right[4])
        return (delta_rho_theta - self.upwind_theta * delta_rho) / self.upwind_scalars[0]

    def _apply(self, delta_left, delta_right, *, velocity_only=False):
        delta_pressure_left = heat_capacity_ratio * self.pressure_left * delta_left[4] / self.left[4]
        delta_pressure_right = heat_capacity_ratio * self.pressure_right * delta_right[4] / self.right[4]
        delta_normal_wind_left = (self.normal_row * delta_left[1:4]).sum(dim=0)
        delta_normal_wind_right = (self.normal_row * delta_right[1:4]).sum(dim=0)
        # d(rho*c) = rho*c/2 * (d rho/rho + d p/p).
        delta_impedance = (
            self.sound_speed_left * (delta_left[0] + self.left[0] * delta_pressure_left / self.pressure_left)
            + self.sound_speed_right * (delta_right[0] + self.right[0] * delta_pressure_right / self.pressure_right)
        ) / 4
        mach_perturbation = torch.where(self.mach_from_left[None], delta_left[:5], delta_right[:5])
        delta_mach = (self.mach_gradient * mach_perturbation).sum(dim=0)
        delta_pressure_scale = torch.where(self.pressure_scale_active, self.pressure_scale_derivative * delta_mach, 0.0)
        delta_normal = (delta_normal_wind_left + delta_normal_wind_right) / 2 - (
            delta_pressure_right - delta_pressure_left
        ) / (2 * self.impedance)
        delta_normal += (self.pressure_right - self.pressure_left) * delta_impedance / (2 * self.impedance**2)
        if self.wall is not None:
            delta_normal = torch.where(self.wall, 0.0, delta_normal)
        mean_v0 = (delta_left[1] + delta_right[1]) / 2
        mean_v1 = (delta_left[2] + delta_right[2]) / 2
        mean_v2 = (delta_left[3] + delta_right[3]) / 2
        mean_normal = self.normal_row[0] * mean_v0 + self.normal_row[1] * mean_v1 + self.normal_row[2] * mean_v2
        normal_shift = (delta_normal - mean_normal) / self.normal_scale
        v0 = mean_v0 + normal_shift if self.direction == 0 else mean_v0
        v1 = mean_v1 + normal_shift if self.direction == 1 else mean_v1
        v2 = mean_v2 + normal_shift if self.direction == 2 else mean_v2
        delta_common_covariant_velocity = torch.stack((v0, v1, v2))
        if velocity_only:
            return delta_common_covariant_velocity
        delta_pressure_impedance = self.pressure_scale * delta_impedance + self.impedance * delta_pressure_scale
        delta_pressure = (delta_pressure_left + delta_pressure_right) / 2 - self.pressure_impedance * (
            delta_normal_wind_right - delta_normal_wind_left
        ) / 2
        delta_pressure -= delta_pressure_impedance * (self.normal_wind_right - self.normal_wind_left) / 2
        delta_kinetic_energy = (
            self.common_contravariant_velocity[0] * v0
            + self.common_contravariant_velocity[1] * v1
            + self.common_contravariant_velocity[2] * v2
        )
        if self.scalar_indices == [0, 4]:
            left_scalars = torch.stack((delta_left[0], delta_left[4]))
            right_scalars = torch.stack((delta_right[0], delta_right[4]))
            delta_scalars = torch.where(self.upwind[None], left_scalars, right_scalars)
        else:
            delta_scalars = torch.where(
                self.upwind[None], delta_left[self.scalar_index], delta_right[self.scalar_index]
            )
        delta_normal_velocity = delta_normal * self.normal_scale
        delta_flux = delta_normal_velocity * self.upwind_scalars + self.normal_velocity * delta_scalars
        return delta_normal_velocity, delta_pressure, delta_kinetic_energy, delta_common_covariant_velocity, delta_flux


class EulerLinearization:
    """Cache an Euler base state for full and complementary Jacobian actions."""

    supports_scale = True

    def __init__(self, rhs, state):
        self.rhs = rhs
        self.shape = state.shape
        self.state = state.reshape((-1,) + rhs.metric.volume.height.shape)
        self.scalars = self.state[rhs.scalar_index]
        fields = rhs.metric.volume
        self.contravariant_velocity = fields.raise_velocity(self.state[1:4])
        self.pressure = pressure(self.state[4])
        self.traces = rhs._extrapolate(self.state, self.contravariant_velocity)
        faces = rhs._faces_with_neighbours(self.traces)
        self.interfaces = tuple(
            AcousticLinearization(
                face[geometry.left],
                face[geometry.right],
                geometry,
                direction,
                mach_cutoff=rhs.config.riemann_mach_cutoff,
                scalar_indices=rhs.scalar_indices,
                scalar_index=rhs.scalar_index,
            )
            for direction, (geometry, face) in enumerate(zip(rhs._static_interfaces(), faces))
        )
        # RHS evaluations overwrite diagnostics; keep separate linearization data.
        gradient = []
        common_velocity_traces = []
        for direction, interface in enumerate(self.interfaces):
            common_velocity_traces.append(interface_trace(interface.common_covariant_velocity, direction))
            gradient.append(
                apply_op(self.pressure, rhs.derivatives[direction])
                + apply_op(interface_trace(interface.common_pressure, direction), rhs.corrections[direction])
            )
        self.pressure_acceleration = -torch.stack(gradient) / self.state[0]
        self.common_velocity_traces = tuple(common_velocity_traces)
        self.base_relative_curl = rhs.relative_covariant_curl(self.state[1:4], self.common_velocity_traces)
        self.horizontal_vertical_velocity_gradient = None

        vertical, geometry = self.interfaces[2], rhs._static_interfaces()[2]
        face_mass_flux = interface_trace(geometry.jacobian * vertical.normal_velocity * vertical.upwind_scalars[0], 2)
        mass_flux = fields.volume_jacobian * self.contravariant_velocity[2] * self.state[0]
        self.vertical_mass_tendency = (
            -(apply_op(mass_flux, rhs.derivatives[2]) + apply_op(face_mass_flux, rhs.corrections[2]))
            / fields.volume_jacobian
        )
        self.theta = self.state[4] / self.state[0]
        element_trace = apply_op(self.theta, rhs.extrapolations[2])
        self.vertical_theta_gradient = apply_op(self.theta, rhs.derivatives[2]) + apply_op(
            element_trace, rhs.corrections[2]
        )
        self.vertical_face_mass_flux = face_mass_flux
        self.vertical_theta_jump = interface_trace(vertical.upwind_theta, 2) - element_trace
        self._face_work_geometry = tuple(
            (
                fields.volume_jacobian[interior].contiguous(),
                fields.G_contra[direction + 1, 1:4][interior].contiguous(),
            )
            for direction, (fields, interior) in enumerate(zip(rhs.metric.faces, rhs.mid_faces))
        )

    def explicit(self, vector):
        """Apply the complementary Jacobian directly, with its DFR curl terms."""
        if self.horizontal_vertical_velocity_gradient is None:
            self.horizontal_vertical_velocity_gradient = torch.stack(
                [
                    apply_op(self.state[3], self.rhs.derivatives[a])
                    + apply_op(self.common_velocity_traces[a][2], self.rhs.corrections[a])
                    for a in range(2)
                ]
            )
        return self(vector, explicit=True)

    def __call__(self, vector, *, explicit=False, scale: float = 1.0):
        rhs, fields, base_state = self.rhs, self.rhs.metric.volume, self.state
        delta_state = _aligned_compiled_input(vector.reshape(base_state.shape).to(base_state.dtype))
        if delta_state.is_cuda:
            (
                prepare_volume,
                prepare_volume_5var,
                finalize_trace_5var,
                direction_action,
                vertical_rho_theta,
                momentum_action,
            ) = _compiled_volume_actions()
            derivative_operators = rhs.dfr_derivatives(_compiled_dfr_derivative())
        else:
            prepare_volume = _prepare_volume
            prepare_volume_5var = _prepare_volume_5var
            finalize_trace_5var = _finalize_trace_5var
            direction_action = _direction_action
            vertical_rho_theta = _vertical_rho_theta
            momentum_action = _momentum_action
            derivative_operators = rhs.dfr_derivatives(_apply_dfr_derivative)

        is_standard_5var = (
            base_state.shape[0] == 5 and rhs.scalar_indices == [0, 4] and rhs.log_scalar_indices == [0, 4]
        )
        if is_standard_5var:
            (
                delta_contravariant_velocity,
                delta_pressure,
                delta_energy,
                delta_scalars,
                extrap_input,
                work_flux,
                delta_theta,
            ) = prepare_volume_5var(
                fields.G_contra[1:4, 1:4],
                fields.volume_jacobian,
                delta_state,
                self.pressure,
                base_state,
                self.contravariant_velocity,
                self.theta,
            )
            traces = tuple(
                finalize_trace_5var(
                    apply_op(extrap_input, operator),
                    base_trace,
                    apply_op(work_flux[direction], operator),
                    face_jacobian,
                    inverse_row,
                    direction,
                )
                for direction, (operator, base_trace, (face_jacobian, inverse_row)) in enumerate(
                    zip(rhs.extrapolations, self.traces, self._face_work_geometry)
                )
            )
            del extrap_input, work_flux
        else:
            delta_contravariant_velocity, delta_pressure, delta_energy = prepare_volume(
                fields.G_contra[1:4, 1:4], delta_state, self.pressure, base_state, self.contravariant_velocity
            )
            delta_scalars = delta_state[rhs.scalar_index]
            delta_theta = None
            relative_scalar_perturbation = delta_state[rhs.log_scalar_index] / base_state[rhs.log_scalar_index]
            traces = []
            for operator, base_trace in zip(rhs.extrapolations, self.traces):
                trace = torch.empty_like(base_trace)
                trace[1:4] = apply_op(delta_state[1:4], operator)
                trace[rhs.log_scalar_index] = base_trace[rhs.log_scalar_index] * apply_op(
                    relative_scalar_perturbation, operator
                )
                if base_state.shape[0] > 6:
                    mixing_ratio = base_state[6:] / base_state[0]
                    delta_mixing_ratio = (delta_state[6:] - mixing_ratio * delta_state[0]) / base_state[0]
                    trace[6:] = trace[0] * base_trace[6:] / base_trace[0]
                    trace[6:] += base_trace[0] * apply_op(delta_mixing_ratio, operator)
                traces.append(trace)
            rhs._match_pressure_work(delta_contravariant_velocity, traces)
            traces = tuple(traces)
            del trace, relative_scalar_perturbation
        # Consume the face arrays one direction at a time to limit peak memory.
        faces = list(rhs._faces_with_neighbours(traces))
        del traces
        scalar_changes = []
        pressure_gradient, energy_gradient = [], []
        velocity_traces = []
        for direction, (geometry, interface) in enumerate(zip(rhs._static_interfaces(), self.interfaces)):
            face = faces[direction]
            left, right = geometry.left, geometry.right
            face_left, face_right = face[left], face[right]
            if explicit and direction == 2:
                delta_covariant_velocity = interface(face_left, face_right, velocity_only=True)
            else:
                _, face_pressure, face_energy, delta_covariant_velocity, face_flux = interface(face_left, face_right)
                scalar_change, p_grad, e_grad, mass_flux_trace = direction_action(
                    derivative_operators[direction],
                    fields.volume_jacobian,
                    delta_contravariant_velocity[direction],
                    self.scalars,
                    self.contravariant_velocity[direction],
                    delta_scalars,
                    geometry.jacobian,
                    face_flux,
                    delta_pressure,
                    face_pressure,
                    delta_energy,
                    face_energy,
                    direction,
                )
                if direction == 2:
                    if delta_theta is None:
                        delta_theta = (delta_state[4] - self.theta * delta_state[0]) / base_state[0]
                    # Extrapolate before the compiled helper, which then holds no matrix product.
                    delta_element_trace = apply_op(delta_theta, rhs.extrapolations[2])
                    scalar_change[rhs.rho_theta_position] = vertical_rho_theta(
                        derivative_operators[2],
                        delta_state,
                        delta_theta,
                        delta_element_trace,
                        face_left[0],
                        face_left[4],
                        face_right[0],
                        face_right[4],
                        interface.upwind,
                        interface.upwind_theta,
                        interface.upwind_scalars[0],
                        scalar_change[0],
                        delta_contravariant_velocity[2],
                        mass_flux_trace,
                        rho=base_state[0],
                        theta=self.theta,
                        velocity=self.contravariant_velocity[2],
                        mass_tendency=self.vertical_mass_tendency,
                        theta_gradient=self.vertical_theta_gradient,
                        mass_flux_trace=self.vertical_face_mass_flux,
                        theta_jump=self.vertical_theta_jump,
                        volume_jacobian=fields.volume_jacobian,
                    )
                scalar_changes.append(scalar_change)
                pressure_gradient.append(p_grad)
                energy_gradient.append(e_grad)
                del scalar_change, face_flux, face_pressure, face_energy, mass_flux_trace
            velocity_traces.append(interface_trace(delta_covariant_velocity, direction))
            faces[direction] = None
            del face, face_left, face_right, delta_covariant_velocity
        del delta_pressure, delta_energy
        derivatives = rhs._curl_derivatives(delta_state[1:4], tuple(velocity_traces), derivative_operators)
        result = momentum_action(
            tuple(scalar_changes),
            delta_state,
            base_state[0],
            tuple(pressure_gradient),
            tuple(energy_gradient),
            self.pressure_acceleration[:2] if explicit else self.pressure_acceleration,
            derivatives,
            self.contravariant_velocity,
            self.base_relative_curl,
            fields.frame_vorticity,
            delta_contravariant_velocity,
            None if rhs.damping is None else rhs.damping[0],
            horizontal_v3_gradient=self.horizontal_vertical_velocity_gradient if explicit else None,
            horizontal_frame_gradient=fields.horizontal_frame_gradient if explicit else None,
            scale=scale,
        )
        return result.reshape(self.shape).flatten()
