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
from ..common.tensor_math import antisymmetric_velocity_contraction, relative_covariant_curl, spatial_matrix_vector
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
    actions = (_prepare_volume, _scalar_flux, _vertical_rho_theta, _momentum_action)
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


def _prepare_volume(inverse_spatial, delta_state, pressure, state, velocity):
    delta_velocity = spatial_matrix_vector(inverse_spatial, delta_state[1:4])
    delta_pressure = heat_capacity_ratio * pressure * delta_state[4] / state[4]
    delta_energy = (velocity * delta_state[1:4]).sum(dim=0)
    return delta_velocity, delta_pressure, delta_energy


def _scalar_flux(jacobian, delta_velocity, scalars, velocity, delta_scalars):
    return jacobian * (delta_velocity * scalars + velocity * delta_scalars)


def _vertical_rho_theta(
    operator,
    delta_state,
    delta_theta_trace,
    delta_mass_tendency,
    delta_velocity,
    *,
    rho,
    theta,
    velocity,
    mass_tendency,
    theta_gradient,
):
    """Linearize the vertical thermodynamic tendency with its common theta trace."""
    delta_theta = (delta_state[4] - theta * delta_state[0]) / rho
    return (
        delta_theta * mass_tendency
        + theta * delta_mass_tendency
        - (delta_state[0] * velocity + rho * delta_velocity) * theta_gradient
        - rho * velocity * _apply_dfr_derivative(operator, delta_theta, delta_theta_trace)
    )


def _momentum_action(
    result,
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
):
    """Assemble the full or complementary momentum Jacobian action."""
    momentum = slice(1, 1 + len(pressure_gradients))
    result[momentum] = -torch.stack(pressure_gradients) / rho - torch.stack(energy_gradients)
    result[momentum] -= pressure_acceleration * (delta_state[0] / rho)
    if horizontal_v3_gradient is None:
        result[1:4] += antisymmetric_velocity_contraction(relative_covariant_curl(derivatives), velocity)
    else:
        dv2_dx1, dv3_dx1, dv1_dx2, dv3_dx2, dv1_dx3, dv2_dx3 = derivatives
        curl_12 = dv2_dx1 - dv1_dx2
        result[1:4] += torch.stack(
            (
                velocity[1] * curl_12 + velocity[2] * (dv3_dx1 - dv1_dx3),
                -velocity[0] * curl_12 + velocity[2] * (dv3_dx2 - dv2_dx3),
                -velocity[0] * dv3_dx1 - velocity[1] * dv3_dx2,
            )
        )
    base_vorticity = antisymmetric_velocity_contraction(base_curl, delta_velocity)
    frame_vorticity = antisymmetric_velocity_contraction(frame_curl, delta_velocity)
    if horizontal_v3_gradient is not None:
        base_vorticity[2] = -(delta_velocity[:2] * horizontal_v3_gradient).sum(dim=0)
        frame_vorticity[2] = -(delta_velocity[:2] * horizontal_frame_gradient).sum(dim=0)
    result[1:4] += base_vorticity
    result[1:4] += frame_vorticity
    if damping is not None:
        result[1:4] -= damping * delta_state[1:4]
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
        delta = torch.where(self.upwind[None], delta_left[self.scalar_index], delta_right[self.scalar_index])
        return (delta[self.rho_theta_position] - self.upwind_theta * delta[0]) / self.upwind_scalars[0]

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
        delta_pressure_scale = torch.where(
            self.pressure_scale_active, self.pressure_scale_derivative * delta_mach, torch.zeros_like(delta_mach)
        )
        delta_normal = (delta_normal_wind_left + delta_normal_wind_right) / 2 - (
            delta_pressure_right - delta_pressure_left
        ) / (2 * self.impedance)
        delta_normal += (self.pressure_right - self.pressure_left) * delta_impedance / (2 * self.impedance**2)
        if self.wall is not None:
            delta_normal = torch.where(self.wall, torch.zeros_like(delta_normal), delta_normal)
        delta_common_covariant_velocity = (delta_left[1:4] + delta_right[1:4]) / 2
        delta_common_covariant_velocity[self.direction] += (
            delta_normal - (self.normal_row * delta_common_covariant_velocity).sum(dim=0)
        ) / self.normal_scale
        if velocity_only:
            return delta_common_covariant_velocity
        delta_pressure_impedance = self.pressure_scale * delta_impedance + self.impedance * delta_pressure_scale
        delta_pressure = (delta_pressure_left + delta_pressure_right) / 2 - self.pressure_impedance * (
            delta_normal_wind_right - delta_normal_wind_left
        ) / 2
        delta_pressure -= delta_pressure_impedance * (self.normal_wind_right - self.normal_wind_left) / 2
        delta_kinetic_energy = (self.common_contravariant_velocity * delta_common_covariant_velocity).sum(dim=0)
        delta_scalars = torch.where(self.upwind[None], delta_left[self.scalar_index], delta_right[self.scalar_index])
        delta_normal_velocity = delta_normal * self.normal_scale
        delta_flux = delta_normal_velocity * self.upwind_scalars + self.normal_velocity * delta_scalars
        return delta_normal_velocity, delta_pressure, delta_kinetic_energy, delta_common_covariant_velocity, delta_flux


class EulerLinearization:
    """Cache an Euler base state for full and complementary Jacobian actions."""

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
        theta_trace = interface_trace(vertical.upwind_theta, 2)
        self.vertical_theta_gradient = apply_op(self.theta, rhs.derivatives[2]) + apply_op(
            theta_trace, rhs.corrections[2]
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

    def __call__(self, vector, *, explicit=False):
        rhs, fields, base_state = self.rhs, self.rhs.metric.volume, self.state
        delta_state = _aligned_compiled_input(vector.reshape(base_state.shape).to(base_state.dtype))
        if delta_state.is_cuda:
            prepare_volume, scalar_flux, vertical_rho_theta, momentum_action = _compiled_volume_actions()
            derivative_operators = rhs.dfr_derivatives(_compiled_dfr_derivative())
        else:
            prepare_volume, scalar_flux, vertical_rho_theta, momentum_action = (
                _prepare_volume,
                _scalar_flux,
                _vertical_rho_theta,
                _momentum_action,
            )
            derivative_operators = rhs.dfr_derivatives(_apply_dfr_derivative)

        def derivative(direction, value, common):
            return derivative_operators[direction](value, common)

        delta_contravariant_velocity, delta_pressure, delta_energy = prepare_volume(
            fields.G_contra[1:4, 1:4], delta_state, self.pressure, base_state, self.contravariant_velocity
        )
        delta_scalars = delta_state[rhs.scalar_index]
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
        # Consume the face arrays one direction at a time to limit peak memory.
        faces = list(rhs._faces_with_neighbours(tuple(traces)))
        del traces, trace, relative_scalar_perturbation
        result = torch.zeros_like(delta_state)
        scalar_tendency = torch.zeros_like(delta_scalars)
        pressure_gradient, energy_gradient = [], []
        velocity_traces = []
        for direction, (geometry, interface) in enumerate(zip(rhs._static_interfaces(), self.interfaces)):
            face = faces[direction]
            left, right = geometry.left, geometry.right
            if explicit and direction == 2:
                delta_covariant_velocity = interface(face[left], face[right], velocity_only=True)
            else:
                _, face_pressure, face_energy, delta_covariant_velocity, face_flux = interface(face[left], face[right])
                flux = scalar_flux(
                    fields.volume_jacobian,
                    delta_contravariant_velocity[direction],
                    self.scalars,
                    self.contravariant_velocity[direction],
                    delta_scalars,
                )
                scalar_trace = interface_trace(geometry.jacobian * face_flux, direction)
                scalar_change = -derivative(direction, flux, scalar_trace) / fields.volume_jacobian
                if direction == 2:
                    scalar_change[rhs.rho_theta_position] = vertical_rho_theta(
                        derivative_operators[2],
                        delta_state,
                        interface_trace(interface.delta_upwind_theta(face[left], face[right]), 2),
                        scalar_change[0],
                        delta_contravariant_velocity[2],
                        rho=base_state[0],
                        theta=self.theta,
                        velocity=self.contravariant_velocity[2],
                        mass_tendency=self.vertical_mass_tendency,
                        theta_gradient=self.vertical_theta_gradient,
                    )
                scalar_tendency += scalar_change
                pressure_gradient.append(
                    derivative(direction, delta_pressure, interface_trace(face_pressure, direction))
                )
                energy_gradient.append(derivative(direction, delta_energy, interface_trace(face_energy, direction)))
                del scalar_trace, scalar_change, face_flux, flux, face_pressure, face_energy
            velocity_traces.append(interface_trace(delta_covariant_velocity, direction))
            faces[direction] = None
            del face, delta_covariant_velocity
        result[rhs.scalar_index] = scalar_tendency
        del delta_pressure, delta_energy
        derivatives = rhs._curl_derivatives(delta_state[1:4], tuple(velocity_traces), derivative_operators)
        result = momentum_action(
            result,
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
        )
        return result.reshape(self.shape).flatten()
