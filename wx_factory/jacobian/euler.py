"""Analytic actions of the static-geometry Euler spatial Jacobians.

Upwind branches are frozen at the base state, including
the nonlinear RHS's left-state convention at zero normal velocity.
"""

import torch

from ..common.definitions import heat_capacity_ratio
from ..common.matmul import apply_op
from ..common.tensor_math import antisymmetric_velocity_contraction, spatial_matrix_vector
from ..pde.euler_interfaces import acoustic_interface, low_mach_scales, physical_mach_number, pressure


def _mach_gradient(state, inverse_spatial, sound_speed, mach):
    """Differentiate sqrt(g^{ij} v_i v_j)/c with respect to (rho, v_i, rho theta)."""
    safe_mach = torch.where(mach > 0, mach, torch.ones_like(mach))
    velocity_gradient = spatial_matrix_vector(inverse_spatial, state[1:4]) / (sound_speed.square() * safe_mach)
    return torch.cat(
        ((mach / (2 * state[0]))[None], velocity_gradient, (-heat_capacity_ratio * mach / (2 * state[4]))[None]),
        dim=0,
    )


class AcousticLinearization:
    """Cache an acoustic face state and apply its linearized interface solver."""

    def __init__(self, left, right, geometry, direction, *, mach_cutoff=0.3):
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
        self.upwind_scalars = torch.where(self.upwind[None], left[[0, 4]], right[[0, 4]])
        self.common_contravariant_velocity = spatial_matrix_vector(self.inverse_spatial, self.common_covariant_velocity)

    def __call__(self, delta_left, delta_right, *, velocity_only=False):
        delta_pressure_left = heat_capacity_ratio * self.pressure_left * delta_left[4] / self.left[4]
        delta_pressure_right = heat_capacity_ratio * self.pressure_right * delta_right[4] / self.right[4]
        delta_normal_wind_left = (self.normal_row * delta_left[1:4]).sum(dim=0)
        delta_normal_wind_right = (self.normal_row * delta_right[1:4]).sum(dim=0)
        # d(rho*c) = rho*c/2 * (d rho/rho + d p/p).
        delta_impedance = (
            self.sound_speed_left * (delta_left[0] + self.left[0] * delta_pressure_left / self.pressure_left)
            + self.sound_speed_right * (delta_right[0] + self.right[0] * delta_pressure_right / self.pressure_right)
        ) / 4
        mach_perturbation = torch.where(self.mach_from_left[None], delta_left, delta_right)
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
        delta_scalars = torch.where(self.upwind[None], delta_left[[0, 4]], delta_right[[0, 4]])
        delta_normal_velocity = delta_normal * self.normal_scale
        delta_flux = delta_normal_velocity * self.upwind_scalars + self.normal_velocity * delta_scalars
        return delta_normal_velocity, delta_pressure, delta_kinetic_energy, delta_common_covariant_velocity, delta_flux


class EulerLinearization:
    """Cache an Euler base state for full and complementary Jacobian actions."""

    def __init__(self, rhs, state):
        self.rhs = rhs
        self.shape = state.shape
        self.state = state.reshape((-1,) + rhs.metric.volume.height.shape)
        self.scalars = self.state[[0, 4]]
        fields = rhs.metric.volume
        self.contravariant_velocity = fields.raise_velocity(self.state[1:4])
        self.pressure = pressure(self.state[4])
        self.traces = rhs._extrapolate(self.state)
        faces = rhs._faces_with_neighbours(self.traces)
        self.interfaces = tuple(
            AcousticLinearization(
                face[geometry.left],
                face[geometry.right],
                geometry,
                direction,
                mach_cutoff=rhs.config.riemann_mach_cutoff,
            )
            for direction, (geometry, face) in enumerate(zip(rhs._static_interfaces(), faces))
        )
        # RHS evaluations overwrite diagnostics; keep separate linearization data.
        gradient = []
        common_velocity_traces = []
        for direction, (geometry, interface, face) in enumerate(zip(rhs._static_interfaces(), self.interfaces, faces)):
            left, right = geometry.left, geometry.right
            pressure_trace = torch.zeros_like(face[0])
            pressure_trace[left], pressure_trace[right] = interface.common_pressure, interface.common_pressure
            velocity_trace = torch.zeros_like(face[1:4])
            velocity_trace[left], velocity_trace[right] = (
                interface.common_covariant_velocity,
                interface.common_covariant_velocity,
            )
            common_velocity_traces.append(velocity_trace[rhs.mid_faces[direction]])
            gradient.append(
                apply_op(self.pressure, rhs.derivatives[direction])
                + apply_op(pressure_trace[rhs.mid_faces[direction]], rhs.corrections[direction])
            )
        self.pressure_acceleration = -torch.stack(gradient) / self.state[0]
        self.common_velocity_traces = tuple(common_velocity_traces)
        self.base_relative_curl = rhs.relative_covariant_curl(self.state[1:4], self.common_velocity_traces)
        self.horizontal_vertical_velocity_gradient = None

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
        delta_state = vector.reshape(base_state.shape).to(base_state.dtype)
        delta_scalars = delta_state[[0, 4]]
        relative_scalar_perturbation = delta_scalars / self.scalars
        traces = []
        for operator, base_trace in zip(rhs.extrapolations, self.traces):
            trace = torch.empty_like(base_trace)
            trace[1:4] = apply_op(delta_state[1:4], operator)
            trace[[0, 4]] = base_trace[[0, 4]] * apply_op(relative_scalar_perturbation, operator)
            traces.append(trace)
        faces = rhs._faces_with_neighbours(tuple(traces))
        scalar_traces, pressure_traces, energy_traces, velocity_traces = [], [], [], []
        for direction, (geometry, interface, face) in enumerate(zip(rhs._static_interfaces(), self.interfaces, faces)):
            left, right = geometry.left, geometry.right
            if explicit and direction == 2:
                delta_covariant_velocity = interface(face[left], face[right], velocity_only=True)
                trace = torch.zeros_like(face[1:4])
                trace[left], trace[right] = delta_covariant_velocity, delta_covariant_velocity
                velocity_traces.append(trace[rhs.mid_faces[direction]])
                continue
            _, delta_pressure, delta_kinetic_energy, delta_covariant_velocity, flux = interface(face[left], face[right])
            flux = geometry.jacobian * flux
            scalar_trace = face.new_zeros((2,) + face.shape[1:])
            scalar_trace[left], scalar_trace[right] = flux, flux
            scalar_traces.append(scalar_trace[rhs.mid_faces[direction]])
            for value, template, output in (
                (delta_pressure, face[0], pressure_traces),
                (delta_kinetic_energy, face[0], energy_traces),
                (delta_covariant_velocity, face[1:4], velocity_traces),
            ):
                trace = torch.zeros_like(template)
                trace[left], trace[right] = value, value
                output.append(trace[rhs.mid_faces[direction]])
        delta_contravariant_velocity = fields.raise_velocity(delta_state[1:4])
        delta_pressure = heat_capacity_ratio * self.pressure * delta_state[4] / base_state[4]
        delta_energy = (self.contravariant_velocity * delta_state[1:4]).sum(dim=0)
        result = torch.zeros_like(delta_state)
        scalar_tendency = torch.zeros_like(delta_scalars)
        pressure_gradient, energy_gradient = [], []
        for direction in range(2 if explicit else 3):
            flux = fields.volume_jacobian * (
                delta_contravariant_velocity[direction] * self.scalars
                + self.contravariant_velocity[direction] * delta_scalars
            )
            scalar_tendency -= (
                apply_op(flux, rhs.derivatives[direction])
                + apply_op(scalar_traces[direction], rhs.corrections[direction])
            ) / fields.volume_jacobian
            pressure_gradient.append(
                apply_op(delta_pressure, rhs.derivatives[direction])
                + apply_op(pressure_traces[direction], rhs.corrections[direction])
            )
            energy_gradient.append(
                apply_op(delta_energy, rhs.derivatives[direction])
                + apply_op(energy_traces[direction], rhs.corrections[direction])
            )
        result[[0, 4]] = scalar_tendency
        momentum = slice(1, 3 if explicit else 4)
        result[momentum] = -torch.stack(pressure_gradient) / base_state[0] - torch.stack(energy_gradient)
        pressure_acceleration = self.pressure_acceleration[:2] if explicit else self.pressure_acceleration
        result[momentum] -= pressure_acceleration * (delta_state[0] / base_state[0])
        result[1:4] += rhs.relative_vorticity_tendency(
            delta_state[1:4], self.contravariant_velocity, tuple(velocity_traces), explicit=explicit
        )
        base_vorticity = antisymmetric_velocity_contraction(self.base_relative_curl, delta_contravariant_velocity)
        frame_vorticity = antisymmetric_velocity_contraction(fields.frame_vorticity, delta_contravariant_velocity)
        if explicit:
            base_vorticity[2] = -(delta_contravariant_velocity[:2] * self.horizontal_vertical_velocity_gradient).sum(
                dim=0
            )
            frame_vorticity[2] = -(delta_contravariant_velocity[:2] * fields.horizontal_frame_gradient).sum(dim=0)
        result[1:4] += base_vorticity
        result[1:4] += frame_vorticity
        if rhs.damping is not None:
            result[1:4] -= rhs.damping[0] * delta_state[1:4]
        return result.reshape(self.shape).flatten()
