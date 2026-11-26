"""Euler RHS in Lamb form."""

import math
from copy import copy
from functools import cache

import numpy
import torch
from mpi4py import MPI
from torch import Tensor

from ..common.definitions import heat_capacity_ratio
from ..common.matmul import DFRDerivative, apply_op
from ..common.tensor_math import antisymmetric_velocity_contraction, relative_covariant_curl, spatial_matrix_vector
from ..init.dcmip import dcmip_schar_damping_coeffs
from ..pde.euler_interfaces import (
    EulerInterfaceGeometry,
    acoustic_interface,
    interface_slices,
    interface_trace,
    pressure,
)


def _reflect_vertical_walls(vertical: Tensor, width: int, bottom_coeff: Tensor, top_coeff: Tensor) -> None:
    """Fill wall ghost faces by reflecting the normal velocity."""
    bottom = vertical[..., 1, :, :, :width]
    bottom_v3 = bottom[3] - (bottom_coeff[0] * bottom[1] + bottom_coeff[1] * bottom[2] + bottom_coeff[2] * bottom[3])
    vertical[..., 0, :, :, width:] = torch.cat((bottom[:3], bottom_v3.unsqueeze(0), bottom[4:]), dim=0)
    top = vertical[..., -2, :, :, width:]
    top_v3 = top[3] - (top_coeff[0] * top[1] + top_coeff[1] * top[2] + top_coeff[2] * top[3])
    vertical[..., -1, :, :, :width] = torch.cat((top[:3], top_v3.unsqueeze(0), top[4:]), dim=0)


@cache
def _compiled_reflect_vertical_walls():
    return torch.compile(
        _reflect_vertical_walls,
        fullgraph=False,
        dynamic=False,
        options={
            "compile_threads": 1,
            "emulate_precision_casts": True,
            "eager_numerics.division_rounding": True,
        },
    )


class RhsEuler:
    def __init__(self, geometry, operators, metric, process_topology, config):
        self.geom = geometry
        self.ops = operators
        self.metric = metric
        self.ptopo = process_topology
        self.config = config
        self.scalar_indices = [0, 4]
        self.log_scalar_indices = [0, 4]
        if config.case_number in (41, 410, 411, 412, 413):
            self.scalar_indices += [5, 6]
            self.log_scalar_indices += [5]
        self.rho_theta_position = self.scalar_indices.index(4)
        # Reuse device indices instead of uploading Python lists on each action.
        self.scalar_index = torch.tensor(self.scalar_indices, dtype=torch.long, device=metric.volume.height.device)
        self.log_scalar_index = torch.tensor(
            self.log_scalar_indices, dtype=torch.long, device=metric.volume.height.device
        )
        self.context = geometry.context
        self.periodic_vertical = config.vertical_boundary == "periodic"
        if not math.isfinite(config.riemann_mach_cutoff) or not 0 < config.riemann_mach_cutoff <= 1:
            raise ValueError("riemann_mach_cutoff must be finite and in (0, 1]")
        self.extrapolations = (operators.extrap_x, operators.extrap_y, operators.extrap_z)
        self.derivatives = (operators.derivative_x, operators.derivative_y, operators.derivative_z)
        self.corrections = (operators.correction_WE, operators.correction_SN, operators.correction_DU)
        self.mid_faces = (
            (..., slice(1, -1), slice(None)),
            (..., slice(1, -1), slice(None), slice(None)),
            (..., slice(1, -1), slice(None), slice(None), slice(None)),
        )
        self.damping = None
        if config.case_number in (21, 22):
            rate, *reference_contravariant = dcmip_schar_damping_coeffs(
                metric, geometry, shear=config.case_number == 22
            )
            self.damping = (rate, metric.volume.lower_velocity(torch.stack(reference_contravariant)))
        self.diagnostics = {}
        self._interface_geometry = None
        self._dfr_derivatives = None
        self._vertical_wall_coeffs = None

    def wave_rate(self, state: Tensor) -> float:
        """Estimate the DFR wave rate in reference-element coordinates for RK startup."""
        rates = [0.0] * 3
        samples = [(state, self.metric.volume.G_contra[1:4, 1:4])]
        samples.extend(
            (trace, fields.G_contra[1:4, 1:4][interior])
            for trace, fields, interior in zip(self._extrapolate(state), self.metric.faces, self.mid_faces)
        )
        for values, inverse in samples:
            sound = (heat_capacity_ratio * pressure(values[4]) / values[0]).sqrt()
            velocity = spatial_matrix_vector(inverse, values[1:4])
            for direction in range(3):
                speed = velocity[direction].abs() + sound * inverse[direction, direction].sqrt()
                rates[direction] = max(rates[direction], speed.max().item())
        damping = 0.0 if self.damping is None else self.damping[0].max().item()
        maxima = numpy.array([*rates, damping], dtype=numpy.float64)
        self.context.comm.Allreduce(MPI.IN_PLACE, maxima, op=MPI.MAX)
        derivative = self.ops.diff_solpt
        correction = self.ops.correction
        extrapolation = torch.stack((self.ops.extrap_west, self.ops.extrap_east))
        # The metric already includes reference-element scaling.
        stencil_norm = (derivative.abs().sum(1) + correction.abs() @ extrapolation.abs().sum(1)).max().item()
        return float(stencil_norm * maxima[:3].sum() + maxima[3])

    def dfr_derivatives(self, action):
        """Cache reference-element stencils independently of the evolving state."""
        if self._dfr_derivatives is None:
            self._dfr_derivatives = tuple(
                DFRDerivative(derivative, correction, action=action)
                for derivative, correction in zip(self.derivatives, self.corrections)
            )
        return self._dfr_derivatives

    def invalidate_geometry(self):
        """Release cached fields before rebuilding terrain on the shared GPU."""
        self._interface_geometry = None
        self._vertical_wall_coeffs = None
        if hasattr(self, "_double_rhs"):
            del self._double_rhs
        self.damping = None

    def refresh_geometry(self):
        """Refresh the operator after a frozen-grid terrain update."""
        self.invalidate_geometry()
        if self.config.case_number in (21, 22):
            rate, *reference_contravariant = dcmip_schar_damping_coeffs(
                self.metric, self.geom, shear=self.config.case_number == 22
            )
            self.damping = (rate, self.metric.volume.lower_velocity(torch.stack(reference_contravariant)))

    def jacobian_rhs(self, state: Tensor) -> Tensor:
        """Evaluate finite-difference probes in float64 to reduce cancellation."""
        return self.double_precision_rhs()(state.to(torch.float64))

    def double_precision_rhs(self):
        """Return the float64 operator for implicit solves and Jacobian probes."""
        if not hasattr(self, "_double_rhs"):
            rhs = copy(self)
            rhs.metric = copy(self.metric)
            collections = []
            for fields in (self.metric.volume, *self.metric.faces):
                promoted = copy(fields)
                for name in ("G_cov", "G_contra", "volume_jacobian", "geopotential"):
                    setattr(promoted, name, getattr(fields, name).to(torch.float64))
                promoted._runtime_frame_vorticity = fields.frame_vorticity.to(torch.float64)
                promoted._runtime_vertical_frame_gradient = fields.vertical_frame_gradient.to(torch.float64)
                promoted._runtime_horizontal_frame_gradient = fields.horizontal_frame_gradient.to(torch.float64)
                collections.append(promoted)
            rhs.metric.volume = collections[0]
            rhs.metric.faces = tuple(collections[1:])
            for name in ("extrapolations", "derivatives", "corrections"):
                setattr(rhs, name, tuple(operator.to(torch.float64) for operator in getattr(self, name)))
            if self.damping is not None:
                rhs.damping = tuple(value.to(torch.float64) for value in self.damping)
            rhs.diagnostics = {}
            rhs._dfr_derivatives = None
            rhs._interface_geometry = None
            rhs._vertical_wall_coeffs = None
            self._double_rhs = rhs
        return self._double_rhs

    def _extrapolate(self, state: Tensor, contravariant: Tensor | None = None) -> tuple[Tensor, ...]:
        if contravariant is None:
            contravariant = self.metric.volume.raise_velocity(state[1:4])
        log_scalars = torch.log(state[self.log_scalar_index])
        traces = tuple(self._extrapolate_direction(state, operator, log_scalars) for operator in self.extrapolations)
        self._match_pressure_work(contravariant, traces)
        return traces

    def _match_pressure_work(self, contravariant: Tensor, traces: tuple[Tensor, ...], directions=(0, 1, 2)):
        """Match each face velocity to the discrete pressure work.

        Pressure lifting works against R(J u^m) / J_f, where R extrapolates to the face.
        Matching this contravariant component makes the velocity-jump pressure penalty dissipative.
        Shifting only v_m preserves tangential velocity. The correction is linear in velocity,
        so the Jacobian applies the same operation to perturbations.
        """
        volume_jacobian = self.metric.volume.volume_jacobian
        for direction, trace in zip(directions, traces):
            fields, interior = self.metric.faces[direction], self.mid_faces[direction]
            work_velocity = apply_op(volume_jacobian * contravariant[direction], self.extrapolations[direction])
            work_velocity = work_velocity / fields.volume_jacobian[interior]
            inverse_row = fields.G_contra[direction + 1, 1:4][interior]
            traced_velocity = (inverse_row * trace[1:4]).sum(dim=0)
            trace[1 + direction] += (work_velocity - traced_velocity) / inverse_row[direction]

    def _extrapolate_direction(self, state, operator, log_scalars=None):
        trace = state.new_empty(state.shape[:-1] + (operator.shape[-1],))
        trace[1:4] = apply_op(state[1:4], operator)
        if log_scalars is None:
            log_scalars = torch.log(state[self.log_scalar_index])
        trace[self.log_scalar_index] = torch.exp(apply_op(log_scalars, operator))
        if state.shape[0] > 6:
            # Potential vorticity can vanish, so its mixing ratio is extrapolated without a logarithm.
            trace[6:] = trace[0] * apply_op(state[6:] / state[0], operator)
        return trace

    def _faces_with_neighbours(self, traces: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        width = self.geom.itf_size
        # Payload slots 1:4 carry relative covariant velocity in the reference basis.
        request = self.ptopo.start_exchange_euler_3d(
            traces[1][..., 0, :, :width],
            traces[1][..., -1, :, width:],
            traces[0][..., 0, :width],
            traces[0][..., -1, width:],
            self.geom.boundary_sn_new,
            self.geom.boundary_we_new,
            flip_dim=(-3, -1),
            covariant=True,
        )
        full_faces = []
        for fields, trace, interior in zip(self.metric.faces, traces, self.mid_faces):
            # Unused outer halo slots are never read by interface_slices.
            face_state = trace.new_empty((trace.shape[0],) + fields.height.shape)
            face_state[interior] = trace
            full_faces.append(face_state)
        south, north, west, east = request.wait()
        neighbours = (
            (0, (..., 0, slice(width, None)), west),
            (0, (..., -1, slice(None, width)), east),
            (1, (..., 0, slice(None), slice(width, None)), south),
            (1, (..., -1, slice(None), slice(None, width)), north),
        )
        for direction, face, received in neighbours:
            full_faces[direction][face] = received

        self._fill_vertical_boundaries(full_faces[2])
        return tuple(full_faces)

    def _vertical_faces(self, trace: Tensor) -> Tensor:
        face_state = trace.new_empty((trace.shape[0],) + self.metric.faces[2].height.shape)
        face_state[self.mid_faces[2]] = trace
        self._fill_vertical_boundaries(face_state)
        return face_state

    def _get_vertical_wall_coeffs(self) -> tuple[Tensor, Tensor]:
        """Cache the metric factors that reverse the normal wind at each wall."""
        if self._vertical_wall_coeffs is None:
            width = self.geom.itf_size
            coeffs = []
            for ghost_face in (
                (..., 0, slice(None), slice(None), slice(width, None)),
                (..., -1, slice(None), slice(None), slice(None, width)),
            ):
                inverse_spatial = self.metric.faces[2].G_contra[1:4, 1:4][ghost_face]
                coeffs.append(2 * inverse_spatial[2] / inverse_spatial[2, 2])
            self._vertical_wall_coeffs = tuple(coeffs)
        return self._vertical_wall_coeffs

    def _fill_vertical_boundaries(self, vertical: Tensor):
        width = self.geom.itf_size
        if self.periodic_vertical:
            vertical[..., 0, :, :, width:] = vertical[..., -2, :, :, width:]
            vertical[..., -1, :, :, :width] = vertical[..., 1, :, :, :width]
            return
        bottom_coeff, top_coeff = self._get_vertical_wall_coeffs()
        if vertical.is_cuda:
            _compiled_reflect_vertical_walls()(vertical, width, bottom_coeff, top_coeff)
        else:
            _reflect_vertical_walls(vertical, width, bottom_coeff, top_coeff)

    def _static_interfaces(self) -> tuple[EulerInterfaceGeometry, ...]:
        """Cache face coefficients for this static geometry and precision."""
        if self._interface_geometry is None:
            interfaces = []
            for direction, fields in enumerate(self.metric.faces):
                left, right = interface_slices(direction, self.geom.num_solpts)
                inverse_spatial = fields.G_contra[1:4, 1:4][left]
                geopotential = fields.geopotential[left]
                jacobian = fields.volume_jacobian[left]
                normal_scale = torch.sqrt(inverse_spatial[direction, direction])
                normal_row = inverse_spatial[direction] / normal_scale
                wall = None
                if direction == 2 and not self.periodic_vertical:
                    wall = torch.zeros_like(geopotential, dtype=torch.bool)
                    wall[0], wall[-1] = True, True
                elif (
                    direction != 2
                    and getattr(self.geom, "is_y_invariant_slab", False)
                    and self.config.lateral_boundary == "wall"
                ):
                    wall = torch.zeros_like(geopotential, dtype=torch.bool)
                    if direction == 0:
                        wall[:, :, 0], wall[:, :, -1] = True, True
                    else:
                        wall[:, 0], wall[:, -1] = True, True
                interfaces.append(
                    EulerInterfaceGeometry(
                        left=left,
                        right=right,
                        inverse_spatial=inverse_spatial,
                        geopotential=geopotential,
                        jacobian=jacobian,
                        normal_scale=normal_scale,
                        normal_row=normal_row,
                        wall=wall,
                    )
                )
            self._interface_geometry = tuple(interfaces)
        return self._interface_geometry

    def _interface_data(self, full_faces: tuple[Tensor, ...], directions=(0, 1, 2)):
        scalar_fluxes, pressure_traces, energy_traces = [], [], []
        jumps, normal_velocities, velocity_traces, theta_traces = [], [], [], []
        for direction, state in zip(directions, full_faces):
            geometry = self._static_interfaces()[direction]
            left, right = geometry.left, geometry.right
            normal_velocity, common_pressure, common_energy, common_covariant_velocity = acoustic_interface(
                state[left],
                state[right],
                geometry.inverse_spatial,
                direction,
                geometry.geopotential,
                wall=geometry.wall,
                return_velocity=True,
                normal_geometry=(geometry.normal_scale, geometry.normal_row),
                mach_cutoff=self.config.riemann_mach_cutoff,
            )
            scalar_state = torch.where(
                normal_velocity[None] >= 0, state[left][self.scalar_index], state[right][self.scalar_index]
            )
            common_flux = geometry.jacobian * normal_velocity * scalar_state
            scalar_fluxes.append(interface_trace(common_flux, direction))
            theta_traces.append(interface_trace(scalar_state[self.rho_theta_position] / scalar_state[0], direction))
            pressure_traces.append(interface_trace(common_pressure, direction))
            energy_traces.append(interface_trace(common_energy, direction))
            velocity_traces.append(interface_trace(common_covariant_velocity, direction))
            jumps.append(pressure(state[right][4]) - pressure(state[left][4]))
            normal_velocities.append(normal_velocity / geometry.normal_scale)
        self.diagnostics["interface_pressure_jumps"] = tuple(jumps)
        self.diagnostics["interface_normal_winds"] = tuple(normal_velocities)
        return scalar_fluxes, pressure_traces, energy_traces, velocity_traces, theta_traces

    def _vertical_rho_theta_tendency(self, state, vertical_velocity, mass_tendency, theta_trace, mass_flux_trace):
        """Advect theta within each element and apply a face-mass-flux penalty.

        Direct theta advection avoids subtracting scalar flux divergences via a discrete product rule.
        Weighting the upwind correction by the common mass flux makes it vanish at zero face wind.
        """
        theta = state[4] / state[0]
        element_trace = apply_op(theta, self.extrapolations[2])
        theta_gradient = apply_op(theta, self.derivatives[2]) + apply_op(element_trace, self.corrections[2])
        penalty = apply_op(mass_flux_trace * (theta_trace - element_trace), self.corrections[2])
        volume_jacobian = self.metric.volume.volume_jacobian
        return theta * mass_tendency - state[0] * vertical_velocity * theta_gradient - penalty / volume_jacobian

    def _curl_derivatives(self, covariant: Tensor, velocity_traces: tuple[Tensor, ...], derivative_operators=None):
        """Batch only the two off-diagonal velocity derivatives needed in each direction."""
        if velocity_traces is None:
            raise ValueError("DFR curl requires common velocity traces from the Riemann solver")
        selections = (slice(1, 3), slice(0, 3, 2), slice(0, 2))
        derivatives = []
        for direction, selection in enumerate(selections):
            if derivative_operators is None:
                value = apply_op(covariant[selection], self.derivatives[direction])
                value += apply_op(velocity_traces[direction][selection], self.corrections[direction])
            else:
                value = derivative_operators[direction](covariant[selection], velocity_traces[direction][selection])
            derivatives.append(value)
        return (
            derivatives[0][0],
            derivatives[0][1],
            derivatives[1][0],
            derivatives[1][1],
            derivatives[2][0],
            derivatives[2][1],
        )

    def relative_vorticity_tendency(
        self, covariant: Tensor, contravariant: Tensor, velocity_traces: tuple[Tensor, ...], *, explicit=False
    ) -> Tensor:
        """Evaluate u^j (v_{j,i} - v_{i,j}) or its complementary terms.

        The DFR endpoint values are common velocities from the Riemann solver.
        """
        if explicit:
            dv2_dx1, dv3_dx1, dv1_dx2, dv3_dx2, dv1_dx3, dv2_dx3 = self._curl_derivatives(covariant, velocity_traces)
            curl_12 = dv2_dx1 - dv1_dx2
            return torch.stack(
                (
                    contravariant[1] * curl_12 + contravariant[2] * (dv3_dx1 - dv1_dx3),
                    -contravariant[0] * curl_12 + contravariant[2] * (dv3_dx2 - dv2_dx3),
                    -contravariant[0] * dv3_dx1 - contravariant[1] * dv3_dx2,
                )
            )
        antisymmetric = self.relative_covariant_curl(covariant, velocity_traces)
        return antisymmetric_velocity_contraction(antisymmetric, contravariant)

    def relative_covariant_curl(self, covariant: Tensor, velocity_traces: tuple[Tensor, ...]) -> Tensor:
        """Return v_{j,i} - v_{i,j} with leading spatial axes (i, j)."""
        return relative_covariant_curl(self._curl_derivatives(covariant, velocity_traces))

    def linearize(self, state: Tensor):
        from ..jacobian.euler import EulerLinearization

        return EulerLinearization(self, state)

    def implicit(self, state: Tensor) -> Tensor:
        """Return the column-local transport, pressure, energy, and vertical curl terms."""
        given_shape = state.shape
        state = state.reshape((-1,) + self.metric.volume.height.shape)
        fields = self.metric.volume
        contravariant_velocity = fields.raise_velocity(state[1:4])
        trace = self._extrapolate_direction(state, self.extrapolations[2])
        self._match_pressure_work(contravariant_velocity, (trace,), (2,))
        faces = self._vertical_faces(trace)
        scalar_traces, pressure_traces, energy_traces, velocity_traces, theta_traces = self._interface_data(
            (faces,), (2,)
        )

        def derivative(value, common):
            return apply_op(value, self.derivatives[2]) + apply_op(common, self.corrections[2])

        result = torch.zeros_like(state)
        flux = fields.volume_jacobian * contravariant_velocity[2] * state[self.scalar_index]
        scalar_tendency = -derivative(flux, scalar_traces[0]) / fields.volume_jacobian
        scalar_tendency[self.rho_theta_position] = self._vertical_rho_theta_tendency(
            state, contravariant_velocity[2], scalar_tendency[0], theta_traces[0], scalar_traces[0][0]
        )
        result[self.scalar_index] = scalar_tendency
        energy = (state[1:4] * contravariant_velocity).sum(dim=0) / 2 + fields.geopotential
        result[3] = -derivative(pressure(state[4]), pressure_traces[0]) / state[0]
        result[3] -= derivative(energy, energy_traces[0])
        result[3] += (contravariant_velocity[:2] * derivative(state[1:3], velocity_traces[0][:2])).sum(dim=0)
        result[3] += (contravariant_velocity[:2] * fields.vertical_frame_gradient).sum(dim=0)
        return result.reshape(given_shape)

    def explicit(self, state: Tensor) -> Tensor:
        """Return horizontal transport and gradients, complementary curl terms, and damping."""
        return self(state, explicit=True)

    def linearize_partitions(self, state: Tensor):
        from ..jacobian.euler_vertical import EulerPartitionLinearization

        return EulerPartitionLinearization(self, state)

    def __call__(self, state: Tensor, *, explicit=False) -> Tensor:
        given_shape = state.shape
        state = state.reshape((-1,) + self.metric.volume.height.shape)
        fields = self.metric.volume
        contravariant_velocity = fields.raise_velocity(state[1:4])
        traces = self._extrapolate(state, contravariant_velocity)
        full_faces = self._faces_with_neighbours(traces)
        scalar_fluxes, pressure_traces, energy_traces, velocity_traces, theta_traces = self._interface_data(full_faces)
        kinetic_energy = (state[1:4] * contravariant_velocity).sum(dim=0) / 2
        thermodynamic_pressure = pressure(state[4])
        mechanical_energy = kinetic_energy + fields.geopotential
        result = torch.zeros_like(state)
        scalars = state[self.scalar_index]
        scalar_tendency = torch.zeros_like(scalars)
        pressure_gradient, energy_gradient = [], []
        for direction in range(2 if explicit else 3):
            volume_flux = fields.volume_jacobian * contravariant_velocity[direction] * scalars
            divergence = apply_op(volume_flux, self.derivatives[direction]) + apply_op(
                scalar_fluxes[direction], self.corrections[direction]
            )
            scalar_change = -divergence / fields.volume_jacobian
            if direction == 2:
                scalar_change[self.rho_theta_position] = self._vertical_rho_theta_tendency(
                    state, contravariant_velocity[2], scalar_change[0], theta_traces[2], scalar_fluxes[2][0]
                )
            scalar_tendency += scalar_change
            pressure_gradient.append(
                apply_op(thermodynamic_pressure, self.derivatives[direction])
                + apply_op(pressure_traces[direction], self.corrections[direction])
            )
            energy_gradient.append(
                apply_op(mechanical_energy, self.derivatives[direction])
                + apply_op(energy_traces[direction], self.corrections[direction])
            )
        if explicit:
            pressure_gradient.append(torch.zeros_like(state[0]))
            energy_gradient.append(torch.zeros_like(state[0]))
        pressure_acceleration = -torch.stack(pressure_gradient) / state[0]
        energy_acceleration = -torch.stack(energy_gradient)
        relative_vorticity = self.relative_vorticity_tendency(
            state[1:4], contravariant_velocity, velocity_traces, explicit=explicit
        )
        frame_vorticity = antisymmetric_velocity_contraction(fields.frame_vorticity, contravariant_velocity)
        if explicit:
            frame_vorticity[2] = -(contravariant_velocity[:2] * fields.horizontal_frame_gradient).sum(dim=0)
        result[self.scalar_index] = scalar_tendency
        result[1:4] = pressure_acceleration + energy_acceleration + relative_vorticity + frame_vorticity
        if self.damping is not None:
            rate, reference = self.damping
            result[1:4] -= rate * (state[1:4] - reference)
        self.diagnostics.update(
            pressure_acceleration=pressure_acceleration,
            energy_acceleration=energy_acceleration,
            relative_vorticity=relative_vorticity,
            frame_vorticity=frame_vorticity,
        )
        return result.reshape(given_shape)
