"""Euler RHS in Lamb form. The state is (rho, v_1, v_2, v_3, rho*theta), where v_i=G_ij*u^j."""

import math
from copy import copy

import torch
from torch import Tensor

from ..common.matmul import DFRDerivative, apply_op
from ..common.tensor_math import antisymmetric_velocity_contraction, relative_covariant_curl
from ..init.dcmip import dcmip_schar_damping_coeffs
from ..pde.euler_interfaces import EulerInterfaceGeometry, acoustic_interface, interface_slices, pressure


class RhsEuler:
    def __init__(self, geometry, operators, metric, process_topology, config):
        self.geom = geometry
        self.ops = operators
        self.metric = metric
        self.ptopo = process_topology
        self.config = config
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
            self._double_rhs = rhs
        return self._double_rhs

    def _extrapolate(self, state: Tensor) -> tuple[Tensor, ...]:
        log_scalars = torch.log(state[[0, 4]])
        traces = []
        for operator in self.extrapolations:
            trace = state.new_empty(state.shape[:-1] + (operator.shape[-1],))
            trace[1:4] = apply_op(state[1:4], operator)
            trace[[0, 4]] = torch.exp(apply_op(log_scalars, operator))
            traces.append(trace)
        return tuple(traces)

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
            face_state = trace.new_zeros((trace.shape[0],) + fields.height.shape)
            # Only unused outer faces retain these values; used traces are filled below.
            face_state[[0, 4]] = 1
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
        face_state = trace.new_zeros((trace.shape[0],) + self.metric.faces[2].height.shape)
        face_state[[0, 4]] = 1
        face_state[self.mid_faces[2]] = trace
        self._fill_vertical_boundaries(face_state)
        return face_state

    def _fill_vertical_boundaries(self, vertical: Tensor):
        width = self.geom.itf_size
        if self.periodic_vertical:
            vertical[..., 0, :, :, width:] = vertical[..., -2, :, :, width:]
            vertical[..., -1, :, :, :width] = vertical[..., 1, :, :, :width]
            return
        for ghost_face, interior_face in (
            (
                (..., 0, slice(None), slice(None), slice(width, None)),
                (..., 1, slice(None), slice(None), slice(None, width)),
            ),
            (
                (..., -1, slice(None), slice(None), slice(None, width)),
                (..., -2, slice(None), slice(None), slice(width, None)),
            ),
        ):
            vertical[ghost_face] = vertical[interior_face]
            covariant_velocity = vertical[1:4][ghost_face]
            inverse_spatial = self.metric.faces[2].G_contra[1:4, 1:4][ghost_face]
            contravariant_normal_velocity = (inverse_spatial[2] * covariant_velocity).sum(dim=0)
            reflected = covariant_velocity.clone()
            reflected[2] -= 2 * contravariant_normal_velocity / inverse_spatial[2, 2]
            vertical[1:4][ghost_face] = reflected

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
        jumps, normal_velocities, velocity_traces = [], [], []
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
            scalar_flux = state.new_zeros((2,) + state.shape[1:])
            scalar_state = torch.where(normal_velocity[None] >= 0, state[left][[0, 4]], state[right][[0, 4]])
            common_flux = geometry.jacobian * normal_velocity * scalar_state
            scalar_flux[left], scalar_flux[right] = common_flux, common_flux
            numerical_pressure, numerical_energy = torch.zeros_like(state[0]), torch.zeros_like(state[0])
            common_velocity_trace = torch.zeros_like(state[1:4])
            common_velocity_trace[left], common_velocity_trace[right] = (
                common_covariant_velocity,
                common_covariant_velocity,
            )
            velocity_traces.append(common_velocity_trace[self.mid_faces[direction]])
            numerical_pressure[left], numerical_pressure[right] = common_pressure, common_pressure
            numerical_energy[left], numerical_energy[right] = common_energy, common_energy
            scalar_fluxes.append(scalar_flux[self.mid_faces[direction]])
            pressure_traces.append(numerical_pressure[self.mid_faces[direction]])
            energy_traces.append(numerical_energy[self.mid_faces[direction]])
            jumps.append(pressure(state[right][4]) - pressure(state[left][4]))
            normal_velocities.append(normal_velocity / geometry.normal_scale)
        self.diagnostics["interface_pressure_jumps"] = tuple(jumps)
        self.diagnostics["interface_normal_winds"] = tuple(normal_velocities)
        return scalar_fluxes, pressure_traces, energy_traces, velocity_traces

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
        trace = apply_op(state, self.extrapolations[2])
        trace[[0, 4]] = torch.exp(apply_op(torch.log(state[[0, 4]]), self.extrapolations[2]))
        faces = self._vertical_faces(trace)
        scalar_traces, pressure_traces, energy_traces, velocity_traces = self._interface_data((faces,), (2,))
        contravariant_velocity = fields.raise_velocity(state[1:4])

        def derivative(value, common):
            return apply_op(value, self.derivatives[2]) + apply_op(common, self.corrections[2])

        result = torch.zeros_like(state)
        flux = fields.volume_jacobian * contravariant_velocity[2] * state[[0, 4]]
        result[[0, 4]] = -derivative(flux, scalar_traces[0]) / fields.volume_jacobian
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
        traces = self._extrapolate(state)
        full_faces = self._faces_with_neighbours(traces)
        scalar_fluxes, pressure_traces, energy_traces, velocity_traces = self._interface_data(full_faces)
        contravariant_velocity = fields.raise_velocity(state[1:4])
        kinetic_energy = (state[1:4] * contravariant_velocity).sum(dim=0) / 2
        thermodynamic_pressure = pressure(state[4])
        mechanical_energy = kinetic_energy + fields.geopotential
        result = torch.zeros_like(state)
        scalars = state[[0, 4]]
        scalar_tendency = torch.zeros_like(scalars)
        pressure_gradient, energy_gradient = [], []
        for direction in range(2 if explicit else 3):
            volume_flux = fields.volume_jacobian * contravariant_velocity[direction] * scalars
            scalar_tendency -= (
                apply_op(volume_flux, self.derivatives[direction])
                + apply_op(scalar_fluxes[direction], self.corrections[direction])
            ) / fields.volume_jacobian
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
        result[[0, 4]] = scalar_tendency
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
