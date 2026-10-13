"""DFR tracer transport with analytic DCMIP winds and an internal time coordinate."""

import math
from copy import copy

import torch

from ..common.definitions import Rd, gravity, p0
from ..common.matmul import apply_op
from ..init.dcmip import dcmip_prescribed_rho_theta
from ..pde.euler_interfaces import interface_slices


class RhsPrescribedTransport:
    """Evolve tracer densities and t/period, with meteorology prescribed analytically.

    The last component is a local clock, allowing exponential stages to differentiate
    the time-dependent wind. The adapter restores rho, v_i and rho*theta for output.
    """

    def __init__(self, geometry, operators, metric, config, num_tracers):
        self.geom = geometry
        self.ops = operators
        self.metric = metric
        self.config = config
        self.num_tracers = num_tracers
        self.shape = (num_tracers + 1,) + geometry.grid_shape_3d_new
        self.period = 86400.0 if config.case_number == 12 else 12 * 86400.0
        self.time = 0.0
        self.mid_faces = (
            (..., slice(1, -1), slice(None)),
            (..., slice(1, -1), slice(None), slice(None)),
            (..., slice(1, -1), slice(None), slice(None), slice(None)),
        )

    def wind(self, fields, clock):
        """Return contravariant wind at analytic volume or face positions; clock=t/period."""
        clock = torch.as_tensor(clock, dtype=fields.height.dtype, device=fields.height.device)
        radial = fields.radial
        longitude = torch.atan2(radial[1], radial[0])
        latitude = torch.asin(torch.clamp(radial[2], -1, 1))
        east = torch.stack((-torch.sin(longitude), torch.cos(longitude), torch.zeros_like(longitude)))
        north = torch.linalg.cross(radial, east, dim=0)
        height = fields.height
        pressure = p0 * torch.exp(-gravity * height / (Rd * 300.0))
        if self.config.case_number == 11:
            eastward_wind, northward_wind, vertical_wind = self._deformational_wind(
                longitude, latitude, pressure, clock
            )
        elif self.config.case_number == 12:
            eastward_wind, northward_wind, vertical_wind = self._hadley_wind(latitude, height, pressure, clock)
        else:
            eastward_wind, northward_wind, vertical_wind = self._solid_body_wind(longitude, latitude)
        physical = east * eastward_wind + north * northward_wind + radial * vertical_wind
        return fields.raise_velocity(fields.covariant_from_physical(physical))

    def _deformational_wind(self, longitude, latitude, pressure, clock):
        radius = self.geom.earth_radius
        phase = longitude - 2 * math.pi * clock
        top_pressure = p0 * math.exp(-gravity * self.geom.ztop / (Rd * 300.0))
        pressure_width = 0.2 * top_pressure
        lower = torch.exp((pressure - p0) / pressure_width)
        upper = torch.exp((top_pressure - pressure) / pressure_width)
        vertical_shape = 1 + math.exp((top_pressure - p0) / pressure_width) - lower - upper
        pressure_velocity_amplitude = 23000 * math.pi / self.period
        deformation_speed = 10 * radius / self.period
        eastward_wind = (
            deformation_speed * torch.sin(phase) ** 2 * torch.sin(2 * latitude) * torch.cos(math.pi * clock)
            + 2 * math.pi * radius / self.period * torch.cos(latitude)
            + pressure_velocity_amplitude
            * radius
            / pressure_width
            * torch.cos(phase)
            * torch.cos(latitude) ** 2
            * torch.cos(2 * math.pi * clock)
            * (upper - lower)
        )
        northward_wind = deformation_speed * torch.sin(2 * phase) * torch.cos(latitude) * torch.cos(math.pi * clock)
        vertical_wind = (
            -Rd
            * 300
            / (gravity * pressure)
            * pressure_velocity_amplitude
            * torch.sin(phase)
            * torch.cos(latitude)
            * torch.cos(2 * math.pi * clock)
            * vertical_shape
        )
        return eastward_wind, northward_wind, vertical_wind

    def _hadley_wind(self, latitude, height, pressure, clock):
        radius = self.geom.earth_radius
        density_ratio = p0 / pressure
        eastward_wind = 40 * torch.cos(latitude)
        northward_wind = (
            -density_ratio
            * radius
            * 0.15
            * math.pi
            / (5 * self.geom.ztop)
            * torch.cos(latitude)
            * torch.sin(5 * latitude)
            * torch.cos(math.pi * height / self.geom.ztop)
            * torch.cos(math.pi * clock)
        )
        vertical_wind = (
            density_ratio
            * 0.15
            / 5
            * (-2 * torch.sin(5 * latitude) * torch.sin(latitude) + 5 * torch.cos(latitude) * torch.cos(5 * latitude))
            * torch.sin(math.pi * height / self.geom.ztop)
            * torch.cos(math.pi * clock)
        )
        return eastward_wind, northward_wind, vertical_wind

    def _solid_body_wind(self, longitude, latitude):
        radius = self.geom.earth_radius
        angle = math.pi / 6
        speed = 2 * math.pi * radius / self.period
        eastward_wind = speed * (
            torch.cos(latitude) * math.cos(angle) + torch.sin(latitude) * torch.cos(longitude) * math.sin(angle)
        )
        northward_wind = -speed * torch.sin(longitude) * math.sin(angle)
        vertical_wind = torch.zeros_like(eastward_wind)
        return eastward_wind, northward_wind, vertical_wind

    def restore(self, state, time):
        prescribed = state.clone()
        rho, rho_theta = dcmip_prescribed_rho_theta(self.geom)
        prescribed[0], prescribed[4] = rho, rho_theta
        prescribed[1:4] = self.metric.volume.lower_velocity(self.wind(self.metric.volume, time / self.period))
        return prescribed

    def _traces(self, state):
        traces = [apply_op(state, operator) for operator in (self.ops.extrap_x, self.ops.extrap_y, self.ops.extrap_z)]
        width = self.geom.itf_size
        request = self.geom.process_topology.start_exchange_scalars(
            traces[1][..., 0, :, :width],
            traces[1][..., -1, :, width:],
            traces[0][..., 0, :width],
            traces[0][..., -1, width:],
            boundary_shape=self.geom.boundary_sn_new.shape,
            flip_dim=(-3, -1),
        )
        faces = []
        for fields, trace, interior in zip(self.metric.faces, traces, self.mid_faces):
            face = state.new_zeros((state.shape[0],) + fields.height.shape)
            face[interior] = trace
            faces.append(face)
        south, north, west, east = request.wait()
        faces[0][..., 0, width:], faces[0][..., -1, :width] = west, east
        faces[1][..., 0, :, width:], faces[1][..., -1, :, :width] = south, north
        faces[2][..., 0, :, :, width:] = faces[2][..., 1, :, :, :width]
        faces[2][..., -1, :, :, :width] = faces[2][..., -2, :, :, width:]
        return faces

    def _velocities(self, state, faces, derivative=False):
        values = []
        for fields, clock in zip((self.metric.volume, *self.metric.faces), (state[-1], *(face[-1] for face in faces))):
            if derivative:
                # Differentiate only the analytic wind formula; MPI transport remains explicit.
                with torch.inference_mode(False):
                    value, tangent = torch.func.jvp(
                        lambda time, fields=fields: self.wind(fields, time), (clock.clone(),), (torch.ones_like(clock).clone(),)
                    )
                values.append((value, tangent))
            else:
                values.append(self.wind(fields, clock))
        return values

    def _divergence(self, tracers, faces, velocities, base_faces=None, velocity_derivatives=None):
        jacobian = self.metric.volume.volume_jacobian
        result = torch.zeros_like(tracers)
        for direction, (fields, face, interior, derivative, correction) in enumerate(
            zip(
                self.metric.faces,
                faces,
                self.mid_faces,
                (self.ops.derivative_x, self.ops.derivative_y, self.ops.derivative_z),
                (self.ops.correction_WE, self.ops.correction_SN, self.ops.correction_DU),
            )
        ):
            left, right = interface_slices(direction, self.geom.num_solpts)
            contravariant_normal_velocity = velocities[direction + 1][direction][left]
            if direction == 2:
                contravariant_normal_velocity = contravariant_normal_velocity.clone()
                contravariant_normal_velocity[0], contravariant_normal_velocity[-1] = 0, 0
            upwind = contravariant_normal_velocity >= 0
            common_tracer_flux = (
                fields.volume_jacobian[left]
                * contravariant_normal_velocity
                * torch.where(upwind, face[left][:-1], face[right][:-1])
            )
            if base_faces is not None:
                base_face_state = base_faces[direction]
                delta_contravariant_normal_velocity = (
                    velocity_derivatives[direction + 1][direction][left] * face[left][-1]
                )
                if direction == 2:
                    delta_contravariant_normal_velocity = delta_contravariant_normal_velocity.clone()
                    delta_contravariant_normal_velocity[0], delta_contravariant_normal_velocity[-1] = 0, 0
                common_tracer_flux += (
                    fields.volume_jacobian[left]
                    * delta_contravariant_normal_velocity
                    * torch.where(upwind, base_face_state[left][:-1], base_face_state[right][:-1])
                )
            flux_faces = torch.zeros_like(face[:-1])
            flux_faces[left], flux_faces[right] = common_tracer_flux, common_tracer_flux
            flux = jacobian * velocities[0][direction] * tracers
            result -= (apply_op(flux, derivative) + apply_op(flux_faces[interior], correction)) / jacobian
        return result

    def __call__(self, state):
        if state.numel() != math.prod(self.shape):
            full = state.reshape((-1,) + self.shape[1:])
            augmented = torch.cat((full[5:], torch.full_like(full[:1], self.time / self.period)))
            result = torch.zeros_like(full)
            result[5:] = self(augmented)[:-1]
            return result.reshape_as(state)
        augmented = state.reshape(self.shape)
        faces = self._traces(augmented)
        velocities = self._velocities(augmented, faces)
        result = torch.zeros_like(augmented)
        result[:-1] = self._divergence(augmented[:-1], faces, velocities)
        result[-1] = 1 / self.period
        return result.reshape_as(state)

    def linearize(self, state):
        augmented = state.reshape(self.shape)
        faces = self._traces(augmented)
        pairs = self._velocities(augmented, faces, derivative=True)
        velocities, derivatives = zip(*pairs)

        def action(vector):
            delta = vector.reshape(self.shape)
            delta_faces = self._traces(delta)
            result = torch.zeros_like(delta)
            result[:-1] = self._divergence(delta[:-1], delta_faces, velocities, faces, derivatives)
            # The volume velocity depends on the local clock as well as the face clock.
            clock_flux_perturbation = derivatives[0][:, None] * delta[-1] * augmented[:-1][None]
            jacobian = self.metric.volume.volume_jacobian
            for direction, operator in enumerate((self.ops.derivative_x, self.ops.derivative_y, self.ops.derivative_z)):
                result[:-1] -= apply_op(jacobian * clock_flux_perturbation[direction], operator) / jacobian
            return result.reshape_as(vector)

        return action

    def double_precision_rhs(self):
        if not hasattr(self, "_double_rhs"):
            rhs = copy(self)
            rhs.metric = copy(self.metric)
            rhs.metric.volume = copy(self.metric.volume)
            rhs.metric.faces = tuple(copy(fields) for fields in self.metric.faces)
            for fields in (rhs.metric.volume, *rhs.metric.faces):
                for name in ("G_cov", "G_contra", "volume_jacobian", "physical_basis", "radial", "height"):
                    setattr(fields, name, getattr(fields, name).to(torch.float64))
            rhs.ops = copy(self.ops)
            for name in (
                "extrap_x",
                "extrap_y",
                "extrap_z",
                "derivative_x",
                "derivative_y",
                "derivative_z",
                "correction_WE",
                "correction_SN",
                "correction_DU",
            ):
                setattr(rhs.ops, name, getattr(self.ops, name).to(torch.float64))
            self._double_rhs = rhs
        self._double_rhs.time = self.time
        return self._double_rhs

    def jacobian_rhs(self, state):
        return self.double_precision_rhs()(state.to(torch.float64))
