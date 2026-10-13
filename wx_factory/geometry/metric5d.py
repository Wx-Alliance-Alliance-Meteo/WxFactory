"""Static five-dimensional geometry for coordinate-relative Euler momentum.

Indices are (time, xi1, xi2, xi3, auxiliary). Spatial coordinates xi are the
element coordinates in [-1, 1]; panel coordinates are (x1, x2, eta).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, fields

import torch
from torch import Tensor

from ..common.definitions import gravity
from ..common.tensor_math import spatial_matrix_vector


def panel_frame(longitude: float, latitude: float, angle: float, like: Tensor) -> Tensor:
    """Columns are the panel's Cartesian orthonormal vectors (e1, e2, e0)."""
    east = like.new_tensor([-math.sin(longitude), math.cos(longitude), 0.0])
    north = like.new_tensor(
        [-math.sin(latitude) * math.cos(longitude), -math.sin(latitude) * math.sin(longitude), math.cos(latitude)]
    )
    radial = like.new_tensor(
        [math.cos(latitude) * math.cos(longitude), math.cos(latitude) * math.sin(longitude), math.sin(latitude)]
    )
    return torch.stack(
        (math.cos(angle) * east - math.sin(angle) * north, math.sin(angle) * east + math.cos(angle) * north, radial),
        dim=1,
    )


def unit_sphere_basis(coordinates: Tensor, frame: Tensor) -> tuple[Tensor, Tensor]:
    """Return the radial unit vector and its two panel-angle derivatives."""
    x, y = torch.tan(coordinates[0]), torch.tan(coordinates[1])
    delta = torch.sqrt(1 + x * x + y * y)
    raw = torch.stack((x, y, torch.ones_like(x)))
    local = raw / delta
    derivatives = []
    for axis, tangent in enumerate((x, y)):
        direction = torch.zeros_like(raw)
        direction[axis] = 1
        derivatives.append((1 + tangent * tangent) * (direction / delta - raw * tangent / delta**3))
    return torch.einsum("ab,b...->a...", frame, local), torch.einsum(
        "ab,bi...->ai...", frame, torch.stack(derivatives, dim=1)
    )


def mapping_derivatives(height_map: Callable[[Tensor], Tensor], coordinates: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return the height map, its gradient, and its Hessian at the supplied points.

    Automatic differentiation is used only at geometry setup. The returned
    tensors are detached from the computation graph.
    """
    with torch.inference_mode(False), torch.enable_grad():
        points = coordinates.detach().clone().requires_grad_(True)
        height = height_map(points) + (points * 0).sum(dim=0)
        gradient = torch.autograd.grad(height.sum(), points, create_graph=True)[0]
        rows = []
        for i in range(3):
            # Linear maps have a constant gradient, which still needs a zero Hessian.
            component = gradient[i] + (points * 0).sum(dim=0)
            rows.append(torch.autograd.grad(component.sum(), points, retain_graph=True)[0])
        return height.detach(), gradient.detach(), torch.stack(rows).detach()


@dataclass
class Metric5DFields:
    """5D tensors and derivatives at one collection of volume or face points.

    G_cov[A,B,...], G_contra[A,B,...], dG_cov[k,A,B,...]. The derivative
    index k=0,1,2 labels xi1,xi2,xi3, unlike the five-dimensional indices A,B.
    """

    G_cov: Tensor
    G_contra: Tensor
    dG_cov: Tensor
    volume_jacobian: Tensor
    geopotential: Tensor
    height: Tensor
    height_gradient: Tensor
    physical_basis: Tensor
    radial: Tensor

    @property
    def frame_vorticity(self) -> Tensor:
        cached = getattr(self, "_runtime_frame_vorticity", None)
        if cached is not None:
            return cached
        derivatives = self.dG_cov[:, 0, 1:4]
        return derivatives - derivatives.transpose(0, 1)

    @property
    def vertical_frame_gradient(self) -> Tensor:
        """Return G_{0alpha,3}, alpha=1,2, in reference coordinates."""
        cached = getattr(self, "_runtime_vertical_frame_gradient", None)
        return self.dG_cov[2, 0, 1:3] if cached is None else cached

    @property
    def horizontal_frame_gradient(self) -> Tensor:
        """Return G_{03,alpha}, alpha=1,2, in reference coordinates."""
        cached = getattr(self, "_runtime_horizontal_frame_gradient", None)
        return self.dG_cov[:2, 0, 3] if cached is None else cached

    def raise_velocity(self, relative_covariant: Tensor) -> Tensor:
        spatial_metric = getattr(self, "_transport_G_contra", self.G_contra[1:4, 1:4])
        return spatial_matrix_vector(spatial_metric, relative_covariant)

    def lower_velocity(self, contravariant: Tensor) -> Tensor:
        spatial_metric = getattr(self, "_transport_G_cov", self.G_cov[1:4, 1:4])
        return spatial_matrix_vector(spatial_metric, contravariant)

    def physical_velocity(self, relative_covariant: Tensor) -> Tensor:
        """Return relative Cartesian wind in m/s."""
        return torch.einsum("ai...,i...->a...", self.physical_basis, self.raise_velocity(relative_covariant))

    def covariant_from_physical(self, physical: Tensor) -> Tensor:
        return torch.einsum("ai...,a...->i...", self.physical_basis, physical)

    def reflect(self, relative_covariant: Tensor, direction: int) -> Tensor:
        """Reflect across a constant-xi face, preserving its physical tangential wind."""
        reflected = relative_covariant.clone()
        velocity = self.raise_velocity(relative_covariant)
        reflected[direction] -= 2 * velocity[direction] / self.G_contra[direction + 1, direction + 1]
        return reflected

    def cache_frame_derivatives(self) -> None:
        """Keep the small frame coefficients on-device when the full dG tensor is offloaded."""
        derivatives = self.dG_cov[:, 0, 1:4]
        self._runtime_frame_vorticity = (derivatives - derivatives.transpose(0, 1)).to(self.G_cov.device)
        self._runtime_vertical_frame_gradient = derivatives[2, :2].to(self.G_cov.device).clone()
        self._runtime_horizontal_frame_gradient = derivatives[:2, 2].to(self.G_cov.device).clone()


def _gnomonic_metric(x: Tensor, y: Tensor, height: Tensor) -> tuple[Tensor, Tensor]:
    """Spatial metric of the unit sphere and its panel-coordinate derivatives."""
    secant2_x, secant2_y, delta_squared = 1 + x * x, 1 + y * y, 1 + x * x + y * y
    shape = height.shape
    sphere_metric = height.new_zeros((3, 3) + shape)
    sphere_metric[0, 0], sphere_metric[0, 1] = (
        secant2_x * secant2_x * secant2_y / delta_squared**2,
        -x * y * secant2_x * secant2_y / delta_squared**2,
    )
    sphere_metric[1, 0], sphere_metric[1, 1] = (
        sphere_metric[0, 1],
        secant2_x * secant2_y * secant2_y / delta_squared**2,
    )
    sphere_metric_numerator = sphere_metric * delta_squared**2
    numerator_derivatives = height.new_zeros((3, 3, 3) + shape)
    numerator_derivatives[0, 0, 0], numerator_derivatives[0, 0, 1], numerator_derivatives[0, 1, 1] = (
        4 * x * secant2_x * secant2_x * secant2_y,
        -y * secant2_x * secant2_y * (1 + 3 * x * x),
        2 * x * secant2_x * secant2_y * secant2_y,
    )
    numerator_derivatives[1, 0, 0], numerator_derivatives[1, 0, 1], numerator_derivatives[1, 1, 1] = (
        2 * y * secant2_x * secant2_x * secant2_y,
        -x * secant2_x * secant2_y * (1 + 3 * y * y),
        4 * y * secant2_x * secant2_y * secant2_y,
    )
    numerator_derivatives[:, 1, 0] = numerator_derivatives[:, 0, 1]
    delta_gradient = torch.stack((2 * x * secant2_x, 2 * y * secant2_y, torch.zeros_like(height)))
    sphere_metric_derivatives = (
        numerator_derivatives / delta_squared**2
        - 2 * sphere_metric_numerator[None] * delta_gradient[:, None, None] / delta_squared**3
    )

    return sphere_metric, sphere_metric_derivatives


def _rotation_covector(x: Tensor, y: Tensor, frame: Tensor, rotation_speed: float, height: Tensor):
    """Rotation contribution to G_0i on the unit sphere, before reference scaling."""
    secant2_x, secant2_y, delta_squared = 1 + x * x, 1 + y * y, 1 + x * x + y * y
    delta_gradient = torch.stack((2 * x * secant2_x, 2 * y * secant2_y, torch.zeros_like(height)))
    shape = height.shape
    rotation_axis_x, rotation_axis_y, rotation_axis_z = frame[2]
    rotation_covector = torch.stack(
        (
            rotation_speed * secant2_x * (rotation_axis_y - rotation_axis_z * y) / delta_squared,
            rotation_speed * secant2_y * (rotation_axis_z * x - rotation_axis_x) / delta_squared,
            torch.zeros_like(height),
        )
    )
    rotation_numerator_derivatives = height.new_zeros((3, 3) + shape)
    rotation_numerator_derivatives[0, 0] = 2 * rotation_speed * x * secant2_x * (rotation_axis_y - rotation_axis_z * y)
    rotation_numerator_derivatives[0, 1] = rotation_speed * secant2_x * secant2_y * rotation_axis_z
    rotation_numerator_derivatives[1, 0] = -rotation_speed * secant2_x * secant2_y * rotation_axis_z
    rotation_numerator_derivatives[1, 1] = 2 * rotation_speed * y * secant2_y * (rotation_axis_z * x - rotation_axis_x)
    rotation_derivatives = (
        rotation_numerator_derivatives / delta_squared
        - rotation_covector[None] * delta_gradient[:, None] / delta_squared
    )

    return rotation_covector, rotation_derivatives


def evaluate_metric5d(
    coordinates: Tensor,
    height_map: Callable[[Tensor], Tensor],
    *,
    radius: float,
    rotation_speed: float,
    frame: Tensor,
    scales: tuple[float, float, float],
    deep: bool,
    gravity_surface: float = gravity,
    derivative_device: torch.device | str | None = None,
) -> Metric5DFields:
    """Build the 5D metric and analytic first derivatives in reference coordinates.

    The supplied coordinates are unscaled panel angles and eta. ``scales`` are
    their affine half-element widths. Deep mode includes the centrifugal
    potential; shallow mode uses the spherical effective potential g*z.
    """
    if coordinates.dtype != torch.float64:
        raise ValueError("Construct the 5D metric in float64 before casting runtime fields")
    if min(scales) <= 0 or radius <= 0:
        raise ValueError("The radius and reference-element scales must be positive")
    height, height_gradient, height_hessian = mapping_derivatives(height_map, coordinates)
    if not bool(
        torch.isfinite(height).all() and torch.isfinite(height_gradient).all() and torch.isfinite(height_hessian).all()
    ):
        raise ValueError("The height map and its first two derivatives must be finite")
    if not bool((height_gradient[2] > 0).all() and (radius + height > 0).all()):
        raise ValueError("The radial coordinate must be orientation-preserving with positive radius")
    radial, sphere_basis = unit_sphere_basis(coordinates, frame)
    x, y = torch.tan(coordinates[0]), torch.tan(coordinates[1])
    secant2_x, secant2_y, delta_squared = 1 + x * x, 1 + y * y, 1 + x * x + y * y
    shape = height.shape
    sphere_metric, sphere_metric_derivatives = _gnomonic_metric(x, y, height)
    rotation_covector, rotation_derivatives = _rotation_covector(x, y, frame, rotation_speed, height)

    radial_distance = radius + height if deep else torch.full_like(height, radius)
    radial_gradient = height_gradient if deep else torch.zeros_like(height_gradient)
    effective_geopotential = gravity_surface * radius * height / (radius + height) if deep else gravity_surface * height
    gravity_field = gravity_surface * (radius / (radius + height)) ** 2 if deep else gravity_surface
    effective_geopotential_gradient = gravity_field * height_gradient
    spatial_metric = (
        radial_distance * radial_distance * sphere_metric + height_gradient[:, None] * height_gradient[None, :]
    )
    spatial_metric_derivatives = (
        2 * radial_distance * radial_gradient[:, None, None] * sphere_metric
        + radial_distance * radial_distance * sphere_metric_derivatives
        + height_hessian[:, :, None] * height_gradient[None, None, :]
        + height_gradient[None, :, None] * height_hessian[:, None, :]
    )
    frame_covector = radial_distance * radial_distance * rotation_covector
    frame_covector_derivatives = (
        2 * radial_distance * radial_gradient[:, None] * rotation_covector
        + radial_distance * radial_distance * rotation_derivatives
    )

    scale = height.new_tensor(scales).reshape((3,) + (1,) * height.ndim)
    spatial_metric = spatial_metric * scale[:, None] * scale[None, :]
    spatial_metric_derivatives = (
        spatial_metric_derivatives * scale[:, None, None] * scale[None, :, None] * scale[None, None, :]
    )
    frame_covector = frame_covector * scale
    frame_covector_derivatives = frame_covector_derivatives * scale[:, None] * scale[None, :]
    reference_height_gradient = height_gradient * scale
    effective_geopotential_gradient = effective_geopotential_gradient * scale
    inverse_spatial = torch.linalg.inv(spatial_metric.movedim((0, 1), (-2, -1))).movedim((-2, -1), (0, 1))
    raised_frame_covector = torch.einsum("ij...,j...->i...", inverse_spatial, frame_covector)
    frame_speed_squared = (frame_covector * raised_frame_covector).sum(dim=0)
    gravitational_geopotential = effective_geopotential
    if deep:
        # Eq. (3.8): differentiate the contraction using d(S^-1)=-S^-1 dS S^-1.
        effective_geopotential = gravitational_geopotential - 0.5 * frame_speed_squared
        effective_geopotential_gradient = (
            effective_geopotential_gradient
            - torch.einsum("i...,ki...->k...", raised_frame_covector, frame_covector_derivatives)
            + 0.5
            * torch.einsum(
                "i...,kij...,j...->k...", raised_frame_covector, spatial_metric_derivatives, raised_frame_covector
            )
        )
    G_cov = height.new_zeros((5, 5) + shape)
    G_cov[0, 0] = -2 * effective_geopotential
    G_cov[0, 1:4], G_cov[1:4, 0] = frame_covector, frame_covector
    G_cov[1:4, 1:4] = spatial_metric
    G_cov[0, 4] = G_cov[4, 0] = 1
    G_contra = torch.zeros_like(G_cov)
    G_contra[0, 4] = G_contra[4, 0] = 1
    G_contra[1:4, 1:4] = inverse_spatial
    G_contra[1:4, 4], G_contra[4, 1:4] = -raised_frame_covector, -raised_frame_covector
    G_contra[4, 4] = 2 * gravitational_geopotential if deep else 2 * effective_geopotential + frame_speed_squared
    derivatives = torch.zeros(
        (3, 5, 5) + shape,
        dtype=height.dtype,
        device=height.device if derivative_device is None else derivative_device,
    )
    derivatives[:, 0, 0] = -2 * effective_geopotential_gradient
    derivatives[:, 0, 1:4], derivatives[:, 1:4, 0] = frame_covector_derivatives, frame_covector_derivatives
    derivatives[:, 1:4, 1:4] = spatial_metric_derivatives
    # Release construction tensors before allocating the remaining fields on the shared GPU.
    del spatial_metric_derivatives, sphere_metric_derivatives, height_hessian
    del spatial_metric, inverse_spatial, sphere_metric
    del frame_covector_derivatives, rotation_derivatives, rotation_covector
    del frame_covector, raised_frame_covector, frame_speed_squared, effective_geopotential_gradient
    volume_jacobian = (
        radial_distance
        * radial_distance
        * secant2_x
        * secant2_y
        * height_gradient[2].abs()
        / delta_squared**1.5
        * math.prod(scales)
    )
    physical_basis = radial[:, None] * height_gradient[None, :]
    physical_basis[:, :2] += radial_distance * sphere_basis
    physical_basis *= scale[None, :]
    return Metric5DFields(
        G_cov=G_cov,
        G_contra=G_contra,
        dG_cov=derivatives,
        volume_jacobian=volume_jacobian,
        geopotential=effective_geopotential,
        height=height,
        height_gradient=reference_height_gradient,
        physical_basis=physical_basis,
        radial=radial,
    )


class Metric5D:
    """Volume and face geometry for a static cubed-sphere Euler simulation."""

    def __init__(self, geometry, config):
        from .terrain_mapping import TerrainHeightMap

        self.geometry = geometry
        # Prescribed tracer transport does not use metric derivatives at runtime.
        self.derivative_device = "cpu" if config.advection_only else geometry.radians.device
        self.frame = panel_frame(geometry.lon_p, geometry.lat_p, geometry.angle_p, geometry.radians)
        self.height_map = TerrainHeightMap(geometry, config, self.frame)
        self.scales = (geometry.delta_x1 / 2, geometry.delta_x2 / 2, geometry.delta_eta / 2)
        self.terrain_ramping = config.enable_schar_mountain and config.schar_mountain_step > 0

    def evaluate(self, coordinates: Tensor) -> Metric5DFields:
        return evaluate_metric5d(
            coordinates,
            self.height_map,
            radius=self.geometry.earth_radius,
            rotation_speed=self.geometry.rotation_speed,
            frame=self.frame,
            scales=self.scales,
            deep=self.geometry.deep,
            derivative_device=self.derivative_device,
        )

    def build_metric(self):
        # Release old fields before allocating replacements on the shared GPU.
        for name in ("volume", "faces"):
            if hasattr(self, name):
                delattr(self, name)
        geom = self.geometry
        coordinates = getattr(self, "_construction_coordinates", None)
        if coordinates is None:
            coordinates = (torch.stack((geom.radians[0], geom.radians[1], geom.eta_new)),) + tuple(
                torch.stack((angles[0], angles[1], eta))
                for angles, eta in (
                    (geom.radians_itf_i, geom.eta_itf_i_new),
                    (geom.radians_itf_j, geom.eta_itf_j_new),
                    (geom.radians_itf_k, geom.eta_itf_k_new),
                )
            )
            if self.terrain_ramping:
                self._construction_coordinates = coordinates
        if self.derivative_device == "cpu":
            # Keep completed fields off the shared GPU while constructing the next collection.
            collections = []
            for points in coordinates:
                collections.append(self._evaluate_transport_metric(points))
            del coordinates
            for collection in collections:
                # Tracer transport needs only the spatial blocks on the runtime device.
                collection._transport_G_cov = collection.G_cov[1:4, 1:4].to(geom.radians.device)
                collection._transport_G_contra = collection.G_contra[1:4, 1:4].to(geom.radians.device)
                for field in fields(Metric5DFields):
                    if field.name not in ("G_cov", "G_contra", "dG_cov"):
                        value = getattr(collection, field.name).to(geom.radians.device)
                        setattr(collection, field.name, value)
            self.volume, *face_fields = collections
            self.faces = tuple(face_fields)
        else:
            self.volume = self.evaluate(coordinates[0])
            self.faces = tuple(self.evaluate(points) for points in coordinates[1:])

    def _evaluate_transport_metric(self, coordinates: Tensor) -> Metric5DFields:
        # Bound the analytic construction workspace independently of the grid resolution.
        shape = coordinates.shape[1:]
        points = coordinates.reshape(3, -1)
        assembled = {}
        for start in range(0, points.shape[1], 16384):
            stop = min(start + 16384, points.shape[1])
            collection = self.evaluate(points[:, start:stop])
            for field in fields(Metric5DFields):
                value = getattr(collection, field.name)
                if field.name not in assembled:
                    assembled[field.name] = torch.empty(
                        value.shape[:-1] + (points.shape[1],), dtype=value.dtype, device="cpu"
                    )
                assembled[field.name][..., start:stop].copy_(value)
            del collection, value
        return Metric5DFields(
            **{name: value.reshape(value.shape[:-1] + shape) for name, value in assembled.items()}
        )

    def to_runtime(self, dtype: torch.dtype) -> None:
        """Cast runtime fields; retain float64 geometry construction."""
        for collection in (self.volume, *self.faces):
            for field in fields(Metric5DFields):
                setattr(collection, field.name, getattr(collection, field.name).to(dtype))
            for name in ("_transport_G_cov", "_transport_G_contra"):
                if hasattr(collection, name):
                    setattr(collection, name, getattr(collection, name).to(dtype))
            if hasattr(collection, "_runtime_frame_vorticity"):
                collection.cache_frame_derivatives()

    def offload_derivatives(self) -> None:
        """Store static dG on the CPU; retain the frame coefficients on the runtime device."""
        for collection in (self.volume, *self.faces):
            collection.cache_frame_derivatives()
            collection.dG_cov = collection.dG_cov.cpu()

    def physical_winds(self, relative_covariant: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        physical = self.volume.physical_velocity(relative_covariant)
        east, north, radial = self.wind_basis()
        return tuple((physical * basis).sum(dim=0) for basis in (east, north, radial))

    def wind_basis(self) -> tuple[Tensor, Tensor, Tensor]:
        radial = self.volume.radial
        longitude = torch.atan2(radial[1], radial[0])
        east = torch.stack((-torch.sin(longitude), torch.cos(longitude), torch.zeros_like(longitude)))
        north = torch.linalg.cross(radial, east, dim=0)
        return east, north, radial

    def covariant_winds(self, eastward, northward, vertical) -> Tensor:
        east, north, radial = self.wind_basis()
        return self.volume.covariant_from_physical(eastward * east + northward * north + vertical * radial)
