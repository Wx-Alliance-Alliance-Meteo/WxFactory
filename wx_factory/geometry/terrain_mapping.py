"""Terrain height maps differentiated analytically at geometry setup."""

import math

import torch
from torch import Tensor

from .metric5d import unit_sphere_basis


def stretch_vertical_coordinate(eta, stretch):
    if stretch == 0:
        return eta
    return torch.expm1(stretch * eta) / math.expm1(stretch)


def angular_distance_squared(cosine: Tensor) -> Tensor:
    """acos(cosine)^2 with its regular first two derivatives at the centre."""
    distance = torch.acos(torch.clamp(cosine, -1 + 1e-12, 1 - 1e-12)) ** 2
    offset = 1 - cosine
    centre = 2 * offset + offset**2 / 3 + 4 * offset**3 / 45
    return torch.where(offset < 1e-6, centre, distance)


def cosine_of_distance(cosine: Tensor, frequency: float) -> Tensor:
    """cos(frequency*acos(cosine)), regularized at the mountain centre."""
    offset = cosine - 1
    squared = frequency**2
    # The Taylor expansion requires both the angular distance and oscillation phase to be small.
    near_centre = (offset.abs() < 1e-7) & (squared * offset.abs() < 1e-5)
    # Keep the inactive acos derivative finite at the centre.
    safe_cosine = torch.where(near_centre, torch.zeros_like(cosine), cosine)
    angle = torch.acos(torch.clamp(safe_cosine, -1 + 1e-12, 1.0))
    centre = (
        1
        + squared * offset
        + squared * (squared - 1) * offset**2 / 6
        + squared * (squared - 1) * (squared - 4) * offset**3 / 90
    )
    return torch.where(near_centre, centre, torch.cos(frequency * angle))


class TerrainHeightMap:
    """Height maps for the 200/201 mountain variant, Schar terrain, and flat grids.

    Case 201 uses the repository's 2000-m mountain and lapse-rate profile for
    comparison with the flux path; it is not the official DCMIP 2-0-1 profile.
    """

    def __init__(self, geometry, config, frame: Tensor):
        self.frame = frame
        self.terrain_ratio = 1.0
        self.radial_surface = None
        self.tabulated_surface = None
        self.ztop = geometry.ztop
        self.radius = geometry.earth_radius
        self.vertical_coord = geometry.vertical_coord
        self.vertical_stretch = getattr(geometry, "vertical_stretch", 0.0)
        self.scale_large = geometry.sleve_scale_large
        self.scale_small = geometry.sleve_scale_small
        self.case = config.case_number
        self.mountain = self.case in (13, 20, 200, 201)
        self.gaussian = config.enable_schar_mountain and not self.mountain
        if self.mountain:
            longitude, latitude = 3 * math.pi / 2, 0.0
        elif self.gaussian:
            longitude, latitude = config.schar_mountain_longitude, config.schar_mountain_lattitude
            self.amplitude = config.schar_mountain_height
            self.width = config.schar_mountain_radius
            self.wavelength = config.schar_mountain_length
        else:
            longitude, latitude = 0.0, 0.0
        self.centre = frame.new_tensor(
            [math.cos(latitude) * math.cos(longitude), math.cos(latitude) * math.sin(longitude), math.sin(latitude)]
        )

    def decay(self, eta: Tensor, scale: float) -> Tensor:
        if self.vertical_coord == "gal_chen":
            return 1 - eta
        ratio = self.ztop / scale
        return torch.sinh(ratio * (1 - eta)) / math.sinh(ratio)

    def __call__(self, coordinates: Tensor) -> Tensor:
        eta = stretch_vertical_coordinate(coordinates[2], self.vertical_stretch)
        if self.tabulated_surface is not None:
            return self.ztop * eta + self.tabulated_surface(coordinates) * self.decay(eta, self.scale_large)
        if self.radial_surface is not None:
            radial, _ = unit_sphere_basis(coordinates, self.frame.clone())
            return self.ztop * eta + self.radial_surface(radial) * self.decay(eta, self.scale_large)
        if not self.mountain and not self.gaussian:
            return self.ztop * eta
        # Clone inference-mode constants so autograd can differentiate the map.
        radial, _ = unit_sphere_basis(coordinates, self.frame.clone())
        cosine = torch.einsum("i,i...->...", self.centre.clone(), radial)
        if self.mountain:
            radius = 3 * math.pi / 4
            bell = 1000 * (1 + cosine_of_distance(cosine, math.pi / radius))
            bell = torch.where(cosine > math.cos(radius), bell, torch.zeros_like(bell))
            ripple = cosine_of_distance(cosine, 32.0)
        else:
            bell = self.amplitude * torch.exp(-(self.radius**2) * angular_distance_squared(cosine) / self.width**2)
            ripple = cosine_of_distance(cosine, 2 * math.pi * self.radius / self.wavelength)
        large, small = bell / 2, bell * ripple / 2
        return self.ztop * eta + self.terrain_ratio * (
            large * self.decay(eta, self.scale_large) + small * self.decay(eta, self.scale_small)
        )


class TabulatedSurface:
    """Tensor-product cubic spline with not-a-knot endpoint conditions."""

    def __init__(self, x, y, values):
        self.x, self.y = x.clone(), y.clone()
        along_x = self.coefficients(self.x, values)
        along_y = self.coefficients(self.y, along_x.permute(1, 2, 0))
        self.polynomials = along_y.permute(2, 0, 3, 1).clone()

    @staticmethod
    def coefficients(axis, values):
        count = axis.numel()
        if count < 2:
            raise ValueError("Surface interpolation requires at least two nodes per direction")
        widths = axis[1:] - axis[:-1]
        if not bool((widths > 0).all()):
            raise ValueError("Surface interpolation axes must be strictly increasing")
        matrix = axis.new_zeros((count, count))
        right = torch.zeros_like(values)
        slopes = (values[..., 1:] - values[..., :-1]) / widths
        interior = torch.arange(1, count - 1, device=axis.device)
        matrix[interior, interior - 1] = widths[:-1]
        matrix[interior, interior] = 2 * (widths[:-1] + widths[1:])
        matrix[interior, interior + 1] = widths[1:]
        right[..., 1:-1] = 6 * (slopes[..., 1:] - slopes[..., :-1])
        if count == 2:
            matrix[0, 0] = matrix[-1, -1] = 1
        elif count == 3:
            matrix[0, 0], matrix[0, 1] = 1, -1
            matrix[-1, -2], matrix[-1, -1] = 1, -1
        else:
            matrix[0, :3] = torch.stack((1 / widths[0], -1 / widths[0] - 1 / widths[1], 1 / widths[1]))
            matrix[-1, -3:] = torch.stack((1 / widths[-2], -1 / widths[-2] - 1 / widths[-1], 1 / widths[-1]))
        second = torch.linalg.solve(matrix, right.reshape(-1, count).T).T.reshape_as(values)
        return torch.stack(
            (
                values[..., :-1],
                slopes - widths * (2 * second[..., :-1] + second[..., 1:]) / 6,
                second[..., :-1] / 2,
                (second[..., 1:] - second[..., :-1]) / (6 * widths),
            ),
            dim=-1,
        )

    def __call__(self, coordinates):
        x, y = self.x.clone(), self.y.clone()
        x_indices = (torch.searchsorted(x, coordinates[0].contiguous()) - 1).clamp(0, x.numel() - 2)
        y_indices = (torch.searchsorted(y, coordinates[1].contiguous()) - 1).clamp(0, y.numel() - 2)
        dx, dy = coordinates[0] - x[x_indices], coordinates[1] - y[y_indices]
        x_powers = torch.stack((torch.ones_like(dx), dx, dx**2, dx**3), dim=-1)
        y_powers = torch.stack((torch.ones_like(dy), dy, dy**2, dy**3), dim=-1)
        coefficients = self.polynomials.clone()[y_indices, x_indices]
        return (coefficients * y_powers[..., :, None] * x_powers[..., None, :]).sum((-2, -1))


class SampledSphereSurface:
    """A common C2 surface built by blending cubic splines on the six panels."""

    def __init__(self, geometry, surface, frame):
        topology = geometry.process_topology
        cube = topology.gather_cube(geometry.to_old_floor(surface), 2)
        cube = geometry.context.comm.bcast(None if cube is None else cube.cpu().numpy(), root=0)
        cube = torch.as_tensor(cube, dtype=frame.dtype, device=frame.device)
        frames = geometry.context.comm.allgather((topology.my_panel, frame.cpu().numpy()))
        self.frames = torch.stack(
            [
                torch.as_tensor(
                    next(value for panel, value in frames if panel == index), dtype=frame.dtype, device=frame.device
                )
                for index in range(6)
            ]
        )
        self.frame = frame
        elements = geometry.total_num_elements_horizontal
        spacing = math.pi / (2 * elements)
        axis = -math.pi / 4 + spacing * (
            torch.arange(elements, device=frame.device)[:, None] + 0.5 + geometry.solutionPoints[None] / 2
        )
        self.panels = []
        for values in cube:
            self.panels.append(TabulatedSurface(axis.flatten(), axis.flatten(), values))

    def __call__(self, coordinates):
        radial, _ = unit_sphere_basis(coordinates, self.frame.clone())
        height = torch.zeros_like(coordinates[0])
        normalization = torch.zeros_like(height)
        for frame, interpolant in zip(self.frames.clone(), self.panels):
            panel = torch.einsum("ai,a...->i...", frame, radial)
            # Cubic weights and their first two derivatives vanish outside each patch.
            weight = (panel[2] - 0.5).clamp_min(0) ** 3
            angles = torch.stack(
                (torch.atan2(panel[0], panel[2].clamp_min(0.5)), torch.atan2(panel[1], panel[2].clamp_min(0.5)))
            )
            height = height + weight * interpolant(angles)
            normalization = normalization + weight
        return height / normalization
