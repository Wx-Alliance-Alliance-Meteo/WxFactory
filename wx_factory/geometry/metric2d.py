"""Horizontal geometry of the cubed sphere in element reference coordinates."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class HorizontalMetric:
    """Spatial metric, area factor, and frame vorticity in element reference coordinates."""

    covariant: Tensor
    contravariant: Tensor
    volume_jacobian: Tensor
    frame_vorticity: Tensor

    def raise_velocity(self, velocity):
        inverse = self.contravariant
        return torch.stack(
            (
                inverse[0, 0] * velocity[0] + inverse[0, 1] * velocity[1],
                inverse[1, 0] * velocity[0] + inverse[1, 1] * velocity[1],
            )
        )

    def lower_velocity(self, velocity):
        metric = self.covariant
        return torch.stack(
            (
                metric[0, 0] * velocity[0] + metric[0, 1] * velocity[1],
                metric[1, 0] * velocity[0] + metric[1, 1] * velocity[1],
            )
        )

    def to(self, dtype):
        return HorizontalMetric(
            self.covariant.to(dtype),
            self.contravariant.to(dtype),
            self.volume_jacobian.to(dtype),
            self.frame_vorticity.to(dtype),
        )

    def at(self, indices):
        return HorizontalMetric(
            self.covariant[indices],
            self.contravariant[indices],
            self.volume_jacobian[indices],
            self.frame_vorticity[indices],
        )


class Metric2D:
    def __init__(self, geom):
        self.volume = self.evaluate(geom, geom.X, geom.Y)
        self.faces = (self.evaluate(geom, geom.X_itf_i, geom.Y_itf_i), self.evaluate(geom, geom.X_itf_j, geom.Y_itf_j))
        self._set_views()

    @staticmethod
    def evaluate(geom, x, y):
        delta2 = 1 + x.square() + y.square()
        delta = torch.sqrt(delta2)
        scale1, scale2 = geom.delta_x1 / 2, geom.delta_x2 / 2
        factor = geom.earth_radius**2 / delta2.square()
        g11 = factor * (1 + x.square()).square() * (1 + y.square()) * scale1**2
        g12 = -factor * x * y * (1 + x.square()) * (1 + y.square()) * scale1 * scale2
        g22 = factor * (1 + x.square()) * (1 + y.square()).square() * scale2**2
        covariant = torch.stack((g11, g12, g12, g22)).reshape((2, 2) + x.shape)
        determinant = g11 * g22 - g12.square()
        inverse = torch.stack((g22, -g12, -g12, g11)).reshape((2, 2) + x.shape) / determinant
        area = geom.earth_radius**2 * (1 + x.square()) * (1 + y.square()) / delta**3 * scale1 * scale2
        sin_latitude = (
            math.sin(geom.lat_p)
            - x * math.cos(geom.lat_p) * math.sin(geom.angle_p)
            + y * math.cos(geom.lat_p) * math.cos(geom.angle_p)
        ) / delta
        # The exterior derivative of the frame covector is f times the oriented area element.
        frame_vorticity = 2 * geom.rotation_speed * sin_latitude * area
        return HorizontalMetric(covariant, inverse, area, frame_vorticity)

    def _set_views(self):
        self.sqrtG = self.volume.volume_jacobian
        self.sqrtG_itf_i, self.sqrtG_itf_j = (face.volume_jacobian for face in self.faces)
        self.inv_sqrtG = self.sqrtG.reciprocal()
        self.coriolis_f = self.volume.frame_vorticity / self.sqrtG

    def to_runtime(self, dtype):
        self.volume = self.volume.to(dtype)
        self.faces = tuple(face.to(dtype) for face in self.faces)
        self._set_views()
