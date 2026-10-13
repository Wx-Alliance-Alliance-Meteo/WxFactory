"""Five-dimensional metric for a flat Cartesian reference element."""

import math

import torch

from ..common.definitions import gravity
from .metric5d import Metric5D, Metric5DFields


class CartesianMetric5D(Metric5D):
    def __init__(self, geometry, config):
        self.geometry = geometry
        self.gravity = 0 if config.case_number == 651 else gravity
        self.scales = (
            geometry.delta_x1 / 2,
            geometry.delta_x2 / 2,
            geometry.ztop / geometry.num_elements_x3 / 2,
        )

    def evaluate(self, height):
        shape = height.shape
        geopotential = self.gravity * height
        G_cov = height.new_zeros((5, 5) + shape)
        G_contra = torch.zeros_like(G_cov)
        physical_basis = height.new_zeros((3, 3) + shape)
        height_gradient = height.new_zeros((3,) + shape)
        vertical_unit_vector = torch.zeros_like(height_gradient)
        metric_derivatives = height.new_zeros((3, 5, 5) + shape)
        for direction, scale in enumerate(self.scales):
            G_cov[direction + 1, direction + 1] = scale**2
            G_contra[direction + 1, direction + 1] = 1 / scale**2
            physical_basis[direction, direction] = scale
        G_cov[0, 0] = -2 * geopotential
        G_cov[0, 4] = G_cov[4, 0] = 1
        G_contra[0, 4] = G_contra[4, 0] = 1
        G_contra[4, 4] = 2 * geopotential
        metric_derivatives[2, 0, 0] = -2 * self.gravity * self.scales[2]
        height_gradient[2] = self.scales[2]
        vertical_unit_vector[2] = 1
        volume_jacobian = torch.full_like(height, math.prod(self.scales))
        return Metric5DFields(
            G_cov=G_cov,
            G_contra=G_contra,
            dG_cov=metric_derivatives,
            volume_jacobian=volume_jacobian,
            geopotential=geopotential,
            height=height,
            height_gradient=height_gradient,
            physical_basis=physical_basis,
            radial=vertical_unit_vector,
        )

    def build_metric(self):
        geom = self.geometry
        self.volume = self.evaluate(geom.X3)
        self.faces = tuple(
            self.evaluate(points[2]) for points in (geom.gnomonic_itf_i, geom.gnomonic_itf_j, geom.gnomonic_itf_k)
        )

    def wind_basis(self):
        vertical_unit_vector = self.volume.radial
        x_direction = torch.zeros_like(vertical_unit_vector)
        y_direction = torch.zeros_like(vertical_unit_vector)
        x_direction[0] = y_direction[1] = 1
        return x_direction, y_direction, vertical_unit_vector
