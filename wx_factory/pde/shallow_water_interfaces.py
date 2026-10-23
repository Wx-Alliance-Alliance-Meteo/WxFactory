"""Gravity-wave matching for wet shallow-water states in Lamb form."""

import torch

from ..common.definitions import gravity


class GravityWaveInterface:
    """Face traces contain free-surface height and relative covariant velocity."""

    def __init__(self, left, right, metric, bottom, direction):
        self.metric = metric
        self.bottom = bottom
        self.direction = direction
        self.normal_scale = torch.sqrt(metric.contravariant[direction, direction])
        self.normal_row = metric.contravariant[direction] / self.normal_scale
        self.left = left
        self.right = right
        self.normal_left = self.normal_wind(left[1:3])
        self.normal_right = self.normal_wind(right[1:3])
        mean_depth = (left[0] + right[0]) / 2 - bottom
        self.wave_speed = torch.sqrt(gravity * mean_depth)
        self.normal_velocity = (self.normal_left + self.normal_right) / 2
        self.normal_velocity -= gravity * (right[0] - left[0]) / (2 * self.wave_speed)
        self.surface = (left[0] + right[0]) / 2
        self.surface -= self.wave_speed * (self.normal_right - self.normal_left) / (2 * gravity)
        self.depth = self.surface - bottom
        self.velocity = (left[1:3] + right[1:3]) / 2
        self.velocity[direction] += (self.normal_velocity - self.normal_wind(self.velocity)) / self.normal_scale
        self.contravariant = metric.raise_velocity(self.velocity)

    def normal_wind(self, velocity):
        return self.normal_row[0] * velocity[0] + self.normal_row[1] * velocity[1]

    def common_values(self):
        mass_flux = self.metric.volume_jacobian * self.depth * self.normal_scale * self.normal_velocity
        energy = (self.velocity * self.contravariant).sum(dim=0) / 2 + gravity * self.surface
        return torch.stack((mass_flux, energy, self.velocity[0], self.velocity[1]))

    def linearize(self, delta_left, delta_right):
        delta_normal_left = self.normal_wind(delta_left[1:3])
        delta_normal_right = self.normal_wind(delta_right[1:3])
        delta_wave_speed = gravity * (delta_left[0] + delta_right[0]) / (4 * self.wave_speed)
        delta_normal = (delta_normal_left + delta_normal_right) / 2
        delta_normal -= gravity * (delta_right[0] - delta_left[0]) / (2 * self.wave_speed)
        delta_normal += gravity * (self.right[0] - self.left[0]) * delta_wave_speed / (2 * self.wave_speed.square())
        delta_surface = (delta_left[0] + delta_right[0]) / 2
        delta_surface -= (
            delta_wave_speed * (self.normal_right - self.normal_left)
            + self.wave_speed * (delta_normal_right - delta_normal_left)
        ) / (2 * gravity)
        delta_velocity = (delta_left[1:3] + delta_right[1:3]) / 2
        delta_velocity[self.direction] += (delta_normal - self.normal_wind(delta_velocity)) / self.normal_scale
        delta_mass_flux = (
            self.metric.volume_jacobian
            * self.normal_scale
            * (delta_surface * self.normal_velocity + self.depth * delta_normal)
        )
        delta_energy = (self.contravariant * delta_velocity).sum(dim=0) + gravity * delta_surface
        return torch.stack((delta_mass_flux, delta_energy, delta_velocity[0], delta_velocity[1]))
