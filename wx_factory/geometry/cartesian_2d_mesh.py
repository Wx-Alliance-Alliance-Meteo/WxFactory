from typing import Self, Tuple

import numpy

from device import Device
from .geometry import Geometry


class Cartesian2D(Geometry):
    def __init__(
        self: Self,
        domain_x: Tuple[float, float],
        domain_z: Tuple[float, float],
        num_elements_x: int,
        num_elements_z: int,
        num_solpts: int,
        total_num_elements_x: int,
        device: Device,
    ):
        super().__init__(num_solpts, num_elements_x, num_elements_z, total_num_elements_x, device)
        xp = device.xp

        scaled_points = 0.5 * (1.0 + self.solutionPoints)

        # --- Horizontal coord
        Δx1 = (domain_x[1] - domain_x[0]) / num_elements_x
        itf_x1 = xp.linspace(start=domain_x[0], stop=domain_x[1], num=num_elements_x + 1)
        x1 = xp.zeros(num_elements_x * len(self.solutionPoints))
        for i in range(num_elements_x):
            idx = i * num_solpts
            x1[idx : idx + num_solpts] = itf_x1[i] + scaled_points * Δx1

        # --- Vertical coord
        Δx3 = (domain_z[1] - domain_z[0]) / num_elements_z
        x3 = xp.zeros(num_elements_z * len(self.solutionPoints))

        Δx3 = (domain_z[1] - domain_z[0]) / num_elements_z
        itf_x3 = xp.linspace(start=domain_z[0], stop=domain_z[1], num=num_elements_z + 1)

        for i in range(num_elements_z):
            idz = i * num_solpts
            x3[idz : idz + num_solpts] = itf_x3[i] + scaled_points * Δx3

        X1, X3 = xp.meshgrid(x1, x3)

        self.X1_cartesian = X1
        self.X3_cartesian = X3
        self.itf_Z = itf_x3
        self.Δx1 = Δx1
        self.Δx2 = Δx1
        self.Δx3 = Δx3
        self.xperiodic = False

        # TODO : hackathon SG
        self.X1 = xp.zeros((num_elements_z, num_elements_x, num_solpts**2))
        self.X3 = xp.zeros((num_elements_z, num_elements_x, num_solpts**2))
        idx_elem = 0
        for ek in range(num_elements_z):
            for ei in range(num_elements_x):
                #            idx_elem = ei + num_elements_z * ek
                start_i = ei * num_solpts
                end_i = (ei + 1) * num_solpts
                start_k = ek * num_solpts
                end_k = (ek + 1) * num_solpts
                self.X1[ek, ei, :] = X1[start_k:end_k, start_i:end_i].flatten()
                self.X3[ek, ei, :] = X3[start_k:end_k, start_i:end_i].flatten()
                idx_elem += 1

    def to_single_block(self, a):
        """Convert an array of values over this grid (which be may organized as a list of elements)
        into a single block of data (2D or 3D)."""

        # Get the number of variables (first dimension, as a list)
        num_equations = a.shape[:-3]

        # Add other dimensions and reshape to a plottable 2D graph
        tmp_shape = num_equations + (
            self.num_elements_vertical,
            self.num_elements_horizontal,
            self.num_solpts,
            self.num_solpts,
        )
        new_shape = num_equations + (
            self.num_elements_vertical * self.num_solpts,
            self.num_elements_horizontal * self.num_solpts,
        )
        a_new = numpy.swapaxes(a.reshape(tmp_shape), -2, -3).reshape(new_shape)

        return a_new
