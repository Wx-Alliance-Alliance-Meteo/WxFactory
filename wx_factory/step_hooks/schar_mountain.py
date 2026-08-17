import math

import numpy
import torch

from ..common import Configuration
from ..geometry import CubedSphere3D, Metric3DTopo
from . import step_hook


class ScharMountainHook(step_hook.StepHook):
    lambdam: float  # mountain longitude center point (radians)
    phim: float  # mountain latitude center point (radians)
    h0: float  # peak height of the mountain range (m)
    Dm: float  # mountain radius (meters)
    Dxi: float  # Mountain wavelength (meters)
    geom: CubedSphere3D
    metric: Metric3DTopo
    step_to_completion: int

    step: int

    zbot: numpy.ndarray
    zbot_itf_i: numpy.ndarray
    zbot_itf_j: numpy.ndarray

    zbot_new: numpy.ndarray
    zbot_itf_i_new: numpy.ndarray
    zbot_itf_j_new: numpy.ndarray

    def __init__(self, config: Configuration, geom: CubedSphere3D):
        if config is None:
            raise ValueError("The configuration must no be None")
        if type(geom) != CubedSphere3D or geom is None:
            raise TypeError("The Schar waves works only with a 3D cubed sphere")
        self.geom = geom
        self.lambdam = config.schar_mountain_longitude
        self.phim = config.schar_mountain_lattitude
        self.h0 = config.schar_mountain_height
        self.Dm = config.schar_mountain_radius
        self.Dxi = config.schar_mountain_length

        self.step_to_completion = config.schar_mountain_step
        self.step = 0
        self.build()

    def build(self):
        """
        ratio : % of the actual montain to apply and build
        """
        self.zbot = self.build_topo_old(self.geom.coordVec_latlon)
        self.zbot_itf_i = self.build_topo_old(self.geom.coordVec_latlon_itf_i)
        self.zbot_itf_j = self.build_topo_old(self.geom.coordVec_latlon_itf_j)

        self.zbot_new = self.build_topo(self.geom.get_floor(self.geom.polar))
        self.zbot_itf_i_new = self.build_topo(self.geom.get_itf_i_floor(self.geom.polar_itf_i))
        self.zbot_itf_j_new = self.build_topo(self.geom.get_itf_j_floor(self.geom.polar_itf_j))

        # Large-scale part of the mountain, for the SLEVE vertical coordinate (see build_topo)
        self.large = self.build_topo_old(self.geom.coordVec_latlon, large_scale_only=True)
        self.large_itf_i = self.build_topo_old(self.geom.coordVec_latlon_itf_i, large_scale_only=True)
        self.large_itf_j = self.build_topo_old(self.geom.coordVec_latlon_itf_j, large_scale_only=True)

        self.large_new = self.build_topo(self.geom.get_floor(self.geom.polar), large_scale_only=True)
        self.large_itf_i_new = self.build_topo(self.geom.get_itf_i_floor(self.geom.polar_itf_i), large_scale_only=True)
        self.large_itf_j_new = self.build_topo(self.geom.get_itf_j_floor(self.geom.polar_itf_j), large_scale_only=True)
        self.large_itf_i_new[self.geom.floor_west_edge] = 0.0
        self.large_itf_i_new[self.geom.floor_east_edge] = 0.0
        self.large_itf_j_new[self.geom.floor_south_edge] = 0.0
        self.large_itf_j_new[self.geom.floor_north_edge] = 0.0

        self.zbot_itf_i_new[self.geom.floor_west_edge] = 0.0
        self.zbot_itf_i_new[self.geom.floor_east_edge] = 0.0
        self.zbot_itf_j_new[self.geom.floor_south_edge] = 0.0
        self.zbot_itf_j_new[self.geom.floor_north_edge] = 0.0

        diff = self.zbot_new - self.geom.to_new_floor(self.zbot)
        diffn = torch.linalg.norm(diff)

        diffi = self.zbot_itf_i_new - self.geom.to_new_itf_i_floor(self.zbot_itf_i)
        diffin = torch.linalg.norm(diffi)

        diffj = self.zbot_itf_j_new - self.geom.to_new_itf_j_floor(self.zbot_itf_j)
        diffjn = torch.linalg.norm(diffj)

        if diffn > 0.0 or diffin > 0.0 or diffjn > 0.0:
            raise ValueError

    def apply(self, ratio: float):
        # Match cached terrain fields to the runtime geometry precision.
        if self.zbot.dtype != self.geom.dtype:
            for name in (
                "zbot",
                "zbot_itf_i",
                "zbot_itf_j",
                "zbot_new",
                "zbot_itf_i_new",
                "zbot_itf_j_new",
                "large",
                "large_itf_i",
                "large_itf_j",
                "large_new",
                "large_itf_i_new",
                "large_itf_j_new",
            ):
                setattr(self, name, getattr(self, name).to(dtype=self.geom.dtype))

        # Update the geometry object with the new bottom topography
        self.geom.apply_topography(
            self.zbot * ratio,
            self.zbot_itf_i * ratio,
            self.zbot_itf_j * ratio,
            self.zbot_new * ratio,
            self.zbot_itf_i_new * ratio,
            self.zbot_itf_j_new * ratio,
            self.large * ratio,
            self.large_itf_i * ratio,
            self.large_itf_j * ratio,
            self.large_new * ratio,
            self.large_itf_i_new * ratio,
            self.large_itf_j_new * ratio,
        )

        # And regenerate the metric to take this new topography into account
        self.metric.build_metric()

    def process(self, Q: numpy.ndarray, t: float) -> numpy.ndarray:
        if self.step < self.step_to_completion:
            self.step += 1
            ratio = float(self.step) / self.step_to_completion
            self.apply(ratio)
        return Q

    def build_topo_old(self, latlon, large_scale_only: bool = False):
        lat = latlon[1, 0, :, :]
        lon = latlon[0, 0, :, :]
        z = torch.zeros(lat.shape, dtype=lat.dtype)
        z[:, :] = self.topo(lon, lat, large_scale_only)
        return z

    def build_topo(self, latlon, large_scale_only: bool = False):
        return self.topo(latlon[0], latlon[1], large_scale_only)

    def topo(self, lon, lat, large_scale_only: bool = False):
        """
        The Schar mountain: a Gaussian envelope modulated by a short-wavelength ripple.

        Since cos² x = (1 + cos 2x)/2, half of the envelope carries the whole mountain height with
        none of the ripple. That half is the large-scale part h1 asked for by the SLEVE vertical
        coordinate (Schar et al. 2002, eq. 27), the rest is the small-scale part h2.
        """
        r = self.geom.earth_radius * torch.arccos(
            math.sin(self.phim) * torch.sin(lat) + math.cos(self.phim) * torch.cos(lat) * torch.cos(lon - self.lambdam)
        )

        envelope = self.h0 * torch.exp(-(r**2) / self.Dm**2)
        shape = 0.5 if large_scale_only else torch.cos(torch.pi * r / self.Dxi) ** 2

        return envelope * shape
