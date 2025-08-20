from . import post_proccessor
from common import Configuration
from geometry import CubedSphere3D, Metric3DTopo
import math
import numpy

class ScharMountainPostProcessor(post_proccessor.PostProcessor):
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
        self.xp = geom.device.xp

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
        self.zbot_itf_i_new[self.geom.floor_west_edge] = 0.0
        self.zbot_itf_i_new[self.geom.floor_east_edge] = 0.0
        self.zbot_itf_j_new[self.geom.floor_south_edge] = 0.0
        self.zbot_itf_j_new[self.geom.floor_north_edge] = 0.0

        diff = self.zbot_new - self.geom.to_new_floor(self.zbot)
        diffn = self.xp.linalg.norm(diff)

        diffi = self.zbot_itf_i_new - self.geom.to_new_itf_i_floor(self.zbot_itf_i)
        diffin = self.xp.linalg.norm(diffi)

        diffj = self.zbot_itf_j_new - self.geom.to_new_itf_j_floor(self.zbot_itf_j)
        diffjn = self.xp.linalg.norm(diffj)

        if diffn > 0.0 or diffin > 0.0 or diffjn > 0.0:
            raise ValueError
        
    def apply(self, ratio: float):
        # Update the geometry object with the new bottom topography
        self.geom.apply_topography(self.zbot * ratio, self.zbot_itf_i * ratio, self.zbot_itf_j * ratio,
                                    self.zbot_new * ratio, self.zbot_itf_i_new * ratio, self.zbot_itf_j_new * ratio)
        
        # And regenerate the metric to take this new topography into account
        self.metric.build_metric()

    def process(self):
        if self.step < self.step_to_completion:
            self.step += 1
            ratio = float(self.step) / self.step_to_completion
            self.apply(ratio)

    def build_topo_old(self, latlon):
        lat = latlon[1, 0, :, :]
        lon = latlon[0, 0, :, :]
        r = self.geom.earth_radius * self.xp.arccos(
            math.sin(self.phim) * self.xp.sin(lat) + math.cos(self.phim) * self.xp.cos(lat) * self.xp.cos(lon - self.lambdam)
        )
        z = self.xp.zeros(lat.shape, dtype=lat.dtype)
        z[:, :] = self.h0 * self.xp.exp(-(r**2) / self.Dm**2) * self.xp.cos(self.xp.pi * r / self.Dxi) ** 2
        return z

    def build_topo(self, latlon):
        lat = latlon[1]
        lon = latlon[0]
        r = self.geom.earth_radius * self.xp.arccos(
            math.sin(self.phim) * self.xp.sin(lat) + math.cos(self.phim) * self.xp.cos(lat) * self.xp.cos(lon - self.lambdam)
        )

        return self.h0 * self.xp.exp(-(r**2) / self.Dm**2) * self.xp.cos(self.xp.pi * r / self.Dxi) ** 2
