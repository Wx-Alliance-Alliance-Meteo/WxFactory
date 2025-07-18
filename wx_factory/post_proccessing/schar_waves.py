from . import post_proccessor
from common import Configuration
from geometry import CubedSphere3D, Metric3DTopo
import math

class ScharWavesPostProcessor(post_proccessor.PostProcessor):
    lambdam: float  # mountain longitude center point (radians)
    phim: float  # mountain latitude center point (radians)
    h0: float  # peak height of the mountain range (m)
    Dm: float  # mountain radius (meters)
    Dxi: float  # Mountain wavelength (meters)
    geom: CubedSphere3D
    metric: Metric3DTopo
    step_to_completion: int

    step: int

    def __init__(self, config: Configuration, geom: CubedSphere3D, metric: Metric3DTopo):
        if config is None:
            raise ValueError("The configuration must no be None")
        if type(geom) != CubedSphere3D or geom is None:
            raise TypeError("The Schar waves works only with a 3D cubed sphere")
        if type(metric) != Metric3DTopo or metric is None:
            raise TypeError("The Schar waves works only with a 3D metric topology")
        self.geom = geom
        self.xp = geom.device.xp
        self.metric = metric

        self.lambdam = config.schar_waves_longitude
        self.phim = config.schar_waves_lattitude
        self.h0 = config.schar_waves_height
        self.Dm = config.schar_waves_radius
        self.Dxi = config.schar_waves_length
        
        self.step_to_completion = config.schar_waves_step
        self.step = 0

    def build(self, ratio: float):
        """
        ratio : % of the actual montain to apply and build
        """
        zbot = self.build_topo_old(self.geom.coordVec_latlon)
        zbot_itf_i = self.build_topo_old(self.geom.coordVec_latlon_itf_i)
        zbot_itf_j = self.build_topo_old(self.geom.coordVec_latlon_itf_j)

        zbot_new = self.build_topo(self.geom.get_floor(self.geom.polar))
        zbot_itf_i_new = self.build_topo(self.geom.get_itf_i_floor(self.geom.polar_itf_i))
        zbot_itf_j_new = self.build_topo(self.geom.get_itf_j_floor(self.geom.polar_itf_j))
        zbot_itf_i_new[self.geom.floor_west_edge] = 0.0
        zbot_itf_i_new[self.geom.floor_east_edge] = 0.0
        zbot_itf_j_new[self.geom.floor_south_edge] = 0.0
        zbot_itf_j_new[self.geom.floor_north_edge] = 0.0

        diff = zbot_new - self.geom.to_new_floor(zbot)
        diffn = self.xp.linalg.norm(diff)

        diffi = zbot_itf_i_new - self.geom.to_new_itf_i_floor(zbot_itf_i)
        diffin = self.xp.linalg.norm(diffi)

        diffj = zbot_itf_j_new - self.geom.to_new_itf_j_floor(zbot_itf_j)
        diffjn = self.xp.linalg.norm(diffj)

        if diffn > 0.0 or diffin > 0.0 or diffjn > 0.0:
            raise ValueError

        # Update the geometry object with the new bottom topography
        self.geom.apply_topography(zbot * ratio, zbot_itf_i * ratio, zbot_itf_j * ratio, zbot_new * ratio, zbot_itf_i_new * ratio, zbot_itf_j_new * ratio)
        
        # And regenerate the metric to take this new topography into account
        self.metric.build_metric()

    def process(self):
        if self.step < self.step_to_completion:
            self.step += 1
            ratio = float(self.step) / self.step_to_completion
            self.build(ratio)

    def build_topo_old(self, latlon):
        lat = latlon[1, 0, :, :]
        lon = latlon[0, 0, :, :]
        r = self.geom.earth_radius * self.xp.arccos(
            math.sin(self.phim) * self.xp.sin(lat) + math.cos(self.phim) * self.xp.cos(lat) * self.xp.cos(lon - self.lambdam)
        )
        z = self.xp.zeros(lat.shape)
        z[:, :] = self.h0 * self.xp.exp(-(r**2) / self.Dm**2) * self.xp.cos(self.xp.pi * r / self.Dxi) ** 2
        return z

    def build_topo(self, latlon):
        lat = latlon[1]
        lon = latlon[0]
        r = self.geom.earth_radius * self.xp.arccos(
            math.sin(self.phim) * self.xp.sin(lat) + math.cos(self.phim) * self.xp.cos(lat) * self.xp.cos(lon - self.lambdam)
        )

        return self.h0 * self.xp.exp(-(r**2) / self.Dm**2) * self.xp.cos(self.xp.pi * r / self.Dxi) ** 2
