import tempfile
import unittest
from pathlib import Path

import netCDF4
import numpy

from scripts.plot_dcmip21 import GRAVITY, RD, load_sections, time_indices


class MountainWavePlotTestCase(unittest.TestCase):
    def test_sheared_background_is_removed_before_equatorial_interpolation(self):
        latitude = numpy.broadcast_to(numpy.array([-2.0, 2.0])[None, :, None], (4, 2, 12))
        longitude = numpy.broadcast_to((numpy.arange(48).reshape(4, 1, 12) + 0.5) * 7.5, latitude.shape)
        elevation = numpy.broadcast_to(numpy.array([500.0, 15000.0, 29500.0])[None, :, None, None], (4, 3, 2, 12))
        radians = numpy.deg2rad(latitude[:, None])
        temperature = 300 * (1 - 2.5e-4 * 400 / GRAVITY * numpy.sin(radians) ** 2)
        density = numpy.ones_like(elevation)
        zonal = 20 * numpy.cos(radians) * numpy.sqrt(600 / temperature * 2.5e-4 * elevation + temperature / 300)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shear.nc"
            with netCDF4.Dataset(path, "w") as dataset:
                dataset.case_number = 22
                for name, size in (("panel", 4), ("z", 3), ("lat", 2), ("lon", 12), ("time", 3)):
                    dataset.createDimension(name, size)
                floor = ("panel", "lat", "lon")
                volume = ("panel", "z", "lat", "lon")
                for name, dimensions, values in (
                    ("time", ("time",), [2400.0, 3600.0, 7200.0]),
                    ("lats", floor, latitude),
                    ("lons", floor, longitude),
                    ("topo", floor, numpy.zeros_like(latitude)),
                    ("elev", volume, elevation),
                    ("rho", ("time",) + volume, density),
                    ("P", ("time",) + volume, RD * density * temperature),
                    ("U", ("time",) + volume, zonal),
                    ("W", ("time",) + volume, numpy.zeros_like(elevation)),
                ):
                    dataset.createVariable(name, "f8", dimensions)[:] = values
            sections = load_sections(path)
            self.assertEqual(sections["case_number"], 22)
            for field in ("tprime", "w", "uprime"):
                numpy.testing.assert_allclose(sections[field], 0.0, atol=1e-12)
            with self.assertRaisesRegex(ValueError, "contains case"):
                load_sections(path, case=21)
        with self.assertRaisesRegex(ValueError, "outside"):
            time_indices(numpy.array([0.0, 100.0]), [2400.0])
