import os
import unittest
from unittest.mock import patch

import torch

from wx_factory.common import Configuration, load_default_schema
from wx_factory.simulation import Simulation


def cartesian_config(case_number=3):
    return Configuration(
        f"""
[Grid]
grid_type = cartesian3d
lateral_boundary = wall
vertical_boundary = wall
x0 = 0
x1 = 1000
y0 = 0
y1 = 1000
z0 = 0
z1 = 1000
[Test_case]
equations = euler
case_number = {case_number}
bubble_theta = 300
bubble_rad = 250
gresho_mach = 0.1
[Time_integration]
time_integrator = epi2
dt = 0.01
t_end = 0.02
krylov_size = 10
tolerance = 1e-8
[Spatial_discretization]
num_solpts = 3
num_elements_horizontal = 2
num_elements_vertical = 3
[Output_options]
output_freq = 0
stat_freq = 0
output_dir = /tmp/wxfactory-cartesian-tests
[Device]
pytorch_device = cpu
precision = double
""",
        load_default_schema(),
    )


def cartesian_simulation(config):
    with patch.dict(os.environ, {"WX_FACTORY_DIFFERENTIABLE": "1"}):
        return Simulation(config, quiet=True)


class CartesianTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default_device = torch.get_default_device()
        cls.sim = cartesian_simulation(cartesian_config())

    @classmethod
    def tearDownClass(cls):
        torch.set_default_device(cls.default_device)

    def setUp(self):
        self.generator = torch.Generator().manual_seed(37)

    def random_like(self, value):
        return torch.randn(value.shape, generator=self.generator, dtype=value.dtype, device=value.device)
