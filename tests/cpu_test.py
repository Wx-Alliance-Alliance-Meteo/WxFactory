import unittest
from common.device import CpuDevice, Device


class CpuTestCases(unittest.TestCase):
    cpu_device: Device

    def setUp(self) -> None:
        self.cpu_device = CpuDevice()
