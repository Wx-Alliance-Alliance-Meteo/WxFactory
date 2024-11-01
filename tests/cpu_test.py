import unittest
from common.device import CpuDevice, Device


class CpuTestCases(unittest.TestCase):
    """
    Base test case for a test that requires a CPU device
    """

    cpu_device: Device

    def setUp(self) -> None:
        self.cpu_device = CpuDevice()
