import unittest

from mpi4py import MPI

from wx_factory.device import CpuDevice, CudaDevice, PytorchDevice


class WxTestCase(unittest.TestCase):
    def __init__(self, methodName: str, device_name: str = "cpu") -> None:
        super().__init__(methodName)
        self.device_name = device_name
        self.comm = MPI.COMM_WORLD

    def __str__(self):
        return f"{self.__class__.__qualname__}.{self._testMethodName}.{self.device_name}"

    def setUp(self) -> None:
        super().setUp()

        if self.device_name == "torch":
            self.device = PytorchDevice(self.comm)
        elif self.device_name == "cuda":
            try:
                self.device = CudaDevice(self.comm)
            except ValueError:
                self.skipTest("Could not create a CudaDevice, probably no GPU available")
        else:
            self.device = CpuDevice(self.comm)


class WxTestSuite(unittest.TestSuite):
    def run(self, result, debug=False):
        for test in self:
            # print(f"running {test.__class__}.{test._testMethodName}")
            print(f"Running {test}")
            test.run(result)
        return result
