import cpu_test
from common.device import Device, CudaDevice

class CudaTestCases(cpu_test.CpuTestCases):
    gpu_device: Device

    def setUp(self) -> None:
        super().setUp()
        from wx_cupy import num_devices, cuda_avail
        if not cuda_avail:
            reason: str = f'Cannot run test case {self._testMethodName}, no cuda device were found to run the test'
            print(reason)
            self.skipTest(reason)
        self.gpu_device = CudaDevice(list(range(num_devices)))
