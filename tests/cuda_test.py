import cpu_test
from common.device import Device, CudaDevice


class CudaTestCases(cpu_test.CpuTestCases):
    """
    Base test case for a test case that requires a Cuda device

    There is also a CPU device for comparaison tests
    """
    gpu_device: Device

    def setUp(self) -> None:
        super().setUp()
        import wx_cupy

        wx_cupy.init_wx_cupy()

        if not wx_cupy.cuda_avail:
            reason: str = f"Cannot run test case {str(self)}, no cuda device were found to run the test"
            print(reason)
            self.skipTest(reason)
        self.gpu_device = CudaDevice(list(range(wx_cupy.num_devices)))
