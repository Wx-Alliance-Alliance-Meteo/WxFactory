import unittest

from device import wx_cupy


class CudaTestCases(unittest.TestCase):
    """
    Base test case for a test case that requires a Cuda device
    """

    def setUp(self) -> None:
        super().setUp()

        wx_cupy.load_cupy()

        if not wx_cupy.cuda_avail:
            reason: str = f"Cannot run test case {str(self)}, no cuda device were found to run the test"
            print(reason)
            self.skipTest(reason)
