import time
import unittest
from typing import Any
from unittest.runner import _WritelnDecorator

from mpi4py import MPI

from wx_factory.context import Context


class WxTestResult(unittest.TextTestResult):
    def __init__(self, stream: _WritelnDecorator, descriptions: bool, verbosity: int) -> None:
        super().__init__(stream, descriptions, verbosity)
        self.verbose = True

    def startTest(self, test: unittest.TestCase) -> None:
        super().startTest(test)
        self.t0 = time.time()
        if self.verbose:
            self.stream.write(f"Running {test} ... \n    ")
            self.stream.flush()

    def addSkip(self, test: unittest.TestCase, reason: str) -> None:
        super().addSkip(test, reason)
        if self.verbose:
            self.stream.write("SKIP ")

    def stopTest(self, test: unittest.TestCase) -> None:
        self.t1 = time.time()
        super().stopTest(test)
        if self.verbose:
            self.stream.writeln(f"({(self.t1 - self.t0) * 1000:.1f} ms)")

    def addSuccess(self, test: unittest.TestCase) -> None:
        super().addSuccess(test)
        if self.verbose:
            self.stream.write("PASS ")

    def addError(
        self, test: unittest.TestCase, err: tuple[type[BaseException], BaseException, Any] | tuple[None, None, None]
    ) -> None:
        super().addError(test, err)
        if self.verbose:
            self.stream.write("ERROR ")

    def addFailure(
        self, test: unittest.TestCase, err: tuple[type[BaseException], BaseException, Any] | tuple[None, None, None]
    ) -> None:
        super().addFailure(test, err)
        if self.verbose:
            self.stream.write("FAIL ")

    def addExpectedFailure(
        self, test: unittest.TestCase, err: tuple[type[BaseException], BaseException, Any] | tuple[None, None, None]
    ) -> None:
        super().addExpectedFailure(test, err)
        if self.verbose:
            self.stream.write("PASS ")

    def addUnexpectedSuccess(self, test: unittest.TestCase) -> None:
        super().addUnexpectedSuccess(test)
        if self.verbose:
            self.stream.write("FAIL ")


class WxTestCase(unittest.TestCase):
    def __init__(self, methodName: str, device_name: str = "cpu") -> None:
        """
        :param device_name: Name of the device where we want to run the test. Can be either "cpu" or "cuda"
        """
        super().__init__(methodName)
        self.device_name = device_name
        self.comm = MPI.COMM_WORLD  # Default value. Should be overridden by subclasses if needed

    def __str__(self):
        return f"{self.__class__.__qualname__}.{self._testMethodName}.{self.device_name}"

    def setUp(self) -> None:
        super().setUp()

        # Only the Pytorch backend remains. "cuda" asks for GPU tensors (skip if unavailable);
        # anything else keeps the tensors on the host.
        device_type = "cuda" if self.device_name == "cuda" else "cpu"
        self.context = Context(self.comm, device_type=device_type)
        if self.device_name == "cuda" and self.context.torch_device.type != "cuda":
            self.skipTest("No GPU available for the Pytorch backend")


class WxTestRunner(unittest.TextTestRunner):
    resultclass = WxTestResult
