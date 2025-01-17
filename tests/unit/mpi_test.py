import sys
import time
from types import TracebackType
from typing import List, Tuple
import unittest
from unittest.result import TestResult
from unittest.signals import registerResult
import warnings

from mpi4py import MPI
import numpy


test_tag: int = 0


def run_test_on_x_process(test: unittest.TestCase, x: int = 0) -> MPI.Comm:
    """
    Make the test run on `x` processes

    :param test: Test that want to restrict the number of processes
    :param x: Required number of processes, 0 for no restriction
    :return: The new communicator to be use for the tests
    """
    if x == 0:
        return MPI.COMM_WORLD

    if x > MPI.COMM_WORLD.size:
        test.fail("Not enough process to run this test")

    global test_tag
    is_needed: bool = MPI.COMM_WORLD.rank < x
    comm: MPI.Comm = MPI.COMM_WORLD.Split(0 if is_needed else 1, test_tag)
    test_tag += 1  # Test_tag is needed two tests don't interfer between each other
    if not is_needed:
        comm.Disconnect()
        test.skipTest("This process is not needed for this test")

    return comm


class _WritelnDecorator(object):
    """Used to decorate file-like objects with a handy 'writeln' method"""

    def __init__(self, stream):
        self.stream = stream

    def __getattr__(self, attr):
        if attr in ("stream", "__getstate__"):
            raise AttributeError(attr)
        return getattr(self.stream, attr)

    def writeln(self, arg=None):
        if arg:
            self.write(arg)
        self.write("\n")


class MpiTestResult(TestResult):
    """
    Result accumulator
    """

    tests_order: List[unittest.TestCase] = None
    rank: int
    results_as_list: List[int]  # 0=Nothing special, 1=error, 2=fail, 3=skip

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.stream = stream
        self.rank = MPI.COMM_WORLD.rank
        if self.rank == 0:
            self.tests_order = []
        self.results_as_list = []

    def startTest(self, test: unittest.TestCase) -> None:
        super().startTest(test)

    def stopTest(self, test: unittest.TestCase) -> None:
        super().stopTest(test)
        if self.rank == 0:
            self.tests_order.append(test)

    def addSuccess(self, test: unittest.TestCase) -> None:
        super().addSuccess(test)
        self.results_as_list.append(0)

    def addSkip(self, test: unittest.TestCase, reason: str) -> None:
        super().addSkip(test, reason)
        self.results_as_list.append(3)

    def addError(self, test: unittest.TestCase, err: tuple[type[BaseException], BaseException, TracebackType]) -> None:
        super().addError(test, err)
        self.results_as_list.append(1)

    def addFailure(
        self, test: unittest.TestCase, err: tuple[type[BaseException], BaseException, TracebackType]
    ) -> None:
        super().addFailure(test, err)
        self.results_as_list.append(2)

    def getDescription(self, test: unittest.TestCase) -> str:
        return str(test)

    def printErrors(self):
        import wx_mpi

        error_index: int = 0  # Current position in the error list
        failure_index: int = 0  # Current position in the failure list
        skipped_tests: List[int] = []
        success_tests: List[int] = []
        for test_index, result in enumerate(self.results_as_list):
            buffer = numpy.empty((MPI.COMM_WORLD.size))
            to_send = numpy.empty((1))
            to_send[0] = result

            MPI.COMM_WORLD.Gather([to_send, MPI.LONG], [buffer, MPI.LONG], 0)
            if self.rank == 0:
                # Receive the error messages
                errors: List[Tuple[int, str]] = []
                for node_index, item in enumerate(buffer):
                    if item == 1 or item == 2:
                        if node_index == 0:
                            if item == 1:
                                errors.append((0, self.errors[error_index][1]))
                            else:
                                errors.append((0, self.failures[failure_index][1]))
                        else:
                            errors.append((node_index, wx_mpi.receive_string_from(node_index, MPI.COMM_WORLD)))
                if len(errors) > 0:
                    self.printErrorList(errors, self.tests_order[test_index])
                else:
                    skipped: bool = numpy.count_nonzero(buffer == 0) == 0
                    if skipped:
                        skipped_tests.append(test_index)
                    else:
                        success_tests.append(test_index)
            else:
                # Send an error message
                if to_send[0] == 1:
                    wx_mpi.send_string_to(self.errors[error_index][1], 0, MPI.COMM_WORLD)
                    error_index += 1
                elif to_send[0] == 2:
                    wx_mpi.send_string_to(self.failures[failure_index][1], 0, MPI.COMM_WORLD)
                    failure_index += 1

        if len(skipped_tests) > 0:
            self.stream.writeln("SKIPPED")
            for skipped_index in skipped_tests:
                self.stream.writeln(self.getDescription(self.tests_order[skipped_index]))

        if len(success_tests) > 0:
            self.stream.writeln("SUCCESS")
            for success_index in success_tests:
                self.stream.writeln(self.getDescription(self.tests_order[success_index]))

    def printErrorList(self, errors: List[Tuple[int, str]], test: unittest.TestCase):
        self.stream.writeln(f"ERROR {self.getDescription(test)}")

        for error in errors:
            self.stream.writeln(f"On node {error[0]}\n")
            self.stream.writeln(f"{error[1]}\n")


class MpiRunner(object):
    """
    Partial reimplementation of the default TextTestRunner class of unittest to add MPI support
    """

    def __init__(self):
        self.stream = _WritelnDecorator(sys.stderr)
        self.descriptions = True
        self.verbosity = False
        self.failfast = False
        self.buffer = False
        self.tb_locals = False
        self.warnings = None

    def run(self, test: unittest.TestSuite | unittest.TestCase) -> TestResult:
        result = MpiTestResult(self.stream, self.descriptions, self.verbosity)
        registerResult(result)
        result.failfast = self.failfast
        result.buffer = self.buffer
        result.tb_locals = self.tb_locals

        # The actual tests are run there
        with warnings.catch_warnings():
            if self.warnings:
                warnings.simplefilter(self.warnings)
                if self.warnings in ["default", "always"]:
                    warnings.filterwarnings(
                        "module", category=DeprecationWarning, message=r"Please use assert\w+ instead."
                    )

            startTime = time.perf_counter()
            startTestRun = getattr(result, "startTestRun", None)
            if startTestRun is not None:
                startTestRun()
            try:
                test(result)
            finally:
                stopTestRun = getattr(result, "stopTestRun", None)
                if stopTestRun is not None:
                    stopTestRun()
            stopTime = time.perf_counter()

        timeTaken = stopTime - startTime

        result.printErrors()
        num_run = result.testsRun

        final_time = MPI.COMM_WORLD.reduce(timeTaken, MPI.MAX, 0)
        if MPI.COMM_WORLD.rank == 0:
            self.stream.writeln(f"Ran {num_run} test{'s' if num_run > 1 else ''} in {final_time:.3f}s\n")

        return result
