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


def run_test_on_x_process(test: unittest.TestCase, x: int = 0, optional: bool = False) -> MPI.Comm:
    """
    Make the test run on `x` processes

    :param test: Test that want to restrict the number of processes
    :param x: Required number of processes, 0 for no restriction
    :return: The new communicator to be used for the tests
    """
    if x == 0:
        return MPI.COMM_WORLD

    if x > MPI.COMM_WORLD.size:
        if MPI.COMM_WORLD.rank == 0 and not optional:
            test.fail("Not enough process to run this test")
        else:
            test.skipTest(f"Not enough processes to run this test [optional={optional}]")

    is_needed = MPI.COMM_WORLD.rank < x
    comm = MPI.COMM_WORLD.Split(0 if is_needed else 1)

    if not is_needed:
        comm.Disconnect()
        test.skipTest("This process is not needed for this test")

    return comm


class MpiTestCase(unittest.TestCase):
    def __init__(self, num_procs: int, methodName="runTest", optional: bool = False):
        super().__init__(methodName)
        self.num_procs = num_procs
        self.optional = optional

    def setUp(self):
        super().setUp()
        self.comm = run_test_on_x_process(self, self.num_procs, self.optional)


class MpiTestSuite(unittest.TestSuite):
    def run(self, result, debug=False):
        for test in self:
            if MPI.COMM_WORLD.rank == 0:
                print(f"running {test}", flush=True)
            test.run(result)
        return result


class MpiTestResult(TestResult):
    """
    Custom result accumulator: see addCorrectResult.
    """

    _SUCCESS = 0
    _ERROR = 1
    _FAILURE = 2
    _SKIP = 3
    _UNEXPECTED_SUCCESS = 4

    tests_order: List[unittest.TestCase] = None
    results_as_list: List[int]  # 0=Nothing special, 1=error, 2=fail, 3=skip

    def addSuccess(self, test: unittest.TestCase) -> None:
        self.addCorrectResult(test, MpiTestResult._SUCCESS)

    def addSkip(self, test: unittest.TestCase, reason: str) -> None:
        self.addCorrectResult(test, MpiTestResult._SKIP, reason=reason)

    def addError(self, test: unittest.TestCase, err: tuple[type[BaseException], BaseException, TracebackType]) -> None:
        self.addCorrectResult(test, MpiTestResult._ERROR, err=err)

    def addFailure(
        self, test: unittest.TestCase, err: tuple[type[BaseException], BaseException, TracebackType]
    ) -> None:
        self.addCorrectResult(test, MpiTestResult._FAILURE, err=err)

    def addExpectedFailure(self, test, err):
        self.addCorrectResult(test, MpiTestResult._SUCCESS, err=err)

    def addUnexpectedSuccess(self, test):
        self.addCorrectResult(test, MpiTestResult._UNEXPECTED_SUCCESS)

    def addCorrectResult(self, test: unittest.TestCase, result: int, **kwargs) -> None:
        """
        All MPI processes will register the same result:
            - If any process has an error -> Error
            - else if any process has a failure -> Failure
            - else if any process has an unexpected success -> Unexpected success
            - else if allll processes have skipped -> Skip
            - otherwise it's a success!
        """

        err = (None, None, None)
        reason = (test, None)
        if "err" in kwargs:
            err = kwargs["err"]
        if "reason" in kwargs:
            reason = kwargs["reason"]

        all_results = numpy.array(MPI.COMM_WORLD.allgather(result))
        if numpy.any(all_results == MpiTestResult._ERROR):
            super().addError(test, err)
        elif numpy.any(all_results == MpiTestResult._FAILURE):
            super().addFailure(test, err)
        elif numpy.any(all_results == MpiTestResult._UNEXPECTED_SUCCESS):
            super().addUnexpectedSuccess(test)
        elif numpy.all(all_results == MpiTestResult._SKIP):
            super().addSkip(test, reason)
        else:
            super().addSuccess(test)


class MpiRunner(unittest.TextTestRunner):
    """
    Partial reimplementation of the default TextTestRunner class of unittest to add MPI support
    """

    def run(self, test: unittest.TestSuite | unittest.TestCase) -> TestResult:
        result = MpiTestResult(self.stream, self.descriptions, self.verbosity)
        registerResult(result)
        result.failfast = self.failfast
        result.buffer = self.buffer
        result.tb_locals = self.tb_locals

        # The actual tests are run there
        start_time = time.perf_counter()
        result.startTestRun()
        try:
            test(result)
        finally:
            result.stopTestRun()
        stop_time = time.perf_counter()

        time_taken = stop_time - start_time

        num_run = result.testsRun
        final_time = MPI.COMM_WORLD.reduce(time_taken, MPI.MAX, 0)

        if MPI.COMM_WORLD.rank == 0:
            # Print failed and skipped tests, if any
            if len(result.skipped) > 0:
                skipped_tests = "\n  ".join([f"{r[0]}" for r in result.skipped])
                self.stream.writeln(f"\nskipped test(s): \n  {skipped_tests}")

            if len(result.errors) + len(result.failures) + len(result.unexpectedSuccesses) > 0:
                failed_tests = "\n  ".join(
                    [f"{r[0]}" for r in result.errors + result.failures] + [f"{r}" for r in result.unexpectedSuccesses]
                )
                self.stream.writeln(f"\nfailed test(s): \n  {failed_tests}")

            # Print timing
            self.stream.writeln(f"\nRan {num_run} test{'s' if num_run > 1 else ''} in {final_time:.3f}s")

        return result
