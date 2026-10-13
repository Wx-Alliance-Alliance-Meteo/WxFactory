#!/usr/bin/env python3

import argparse
import os
import re
import sys
import unittest

import torch

main_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(main_project_dir)

from mpi_test import MpiRunner

from tests.unit.common.test_process_topology import ExchangeTest, GatherScatterTest
from tests.unit.geometry.test_metric2d_identities import Metric2DIdentityTestCase
from tests.unit.geometry.test_metric5d_mpi import Metric5DSeamTestCase
from tests.unit.output_managers.compare_zarr_to_nc_mpi import CompareZarrToNcTestCase
from tests.unit.restart.test_restart import (
    Euler3DRestartTestCase,
    ShallowWaterRestartTestCase,
)
from tests.unit.rhs.test_euler_mpi import EulerRhsTestCase
from tests.unit.rhs.test_prescribed_transport_mpi import PrescribedTransportTestCase
from tests.unit.rhs.test_terrain_ramp_mpi import TerrainRampTestCase
from tests.unit.solvers.test_exponential_mpi import ExponentialMpiTestCase


def add_test(suite: unittest.TestSuite, test: unittest.TestCase, test_re: re.Pattern | None):
    if test_re is None or test_re.search(str(test)) is not None:
        suite.addTest(test)


def load_tests(test_name: str):
    suite = unittest.TestSuite()

    num_devices = torch.cuda.device_count()
    device_names = ["cpu", "cuda"] if num_devices > 0 else ["cpu"]

    test_re = re.compile(test_name, re.IGNORECASE)
    for name in unittest.defaultTestLoader.getTestCaseNames(EulerRhsTestCase):
        add_test(suite, EulerRhsTestCase(6, name, "cpu"), test_re)
        add_test(suite, EulerRhsTestCase(24, name, "cpu", optional=True), test_re)

    for name in unittest.defaultTestLoader.getTestCaseNames(PrescribedTransportTestCase):
        add_test(suite, PrescribedTransportTestCase(6, name, "cpu"), test_re)
    for name in unittest.defaultTestLoader.getTestCaseNames(TerrainRampTestCase):
        add_test(suite, TerrainRampTestCase(6, name, "cpu"), test_re)

    for depth in ("shallow", "deep"):
        for name in unittest.defaultTestLoader.getTestCaseNames(Metric5DSeamTestCase):
            add_test(suite, Metric5DSeamTestCase(6, name, "cpu", depth=depth), test_re)
            add_test(suite, Metric5DSeamTestCase(24, name, "cpu", optional=True, depth=depth), test_re)
    for name in (
        "test_volume_factor_matches_the_covariant_determinant",
        "test_metrics_are_inverses",
        "test_global_integral_recovers_the_area_of_the_sphere",
    ):
        add_test(suite, Metric2DIdentityTestCase(6, name, "cpu"), test_re)

    add_test(suite, ShallowWaterRestartTestCase(6, "test_read_restart", "cpu"), test_re)
    add_test(suite, Euler3DRestartTestCase(6, "test_read_restart", "cpu"), test_re)

    add_test(suite, ShallowWaterRestartTestCase(24, "test_read_restart", "cpu", optional=True), test_re)
    add_test(suite, Euler3DRestartTestCase(24, "test_read_restart", "cpu", optional=True), test_re)

    add_test(suite, ShallowWaterRestartTestCase(24, "test_multisize", "cpu", optional=True), test_re)
    add_test(suite, Euler3DRestartTestCase(24, "test_multisize", "cpu", optional=True), test_re)

    for dev in device_names:
        add_test(suite, ExchangeTest("vector2d_1d_shape1d", dev), test_re)
        add_test(suite, ExchangeTest("vector2d_1d_shape2d", dev), test_re)
        add_test(suite, ExchangeTest("vector2d_2d_shape1d", dev), test_re)
        add_test(suite, ExchangeTest("vector2d_2d_shape3d", dev), test_re)
        add_test(suite, ExchangeTest("vector3d_1d_shape1d", dev), test_re)
        add_test(suite, ExchangeTest("vector3d_1d_shape2d", dev), test_re)
        add_test(suite, ExchangeTest("vector3d_3d_shape1d", dev), test_re)
        add_test(suite, ExchangeTest("vector3d_4d_shape3d", dev), test_re)
        add_test(suite, ExchangeTest("scalar_1d_shape1d", dev), test_re)
        add_test(suite, ExchangeTest("scalar_1d_shape2d", dev), test_re)
        add_test(suite, ExchangeTest("scalar_1d_shape3d", dev), test_re)
        add_test(suite, ExchangeTest("scalar_2d_shape1d", dev), test_re)
        add_test(suite, ExchangeTest("scalar_2d_shape2d", dev), test_re)

        add_test(suite, GatherScatterTest(6, "gather_scatter_2d", dev), test_re)
        add_test(suite, GatherScatterTest(6, "gather_scatter_elem_2d", dev), test_re)
        add_test(suite, GatherScatterTest(6, "gather_scatter_3d", dev), test_re)
        add_test(suite, GatherScatterTest(6, "gather_scatter_elem_3d", dev), test_re)
        add_test(suite, GatherScatterTest(6, "gather_scatter_elem_4d", dev), test_re)

        add_test(suite, GatherScatterTest(24, "gather_scatter_2d", dev, optional=True), test_re)
        add_test(suite, GatherScatterTest(24, "gather_scatter_elem_2d", dev, optional=True), test_re)
        add_test(suite, GatherScatterTest(24, "gather_scatter_3d", dev, optional=True), test_re)
        add_test(suite, GatherScatterTest(24, "gather_scatter_elem_3d", dev, optional=True), test_re)
        add_test(suite, GatherScatterTest(24, "gather_scatter_elem_4d", dev, optional=True), test_re)

        add_test(suite, GatherScatterTest(54, "gather_scatter_2d", dev, optional=True), test_re)
        add_test(suite, GatherScatterTest(54, "gather_scatter_elem_2d", dev, optional=True), test_re)
        add_test(suite, GatherScatterTest(54, "gather_scatter_3d", dev, optional=True), test_re)
        add_test(suite, GatherScatterTest(54, "gather_scatter_elem_3d", dev, optional=True), test_re)
        add_test(suite, GatherScatterTest(54, "gather_scatter_elem_4d", dev, optional=True), test_re)

        add_test(suite, GatherScatterTest(24, "fail_wrong_num_proc", dev, optional=True), test_re)  # Needs 24+ procs
        add_test(suite, GatherScatterTest(6, "fail_not_square", dev), test_re)
        add_test(suite, GatherScatterTest(6, "fail_not_cube", dev), test_re)
        add_test(suite, GatherScatterTest(6, "fail_wrong_num_dim", dev), test_re)

    add_test(
        suite, ExponentialMpiTestCase(2, "test_distributed_phi_combination_against_dense_reference", "cpu"), test_re
    )

    add_test(suite, CompareZarrToNcTestCase(6, "test_compare_zarr_to_nc"), test_re)

    return suite


def regular_run(runner, args):
    result = runner.run(load_tests(args.test_name))

    if not result.wasSuccessful():
        raise SystemExit(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage="mpirun -n [6+] %(prog)s [options]", description="Run the suite of multi-process tests."
    )
    parser.add_argument(
        "test_name",
        nargs="?",
        default="",
        type=str,
        help="Will only run tests whose name or type matches this regular expression.",
    )
    parser.add_argument("--no-buffer", action="store_true", help="Print all test output to terminal")
    parser.add_argument("--failfast", action="store_true", help="Stop running tests after 1 failure")
    args = parser.parse_args()

    runner = MpiRunner(buffer=not args.no_buffer, verbosity=0, failfast=args.failfast)

    regular_run(runner, args)
