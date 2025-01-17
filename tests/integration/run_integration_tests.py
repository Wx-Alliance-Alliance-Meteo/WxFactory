#!/usr/bin/env python3

import argparse
import os
import sys

main_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
main_module_dir = os.path.join(main_project_dir, "wx_factory")
sys.path.append(main_project_dir)
sys.path.append(main_module_dir)

from tests.integration.test_integration_state import StateIntegrationTestCases
import tests.unit.mpi_test as mpi_test


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("problems", type=str, nargs="+")
    args = parser.parse_args()

    runner = mpi_test.MpiRunner()
    results = []
    for problem in args.problems:
        results.append(runner.run(StateIntegrationTestCases(os.path.join(main_project_dir, "tests/data", problem))))

    if not all(r.wasSuccessful() for r in results):
        sys.exit(-1)
