#!/usr/bin/env python3

import argparse
import os
import sys

main_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(main_project_dir)

from tests.integration.test_integration_state import StateIntegrationTestCases
import tests.unit.mpi_test as mpi_test

test_cases_dir = "tests/data/integration"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("problems", type=str, nargs="+")
    parser.add_argument("--no-buffer", action="store_true", help="Display output as the test runs")
    args = parser.parse_args()

    runner = mpi_test.MpiRunner(buffer=not args.no_buffer)
    results = []
    for problem in args.problems:
        problem_dir = os.path.join(main_project_dir, test_cases_dir, problem)
        result = runner.run(StateIntegrationTestCases(problem_dir))
        results.append(result)

    if not all(r.wasSuccessful() for r in results):
        raise SystemExit(-1)
