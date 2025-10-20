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

test_cases_dir = "tests/data/integration"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("problems", type=str, nargs="+")
    args = parser.parse_args()

    runner = mpi_test.MpiRunner(buffer=True)
    results = []
    for problem in args.problems:
        problem_dir = os.path.join(main_project_dir, test_cases_dir, problem)
        result = runner.run(StateIntegrationTestCases(problem_dir))
        results.append(result)

    if not all(r.wasSuccessful() for r in results):
        raise SystemExit(-1)
