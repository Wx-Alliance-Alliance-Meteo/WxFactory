#!/usr/bin/env python3

import argparse
import os
import sys

from mpi4py import MPI

main_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(main_project_dir)

from tests.integration.test_integration_state import StateIntegrationTestCases
from tests.unit import mpi_test

test_cases_dir = "tests/data/integration"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("problems", type=str, nargs="+")
    parser.add_argument("--no-buffer", action="store_true", help="Display output as the test runs")
    parser.add_argument("--failfast", action="store_true", help="Stop running tests after 1 failure")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device (cpu/cuda) on which the test should run",
    )
    args = parser.parse_args()

    runner = mpi_test.MpiRunner(buffer=not args.no_buffer, verbosity=0, failfast=args.failfast)
    results = []
    for problem in args.problems:
        problem_dir = os.path.join(main_project_dir, test_cases_dir, problem)
        if MPI.COMM_WORLD.rank == 0:
            print(f"Running integration test for {os.path.basename(os.path.normpath(problem_dir))}")
        result = runner.run(StateIntegrationTestCases(problem_dir, device_name=args.device))
        results.append(result)

    if not all(r.wasSuccessful() for r in results):
        raise SystemExit(-1)
