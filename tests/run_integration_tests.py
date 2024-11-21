import os
import sys
import mpi_test
import argparse

main_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(main_project_dir)

from tests.integration.test_integration_state import StateIntegrationTestCases


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("problem", type=str)
    args = parser.parse_args()

    runner = mpi_test.MpiRunner()
    runner.run(StateIntegrationTestCases(os.path.join(main_project_dir, "tests/data", args.problem)))
