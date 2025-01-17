#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
WX_DIR=${SCRIPT_DIR}/..

cd ${WX_DIR}

${WX_DIR}/tests/unit/run_tests.py
mpirun -n 6 ${WX_DIR}/tests/unit/run_mpi_tests.py
${WX_DIR}/tests/integration/run_all_integration_tests.sh
