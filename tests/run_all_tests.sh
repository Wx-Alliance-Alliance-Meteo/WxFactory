#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
WX_DIR=${SCRIPT_DIR}/..

cd ${WX_DIR}

${WX_DIR}/tests/unit/run_tests.py || exit -1
mpirun -n 24 ${WX_DIR}/tests/unit/run_mpi_tests.py || exit -1
${WX_DIR}/tests/integration/run_all_integration_tests.sh || exit -1
