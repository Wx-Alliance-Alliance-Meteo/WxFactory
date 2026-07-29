#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
WX_DIR=${SCRIPT_DIR}/../..

# ${MPIRUN} -n 2 python tests/integration/run_integration_tests.py small_cartesian2d_problem || exit -1
# ${WX_DIR}/scripts/run.sh -n 1 ./tests/integration/run_integration_tests.py small_cartesian2d_problem || exit -1
${WX_DIR}/scripts/run.sh -n 24 ./tests/integration/run_integration_tests.py small_sw_case5 $@ || exit -1
# ${MPIRUN} -n 24 ${PYTHON} ./tests/integration/run_integration_tests.py small_dcmip21 || exit -1
