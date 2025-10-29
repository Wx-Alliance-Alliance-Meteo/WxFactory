#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
TMP_OUT=a.out

if nvcc -arch native ${SCRIPT_DIR}/test_nvcc.cu && ./${TMP_OUT}; then
    echo "yes"
else
    echo "no"
fi

rm -f ${TMP_OUT}
