#!/usr/bin/env bash

# This script tries to determine whether the loaded MPI is CUDA-aware
# - It tries to compile a program that uses MPI extensions. If that works, the program will tell us
#   if there is CUDA support
# - It tries to use the `ompi_info` utility to get MPI config parameters, but this is only valid for OpenMPI,
#   so if it's another MPI version, we still don't know.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
TMP_OUT=a.out

if ! which mpirun 2>/dev/null; then
    echo "Can't find MPI"
    exit 1
fi

if mpicc ${SCRIPT_DIR}/cuda_awareness.c && ./${TMP_OUT}; then
    if ! ./${TMP_OUT}; then
        echo "No"
        result=4
    elif ./${TMP_OUT} | grep "does not\|cannot"; then
        echo "No"
        result=3
    else
        echo "Yes"
        result=0
    fi
else
    echo "Probably not"
    result=2
fi

rm -f ${TMP_OUT}

[ "x0" == "x${result}" ] && exit $result

result=-1
if which ompi_info 2>/dev/null; then 
    output="$(ompi_info --parsable --all | grep mpi_built_with_cuda_support:value)"
    echo ${output}
    if echo ${output} | grep true; then
        echo "Yes"
        result=0
    fi
fi

[ "x0" != "x${result}" ] && echo "No"

exit $result
